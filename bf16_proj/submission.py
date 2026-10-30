# okc: problem=muse_bf16_proj
"""Dense projections out = x @ W[:n].T, one kernel launch per call.

At 1-32 tokens these are memory-bound: the time floor is reading W once at 4.8 TB/s.

CUDA kernel (m <= 32), compiled with load_inline in prepare():
- register-direct tensor cores: each lane loads 16 contiguous bytes of a W row with
  ld.global.nc.L1::no_allocate and feeds them straight to mma.sync m16n8k16. A dot
  product does not care about the order of k, so k is relabelled to make each lane's
  16 bytes exactly its fragment slots; x is read with the same relabelling;
- the 8 warps of a CTA split K and add their partials in shared memory;
- small-N layers (router, sliced unpooler) also split K across the CTAs of a thread
  block cluster; the leader CTA reads the others' partials from their shared memory
  (DSMEM). One launch, no atomics, no workspace;
- router: W = hi + lo, two BF16 halves stored side by side as one [256, 16384] matrix;
  x is read at k mod 8192, so both halves meet the same x in the FP32 accumulator.
Triton (_skinny_gemm) and cuBLAS stay as fallbacks; PLAN says which runs where.

BENCH = True (check runs): prepare() measures HBM read bandwidth, then times every
candidate per scored shape the way the judge does (CUDA graph replay, cold L2),
validates it, prints a table and uses the fastest.
"""

import importlib
import math
import platform
import time
from typing import TypedDict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

BENCH = True


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


# ---------------------------------------------------------------- CUDA kernels

CUDA_KERNELS = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <stdint.h>
namespace cg = cooperative_groups;

__device__ __forceinline__ int4 ld_stream(const void* p) {
    int4 v;
    asm("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ void mma_bf16(float* c, int a0, int a1, int a2, int a3, int b0, int b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// out[M, N] = x[M, KX] @ W[N, KW]^T for M <= 8 * MT, with W read at column k and x at k % KX.
// grid (N / (16 * RT), ksplit), 256 threads; ksplit > 1 needs a (1, ksplit, 1) cluster.
// Lane (g = lane / 4, t = lane % 4) holds W rows g and g + 8 of each 16-row tile and token g
// of each 8-token tile; per 32-wide k block it loads k in [8t, 8t + 8) as one int4. MMA step 0
// uses k 8t + 0..3 (fragment slots 2t, 2t+1, 2t+8, 2t+9), step 1 uses k 8t + 4..7.
template <int RT, int MT, int U, bool OUT_F32>
__global__ void __launch_bounds__(256) skinny_mma_kernel(
        const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
        void* __restrict__ out, int M, int N, int KX, int KW, int k_cta) {
    constexpr int E = RT * MT * 4 * 32;  // accumulator floats per warp
    __shared__ float red[8][E];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int row0 = blockIdx.x * 16 * RT;
    const int kw = k_cta >> 3;  // k per warp
    const int k_begin = blockIdx.y * k_cta + warp * kw;

    float acc[RT][MT][4];
#pragma unroll
    for (int rt = 0; rt < RT; ++rt)
#pragma unroll
        for (int nt = 0; nt < MT; ++nt)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[rt][nt][j] = 0.f;

    const __nv_bfloat16* wrow[RT];
#pragma unroll
    for (int rt = 0; rt < RT; ++rt) wrow[rt] = w + (size_t)(row0 + rt * 16 + g) * KW + t * 8;
    const size_t w8 = (size_t)8 * KW;  // row g + 8
    const __nv_bfloat16* xrow[MT];
    bool xok[MT];
#pragma unroll
    for (int nt = 0; nt < MT; ++nt) {
        const int tok = nt * 8 + g;
        xok[nt] = tok < M;
        xrow[nt] = x + (size_t)(xok[nt] ? tok : 0) * KX + t * 8;
    }

    for (int kb = k_begin; kb < k_begin + kw; kb += 32 * U) {
        const int kx = kb % KX;
        int4 wa[U][RT], wb[U][RT], xv[U][MT];
#pragma unroll
        for (int u = 0; u < U; ++u) {
#pragma unroll
            for (int rt = 0; rt < RT; ++rt) {
                wa[u][rt] = ld_stream(wrow[rt] + kb + u * 32);
                wb[u][rt] = ld_stream(wrow[rt] + w8 + kb + u * 32);
            }
#pragma unroll
            for (int nt = 0; nt < MT; ++nt)
                xv[u][nt] = xok[nt] ? __ldg(reinterpret_cast<const int4*>(xrow[nt] + kx + u * 32))
                                    : make_int4(0, 0, 0, 0);
        }
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int rt = 0; rt < RT; ++rt)
#pragma unroll
                for (int nt = 0; nt < MT; ++nt) {
                    mma_bf16(acc[rt][nt], wa[u][rt].x, wb[u][rt].x, wa[u][rt].y, wb[u][rt].y,
                             xv[u][nt].x, xv[u][nt].y);
                    mma_bf16(acc[rt][nt], wa[u][rt].z, wb[u][rt].z, wa[u][rt].w, wb[u][rt].w,
                             xv[u][nt].z, xv[u][nt].w);
                }
    }

    // The 8 warps of the CTA: sum their partials into red[0].
#pragma unroll
    for (int rt = 0; rt < RT; ++rt)
#pragma unroll
        for (int nt = 0; nt < MT; ++nt)
#pragma unroll
            for (int j = 0; j < 4; ++j) red[warp][((rt * MT + nt) * 4 + j) * 32 + lane] = acc[rt][nt][j];
    __syncthreads();
    for (int e = threadIdx.x; e < E; e += 256) {
        float s = 0.f;
#pragma unroll
        for (int q = 0; q < 8; ++q) s += red[q][e];
        red[0][e] = s;
    }

    const int ksplit = gridDim.y;
    bool writer = true;
    if (ksplit > 1) {  // the cluster's CTAs: rank 0 adds the others' red[0] through DSMEM
        cg::cluster_group cluster = cg::this_cluster();
        cluster.sync();
        writer = cluster.block_rank() == 0;
        if (writer)
            for (int e = threadIdx.x; e < E; e += 256)
                for (int r = 1; r < ksplit; ++r) red[0][e] += cluster.map_shared_rank(&red[0][0], r)[e];
        cluster.sync();  // the others' shared memory must outlive rank 0's reads
    } else {
        __syncthreads();
    }
    if (!writer) return;
    for (int e = threadIdx.x; e < E; e += 256) {
        const int ln = e & 31, j = (e >> 5) & 3, nt = (e >> 7) % MT, rt = (e >> 7) / MT;
        const int row = row0 + rt * 16 + (ln >> 2) + (j >= 2 ? 8 : 0);
        const int tok = nt * 8 + 2 * (ln & 3) + (j & 1);
        if (tok < M) {
            if (OUT_F32) reinterpret_cast<float*>(out)[(size_t)tok * N + row] = red[0][e];
            else reinterpret_cast<__nv_bfloat16*>(out)[(size_t)tok * N + row] = __float2bfloat16(red[0][e]);
        }
    }
}

// Bandwidth probe: read n16 int4s, either grid-strided (chunked = 0) or one contiguous chunk per CTA.
template <int UNROLL>
__global__ void read_bw_kernel(const int4* __restrict__ p, long long n16, int chunked, int* sink) {
    long long i, step, end;
    if (chunked) {
        const long long chunk = (n16 + gridDim.x - 1) / gridDim.x;
        i = blockIdx.x * chunk + threadIdx.x;
        end = min(n16, (blockIdx.x + 1) * chunk);
        step = blockDim.x;
    } else {
        i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        end = n16;
        step = (long long)gridDim.x * blockDim.x;
    }
    int acc = 0;
    for (; i + (UNROLL - 1) * step < end; i += UNROLL * step) {
        int4 v[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) v[u] = ld_stream(p + i + u * step);
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    for (; i < end; i += step) {
        const int4 v = ld_stream(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x7fffffff) sink[0] = acc;  // keeps the loads alive
}

template <int RT, int MT, int U>
static cudaError_t launch_rmu(cudaLaunchConfig_t* cfg, bool f32, const __nv_bfloat16* x,
                              const __nv_bfloat16* w, void* out, int M, int N, int KX, int KW, int k_cta) {
    if (f32) return cudaLaunchKernelEx(cfg, skinny_mma_kernel<RT, MT, U, true>, x, w, out, M, N, KX, KW, k_cta);
    return cudaLaunchKernelEx(cfg, skinny_mma_kernel<RT, MT, U, false>, x, w, out, M, N, KX, KW, k_cta);
}

template <int RT, int MT>
static cudaError_t launch_rm(cudaLaunchConfig_t* cfg, int u, bool f32, const __nv_bfloat16* x,
                             const __nv_bfloat16* w, void* out, int M, int N, int KX, int KW, int k_cta) {
    if (u == 8) return launch_rmu<RT, MT, 8>(cfg, f32, x, w, out, M, N, KX, KW, k_cta);
    if (u == 2) return launch_rmu<RT, MT, 2>(cfg, f32, x, w, out, M, N, KX, KW, k_cta);
    return launch_rmu<RT, MT, 4>(cfg, f32, x, w, out, M, N, KX, KW, k_cta);
}

template <int RT>
static cudaError_t launch_r(cudaLaunchConfig_t* cfg, int mt, int u, bool f32, const __nv_bfloat16* x,
                            const __nv_bfloat16* w, void* out, int M, int N, int KX, int KW, int k_cta) {
    if (mt == 1) return launch_rm<RT, 1>(cfg, u, f32, x, w, out, M, N, KX, KW, k_cta);
    if (mt == 2) return launch_rm<RT, 2>(cfg, u, f32, x, w, out, M, N, KX, KW, k_cta);
    return launch_rm<RT, 4>(cfg, u, f32, x, w, out, M, N, KX, KW, k_cta);
}

cudaError_t skinny_mma_launch(const void* x, const void* w, void* out, bool f32, int M, int N, int KX,
                              int KW, int rt, int u, int ksplit, cudaStream_t stream) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(N / (16 * rt), ksplit, 1);
    cfg.blockDim = dim3(256, 1, 1);
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 1;
    attr[0].val.clusterDim.y = ksplit;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = ksplit > 1 ? 1 : 0;
    const int mt = M <= 8 ? 1 : M <= 16 ? 2 : 4;
    const int k_cta = KW / ksplit;
    auto xp = static_cast<const __nv_bfloat16*>(x);
    auto wp = static_cast<const __nv_bfloat16*>(w);
    if (rt == 2) return launch_r<2>(&cfg, mt, u, f32, xp, wp, out, M, N, KX, KW, k_cta);
    return launch_r<1>(&cfg, mt, u, f32, xp, wp, out, M, N, KX, KW, k_cta);
}

cudaError_t read_bw_launch(const void* p, long long bytes, int grid, int block, int unroll, int chunked,
                           int* sink, cudaStream_t stream) {
    const long long n16 = bytes / 16;
    auto q = static_cast<const int4*>(p);
    if (unroll == 16) read_bw_kernel<16><<<grid, block, 0, stream>>>(q, n16, chunked, sink);
    else if (unroll == 4) read_bw_kernel<4><<<grid, block, 0, stream>>>(q, n16, chunked, sink);
    else read_bw_kernel<8><<<grid, block, 0, stream>>>(q, n16, chunked, sink);
    return cudaGetLastError();
}
"""

CUDA_TORCH = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

cudaError_t skinny_mma_launch(const void* x, const void* w, void* out, bool f32, int M, int N, int KX,
                              int KW, int rt, int u, int ksplit, cudaStream_t stream);
cudaError_t read_bw_launch(const void* p, long long bytes, int grid, int block, int unroll, int chunked,
                           int* sink, cudaStream_t stream);

void skinny_mma(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t kx, int64_t rt, int64_t u,
                int64_t ksplit) {
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && out.is_contiguous());
    const cudaError_t err = skinny_mma_launch(
        x.data_ptr(), w.data_ptr(), out.data_ptr(), out.scalar_type() == at::kFloat, x.size(0),
        out.size(1), kx, w.size(1), rt, u, ksplit, at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(err == cudaSuccess, "skinny_mma: ", cudaGetErrorString(err));
}

void read_bw(torch::Tensor buf, int64_t bytes, int64_t grid, int64_t block, int64_t unroll, int64_t chunked,
             torch::Tensor sink) {
    const cudaError_t err = read_bw_launch(buf.data_ptr(), bytes, grid, block, unroll, chunked,
                                           sink.data_ptr<int>(), at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(err == cudaSuccess, "read_bw: ", cudaGetErrorString(err));
}
"""

CPP_DECLS = """
void skinny_mma(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t kx, int64_t rt, int64_t u,
                int64_t ksplit);
void read_bw(torch::Tensor buf, int64_t bytes, int64_t grid, int64_t block, int64_t unroll, int64_t chunked,
             torch::Tensor sink);
"""

_MOD = None


def _cuda():
    """Compile the CUDA kernels once (nvcc, about a minute); None if that fails."""
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load_inline
        t0 = time.time()
        try:
            _MOD = load_inline(
                name="muse_bf16_proj_v2", cpp_sources=CPP_DECLS, cuda_sources=CUDA_KERNELS + CUDA_TORCH,
                functions=["skinny_mma", "read_bw"], extra_cuda_cflags=["-O3"], verbose=False)
            print(f"CUDA compiled in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001 - fall back to Triton / cuBLAS and say why
            print(f"CUDA compile failed after {time.time() - t0:.0f}s: {str(e)[-3000:]}", flush=True)
            _MOD = False
    return _MOD or None


# ---------------------------------------------------------------- Triton kernel

@triton.jit
def _skinny_gemm(x_ptr, w_ptr, out_ptr, ws_ptr, cnt_ptr,
                 M, N, K_X, K_W, stride_xm, stride_wn, stride_om, SPLIT_K,
                 BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
                 OUT_FP32: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    mask_n = offs_n < N
    mask_m = offs_m < M

    k_per = K_W // SPLIT_K
    k_lo = pid_k * k_per
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for k0 in range(k_lo, k_lo + k_per, BLOCK_K):
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + (k0 + offs_k)[None, :],
                    mask=mask_n[:, None], other=0.0)
        kx = k0 % K_X  # the router's [hi | lo] weight reuses x for both halves
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + (kx + offs_k)[None, :],
                    mask=mask_m[:, None], other=0.0)
        acc += tl.dot(w, tl.trans(x))

    out_ptrs = out_ptr + offs_m[None, :] * stride_om + offs_n[:, None]
    out_mask = mask_n[:, None] & mask_m[None, :]
    if SPLIT_K == 1:
        if OUT_FP32:
            tl.store(out_ptrs, acc, mask=out_mask)
        else:
            tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)
    else:
        m_pad = tl.num_programs(2) * BLOCK_M
        tile_ws = offs_m[None, :] * N + offs_n[:, None]
        tl.store(ws_ptr + pid_k * m_pad * N + tile_ws, acc, mask=mask_n[:, None])
        tl.debug_barrier()  # every thread's partial is written before the release below
        tile = pid_m * tl.num_programs(0) + pid_n
        done = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel", scope="gpu")
        if done == SPLIT_K - 1:  # last CTA of this tile: sum the partials
            total = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for s in range(0, SPLIT_K):
                total += tl.load(ws_ptr + s * m_pad * N + tile_ws, mask=mask_n[:, None],
                                 other=0.0, cache_modifier=".cg")
            if OUT_FP32:
                tl.store(out_ptrs, total, mask=out_mask)
            else:
                tl.store(out_ptrs, total.to(tl.bfloat16), mask=out_mask)
            tl.store(cnt_ptr + tile, 0)  # ready for the next call


WS_FLOATS = 16 << 20  # 64 MB of FP32 split-K partials (Triton path)
COUNTERS = 1 << 16
BUCKETS = (1, 8, 16, 32, 128, 256)  # the scored token counts
LAYER_NAMES = ("qkv_gate", "o", "pooler", "unpooler", "router")

# (layer, bucket) -> ("cuda", (rt, u, ksplit)), ("triton", (bm, bn, bk, split, warps, stages))
# or ("torch", None). Filled from BENCH runs.
PLAN = {  # check 1 (job 96a63235): the router beats SGLang; cuBLAS wins every BF16 layer for now
    ("router", 1): ("triton", (16, 32, 256, 32, 4, 5)),
    ("router", 8): ("triton", (16, 32, 256, 32, 4, 5)),
    ("router", 16): ("triton", (16, 32, 256, 32, 4, 5)),
    ("router", 32): ("triton", (32, 32, 128, 32, 4, 5)),
    **{(layer, m): ("torch", None) for layer in ("qkv_gate", "o", "pooler", "unpooler") for m in (1, 8, 16, 32, 128, 256)},
    ("router", 128): ("torch", None),
    ("router", 256): ("torch", None),
}


def _bucket(m):
    return next((b for b in BUCKETS if b >= m), BUCKETS[-1])


def _weight(state, layer):
    """(W [N, K_W] bf16, K_X): the router is [hi | lo] with K_W = 2 * K_X."""
    if layer == "router":
        return state["router_hilo"], state["router"].shape[1]
    w = state[layer]
    return w, w.shape[1]


def _out(x, layer, n):
    return torch.empty((x.shape[0], n), device=x.device,
                       dtype=torch.float32 if layer == "router" else torch.bfloat16)


def _triton(x, state, layer, n, cfg):
    bm, bn, bk, split, warps, stages = cfg
    w, k_x = _weight(state, layer)
    m = x.shape[0]
    m_tiles = triton.cdiv(m, bm)
    if split > 1 and split * m_tiles * bm * n > WS_FLOATS:
        split = 1
    out = _out(x, layer, n)
    _skinny_gemm[(triton.cdiv(n, bn), split, m_tiles)](
        x, w, out, state["ws"], state["cnt"], m, n, k_x, w.shape[1],
        x.stride(0), w.stride(0), out.stride(0), split,
        BLOCK_N=bn, BLOCK_M=bm, BLOCK_K=bk, OUT_FP32=layer == "router",
        num_warps=warps, num_stages=stages)
    return out


def _cuda_proj(x, state, layer, n, cfg):
    rt, u, ksplit = cfg
    w, k_x = _weight(state, layer)
    out = _out(x, layer, n)
    _MOD.skinny_mma(x, w, out, k_x, rt, u, ksplit)
    return out


def _torch(x, state, layer, n):
    if layer == "router":
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    return x @ state[layer][:n].T


def _run(kind, cfg, x, state, layer, n):
    if kind == "cuda" and x.shape[0] <= 32 and _MOD:
        return _cuda_proj(x, state, layer, n, cfg)
    if kind == "triton":
        return _triton(x, state, layer, n, cfg)
    return _torch(x, state, layer, n)


def prepare(weights: Weights):
    dev = weights["router"].device
    state = {name: w.contiguous() for name, w in weights.items()}
    hi = state["router"].to(torch.bfloat16)
    lo = (state["router"] - hi.float()).to(torch.bfloat16)
    state["router_hilo"] = torch.cat([hi, lo], dim=1).contiguous()
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["ws"] = torch.empty(WS_FLOATS, device=dev, dtype=torch.float32)
    state["cnt"] = torch.zeros(COUNTERS, device=dev, dtype=torch.int32)
    state["plan"] = dict(PLAN)
    if BENCH or any(kind == "cuda" for kind, _ in PLAN.values()):
        _cuda()
    if BENCH:
        _env_report()
        _sweep(state)
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    x, cfg = data["x"], data["config"]
    layer, n = cfg["layer"], cfg["n"]
    kind, tile = state["plan"].get((layer, _bucket(x.shape[0])), ("torch", None))
    return _run(kind, tile, x, state, layer, n)


# ---------------------------------------------------------------- benchmark (BENCH only)

FIXED_US = 3.8  # what the judge subtracts per graph replay (3.7-3.9 us in check 1's records)


def _env_report():
    p = torch.cuda.get_device_properties(0)
    parts = [f"gpu {p.name} sm{p.major}{p.minor} sms={p.multi_processor_count} "
             f"l2={p.L2_cache_size >> 20}MB", f"torch {torch.__version__} triton {triton.__version__}"]
    print("ENV " + " | ".join(parts), flush=True)


def _graph_us(fn, flush, reps=30):
    """Median time of one CUDA graph replay of fn with a cold L2, and fn's graph output."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    for a, b in ev:
        flush.sum()  # read more than L2 so W comes from HBM, as in the judge
        a.record()
        g.replay()
        b.record()
    torch.cuda.synchronize()
    ts = sorted(a.elapsed_time(b) * 1e3 for a, b in ev)
    return ts[len(ts) // 2], out


def _bandwidth(flush):
    """What a plain read kernel reaches: net us and TB/s per size, grid, block and unroll."""
    sink = torch.zeros(1, device=flush.device, dtype=torch.int32)
    sizes = {"router 8.4MB": 256 * 8192 * 4, "o 16.8MB": 8192 * 1024 * 2,
             "qkv 37.7MB": 2304 * 8192 * 2, "pooler 67.1MB": 4096 * 8192 * 2}
    src = torch.empty(max(sizes.values()) // 2, device=flush.device, dtype=torch.bfloat16).normal_()
    for name, nbytes in sizes.items():
        best = {}
        for chunked in (0, 1):
            for grid in (132, 264, 528, 1056, 2112):
                for block in (256, 512):
                    for unroll in (4, 8, 16):
                        t, _ = _graph_us(lambda: _MOD.read_bw(src, nbytes, grid, block, unroll, chunked, sink), flush)
                        key = (chunked, grid)
                        if key not in best or t < best[key][0]:
                            best[key] = (t, block, unroll)
        for chunked in (0, 1):
            cells = []
            for grid in (132, 264, 528, 1056, 2112):
                t, block, unroll = best[(chunked, grid)]
                net = t - FIXED_US
                cells.append(f"g{grid}:{net:.2f}us/{nbytes / net / 1e6:.2f}TB/s(b{block}u{unroll})")
            print(f"BW {name} {'chunked' if chunked else 'strided'} " + " ".join(cells), flush=True)


def _variants(m, layer, n, k_w):
    out = [("torch", None)]
    if PLAN.get((layer, m), ("torch",))[0] == "triton":
        out.append(PLAN[(layer, m)])
    if m <= 32 and _MOD:
        for rt in (1, 2):
            for u in (2, 4, 8):
                for ks in (1, 2, 4, 8):
                    ctas = n // (16 * rt) * ks
                    if n % (16 * rt) or k_w % (ks * 8 * 32 * u) or not 64 <= ctas <= 2112:
                        continue
                    out.append(("cuda", (rt, u, ks)))
    return out


def _rel(a, b):
    return float((a.double() - b).norm() / b.norm().clamp(min=1e-30))


def _sweep(state, budget_s=170.0):
    t0 = time.time()
    dev = state["ws"].device
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    flush = torch.empty(max(4 * l2, 256 << 20) // 4, device=dev, dtype=torch.float32).fill_(1.0)
    if _MOD:
        _bandwidth(flush)
    gen = torch.Generator(device=dev).manual_seed(1)
    rows, load_t = [], {}
    for m in BUCKETS:
        for layer in LAYER_NAMES:
            w, k_x = _weight(state, layer)
            n = 1024 if layer == "unpooler" and m <= 8 else w.shape[0]
            x = torch.randn((m, k_x), device=dev, generator=gen).to(torch.bfloat16)
            full = state["router"] if layer == "router" else state[layer][:n]
            ref = x.double() @ full.double().T
            tol = 2e-5 if layer == "router" else 5e-3
            t_sgl, _ = _graph_us(lambda: _sglang(x, state, layer, n), flush)
            results, bad = [], []
            for kind, cfg in _variants(m, layer, n, w.shape[1]):
                if time.time() - t0 > budget_s:
                    break
                try:
                    err = _rel(_run(kind, cfg, x, state, layer, n), ref)
                    t, out = _graph_us(lambda: _run(kind, cfg, x, state, layer, n), flush)
                    err = max(err, _rel(out, ref))  # the graph replays must be right too
                    if err > tol:
                        bad.append(f"{kind}{cfg}:{err:.1e}")
                        continue
                    results.append((t, kind, cfg))
                except Exception as e:  # noqa: BLE001 - a failing variant is reported and skipped
                    bad.append(f"{kind}{cfg}:{type(e).__name__}")
                    print(f"BENCH fail {layer} m={m} {kind}{cfg}: {str(e)[:200]}", flush=True)
            results.sort(key=lambda r: r[0])
            best_t, kind, cfg = results[0]
            state["plan"][(layer, m)] = (kind, cfg)
            sgl, best = t_sgl - FIXED_US, best_t - FIXED_US
            load_t.setdefault(m, [0.0, 0.0])
            load_t[m][0] += sgl
            load_t[m][1] += best
            t_torch = next(t for t, k, _ in results if k == "torch") - FIXED_US
            top = " ".join(f"{t - FIXED_US:.2f}{k[0]}{list(c) if c else ''}" for t, k, c in results[:3])
            rows.append(f"m={m:<3} {layer:<8} sgl={sgl:6.2f} torch={t_torch:6.2f} best={best:6.2f} "
                        f"x{sgl / max(best, 0.05):.2f} | {top}" + (f" | bad {' '.join(bad[:4])}" if bad else ""))
    ratios = [s / max(o, 0.05) for s, o in load_t.values()]
    est = math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if ratios else 0.0
    print("BENCH net us (judge's 3.8us fixed cost removed); c=cuda[rt,u,ksplit] t=triton[...] t=torch", flush=True)
    for r in rows:
        print("BENCH " + r, flush=True)
    print("BENCH loads " + " ".join(f"m={m}:x{s / max(o, 0.05):.2f}" for m, (s, o) in load_t.items())
          + f" | est speedup x{est:.3f} | sweep {time.time() - t0:.0f}s", flush=True)
    print("PLAN = " + repr(state["plan"]), flush=True)


def _sglang(x, state, layer, n):
    """SGLang's dispatch, as baseline.py runs it: the reference point of the sweep."""
    from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
    if layer != "router":
        return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
    if x.shape[0] <= 8:
        return F.linear(x.float(), state["router"])
    return _torch(x, state, layer, n)
