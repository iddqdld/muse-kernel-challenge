# okc: problem=muse_decode_layer
"""Muse Spark Open decoder layer: SGLang's layer (baseline.py, MuseSpark1xDecoderLayer.forward)
with our kernels swapped in where they beat SGLang's:

- BF16 projections (qkv_gate, o, pooler, unpooler) and the router above 8 tokens: our
  register-direct tensor-core GEMV on fragment-packed weights (muse_bf16_proj version 3), at
  the token counts where it beat cuBLAS; SGLang's dispatch elsewhere. Up to 8 tokens the router
  stays SGLang's FP32 F.linear, because SGLang keeps its input in FP32 there.
- Routed experts: our NVFP4 MoE kernels (muse_moe_w4a16 version 2, rounding where SGLang
  rounds) instead of Marlin.
- Everything else is SGLang's, unchanged: QK-norm + RoPE, the FP8 cache store, FA3 attention,
  the gated output norm, both residual boundaries and the routing.
"""

import time
from typing import TypedDict

import torch
import torch.nn.functional as F
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from torch import nn

from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor            # bfloat16 [2304, 8192]: q 16x64, k 2x64, v 2x64, output gate 16x64
    o: torch.Tensor                   # bfloat16 [8192, 1024]
    attn_residual_gate: torch.Tensor  # bfloat16 [8192], the post-attention residual gate's weight
    pre_ffn_norm: torch.Tensor        # bfloat16 [8192] (applied as 1 + w)
    router: torch.Tensor              # float32 [256, 8192]
    router_bias: torch.Tensor         # float32 [256], added to the scores only to choose experts
    pooler: torch.Tensor              # bfloat16 [4096, 8192]
    pre_expert_norm: torch.Tensor     # bfloat16 [4096] (applied as 1 + w)
    w13: torch.Tensor                 # uint8 [32, 8192, 2048]: this GPU's experts, gate then up rows, packed E2M1
    w13_sf: torch.Tensor              # float8_e4m3fn [32, 8192, 256]
    w13_gs: torch.Tensor              # float32 [32]
    w2: torch.Tensor                  # uint8 [32, 4096, 2048]
    w2_sf: torch.Tensor               # float8_e4m3fn [32, 4096, 256]
    w2_gs: torch.Tensor               # float32 [32]
    post_expert_norm: torch.Tensor    # bfloat16 [4096] (applied as 1 + w, before normalizing)
    unpooler: torch.Tensor            # bfloat16 [8192, 4096]; up to 8 tokens only rows 0-1023 (this GPU's slice)
    post_ffn_norm: torch.Tensor       # bfloat16 [8192] (applied as 1 + w, before normalizing)
    ffn_residual_gate: torch.Tensor   # bfloat16 [8192]
    next_pre_attn_norm: torch.Tensor  # bfloat16 [8192], the next layer's input norm (applied as 1 + w)


class Config(TypedDict):
    batch: int            # B sequences
    q_len: int            # L: 1 for decode, 8 for a verify block (causal inside it); 200-16384 on muse_prefill_layer
    tokens: int           # T = B * L, request-major
    hidden: int           # 8192
    num_q_heads: int      # 16 per GPU under TP8
    num_kv_heads: int     # 2
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers (interleaved RoPE); 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)
    n_experts: int        # 256
    top_k: int            # 8
    local_experts: int    # 32: this GPU computes experts local_start .. local_start + 31


class Inputs(TypedDict):
    stream: torch.Tensor      # float32 [T, 8192], the residual stream
    attn_input: torch.Tensor  # bfloat16 [T, 8192], the normalized stream (the previous layer's boundary wrote it)
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    positions: torch.Tensor   # int64 [T], each new token's position
    cache_loc: torch.Tensor   # int64 [T], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule on full-attention decode steps
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    local_start: torch.Tensor # int32 [1], this GPU's first expert (a multiple of 32; its weights are w13[0], ...)
    config: Config


HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
# MuseSpark1xRouter switches to the split BF16 GEMMs above this many tokens.
ROUTER_FP32_MAX_TOKENS = 8
# Up to this many tokens the unpooler runs on this GPU's slice and is all-gathered.
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0


def _gemm(x, w):
    return _bf16_gemm_dispatch_impl(x, w, None)


def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
    """A Muse norm module holding w (applied as 1 + w), as the checkpoint loads it."""
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
    return norm


def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    return gate


# ---------------------------------------------------------------- our CUDA kernels

# BF16 GEMV (from muse_bf16_proj, version 3)
BF16_CUDA = r"""
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
// W is packed (see pack_weight in Python): per 16-row tile and 32-wide k block, 64 int4s, rows
// 0-7 then rows 8-15, lane-major. So one warp load reads 512 contiguous bytes, and a tile's
// whole K range is one linear stream.
template <int RT, int MT, int U, bool OUT_F32>
__global__ void __launch_bounds__(256) skinny_mma_kernel(
        const __nv_bfloat16* __restrict__ x, const int4* __restrict__ w,
        void* __restrict__ out, int M, int N, int KX, int KW, int k_cta) {
    constexpr int E = RT * MT * 4 * 32;  // accumulator floats per warp
    __shared__ float red[4][E];
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

    const size_t nb = KW >> 5;  // 32-wide k blocks per row
    const int4* wt[RT];
#pragma unroll
    for (int rt = 0; rt < RT; ++rt) wt[rt] = w + (size_t)(blockIdx.x * RT + rt) * nb * 64 + lane;
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
                const int4* p = wt[rt] + (size_t)((kb >> 5) + u) * 64;
                wa[u][rt] = ld_stream(p);
                wb[u][rt] = ld_stream(p + 32);
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

    // The 8 warps of the CTA: warps 4-7 hand their partials to warps 0-3, then all threads add
    // the 4 remaining copies into red[0]. Two rounds keep shared memory at 4 copies.
    if (warp >= 4)
#pragma unroll
        for (int rt = 0; rt < RT; ++rt)
#pragma unroll
            for (int nt = 0; nt < MT; ++nt)
#pragma unroll
                for (int j = 0; j < 4; ++j) red[warp - 4][((rt * MT + nt) * 4 + j) * 32 + lane] = acc[rt][nt][j];
    __syncthreads();
    if (warp < 4)
#pragma unroll
        for (int rt = 0; rt < RT; ++rt)
#pragma unroll
            for (int nt = 0; nt < MT; ++nt)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[rt][nt][j] += red[warp][((rt * MT + nt) * 4 + j) * 32 + lane];
    __syncthreads();
    if (warp < 4)
#pragma unroll
        for (int rt = 0; rt < RT; ++rt)
#pragma unroll
            for (int nt = 0; nt < MT; ++nt)
#pragma unroll
                for (int j = 0; j < 4; ++j) red[warp][((rt * MT + nt) * 4 + j) * 32 + lane] = acc[rt][nt][j];
    __syncthreads();
    for (int e = threadIdx.x; e < E; e += 256) red[0][e] = red[0][e] + red[1][e] + red[2][e] + red[3][e];

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

template <int RT, int MT, int U>
static cudaError_t launch_rmu(cudaLaunchConfig_t* cfg, bool f32, const __nv_bfloat16* x,
                              const int4* w, void* out, int M, int N, int KX, int KW, int k_cta) {
    if (f32) return cudaLaunchKernelEx(cfg, skinny_mma_kernel<RT, MT, U, true>, x, w, out, M, N, KX, KW, k_cta);
    return cudaLaunchKernelEx(cfg, skinny_mma_kernel<RT, MT, U, false>, x, w, out, M, N, KX, KW, k_cta);
}

template <int RT, int MT>
static cudaError_t launch_rm(cudaLaunchConfig_t* cfg, int u, bool f32, const __nv_bfloat16* x,
                             const int4* w, void* out, int M, int N, int KX, int KW, int k_cta) {
    if constexpr (RT <= 2) {  // RT 4 with U 8 would hold 256 registers of W
        if (u == 8) return launch_rmu<RT, MT, 8>(cfg, f32, x, w, out, M, N, KX, KW, k_cta);
    }
    if (u == 2) return launch_rmu<RT, MT, 2>(cfg, f32, x, w, out, M, N, KX, KW, k_cta);
    return launch_rmu<RT, MT, 4>(cfg, f32, x, w, out, M, N, KX, KW, k_cta);
}

template <int RT>
static cudaError_t launch_r(cudaLaunchConfig_t* cfg, int mt, int u, bool f32, const __nv_bfloat16* x,
                            const int4* w, void* out, int M, int N, int KX, int KW, int k_cta) {
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
    auto wp = static_cast<const int4*>(w);
    if (rt == 4) return launch_r<4>(&cfg, mt, u, f32, xp, wp, out, M, N, KX, KW, k_cta);
    if (rt == 2) return launch_r<2>(&cfg, mt, u, f32, xp, wp, out, M, N, KX, KW, k_cta);
    return launch_r<1>(&cfg, mt, u, f32, xp, wp, out, M, N, KX, KW, k_cta);
}
"""

BF16_TORCH = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

cudaError_t skinny_mma_launch(const void* x, const void* w, void* out, bool f32, int M, int N, int KX,
                              int KW, int rt, int u, int ksplit, cudaStream_t stream);

void skinny_mma(torch::Tensor x, torch::Tensor w_packed, torch::Tensor out, int64_t kx, int64_t kw,
                int64_t rt, int64_t u, int64_t ksplit) {
    TORCH_CHECK(x.is_contiguous() && w_packed.is_contiguous() && out.is_contiguous());
    const cudaError_t err = skinny_mma_launch(
        x.data_ptr(), w_packed.data_ptr(), out.data_ptr(), out.scalar_type() == at::kFloat, x.size(0),
        out.size(1), kx, kw, rt, u, ksplit, at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(err == cudaSuccess, "skinny_mma: ", cudaGetErrorString(err));
}
"""

BF16_DECLS = """
void skinny_mma(torch::Tensor x, torch::Tensor w_packed, torch::Tensor out, int64_t kx, int64_t kw,
                int64_t rt, int64_t u, int64_t ksplit);
"""

# NVFP4 routed experts (from muse_moe_w4a16, version 2)
MOE_CUDA = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

constexpr int H = 4096, I = 4096, TOPK = 8, LOCAL = 32;
constexpr int NCH = 64;  // 64-wide k chunks per row (K = 4096 in both GEMMs)
constexpr int U = 8;     // chunks per iteration: one int4 of block scales per lane

__device__ __forceinline__ int4 ld_stream(const void* p) {
    int4 v;
    asm("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// E4M3 block scale byte -> BF16 (scale * 2^118) in both halves.
__device__ __forceinline__ uint32_t scale2(uint32_t b) {
    uint32_t h2;
    const unsigned short dup = (unsigned short)(b | (b << 8));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(dup));
    const float f = __half2float(__ushort_as_half((unsigned short)(h2 & 0xffffu))) * 0x1p118f;
    const uint32_t u = __bfloat16_as_ushort(__float2bfloat16_rn(f));
    return u | (u << 16);
}

// Round r of a packed word: the codes in nibbles 3 - r and 7 - r, as BF16 (value * 2^-126),
// times the scale pair s.
template <int R>
__device__ __forceinline__ uint32_t deq(uint32_t q, uint32_t s) {
    const uint32_t t = q << (4 * R);
    const uint32_t e = ((t >> 6) & 0x01C001C0u) | (t & 0x80008000u);
    const __nv_bfloat162 v = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&e),
                                     *reinterpret_cast<const __nv_bfloat162*>(&s));
    return *reinterpret_cast<const uint32_t*>(&v);
}

// The 4 k-steps of one chunk: c holds row g (x: steps 0-1, y: steps 2-3) and row g + 8 (z, w).
__device__ __forceinline__ void dequant_chunk(const int4 c, uint32_t s_lo, uint32_t s_hi, uint32_t (&A)[4][4]) {
    const uint32_t x = c.x, y = c.y, z = c.z, w = c.w;
    A[0][0] = deq<0>(x, s_lo); A[0][1] = deq<0>(z, s_hi); A[0][2] = deq<1>(x, s_lo); A[0][3] = deq<1>(z, s_hi);
    A[1][0] = deq<2>(x, s_lo); A[1][1] = deq<2>(z, s_hi); A[1][2] = deq<3>(x, s_lo); A[1][3] = deq<3>(z, s_hi);
    A[2][0] = deq<0>(y, s_lo); A[2][1] = deq<0>(w, s_hi); A[2][2] = deq<1>(y, s_lo); A[2][3] = deq<1>(w, s_hi);
    A[3][0] = deq<2>(y, s_lo); A[3][1] = deq<2>(w, s_hi); A[3][2] = deq<3>(y, s_lo); A[3][3] = deq<3>(w, s_hi);
}

__device__ __forceinline__ void mma_chunk(float (&acc)[4], const uint32_t (&A)[4][4], const int4 b01, const int4 b23) {
    mma_bf16(acc, A[0][0], A[0][1], A[0][2], A[0][3], b01.x, b01.y);
    mma_bf16(acc, A[1][0], A[1][1], A[1][2], A[1][3], b01.z, b01.w);
    mma_bf16(acc, A[2][0], A[2][1], A[2][2], A[2][3], b23.x, b23.y);
    mma_bf16(acc, A[3][0], A[3][1], A[3][2], A[3][3], b23.z, b23.w);
}

// meta (int32): [0] number of groups; then per group (at most MAX_GROUPS): local expert, first
// position in the route list, route count (at most 32). A group is one pass of an expert.
constexpr int MAX_GROUPS = 160;
constexpr int META_E = 1, META_BEG = 1 + MAX_GROUPS, META_N = 1 + 2 * MAX_GROUPS;

// 1 block, 1024 threads: the local routes (token * 8 + slot) sorted by expert into list, the
// groups of up to 32 routes per expert into meta, and ssq zeroed.
__global__ void __launch_bounds__(1024) moe_route_kernel(const int* __restrict__ ids, int n_ids, int local_start,
                                                         int* __restrict__ meta, int* __restrict__ list,
                                                         float* __restrict__ ssq) {
    __shared__ int cnt[LOCAL], cur[LOCAL];
    if (threadIdx.x < LOCAL) cnt[threadIdx.x] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
        ssq[i] = 0.f;
        const int e = ids[i] - local_start;
        if (e >= 0 && e < LOCAL) atomicAdd(&cnt[e], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int off = 0, ng = 0;
        for (int e = 0; e < LOCAL; ++e) {
            cur[e] = off;
            for (int g0 = 0; g0 < cnt[e] && ng < MAX_GROUPS; g0 += 32, ++ng) {
                meta[META_E + ng] = e;
                meta[META_BEG + ng] = off + g0;
                meta[META_N + ng] = min(32, cnt[e] - g0);
            }
            off += cnt[e];
        }
        meta[0] = ng;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
        const int e = ids[i] - local_start;
        if (e >= 0 && e < LOCAL) list[atomicAdd(&cur[e], 1)] = i;
    }
}

// One warp: NCHUNK chunks of one 16-row tile into acc, against the B rows brow (this lane's
// row at its k offset, nullptr past the routes). All weight loads are issued first.
template <int MT, int NCHUNK>
__device__ __forceinline__ void warp_tile(const int4* __restrict__ cp, const int4* __restrict__ sp,
                                          const __nv_bfloat16* const (&brow)[MT], float (&acc)[MT][4]) {
    int4 w[NCHUNK], s[NCHUNK / U];
#pragma unroll
    for (int u = 0; u < NCHUNK; ++u) w[u] = ld_stream(cp + u * 32);
#pragma unroll
    for (int q = 0; q < NCHUNK / U; ++q) s[q] = ld_stream(sp + q * 32);
#pragma unroll
    for (int u = 0; u < NCHUNK; ++u) {
        const uint32_t sw = reinterpret_cast<const uint32_t*>(&s[u / U])[(u % U) >> 1] >> ((u & 1) * 16);
        uint32_t A[4][4];
        dequant_chunk(w[u], scale2(sw & 0xffu), scale2((sw >> 8) & 0xffu), A);
#pragma unroll
        for (int nt = 0; nt < MT; ++nt) {
            int4 b01 = make_int4(0, 0, 0, 0), b23 = b01;
            if (brow[nt]) {
                const int4* p = reinterpret_cast<const int4*>(brow[nt] + u * 64);
                b01 = __ldg(p);
                b23 = __ldg(p + 1);
            }
            mma_chunk(acc[nt], A, b01, b23);
        }
    }
}

template <int MT>
__device__ __forceinline__ void store_partial(float* red, const float (&acc)[MT][4]) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
    for (int nt = 0; nt < MT; ++nt)
#pragma unroll
        for (int j = 0; j < 4; ++j) red[(warp * MT * 4 + nt * 4 + j) * 32 + lane] = acc[nt][j];
}

// Element e of a warp's MT x 16 x 8 accumulator: (route index within the group, row in the tile).
__device__ __forceinline__ void elem(int e, int& i, int& row) {
    const int ln = e & 31, j = (e >> 5) & 3, nt = e >> 7;
    i = nt * 8 + 2 * (ln & 3) + (j & 1);
    row = (ln >> 2) + (j >= 2 ? 8 : 0);
}

// Gate and up tile `tile` of local expert e for the routes routes[0..n): warps 0-3 take the gate
// rows, warps 4-7 the up rows, each warp a quarter of K. SwiGLU -> act[route].
template <int MT>
__device__ __forceinline__ void up_item(const __nv_bfloat16* __restrict__ x, const int* __restrict__ routes, int n,
                                        int e, int tile, const int4* __restrict__ codes,
                                        const int4* __restrict__ scales, float gsc, __nv_bfloat16* __restrict__ act,
                                        float* red) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, q = warp & 3;
    const size_t tb = (size_t)e * (2 * I / 16) + tile + (warp >= 4 ? I / 16 : 0);
    const __nv_bfloat16* brow[MT];
#pragma unroll
    for (int nt = 0; nt < MT; ++nt) {
        const int i = nt * 8 + g;
        brow[nt] = i < n ? x + (size_t)(routes[i] / TOPK) * H + q * (H / 4) + t * 16 : nullptr;
    }
    float acc[MT][4] = {};
    warp_tile<MT, NCH / 4>(codes + (tb * NCH + q * (NCH / 4)) * 32 + lane,
                           scales + (tb * (NCH / U) + q * (NCH / 4 / U)) * 32 + lane, brow, acc);
    store_partial<MT>(red, acc);
    __syncthreads();
    for (int e2 = threadIdx.x; e2 < MT * 128; e2 += blockDim.x) {
        float gate = 0.f, up = 0.f;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            gate += red[w * MT * 128 + e2];
            up += red[(w + 4) * MT * 128 + e2];
        }
        int i, row;
        elem(e2, i, row);
        if (i < n) {
            // SGLang keeps the GEMM outputs (gate, up) in BF16 before silu_and_mul.
            gate = __bfloat162float(__float2bfloat16_rn(gate * gsc));
            up = __bfloat162float(__float2bfloat16_rn(up * gsc));
            act[(size_t)routes[i] * I + tile * 16 + row] = __float2bfloat16_rn(gate / (1.f + __expf(-gate)) * up);
        }
    }
    __syncthreads();
}

// Down tile `tile` of local expert e: 8 warps, each an eighth of K. y -> ybuf[route] in BF16 (as
// SGLang keeps it), and the sum of squares of y * (1 + w) per route into ssq.
template <int MT>
__device__ __forceinline__ void down_item(const __nv_bfloat16* __restrict__ act, const int* __restrict__ routes, int n,
                                          int e, int tile, const int4* __restrict__ codes,
                                          const int4* __restrict__ scales, float gsc,
                                          const __nv_bfloat16* __restrict__ norm_w, __nv_bfloat16* __restrict__ ybuf,
                                          float* __restrict__ ssq, float* red, float* sq) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const size_t tb = (size_t)e * (H / 16) + tile;
    const __nv_bfloat16* brow[MT];
#pragma unroll
    for (int nt = 0; nt < MT; ++nt) {
        const int i = nt * 8 + g;
        brow[nt] = i < n ? act + (size_t)routes[i] * I + warp * (I / 8) + t * 16 : nullptr;
    }
    if (threadIdx.x < 32) sq[threadIdx.x] = 0.f;
    float acc[MT][4] = {};
    warp_tile<MT, NCH / 8>(codes + (tb * NCH + warp * (NCH / 8)) * 32 + lane,
                           scales + (tb * (NCH / U) + warp) * 32 + lane, brow, acc);
    store_partial<MT>(red, acc);
    __syncthreads();
    for (int e2 = threadIdx.x; e2 < MT * 128; e2 += blockDim.x) {
        float y = 0.f;
#pragma unroll
        for (int w = 0; w < 8; ++w) y += red[w * MT * 128 + e2];
        int i, row;
        elem(e2, i, row);
        if (i < n) {
            const int col = tile * 16 + row;
            const __nv_bfloat16 yb = __float2bfloat16_rn(y * gsc);
            ybuf[(size_t)routes[i] * H + col] = yb;
            const float v = __bfloat162float(yb) * (1.f + __bfloat162float(norm_w[col]));
            atomicAdd(&sq[i], v * v);
        }
    }
    __syncthreads();
    if (threadIdx.x < n) atomicAdd(&ssq[routes[threadIdx.x]], sq[threadIdx.x]);
    __syncthreads();
}

// Persistent: blocks walk the items (group, 16-row tile). 256 threads, 2 blocks per SM.
__global__ void __launch_bounds__(256, 2) moe_up_kernel(
        const __nv_bfloat16* __restrict__ x, const int* __restrict__ meta, const int* __restrict__ list,
        int local_start, const int4* __restrict__ codes, const int4* __restrict__ scales,
        const float* __restrict__ gs, __nv_bfloat16* __restrict__ act) {
    __shared__ float red[8 * 4 * 128];
    const int items = meta[0] * (I / 16);
    for (int item = blockIdx.x; item < items; item += gridDim.x) {
        const int grp = item / (I / 16), tile = item % (I / 16);
        const int e = meta[META_E + grp], n = meta[META_N + grp];
        const int* routes = list + meta[META_BEG + grp];
        const float gsc = gs[local_start + e] * 256.f;
        if (n <= 8) up_item<1>(x, routes, n, e, tile, codes, scales, gsc, act, red);
        else if (n <= 16) up_item<2>(x, routes, n, e, tile, codes, scales, gsc, act, red);
        else up_item<4>(x, routes, n, e, tile, codes, scales, gsc, act, red);
    }
}

__global__ void __launch_bounds__(256, 2) moe_down_kernel(
        const __nv_bfloat16* __restrict__ act, const int* __restrict__ meta, const int* __restrict__ list,
        int local_start, const int4* __restrict__ codes, const int4* __restrict__ scales,
        const float* __restrict__ gs, const __nv_bfloat16* __restrict__ norm_w, __nv_bfloat16* __restrict__ ybuf,
        float* __restrict__ ssq) {
    __shared__ float red[8 * 4 * 128];
    __shared__ float sq[32];
    const int items = meta[0] * (H / 16);
    for (int item = blockIdx.x; item < items; item += gridDim.x) {
        const int grp = item / (H / 16), tile = item % (H / 16);
        const int e = meta[META_E + grp], n = meta[META_N + grp];
        const int* routes = list + meta[META_BEG + grp];
        const float gsc = gs[local_start + e] * 256.f;
        if (n <= 8) down_item<1>(act, routes, n, e, tile, codes, scales, gsc, norm_w, ybuf, ssq, red, sq);
        else if (n <= 16) down_item<2>(act, routes, n, e, tile, codes, scales, gsc, norm_w, ybuf, ssq, red, sq);
        else down_item<4>(act, routes, n, e, tile, codes, scales, gsc, norm_w, ybuf, ssq, red, sq);
    }
}

// grid M, 256 threads, 16 columns each:
// out[t] = sum over local routes of w * y * (1 + post_norm_w) * rsqrt(mean((y * (1 + post_norm_w))^2) + eps).
__global__ void __launch_bounds__(256) moe_combine_kernel(
        const int* __restrict__ ids, const float* __restrict__ wts, int local_start,
        const __nv_bfloat16* __restrict__ norm_w,
        const __nv_bfloat16* __restrict__ ybuf, const float* __restrict__ ssq, __nv_bfloat16* __restrict__ out) {
    const int tok = blockIdx.x, c = threadIdx.x * 16;
    float acc[16] = {}, nw[16];
#pragma unroll
    for (int q = 0; q < 16; ++q) nw[q] = 1.f + __bfloat162float(norm_w[c + q]);
    for (int s = 0; s < TOPK; ++s) {
        const int r = tok * TOPK + s, e = ids[r] - local_start;
        if (e < 0 || e >= LOCAL) continue;
        const float f = wts[r] * rsqrtf(ssq[r] * (1.f / H) + 1e-8f);
        const int4* y = reinterpret_cast<const int4*>(ybuf + (size_t)r * H + c);
        __align__(16) __nv_bfloat16 v[16];
        reinterpret_cast<int4*>(v)[0] = y[0];
        reinterpret_cast<int4*>(v)[1] = y[1];
#pragma unroll
        for (int q = 0; q < 16; ++q) acc[q] += f * nw[q] * __bfloat162float(v[q]);
    }
    __align__(16) __nv_bfloat16 o[16];
#pragma unroll
    for (int q = 0; q < 16; ++q) o[q] = __float2bfloat16_rn(acc[q]);
    int4* dst = reinterpret_cast<int4*>(out + (size_t)tok * H + c);
    dst[0] = reinterpret_cast<const int4*>(o)[0];
    dst[1] = reinterpret_cast<const int4*>(o)[1];
}

cudaError_t moe_launch(const void* x, const int* ids, const float* wts, int M, int local_start,
                       const void* c13, const void* s13, const float* gs13, const void* c2, const void* s2,
                       const float* gs2, const void* norm_w, void* act, void* ybuf, float* ssq, int* meta,
                       int* list, int grid, void* out, cudaStream_t st) {
    const int n_ids = M * TOPK;
    if (n_ids > 32 * MAX_GROUPS - 32 * LOCAL) return cudaErrorInvalidValue;  // groups always fit
    moe_route_kernel<<<1, 1024, 0, st>>>(ids, n_ids, local_start, meta, list, ssq);
    moe_up_kernel<<<grid, 256, 0, st>>>((const __nv_bfloat16*)x, meta, list, local_start, (const int4*)c13,
                                        (const int4*)s13, gs13, (__nv_bfloat16*)act);
    moe_down_kernel<<<grid, 256, 0, st>>>((const __nv_bfloat16*)act, meta, list, local_start, (const int4*)c2,
                                          (const int4*)s2, gs2, (const __nv_bfloat16*)norm_w, (__nv_bfloat16*)ybuf, ssq);
    moe_combine_kernel<<<M, 256, 0, st>>>(ids, wts, local_start, (const __nv_bfloat16*)norm_w,
                                          (const __nv_bfloat16*)ybuf, ssq,
                                          (__nv_bfloat16*)out);
    return cudaGetLastError();
}
"""

MOE_TORCH = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

cudaError_t moe_launch(const void* x, const int* ids, const float* wts, int M, int local_start,
                       const void* c13, const void* s13, const float* gs13, const void* c2, const void* s2,
                       const float* gs2, const void* norm_w, void* act, void* ybuf, float* ssq, int* meta,
                       int* list, int grid, void* out, cudaStream_t st);

void moe_forward(torch::Tensor x, torch::Tensor ids, torch::Tensor wts, int64_t local_start,
                 torch::Tensor c13, torch::Tensor s13, torch::Tensor gs13, torch::Tensor c2, torch::Tensor s2,
                 torch::Tensor gs2, torch::Tensor norm_w, torch::Tensor act, torch::Tensor ybuf, torch::Tensor ssq,
                 torch::Tensor meta, torch::Tensor list, int64_t grid, torch::Tensor out) {
    const cudaError_t err = moe_launch(
        x.data_ptr(), ids.data_ptr<int>(), wts.data_ptr<float>(), x.size(0), local_start, c13.data_ptr(),
        s13.data_ptr(), gs13.data_ptr<float>(), c2.data_ptr(), s2.data_ptr(), gs2.data_ptr<float>(),
        norm_w.data_ptr(), act.data_ptr(), ybuf.data_ptr(), ssq.data_ptr<float>(),
        meta.data_ptr<int>(), list.data_ptr<int>(), grid, out.data_ptr(), at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(err == cudaSuccess, "moe_forward: ", cudaGetErrorString(err));
}
"""

MOE_DECLS = """
void moe_forward(torch::Tensor x, torch::Tensor ids, torch::Tensor wts, int64_t local_start,
                 torch::Tensor c13, torch::Tensor s13, torch::Tensor gs13, torch::Tensor c2, torch::Tensor s2,
                 torch::Tensor gs2, torch::Tensor norm_w, torch::Tensor act, torch::Tensor ybuf, torch::Tensor ssq,
                 torch::Tensor meta, torch::Tensor list, int64_t grid, torch::Tensor out);
"""


def pack_weight(w):
    """[N, K] -> fragment order for skinny_mma_kernel: per 16-row tile and 32-wide k block, rows
    0-7 then rows 8-15, each as lanes (row r, k quarter q) -> 16 bytes, lane = 4 * r + q."""
    n, k = w.shape
    return w.view(n // 16, 2, 8, k // 32, 4, 8).permute(0, 3, 1, 2, 4, 5).contiguous()


# New nibble p of each packed 32-bit word holds original value NIBBLE_SRC[p] of that word.
NIBBLE_SRC = (6, 4, 2, 0, 7, 5, 3, 1)
MAX_GROUPS = 160  # MoE route groups of up to 32 per expert; matches the CUDA constant


def pack_codes(codes):
    """uint8 [E, R, K/2] E2M1 pairs -> [E, R/16 tiles, K/64 chunks, 32 lanes, 16 bytes]: per lane,
    row g then row g + 8, k = 16t .. 16t + 15, each 32-bit word's codes ordered by NIBBLE_SRC."""
    e, r, kb = codes.shape
    c = codes.view(e, r // 16, 2, 8, kb // 32, 4, 8).permute(0, 1, 4, 3, 5, 2, 6).contiguous()
    words = c.view(*c.shape[:-1], 2, 4)  # [..., h, word, byte]
    vals = torch.stack((words & 0xF, words >> 4), dim=-1).flatten(-2)  # [..., 8 values]
    vals = vals[..., list(NIBBLE_SRC)]
    packed = vals[..., 0::2] | (vals[..., 1::2] << 4)
    return packed.reshape(e, r // 16, kb // 32, 32, 16).contiguous()


def pack_scales(sf):
    """E4M3 [E, R, K/16] -> [E, R/16 tiles, K/512 groups, 32 lanes, 16 bytes]: per lane, for 8
    chunks u, the scale of row g then row g + 8 at block 4u + t of the group."""
    e, r, nb = sf.shape
    s = sf.view(torch.uint8).view(e, r // 16, 2, 8, nb // 32, 8, 4)
    return s.permute(0, 1, 4, 3, 6, 5, 2).reshape(e, r // 16, nb // 32, 32, 16).contiguous()


_BF16 = None
_MOE = None


def _compile():
    """Both CUDA modules (nvcc, about 40 s each)."""
    global _BF16, _MOE
    from torch.utils.cpp_extension import load_inline
    t0 = time.time()
    if _BF16 is None:
        _BF16 = load_inline(name="muse_dl_gemv_v1", cpp_sources=BF16_DECLS, cuda_sources=BF16_CUDA + BF16_TORCH,
                            functions=["skinny_mma"], extra_cuda_cflags=["-O3"], verbose=False)
    if _MOE is None:
        _MOE = load_inline(name="muse_dl_moe_v1", cpp_sources=MOE_DECLS, cuda_sources=MOE_CUDA + MOE_TORCH,
                           functions=["moe_forward"], extra_cuda_cflags=["-O3"], verbose=False)
    print(f"CUDA compiled in {time.time() - t0:.0f}s", flush=True)


BUCKETS = (1, 8, 16, 32, 128, 256)  # token counts of the scored loads
# (weight, token bucket) -> (rt, u, ksplit) of our GEMV, where it beat cuBLAS on muse_bf16_proj
# (check 67996f44). Missing entries use SGLang's dispatch. The router runs here only above 8
# tokens: up to 8, SGLang keeps the router input in FP32 (F.linear on the FP32 weight).
GEMV_PLAN = {
    ("qkv_gate", 1): (1, 8, 2), ("qkv_gate", 8): (1, 8, 2), ("qkv_gate", 16): (2, 2, 4),
    ("o", 1): (1, 4, 1),
    ("pooler", 1): (1, 8, 2), ("pooler", 8): (1, 8, 2), ("pooler", 16): (2, 8, 2),
    ("unpooler", 1): (1, 4, 4), ("unpooler", 8): (1, 4, 4),
    ("router", 16): (1, 8, 8), ("router", 32): (1, 8, 8),
}


def _gemv(x, state, name, n):
    """x @ W[:n].T for a dense weight: our GEMV where GEMV_PLAN has the shape, else SGLang's."""
    m = x.shape[0]
    cfg = GEMV_PLAN.get((name, next((b for b in BUCKETS if b >= m), 0)))
    if cfg is None or m > 32:
        return _gemm(x, state["dense"][name][:n])
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    k = x.shape[1]
    _BF16.skinny_mma(x.contiguous(), state["packed"][name], out, k, k, *cfg)
    return out


def _router_split(x, state):
    """Router logits above 8 tokens: W = hi + lo in BF16, FP32 accumulation (as SGLang)."""
    m = x.shape[0]
    cfg = GEMV_PLAN.get(("router", next((b for b in BUCKETS if b >= m), 0)))
    if cfg is None or m > 32:
        logits = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        logits += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return logits
    out = torch.empty((m, N_EXPERTS), device=x.device, dtype=torch.float32)
    _BF16.skinny_mma(x.contiguous(), state["packed"]["router"], out, HIDDEN, 2 * HIDDEN, *cfg)
    return out


def _experts(x, topk_weights, local_ids, state):
    """Our NVFP4 MoE kernels: x bf16 [T, 4096], local expert ids 0-31 (others elsewhere) ->
    the post-expert-normed, weighted sum per token, bf16 [T, 4096]."""
    moe = state["moe"]
    m = x.shape[0]
    dev = x.device
    out = torch.empty((m, D_EXPERT), device=dev, dtype=torch.bfloat16)
    act = torch.empty((m * TOP_K, D_EXPERT), device=dev, dtype=torch.bfloat16)
    ybuf = torch.empty((m * TOP_K, D_EXPERT), device=dev, dtype=torch.bfloat16)
    ssq = torch.empty((m * TOP_K,), device=dev, dtype=torch.float32)
    routes = torch.empty((m * TOP_K,), device=dev, dtype=torch.int32)
    _MOE.moe_forward(x.contiguous(), local_ids, topk_weights.to(torch.float32).contiguous(), 0,
                     moe["c13"], moe["s13"], moe["gs13"], moe["c2"], moe["s2"], moe["gs2"], moe["norm_w"],
                     act, ybuf, ssq, moe["meta"], routes, moe["grid"], out)
    return out


def prepare(weights):
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    # [2^21, 64] float32 cos | sin, as SGLang's RotaryEmbedding builds it.
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, RMS_EPS)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, RMS_EPS)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN, POST_NORM_EPS,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    dense = {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")}
    state = {
        "dense": dense,
        "packed": {**{name: pack_weight(w) for name, w in dense.items()},
                   "router": pack_weight(torch.cat([hi, lo], dim=1).contiguous())},
        "router": router.contiguous(),
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_norm": _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, RMS_EPS),
        "moe": {
            "c13": pack_codes(weights["w13"].contiguous()), "s13": pack_scales(weights["w13_sf"].contiguous()),
            "c2": pack_codes(weights["w2"].contiguous()), "s2": pack_scales(weights["w2_sf"].contiguous()),
            "gs13": weights["w13_gs"].contiguous(), "gs2": weights["w2_gs"].contiguous(),
            "norm_w": weights["post_expert_norm"].contiguous(),  # BF16, as given; the kernels use 1 + w
            "meta": torch.zeros(1 + 3 * MAX_GROUPS, device=device, dtype=torch.int32),
            "grid": 2 * torch.cuda.get_device_properties(device).multi_processor_count,
        },
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn),
    }
    del freqs
    _compile()
    torch.cuda.synchronize()
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    # Attention: MuseSpark1xAttention.forward.
    qkv = _gemv(data["attn_input"], state, "qkv_gate", hq * d * 2 + hkv * d * 2)
    if cfg["rope_theta"]:
        q, k = state["qk_norm"](qkv, data["positions"], state["cos_sin"])
    else:
        q, k = state["qk_norm"](qkv)
    v = qkv[:, hq * d + hkv * d: hq * d + 2 * hkv * d]
    gate = qkv[:, hq * d + 2 * hkv * d:]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k = k.reshape(tokens, hkv * d)
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
    # The pool keeps FP8 rows as bytes; slot 0 is a live row here (SGLang reserves it for padding).
    store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)
    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q.view(tokens, hq, d).to(FP8),
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    o = state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)
    attn_output = _gemv(o, state, "o", HIDDEN)  # all-reduce: identity

    # Post-attention boundary, router and MoE block: MuseSpark1xMoE._forward_experts.
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
    # The router and routing run on a side stream, overlapping the pooler on the main one.
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = _gemv(ffn_input, state, "pooler", D_EXPERT)
    with torch.cuda.stream(side_stream):
        logits = _router_split(ffn_input, state) if split_router else F.linear(router_input, state["router"])
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
    # This GPU's experts start at local_start: as SGLang's expert_map on that EP rank.
    local_ids = ((topk_ids - data["local_start"]) % N_EXPERTS).to(torch.int32).contiguous()
    # all-reduce: identity
    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemv(expert_output, state, "unpooler", UNPOOL_SLICE)
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = _gemv(expert_output, state, "unpooler", HIDDEN)

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
