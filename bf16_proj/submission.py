# okc: problem=muse_bf16_proj
"""Dense projections out = x @ W[:n].T, one kernel launch per call.

At 1-32 tokens these are memory-bound: the time floor is reading W once at 4.8 TB/s.

CUDA kernels, compiled with load_inline in prepare():
- register-direct tensor cores: each lane loads 16 contiguous bytes of a W row with
  ld.global.nc.L1::no_allocate and feeds them straight to mma.sync m16n8k16. A dot
  product does not care about the order of k, so k is relabelled to make each lane's
  16 bytes exactly its fragment slots; x is read with the same relabelling;
- weight formats: Plain (pack_weight) and a lossless 12-bit code (sign + mantissa byte,
  4-bit exponent code, a few exception slots per tile for tiny values) decoded in
  registers: 25% fewer bytes from HBM (pack_c12; layouts C12 and C12W);
- ksplit_kernel: the 8 warps of a CTA split K and add their partials in shared memory;
- K can also split across the CTAs of a thread block cluster; the leader CTA reads the
  others' partials from their shared memory (DSMEM). One launch, no atomics, no workspace;
- router: W = hi + lo, two BF16 halves stored side by side as one [256, 16384] matrix;
  x is read at k mod 8192, so both halves meet the same x in the FP32 accumulator.
cuBLAS stays where it wins: SGLang's own dispatch, or the same GEMM on another weight
layout (_mm). PLAN says which runs where.

BENCH = True (check runs): prepare() times every candidate per scored shape the way
the judge does (CUDA graph replay, cold L2), validates it, prints a table and uses
the fastest.
"""

import math
import time
from typing import TypedDict

import torch
import torch.nn.functional as F

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

// volatile: keeps every load of an unrolled step ahead of the step's MMAs. Without it, ptxas
// issued only a few of the U loads before the first use (U = 16 compiled to 40 registers).
__device__ __forceinline__ int4 ld_stream(const void* p) {
    int4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ int2 ld_stream2(const void* p) {
    int2 v;
    asm volatile("ld.global.nc.L1::no_allocate.v2.s32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

__device__ __forceinline__ void mma_bf16(float* c, int a0, int a1, int a2, int a3, int b0, int b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Two register stages over k in [k_begin, k_end), 32 * U per step (k_end - k_begin is a multiple):
// the loads of step i + 1 are in flight while step i computes.
template <int U, class Stage, class Load, class Compute>
__device__ __forceinline__ void pipeline2(int k_begin, int k_end, Load& load, Compute& compute) {
    Stage A, B;
    load(A, k_begin);
    for (int kb = k_begin + 32 * U;; kb += 64 * U) {
        const bool more = kb < k_end;
        if (more) load(B, kb);
        compute(A);
        if (!more) break;
        const bool more2 = kb + 32 * U < k_end;
        if (more2) load(A, kb + 32 * U);
        compute(B);
        if (!more2) break;
    }
}

// The end of ksplit_kernel. The 8 warps of the CTA: warps 4-7 hand their partials to warps 0-3,
// then all threads add the 4 remaining copies into red[0]. Two rounds keep shared memory at 4
// copies. With ksplit > 1 the cluster's CTAs reduce-scatter: rank r sums its 1/ksplit slice of
// the elements over all ranks' red[0] through DSMEM (independent loads, all ranks at once) and
// stores that slice. Tokens m0 .. m0 + M - 1 of rows row0 .. row0 + 16 * RT - 1.
template <int RT, int MT, bool OUT_F32>
__device__ __forceinline__ void reduce_store(float (&acc)[RT][MT][4], float (*red)[RT * MT * 128],
                                             void* out, int row0, int m0, int M, int N) {
    constexpr int E = RT * MT * 128;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
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
#pragma unroll 1  // rolled loops keep the kernel small: its code comes from HBM on every call
    for (int e = threadIdx.x; e < E; e += 256) red[0][e] = red[0][e] + red[1][e] + red[2][e] + red[3][e];

    auto store = [&](int e, float v) {
        const int ln = e & 31, j = (e >> 5) & 3, nt = (e >> 7) % MT, rt = (e >> 7) / MT;
        const int row = row0 + rt * 16 + (ln >> 2) + (j >= 2 ? 8 : 0);
        const int tok = nt * 8 + 2 * (ln & 3) + (j & 1);
        if (tok < M) {
            const size_t o = (size_t)(m0 + tok) * N + row;
            if (OUT_F32) reinterpret_cast<float*>(out)[o] = v;
            else reinterpret_cast<__nv_bfloat16*>(out)[o] = __float2bfloat16(v);
        }
    };
    const int ksplit = gridDim.y;
    if (ksplit > 1) {
        cg::cluster_group cluster = cg::this_cluster();
        cluster.sync();
        const int chunk = E / ksplit, e0 = (int)cluster.block_rank() * chunk;  // E % ksplit == 0
        float v[2] = {0.f, 0.f};  // chunk <= 512 = 2 elements per thread
#pragma unroll
        for (int i = 0; i < 2; ++i)
            if (i * 256 + (int)threadIdx.x < chunk)
#pragma unroll 4
                for (int q = 0; q < ksplit; ++q)
                    v[i] += cluster.map_shared_rank(&red[0][0], q)[e0 + i * 256 + threadIdx.x];
        cluster.sync();  // the others' shared memory must outlive this CTA's reads
#pragma unroll
        for (int i = 0; i < 2; ++i)
            if (i * 256 + (int)threadIdx.x < chunk) store(e0 + i * 256 + threadIdx.x, v[i]);
        return;
    }
    __syncthreads();
#pragma unroll 1
    for (int e = threadIdx.x; e < E; e += 256) store(e, red[0][e]);
}

// One bf16x2 fragment register from the 12-bit code (see pack_c12 in Python). s holds the
// sign | mantissa bytes of values 4 * (J / 2) .. + 3; pair J is bytes 2 * (J % 2) and + 1.
// q holds 8 exponent codes: pair J's two codes are nibbles J and J + 4, so one shift puts
// them at bits 7-10 and 23-26. Result: sign << 15 | (code + base) << 7 | mantissa, per half.
template <int J>
__device__ __forceinline__ uint32_t dec_pair(uint32_t s, uint32_t q, uint32_t basew) {
    const uint32_t b = __byte_perm(s, 0u, (J & 1) ? 0x4342u : 0x4140u);  // 0x00 s1m1 00 s0m0
    const uint32_t e = (J == 0 ? q << 7 : J == 1 ? q << 3 : J == 2 ? q >> 1 : q >> 5) & 0x07800780u;
    return (b & 0x00800080u) * 0xFFu + b + e + basew;  // sign bit 7 -> 15: s * 0x7F80 + s * 0x80
}

// Weight formats. Lane (g = lane / 4, t = lane % 4) holds W rows g and g + 8 of a 16-row tile;
// per 32-wide k block it needs k in [8t, 8t + 8) of both rows: a[0..3] (row g) and b[0..3]
// (row g + 8) are the bf16x2 registers for k pairs (0,1), (2,3), (4,5), (6,7). MMA step 0 uses
// k 8t + 0..3 (fragment slots 2t, 2t+1, 2t+8, 2t+9), step 1 uses k 8t + 4..7; x is read with
// the same relabelling. A format stores NK k blocks of a tile as one unit of KB int4s, so a
// tile's whole K range is one linear stream. EXC: the format has an exception list.
struct Plain {  // pack_weight: per k block 64 int4s, rows 0-7 then rows 8-15, lane-major
    static constexpr int NK = 1, KB = 64;
    static constexpr bool EXC = false;
    struct Raw { int4 a, b; };
    __device__ static Raw load(const int4* unit, int lane) {
        return {ld_stream(unit + lane), ld_stream(unit + 32 + lane)};
    }
    __device__ static void frag(const Raw& r, int, uint32_t, int (&a)[4], int (&b)[4]) {
        a[0] = r.a.x; a[1] = r.a.y; a[2] = r.a.z; a[3] = r.a.w;
        b[0] = r.b.x; b[1] = r.b.y; b[2] = r.b.z; b[3] = r.b.w;
    }
};

// One k block of the 12-bit code: s = sign | mantissa of rows g and g + 8, q0 / q1 = their codes.
__device__ __forceinline__ void c12_frag(int4 s, uint32_t q0, uint32_t q1, uint32_t basew, int (&a)[4],
                                         int (&b)[4]) {
    a[0] = dec_pair<0>(s.x, q0, basew); a[1] = dec_pair<1>(s.x, q0, basew);
    a[2] = dec_pair<2>(s.y, q0, basew); a[3] = dec_pair<3>(s.y, q0, basew);
    b[0] = dec_pair<0>(s.z, q1, basew); b[1] = dec_pair<1>(s.z, q1, basew);
    b[2] = dec_pair<2>(s.w, q1, basew); b[3] = dec_pair<3>(s.w, q1, basew);
}

struct C12 {  // pack_c12 "w12": per k block 768 bytes, 512 sign | mantissa (16 B per lane), then
              // 256 code bytes (8 B per lane)
    static constexpr int NK = 1, KB = 48;
    static constexpr bool EXC = true;
    struct Raw { int4 s; int2 q; };
    __device__ static Raw load(const int4* unit, int lane) {
        return {ld_stream(unit + lane), ld_stream2(reinterpret_cast<const int2*>(unit + 32) + lane)};
    }
    __device__ static void frag(const Raw& r, int, uint32_t basew, int (&a)[4], int (&b)[4]) {
        c12_frag(r.s, r.q.x, r.q.y, basew, a, b);
    }
};

struct C12W {  // pack_c12 "w12w": per 2 k blocks 1536 bytes, sign | mantissa of block 0 and of
               // block 1 (16 B per lane each), then the codes of both (16 B per lane): 3 x 16 B loads
    static constexpr int NK = 2, KB = 96;
    static constexpr bool EXC = true;
    struct Raw { int4 s0, s1, q; };
    __device__ static Raw load(const int4* unit, int lane) {
        return {ld_stream(unit + lane), ld_stream(unit + 32 + lane), ld_stream(unit + 64 + lane)};
    }
    __device__ static void frag(const Raw& r, int j, uint32_t basew, int (&a)[4], int (&b)[4]) {
        if (j) c12_frag(r.s1, r.q.z, r.q.w, basew, a, b);
        else c12_frag(r.s0, r.q.x, r.q.y, basew, a, b);
    }
};

// Weights are stored in stream order (pack_stream in Python), specific to one launch geometry:
// a warp's step is RT * U / NK units, and at step s all G warps of the grid read one contiguous
// region: warp gw's step s is at (s * G + gw) * step. So the whole grid sweeps HBM front to back
// instead of 1000+ warps each walking their own far-apart stream.
// acc[rt] += W[rows of tile rt] @ x^T over k in [k_begin, k_end), for tokens 0 .. Mz - 1 of x
// (row stride KX; W column k meets x column k % KX, which lets the router read [hi | lo]).
template <class F, int RT, int MT, int U>
__device__ __forceinline__ void mainloop(float (&acc)[RT][MT][4], const int4* wq, size_t sstride,
                                         const __nv_bfloat16* x, int KX, int Mz, int k_begin, int k_end,
                                         uint32_t basew) {
    constexpr int VU = U / F::NK;
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const __nv_bfloat16* xrow[MT];
    bool xok[MT];
#pragma unroll
    for (int nt = 0; nt < MT; ++nt) {
        const int tok = nt * 8 + g;
        xok[nt] = tok < Mz;
        xrow[nt] = x + (size_t)(xok[nt] ? tok : 0) * KX + t * 8;
    }
    static_assert(U % F::NK == 0, "a step holds whole format units");
    struct Stage { typename F::Raw w[VU][RT]; int4 xv[U][MT]; };
    auto load = [&](Stage& st, int kb) {
        const int kx = kb % KX;
        const int4* p = wq + (size_t)((kb - k_begin) / (32 * U)) * sstride;
#pragma unroll
        for (int rt = 0; rt < RT; ++rt)
#pragma unroll
            for (int v = 0; v < VU; ++v) st.w[v][rt] = F::load(p + (rt * VU + v) * F::KB, lane);
#pragma unroll
        for (int u = 0; u < U; ++u) {
#pragma unroll
            for (int nt = 0; nt < MT; ++nt)
                st.xv[u][nt] = xok[nt] ? __ldg(reinterpret_cast<const int4*>(xrow[nt] + kx + u * 32))
                                       : make_int4(0, 0, 0, 0);
        }
    };
    auto compute = [&](const Stage& st) {
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int rt = 0; rt < RT; ++rt) {
                int a[4], b[4];
                F::frag(st.w[u / F::NK][rt], u % F::NK, basew, a, b);
#pragma unroll
                for (int nt = 0; nt < MT; ++nt) {
                    mma_bf16(acc[rt][nt], a[0], b[0], a[1], b[1], st.xv[u][nt].x, st.xv[u][nt].y);
                    mma_bf16(acc[rt][nt], a[2], b[2], a[3], b[3], st.xv[u][nt].z, st.xv[u][nt].w);
                }
            }
    };
    pipeline2<U, Stage>(k_begin, k_end, load, compute);
}

// C12 exceptions. pack_c12 stores them in EXC_CAP (64) slots per tile, sorted by k, padded with
// 0xFFFF: entry (k | row << 16, w_true | w_decoded << 16). Lane i loads slots i and 32 + i at
// kernel start (no dependent load, so the latency hides under the main loop).
constexpr int EXC_CAP = 64;
template <int RT>
__device__ __forceinline__ void exc_prefetch(int2 (&ec)[RT][2], const int2* __restrict__ exc, int tile0) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int rt = 0; rt < RT; ++rt)
#pragma unroll
        for (int c = 0; c < 2; ++c) ec[rt][c] = __ldg(exc + (size_t)(tile0 + rt) * EXC_CAP + c * 32 + lane);
}

// For the exceptions with k in [k_lo, k_hi), the lanes holding the row add
// w_true * x - w_decoded * x for their two tokens per 8-token tile.
template <int RT, int MT>
__device__ __forceinline__ void apply_exceptions(float (&acc)[RT][MT][4], const int2 (&ec)[RT][2],
                                                 const __nv_bfloat16* x, int K, int Mz, int k_lo, int k_hi) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int rt = 0; rt < RT; ++rt)
#pragma unroll
        for (int c = 0; c < 2; ++c)
            for (int i = 0; i < 32; ++i) {
                const int ex = __shfl_sync(0xFFFFFFFFu, ec[rt][c].x, i);
                if (ex == 0xFFFF) break;  // padding: the rest of the slots are empty
                const int ey = __shfl_sync(0xFFFFFFFFu, ec[rt][c].y, i);
                const int k = ex & 0xFFFF, r = ex >> 16;
                if (k < k_lo || k >= k_hi || (r & 7) != g) continue;
                const float wt = __uint_as_float((uint32_t)ey << 16);
                const float wd = __uint_as_float((uint32_t)ey & 0xFFFF0000u);
                const bool hi = r >= 8;
#pragma unroll
                for (int nt = 0; nt < MT; ++nt) {
                    const int tok = nt * 8 + 2 * t;
                    const float x0 = tok < Mz ? __bfloat162float(x[(size_t)tok * K + k]) : 0.f;
                    const float x1 = tok + 1 < Mz ? __bfloat162float(x[(size_t)(tok + 1) * K + k]) : 0.f;
                    float c0 = hi ? acc[rt][nt][2] : acc[rt][nt][0];
                    float c1 = hi ? acc[rt][nt][3] : acc[rt][nt][1];
                    c0 = fmaf(wt, x0, fmaf(-wd, x0, c0));
                    c1 = fmaf(wt, x1, fmaf(-wd, x1, c1));
                    if (hi) { acc[rt][nt][2] = c0; acc[rt][nt][3] = c1; }
                    else    { acc[rt][nt][0] = c0; acc[rt][nt][1] = c1; }
                }
            }
}

template <int RT, int MT>
__device__ __forceinline__ void zero(float (&acc)[RT][MT][4]) {
#pragma unroll
    for (int rt = 0; rt < RT; ++rt)
#pragma unroll
        for (int nt = 0; nt < MT; ++nt)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[rt][nt][j] = 0.f;
}

// out[M, N] = x[M, KX] @ W[N, KW]^T. 256 threads; grid (N / (16 * RT), ksplit, ceil(M / 32));
// ksplit > 1 runs as a (1, ksplit, 1) cluster; blockIdx.z picks 32 tokens (all groups read the
// same W stream). A CTA owns RT tiles (16 * RT rows); its 8 warps split the CTA's K range and
// add their partials in shared memory (reduce_store). w is in stream order for this geometry.
template <class F, int RT, int MT, int U, bool OUT_F32>
__global__ void __launch_bounds__(256) ksplit_kernel(
        const __nv_bfloat16* __restrict__ x, const int4* __restrict__ w, const int2* __restrict__ exc,
        void* __restrict__ out, int M, int N, int KX, int KW, int k_cta, uint32_t basew) {
    __shared__ float red[4][RT * MT * 128];
    const int warp = threadIdx.x >> 5;
    const int m0 = blockIdx.z * 32, Mz = min(M - m0, 32);
    x += (size_t)m0 * KX;
    const int kw = k_cta >> 3, k_begin = blockIdx.y * k_cta + warp * kw;
    constexpr int STEP = RT * (U / F::NK) * F::KB;  // int4s per warp step
    const int gw = (blockIdx.y * gridDim.x + blockIdx.x) * 8 + warp;
    const int4* wq = w + (size_t)gw * STEP;
    const size_t sstride = (size_t)gridDim.x * gridDim.y * 8 * STEP;
    int2 ec[RT][2];
    if constexpr (F::EXC) exc_prefetch(ec, exc, blockIdx.x * RT);
    float acc[RT][MT][4];
    zero(acc);
    mainloop<F, RT, MT, U>(acc, wq, sstride, x, KX, Mz, k_begin, k_begin + kw, basew);
    if constexpr (F::EXC) apply_exceptions(acc, ec, x, KX, Mz, k_begin, k_begin + kw);
    reduce_store<RT, MT, OUT_F32>(acc, red, out, blockIdx.x * 16 * RT, m0, Mz, N);
}

struct Args {
    const __nv_bfloat16* x; const int4* w; const int2* exc; void* out;
    int M, N, KX, KW, k_cta; uint32_t basew;
};

template <class F, int RT, int MT, int U, bool F32>
static cudaError_t go(cudaLaunchConfig_t* cfg, const Args& a) {
    auto kern = ksplit_kernel<F, RT, MT, U, F32>;
    if (cfg->gridDim.y > 8) {  // clusters of 16 CTAs are a non-portable size on Hopper
        const cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        if (e != cudaSuccess) return e;
    }
    return cudaLaunchKernelEx(cfg, kern, a.x, a.w, a.exc, a.out, a.M, a.N, a.KX, a.KW, a.k_cta, a.basew);
}

// Instances that fit in registers with two pipeline stages (see _unroll_ok in Python):
// U 8 only at RT 1, MT 1; U 4 at RT 1 or MT <= 2; U 2 always.
template <class F, int RT, int MT, bool F32>
static cudaError_t pick_u(cudaLaunchConfig_t* cfg, int u, const Args& a) {
    if constexpr (RT == 1 && MT == 1) {
        if (u == 8) return go<F, RT, MT, 8, F32>(cfg, a);
    }
    if constexpr (RT == 1 || MT <= 2) {
        if (u == 4) return go<F, RT, MT, 4, F32>(cfg, a);
    }
    if (u == 2) return go<F, RT, MT, 2, F32>(cfg, a);
    return cudaErrorInvalidValue;
}

template <class F, bool F32>
static cudaError_t pick(cudaLaunchConfig_t* cfg, int rt, int mt, int u, const Args& a) {
    if (rt == 1) {
        if (mt == 1) return pick_u<F, 1, 1, F32>(cfg, u, a);
        if (mt == 2) return pick_u<F, 1, 2, F32>(cfg, u, a);
        return pick_u<F, 1, 4, F32>(cfg, u, a);
    }
    if (mt == 1) return pick_u<F, 2, 1, F32>(cfg, u, a);
    if (mt == 2) return pick_u<F, 2, 2, F32>(cfg, u, a);
    return pick_u<F, 2, 4, F32>(cfg, u, a);
}

// fmt 0: Plain (FP32 output only here: the router), 1: C12, 2: C12W.
cudaError_t proj_launch(int fmt, bool f32, const void* x, const void* w, const void* exc, void* out, int M,
                        int N, int KX, int KW, int rt, int u, int ksplit, uint32_t basew, cudaStream_t stream) {
    if (N % (16 * rt) || (fmt > 0 && (f32 || KX != KW))) return cudaErrorInvalidValue;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(N / (16 * rt), ksplit, (M + 31) / 32);
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
    const Args a{static_cast<const __nv_bfloat16*>(x), static_cast<const int4*>(w), static_cast<const int2*>(exc),
                 out, M, N, KX, KW, KW / ksplit, basew};
    if (fmt == 1) return pick<C12, false>(&cfg, rt, mt, u, a);
    if (fmt == 2) return pick<C12W, false>(&cfg, rt, mt, u, a);
    return f32 ? pick<Plain, true>(&cfg, rt, mt, u, a) : pick<Plain, false>(&cfg, rt, mt, u, a);
}
"""

CUDA_TORCH = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

cudaError_t proj_launch(int fmt, bool f32, const void* x, const void* w, const void* exc, void* out, int M,
                        int N, int KX, int KW, int rt, int u, int ksplit, uint32_t basew, cudaStream_t stream);

void proj(torch::Tensor x, torch::Tensor w, torch::Tensor exc, torch::Tensor out, int64_t fmt, int64_t kx,
          int64_t kw, int64_t rt, int64_t u, int64_t ksplit, int64_t basew) {
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && out.is_contiguous());
    const cudaError_t err = proj_launch(
        fmt, out.scalar_type() == at::kFloat, x.data_ptr(), w.data_ptr(), exc.data_ptr(), out.data_ptr(),
        x.size(0), out.size(1), kx, kw, rt, u, ksplit, (uint32_t)basew, at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(err == cudaSuccess, "proj: ", cudaGetErrorString(err));
}
"""

CPP_DECLS = """
void proj(torch::Tensor x, torch::Tensor w, torch::Tensor exc, torch::Tensor out, int64_t fmt, int64_t kx,
          int64_t kw, int64_t rt, int64_t u, int64_t ksplit, int64_t basew);
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
                name="muse_bf16_proj_v5", cpp_sources=CPP_DECLS, cuda_sources=CUDA_KERNELS + CUDA_TORCH,
                functions=["proj"], extra_cuda_cflags=["-O3"], verbose=False)
            print(f"CUDA compiled in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001 - fall back to cuBLAS and say why
            print(f"CUDA compile failed after {time.time() - t0:.0f}s: {str(e)[-3000:]}", flush=True)
            _MOD = False
    return _MOD or None


BUCKETS = (1, 8, 16, 32, 128, 256)  # the scored token counts
LAYER_NAMES = ("qkv_gate", "o", "pooler", "unpooler", "router")

# (layer, bucket) -> (kind, (rt, u, ksplit)) with kind "cuda" (Plain format), "c12" or "c12w"
# (12-bit code); ("mm_t" | "mm_n", None) for cuBLAS on another weight layout; ("torch", None)
# for SGLang's own dispatch. Filled from BENCH runs.
PLAN = {  # check c30eb344 (submit 93e147b6 ran its sweep): our kernels where they beat SGLang by > 2%
    **{(layer, m): ("torch", None) for layer in LAYER_NAMES for m in BUCKETS},
    ("qkv_gate", 1): ("cuda", (1, 4, 1)),
    ("o", 1): ("c12w", (2, 2, 1)),
    ("pooler", 1): ("c12", (1, 4, 1)),
    ("unpooler", 1): ("cuda", (1, 8, 1)),
    ("router", 1): ("cuda", (2, 4, 8)),
    ("qkv_gate", 8): ("cuda", (1, 2, 2)),
    ("o", 8): ("cuda", (2, 2, 1)),
    ("pooler", 8): ("c12", (1, 4, 1)),
    ("unpooler", 8): ("cuda", (1, 8, 1)),
    ("router", 8): ("cuda", (1, 4, 8)),
    ("pooler", 16): ("cuda", (2, 2, 1)),
    ("router", 16): ("cuda", (2, 4, 8)),
    ("qkv_gate", 32): ("mm_t", None),
    ("pooler", 32): ("mm_t", None),
    ("unpooler", 32): ("mm_t", None),
    ("router", 32): ("cuda", (2, 2, 8)),
    ("o", 128): ("mm_n", None),
    ("pooler", 128): ("mm_t", None),
    ("unpooler", 128): ("mm_t", None),
    ("router", 128): ("mm_t", None),
    ("pooler", 256): ("mm_t", None),
    ("router", 256): ("mm_t", None),
}


def _unroll_ok(rt, u, m):
    """The (RT, U) instances the CUDA launchers compile for this token count (no spills)."""
    mt = 1 if m <= 8 else 2 if m <= 16 else 4
    return u == 2 or (u == 4 and (rt == 1 or mt <= 2)) or (u == 8 and rt == 1 and mt == 1)


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


def pack_weight(w):
    """[N, K] -> Plain units [N / 16, K / 32, 1024 bytes]: per 16-row tile and 32-wide k block,
    rows 0-7 then rows 8-15, each as lanes (row r, k quarter q) -> 16 bytes, lane = 4 * r + q."""
    n, k = w.shape
    units = w.view(n // 16, 2, 8, k // 32, 4, 8).permute(0, 3, 1, 2, 4, 5).contiguous()
    return units.view(torch.uint8).view(n // 16, k // 32, 1024)


def pack_stream(units, nk, rt, u, ks):
    """Units [tiles, n_units, bytes] (a unit holds nk k blocks of a tile) -> the order in which
    ksplit_kernel with (rt, u, ks) reads them: a warp step is rt * (u / nk) units, and step s of
    warp gw = (by * GX + bx) * 8 + warp sits at (s * G + gw) * step, so at each step the whole grid
    reads one contiguous region. Warp (bx, by, warp) owns tiles bx * rt .. + rt - 1 and units
    (by * 8 + warp) * per_warp .. + per_warp - 1 of each."""
    tiles, nu, nbytes = units.shape
    vu, per_warp = u // nk, nu // (ks * 8)
    shape = (per_warp // vu, ks, tiles // rt, 8, rt, vu)
    ix = [torch.arange(d, device=units.device).view([-1 if i == j else 1 for j in range(6)])
          for i, d in enumerate(shape)]
    s, by, bx, warp, r, v = ix
    src = (bx * rt + r) * nu + (by * 8 + warp) * per_warp + s * vu + v
    return units.reshape(tiles * nu, nbytes)[src.reshape(-1)].contiguous()


def _stream(state, layer, n, kind, rt, u, ks):
    """The stream-ordered weights for one launch geometry, packed once and cached."""
    key = (layer, n, kind, rt, u, ks)
    if key not in state["streams"]:
        units = state["units"][layer] if kind == "cuda" else state["c12"][layer]["w12" if kind == "c12" else "w12w"]
        state["streams"][key] = pack_stream(units[: n // 16], 2 if kind == "c12w" else 1, rt, u, ks)
    return state["streams"][key]


EXC_CAP = 64  # exception slots per 16-row tile (the CUDA constant of the same name)


def pack_c12(w):
    """BF16 [N, K] -> lossless 12-bit code (C12 and C12W formats), or None if it does not fit.

    base = (largest exponent field) - 15. A value with exponent e in [base, base + 15] keeps its
    sign and 7 mantissa bits as one byte and stores e - base as a 4-bit code. A value below the
    window (tiny values, zeros; about 1e-4 of Gaussian weights) gets code 0, so it decodes to
    sign | base | mantissa, and goes to its tile's exception slots with its true and decoded
    bits: EXC_CAP slots per tile, sorted by k, padded with k = 0xFFFF.
    Per 16-row tile and 32-wide k block, lane 4g + t holds 16 sign | mantissa bytes (row g then
    row g + 8, k 8t .. 8t + 7) and 8 code bytes: one 32-bit word per row, pair j (k 2j, 2j + 1)
    at nibbles j and j + 4.
      w12:  per k block 768 bytes: the 512 sign | mantissa bytes, then the 256 code bytes;
      w12w: per 2 k blocks 1536 bytes: sign | mantissa of both blocks, then 16 code bytes per lane.
    """
    n, k = w.shape
    tiles, nb = n // 16, k // 32
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e = (bits >> 7) & 0xFF
    base = max(int(e.max()) - 15, 0)
    exc = e < base
    rows, cols = exc.nonzero(as_tuple=True)
    tile_of = rows.long() // 16
    counts = torch.bincount(tile_of, minlength=tiles)
    if int(counts.max()) > EXC_CAP:
        return None
    code = torch.where(exc, 0, e - base)
    sm = ((bits >> 8) & 0x80) | (bits & 0x7F)

    def frag(a):  # [N, K] -> [tile, k block, g, t, row half, 8 values]
        return a.view(tiles, 2, 8, nb, 4, 8).permute(0, 3, 2, 4, 1, 5)

    c = frag(code)[..., [0, 2, 4, 6, 1, 3, 5, 7]]  # nibble order inside each 32-bit word
    sm_b = frag(sm).to(torch.uint8).reshape(tiles, nb, 512)
    ex_b = (c[..., 0::2] | (c[..., 1::2] << 4)).to(torch.uint8).reshape(tiles, nb, 32, 8)

    order = torch.argsort(tile_of * k + cols)  # nonzero() is row-major: sort by tile, then k
    rows, cols, tile_of = rows[order].long(), cols[order].long(), tile_of[order]
    true_bits = bits[rows, cols].long()
    hi = true_bits | (((true_bits & 0x807F) | (base << 7)) << 16)
    slot = tile_of * EXC_CAP + torch.arange(len(rows), device=w.device) - (counts.cumsum(0) - counts)[tile_of]
    slots = torch.zeros((tiles * EXC_CAP, 2), dtype=torch.int64, device=w.device)
    slots[:, 0] = 0xFFFF
    slots[slot, 0] = cols | ((rows % 16) << 16)
    slots[slot, 1] = torch.where(hi >= 1 << 31, hi - (1 << 32), hi)
    return {
        "w12": torch.cat([sm_b, ex_b.reshape(tiles, nb, 256)], 2).contiguous(),
        "w12w": torch.cat([sm_b.reshape(tiles, nb // 2, 1024),
                           ex_b.reshape(tiles, nb // 2, 2, 32, 8).transpose(2, 3).reshape(tiles, nb // 2, 512)],
                          2).contiguous(),
        "exc": slots.to(torch.int32).contiguous(),
        "basew": base << 7 | base << 23,
    }


def _proj(kind, x, state, layer, n, cfg):
    """Our kernel: kind "cuda" (Plain format), "c12" (C12) or "c12w" (C12W)."""
    rt, u, ksplit = cfg
    out = _out(x, layer, n)
    w = _stream(state, layer, n, kind, rt, u, ksplit)
    if kind == "cuda":
        k_x = state["router"].shape[1] if layer == "router" else x.shape[1]
        _MOD.proj(x, w, state["none"], out, 0, k_x, state["units"][layer].shape[1] * 32, rt, u, ksplit, 0)
    else:
        c = state["c12"][layer]
        _MOD.proj(x, w, c["exc"], out, 1 if kind == "c12" else 2, x.shape[1], x.shape[1], rt, u, ksplit,
                  c["basew"])
    return out


def _torch(x, state, layer, n):
    if layer == "router":
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
    return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)  # SGLang's own pick: never slower


def _mm(kind, x, state, layer, n):
    """cuBLAS with another weight layout, which makes it pick other kernels. mm_t: a [K, N]
    copy of W; mm_n: W [N, K] as a transposed view. The router runs as one [x | x] @ [hi | lo]^T."""
    if layer == "router":
        w = state["router_hilo_t"] if kind == "mm_t" else state["router_hilo"].t()
        return torch.mm(torch.cat([x, x], 1), w, out_dtype=torch.float32)
    w = state["wt"][layer][:, :n] if kind == "mm_t" else state[layer][:n].t()
    return torch.mm(x, w)


def _run(kind, cfg, x, state, layer, n):
    if kind in ("cuda", "c12", "c12w") and _MOD:
        if not kind.startswith("c12") or state["c12"].get(layer) is not None:
            return _proj(kind, x, state, layer, n, cfg)
    if kind in ("mm_t", "mm_n"):
        return _mm(kind, x, state, layer, n)
    return _torch(x, state, layer, n)


def prepare(weights: Weights):
    dev = weights["router"].device
    state = {name: w.contiguous() for name, w in weights.items()}
    hi = state["router"].to(torch.bfloat16)
    lo = (state["router"] - hi.float()).to(torch.bfloat16)
    state["router_hilo"] = torch.cat([hi, lo], dim=1).contiguous()
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["router_hilo_t"] = state["router_hilo"].t().contiguous()
    if BENCH or any(kind == "mm_t" for kind, _ in PLAN.values()):
        state["wt"] = {name: state[name].t().contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")}
    state["plan"] = dict(PLAN)
    state["units"] = {name: pack_weight(state[name]) for name in ("qkv_gate", "o", "pooler", "unpooler")}
    state["units"]["router"] = pack_weight(state["router_hilo"])
    state["c12"] = {name: pack_c12(state[name]) for name in ("qkv_gate", "o", "pooler", "unpooler")}
    state["none"] = torch.zeros(1, device=dev, dtype=torch.int32)  # no exception list (Plain format)
    state["streams"] = {}
    for (layer, m), (kind, cfg) in PLAN.items():  # stream order for every planned launch
        if kind in ("cuda", "c12", "c12w") and (kind == "cuda" or state["c12"].get(layer) is not None):
            n = 1024 if layer == "unpooler" and m <= 8 else state["units"][layer].shape[0] * 16
            _stream(state, layer, n, kind, *cfg)
    if BENCH or any(kind != "torch" for kind, _ in PLAN.values()):
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
             f"l2={p.L2_cache_size >> 20}MB", f"torch {torch.__version__}"]
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


def _variants(state, m, layer, n, k_w):
    out = [("torch", None), ("mm_t", None), ("mm_n", None)]
    if PLAN.get((layer, m), ("torch",))[0] != "torch":
        out.append(PLAN[(layer, m)])
    if not _MOD or (m > 32 and layer != "router"):  # token groups lost 3-7x to cuBLAS (check 45fd2ba0)
        return out
    has_c12 = state["c12"].get(layer) is not None
    for kind in ["cuda"] + (["c12", "c12w"] if has_c12 else []):
        for rt in (1, 2):
            for u in (2, 4, 8):
                for ks in (1, 2, 4, 8, 16):
                    ctas = n // (16 * rt) * ks
                    if (n % (16 * rt) or k_w % (ks * 8 * 32 * u) or not 32 <= ctas <= 2112
                            or not _unroll_ok(rt, u, m)):
                        continue
                    out.append((kind, (rt, u, ks)))
    return out


def _c12_probe(state, layer, n):
    """One-hot rows of x at columns that hold exceptions (values below the code's window), and
    W[:n] at those columns: the kernel's output must equal it bit for bit."""
    w = state[layer][:n]
    base = (state["c12"][layer]["basew"] >> 7) & 0xFF
    exc = ((w.view(torch.int16).to(torch.int32) >> 7) & 0xFF) < base
    cols = torch.cat([exc.nonzero()[:, 1].unique(), torch.arange(32, device=w.device)])[:32]
    x = torch.zeros((32, w.shape[1]), device=w.device, dtype=torch.bfloat16)
    x[torch.arange(32, device=w.device), cols] = 1.0
    return x, w[:, cols].T.contiguous(), int(exc.sum())


def _c12_exact(state, layer, n, kind, cfg, probe, m):
    """Bit-exact on the probe, with as many rows as the token tile class of m holds."""
    x, want, _ = probe
    rows = 8 if m <= 8 else 16 if m <= 16 else 32
    return torch.equal(_proj(kind, x[:rows], state, layer, n, cfg), want[:rows])


def _rel(a, b):
    return float((a.double() - b).norm() / b.norm().clamp(min=1e-30))


def _sweep(state, budget_s=150.0):
    t0 = time.time()
    dev = state["router"].device
    for layer, c in state["c12"].items():
        if c is not None:
            n_exc = _c12_probe(state, layer, state[layer].shape[0])[2]
            print(f"C12 {layer}: base exponent {(c['basew'] >> 7) & 0xFF}, {n_exc} exceptions", flush=True)
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    flush = torch.empty(max(4 * l2, 256 << 20) // 4, device=dev, dtype=torch.float32).fill_(1.0)
    gen = torch.Generator(device=dev).manual_seed(1)
    rows, load_t, probes = [], {}, {}
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
            for kind, cfg in _variants(state, m, layer, n, w.shape[1]):
                if time.time() - t0 > budget_s:
                    break
                try:
                    err = _rel(_run(kind, cfg, x, state, layer, n), ref)
                    t, out = _graph_us(lambda: _run(kind, cfg, x, state, layer, n), flush)
                    err = max(err, _rel(out, ref))  # the graph replays must be right too
                    if err > tol:
                        bad.append(f"{kind}{cfg}:{err:.1e}")
                        continue
                    if kind.startswith("c12"):
                        if (layer, n) not in probes:
                            probes[(layer, n)] = _c12_probe(state, layer, n)
                        if not _c12_exact(state, layer, n, kind, cfg, probes[(layer, n)], m):
                            bad.append(f"{kind}{cfg}:inexact")
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
            tag = {"cuda": "c", "c12": "z", "c12w": "w", "torch": "t", "mm_t": "T", "mm_n": "N"}
            top = " ".join(f"{t - FIXED_US:.2f}{tag[k]}{list(c) if c else ''}" for t, k, c in results[:3])
            rows.append(f"m={m:<3} {layer:<8} sgl={sgl:6.2f} torch={t_torch:6.2f} best={best:6.2f} "
                        f"x{sgl / max(best, 0.05):.2f} | {top}" + (f" | bad {' '.join(bad[:4])}" if bad else ""))
    ratios = [s / max(o, 0.05) for s, o in load_t.values()]
    est = math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if ratios else 0.0
    print("BENCH net us (judge's 3.8us fixed cost removed); [rt,u,ksplit] of c=cuda z=c12 w=c12w; "
          "t=SGLang dispatch, T/N=cuBLAS on W^T copy / W view", flush=True)
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
