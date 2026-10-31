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

// Three register stages: the loads of steps i + 1 and i + 2 are in flight while step i computes.
template <int U, class Stage, class Load, class Compute>
__device__ __forceinline__ void pipeline3(int k_begin, int k_end, Load& load, Compute& compute) {
    const int step = 32 * U, S = (k_end - k_begin) / step;
    Stage A, B, C;
    load(A, k_begin);
    if (S > 1) load(B, k_begin + step);
    for (int s = 0;; s += 3) {
        if (s + 2 < S) load(C, k_begin + (s + 2) * step);
        compute(A);
        if (s + 1 >= S) break;
        if (s + 3 < S) load(A, k_begin + (s + 3) * step);
        compute(B);
        if (s + 2 >= S) break;
        if (s + 4 < S) load(B, k_begin + (s + 4) * step);
        compute(C);
        if (s + 3 >= S) break;
    }
}

// The end of ksplit_kernel. The CTA's NW = WR * WK warps write their partials to shared memory;
// for each of the WR row groups, the sum over its WK k-warps is formed by adjacent threads
// (element idx = wr * E + e). With ksplit > 1 the cluster's CTAs then either reduce-scatter (rank
// r sums its 1/ksplit slice of the elements over all ranks through DSMEM loads and stores that
// slice) or, with push, the other ranks store their sums into rank 0's red (DSMEM stores, no
// round trip) and rank 0 adds them and stores everything (needs ksplit - 1 <= WK).
// Rows row0 .. row0 + 16 * RT * WR - 1, tokens m0 + tok.
template <int RT, int MT, bool OUT_F32, int WR, int WK>
__device__ __forceinline__ void reduce_store(float (&acc)[RT][MT][4], float* red, void* out, int row0, int m0,
                                             int M, int N, int push) {
    constexpr int E = RT * MT * 128, NT = WR * WK * 32, F = WR * E, PER = (F + NT - 1) / NT;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = threadIdx.x;
#pragma unroll
    for (int rt = 0; rt < RT; ++rt)
#pragma unroll
        for (int nt = 0; nt < MT; ++nt)
#pragma unroll
            for (int j = 0; j < 4; ++j) red[warp * E + ((rt * MT + nt) * 4 + j) * 32 + lane] = acc[rt][nt][j];
    __syncthreads();
    float v[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int idx = i * NT + tid;
        v[i] = 0.f;
        if (idx < F)
#pragma unroll
            for (int wk = 0; wk < WK; ++wk) v[i] += red[(wk * WR + idx / E) * E + idx % E];
    }
    auto store = [&](int idx, float s) {
        const int e = idx % E, ln = e & 31, j = (e >> 5) & 3, nt = (e >> 7) % MT, rt = (e >> 7) / MT;
        const int row = row0 + (idx / E * RT + rt) * 16 + (ln >> 2) + (j >= 2 ? 8 : 0);
        const int tok = nt * 8 + 2 * (ln & 3) + (j & 1);
        if (tok < M) {
            const size_t o = (size_t)(m0 + tok) * N + row;
            if (OUT_F32) reinterpret_cast<float*>(out)[o] = s;
            else reinterpret_cast<__nv_bfloat16*>(out)[o] = __float2bfloat16(s);
        }
    };
    const int ksplit = gridDim.y;
    if (ksplit == 1) {
#pragma unroll
        for (int i = 0; i < PER; ++i)
            if (i * NT + tid < F) store(i * NT + tid, v[i]);
        return;
    }
    cg::cluster_group cluster = cg::this_cluster();
    if (push) {
        const int rank = (int)cluster.block_rank();
        cluster.sync();  // rank 0 has read its red before the others write into it
        if (rank) {
            float* dst = cluster.map_shared_rank(red, 0) + (rank - 1) * F;
#pragma unroll
            for (int i = 0; i < PER; ++i)
                if (i * NT + tid < F) dst[i * NT + tid] = v[i];
        }
        cluster.sync();  // the pushed sums are visible in rank 0's red
        if (rank) return;
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            const int idx = i * NT + tid;
            if (idx < F) {
                float s = v[i];
#pragma unroll 1
                for (int q = 0; q < ksplit - 1; ++q) s += red[q * F + idx];
                store(idx, s);
            }
        }
        return;
    }
    __syncthreads();  // every warp has read red before it is overwritten with the CTA sums
#pragma unroll
    for (int i = 0; i < PER; ++i)
        if (i * NT + tid < F) red[i * NT + tid] = v[i];
    cluster.sync();
    const int chunk = F / ksplit, e0 = (int)cluster.block_rank() * chunk;  // F % ksplit == 0
#pragma unroll 1
    for (int i = tid; i < chunk; i += NT) {
        float s = 0.f;
#pragma unroll 4
        for (int q = 0; q < ksplit; ++q) s += cluster.map_shared_rank(red, q)[e0 + i];
        store(e0 + i, s);
    }
    cluster.sync();  // the others' shared memory must outlive this CTA's reads
}

// 8 values of one W row from the 12-bit code (see pack_c12 in Python). L0 / L1: the low bytes of
// values 0-3 / 4-7 (bf16 bits 0-7: exponent LSB and mantissa). C: nibble i holds, in bits 0-2, the
// index of value i's high exponent bits (bf16 bits 8-14) in the table T0 | T1 (bytes base + 0..7);
// bit 3 of nibbles 0, 4, 1, 5, 2, 6, 3, 7 holds the sign of values 0..7. Two prmt look up the
// high bytes, four prmt interleave them with the low bytes, the shifted C adds the signs.
__device__ __forceinline__ void c12_row(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1,
                                        int (&a)[4]) {
    const uint32_t cm = C & 0x77777777u;
    const uint32_t hlo = __byte_perm(T0, T1, cm), hhi = __byte_perm(T0, T1, cm >> 16);
    a[0] = __byte_perm(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);
    a[1] = __byte_perm(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);
    a[2] = __byte_perm(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);
    a[3] = __byte_perm(L1, hhi, 0x7362u) | (C & 0x80008000u);
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

// One k block of the 12-bit code: s = low bytes of rows g and g + 8, q0 / q1 = their code words,
// base = the table's first high-exponent value (bytes base .. base + 7).
__device__ __forceinline__ void c12_frag(int4 s, uint32_t q0, uint32_t q1, uint32_t base, int (&a)[4],
                                         int (&b)[4]) {
    const uint32_t T0 = base * 0x01010101u + 0x03020100u, T1 = T0 + 0x04040404u;
    c12_row(s.x, s.y, q0, T0, T1, a);
    c12_row(s.z, s.w, q1, T0, T1, b);
}

struct C12 {  // pack_c12 "w12": per k block 768 bytes, 512 low bytes (16 B per lane), then
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

struct C12W {  // pack_c12 "w12w": per 2 k blocks 1536 bytes, low bytes of block 0 and of block 1
               // (16 B per lane each), then the codes of both (16 B per lane): 3 x 16 B loads
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
// Formats with EXC: esc is this warp's escape list (pack_escapes in Python): up to 32 entries, one
// per lane, sorted by seq = k block index within the warp * RT + rt: seq << 17 | lane << 12 |
// register << 9 | half << 8 | the 7-bit XOR that turns the decoded high exponent bits into the true ones
// (0xFFFFFFFF ends the list). The owning lane patches its decoded register before the MMA.
// DIAG (timing probes only, wrong results): 1 no decode, 2 no escapes, 4 no MMA, 8 no x loads.
template <class F, int RT, int MT, int U, int DIAG = 0, int P = 2>
__device__ __forceinline__ void mainloop(float (&acc)[RT][MT][4], const int4* wq, size_t sstride,
                                         const __nv_bfloat16* x, int KX, int Mz, int k_begin, int k_end,
                                         uint32_t basew, const uint32_t* esc) {
    constexpr int VU = U / F::NK;
    constexpr bool ESC = F::EXC && !(DIAG & 2);
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const __nv_bfloat16* xrow[MT];
    bool xok[MT];
#pragma unroll
    for (int nt = 0; nt < MT; ++nt) {
        const int tok = nt * 8 + g;
        xok[nt] = tok < Mz;
        xrow[nt] = x + (size_t)(xok[nt] ? tok : 0) * KX + t * 8;
    }
    uint32_t my_e = 0xFFFFFFFFu, nxt = 0xFFFFFFFFu;
    int ep = 0;
    if constexpr (ESC) my_e = __ldg(esc + lane);
    static_assert(U % F::NK == 0, "a step holds whole format units");
    struct Stage { typename F::Raw w[VU][RT]; int4 xv[U][MT]; int s; };
    auto load = [&](Stage& st, int kb) {
        const int kx = kb % KX;
        st.s = (kb - k_begin) / (32 * U);
        const int4* p = wq + (size_t)st.s * sstride;
#pragma unroll
        for (int rt = 0; rt < RT; ++rt)
#pragma unroll
            for (int v = 0; v < VU; ++v) st.w[v][rt] = F::load(p + (rt * VU + v) * F::KB, lane);
#pragma unroll
        for (int u = 0; u < U; ++u) {
#pragma unroll
            for (int nt = 0; nt < MT; ++nt)
                st.xv[u][nt] = (DIAG & 8) ? make_int4(0x3F803F80, 0x3F803F80, 0x3F803F80, 0x3F803F80)
                             : xok[nt] ? __ldg(reinterpret_cast<const int4*>(xrow[nt] + kx + u * 32))
                                       : make_int4(0, 0, 0, 0);
        }
    };
    auto compute = [&](const Stage& st) {
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int rt = 0; rt < RT; ++rt) {
                int a[4], b[4];
                if constexpr (DIAG & 1) {  // the raw bits, no decode
                    const int* r = reinterpret_cast<const int*>(&st.w[u / F::NK][rt]);
#pragma unroll
                    for (int i = 0; i < 4; ++i) { a[i] = r[i]; b[i] = r[i] ^ r[4]; }
                } else {
                    F::frag(st.w[u / F::NK][rt], u % F::NK, basew, a, b);
                }
                if constexpr (ESC) {
                    const uint32_t seq = (uint32_t)((st.s * U + u) * RT + rt);
                    while ((nxt >> 17) == seq) {  // nxt is the same in all lanes: no divergence
                        if (lane == (int)((nxt >> 12) & 31)) {
                            const uint32_t reg = (nxt >> 9) & 7, xr = (nxt & 0x7F) << (8 + ((nxt >> 8) & 1) * 16);
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                if (reg == (uint32_t)i) a[i] ^= xr;
                                if (reg == (uint32_t)(4 + i)) b[i] ^= xr;
                            }
                        }
                        ++ep;
                        nxt = ep < 32 ? __shfl_sync(0xFFFFFFFFu, my_e, ep) : 0xFFFFFFFFu;
                    }
                }
#pragma unroll
                for (int nt = 0; nt < MT; ++nt) {
                    if constexpr (DIAG & 4) {  // fold the operands instead of the MMA
                        acc[rt][nt][0] += __int_as_float((a[0] ^ a[1] ^ a[2] ^ a[3] ^ b[0] ^ b[1] ^ b[2] ^ b[3] ^
                                                          st.xv[u][nt].x ^ st.xv[u][nt].w) & 0x3FFFFFFF);
                    } else {
                        mma_bf16(acc[rt][nt], a[0], b[0], a[1], b[1], st.xv[u][nt].x, st.xv[u][nt].y);
                        mma_bf16(acc[rt][nt], a[2], b[2], a[3], b[3], st.xv[u][nt].z, st.xv[u][nt].w);
                    }
                }
            }
    };
    if constexpr (ESC) nxt = __shfl_sync(0xFFFFFFFFu, my_e, 0);
    if constexpr (P == 3) pipeline3<U, Stage>(k_begin, k_end, load, compute);
    else pipeline2<U, Stage>(k_begin, k_end, load, compute);
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

// out[M, N] = x[M, KX] @ W[N, KW]^T. WR * WK * 32 threads; grid (N / (16 * RT * WR), ksplit, ceil(M / 32));
// ksplit > 1 runs as a (1, ksplit, 1) cluster; blockIdx.z picks 32 tokens (all groups read the
// same W stream). A CTA has WR x WK warps: warp (wr = warp % WR, wk = warp / WR) owns the RT tiles
// (blockIdx.x * WR + wr) * RT .. + RT - 1 and the wk-th of WK slices of the CTA's K range. The WR
// row groups share x through L1; the WK k-warps add their partials in shared memory
// (reduce_store). w is in stream order for this geometry.
template <class F, int RT, int MT, int U, bool OUT_F32, int WR, int WK, int DIAG = 0, int P = 2>
__global__ void __launch_bounds__(WR * WK * 32) ksplit_kernel(
        const __nv_bfloat16* __restrict__ x, const int4* __restrict__ w, const uint32_t* __restrict__ esc,
        void* __restrict__ out, int M, int N, int KX, int KW, int k_cta, uint32_t basew, int push) {
    constexpr int NW = WR * WK;
    __shared__ float red[NW * RT * MT * 128];
    const int warp = threadIdx.x >> 5, wk = warp / WR;
    const int m0 = blockIdx.z * 32, Mz = min(M - m0, 32);
    x += (size_t)m0 * KX;
    const int kw = k_cta / WK, k_begin = blockIdx.y * k_cta + wk * kw;
    constexpr int STEP = RT * (U / F::NK) * F::KB;  // int4s per warp step
    const int gw = (blockIdx.y * gridDim.x + blockIdx.x) * NW + warp;
    const int4* wq = w + (size_t)gw * STEP;
    const size_t sstride = (size_t)gridDim.x * gridDim.y * NW * STEP;
    float acc[RT][MT][4];
    zero(acc);
    mainloop<F, RT, MT, U, DIAG, P>(acc, wq, sstride, x, KX, Mz, k_begin, k_begin + kw, basew, esc + (size_t)gw * 32);
    reduce_store<RT, MT, OUT_F32, WR, WK>(acc, red, out, blockIdx.x * 16 * RT * WR, m0, Mz, N, push);
}

struct Args {
    const __nv_bfloat16* x; const int4* w; const uint32_t* esc; void* out;
    int M, N, KX, KW, k_cta; uint32_t basew; int push;
};

// Timing probe: C12, RT 1, MT 1, U 4, warp grid (1, 8), ksplit 1, with DIAG bits (see mainloop).
cudaError_t diag_launch(const void* x, const void* w, const void* exc, void* out, int M, int N, int K,
                        uint32_t basew, int diag, cudaStream_t stream) {
    const dim3 grid(N / 16, 1, (M + 31) / 32), block(256);
    auto xp = static_cast<const __nv_bfloat16*>(x);
    auto wp = static_cast<const int4*>(w);
    auto ep = static_cast<const uint32_t*>(exc);
    if (diag == 1) ksplit_kernel<C12, 1, 1, 4, false, 1, 8, 1><<<grid, block, 0, stream>>>(xp, wp, ep, out, M, N, K, K, K, basew, 0);
    else if (diag == 2) ksplit_kernel<C12, 1, 1, 4, false, 1, 8, 2><<<grid, block, 0, stream>>>(xp, wp, ep, out, M, N, K, K, K, basew, 0);
    else if (diag == 4) ksplit_kernel<C12, 1, 1, 4, false, 1, 8, 4><<<grid, block, 0, stream>>>(xp, wp, ep, out, M, N, K, K, K, basew, 0);
    else if (diag == 8) ksplit_kernel<C12, 1, 1, 4, false, 1, 8, 8><<<grid, block, 0, stream>>>(xp, wp, ep, out, M, N, K, K, K, basew, 0);
    else if (diag == 15) ksplit_kernel<C12, 1, 1, 4, false, 1, 8, 15><<<grid, block, 0, stream>>>(xp, wp, ep, out, M, N, K, K, K, basew, 0);
    else return cudaErrorInvalidValue;
    return cudaGetLastError();
}

template <class F, int RT, int MT, int U, bool F32, int WR, int WK, int P>
static cudaError_t go(cudaLaunchConfig_t* cfg, const Args& a) {
    auto kern = ksplit_kernel<F, RT, MT, U, F32, WR, WK, 0, P>;
    if (cfg->gridDim.y > 8) {  // clusters of 16 CTAs are a non-portable size on Hopper
        const cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        if (e != cudaSuccess) return e;
    }
    return cudaLaunchKernelEx(cfg, kern, a.x, a.w, a.esc, a.out, a.M, a.N, a.KX, a.KW, a.k_cta, a.basew, a.push);
}

// Warp grids (WR, WK): 0 = (1, 8) 256 threads, 1 = (2, 8) 512, 2 = (2, 16) 1024, 3 = (4, 8) 1024,
// 4 = (1, 8) 256 threads with three pipeline stages: U 4 at RT 1 and MT <= 2; U 2 at RT 1 or MT <= 2.
// 5 = (1, 4) 128 threads; 6 = (1, 4) and 7 = (1, 8) with the push reduction (ksplit - 1 <= WK).
// Instances that fit in registers with two pipeline stages (mirrored by _unroll_ok in Python):
// 128 / 256 threads: U 8 only at RT 1, MT 1; U 4 at RT 1 or MT <= 2; U 2 always.
// 512 threads: RT 1; U 2, or U 4 at MT 1.  1024 threads: RT 1, MT 1, U 1 or 2.
template <class F, int RT, int MT, bool F32, int WR, int WK, int P>
static cudaError_t pick_u(cudaLaunchConfig_t* cfg, int u, const Args& a) {
    constexpr int NT = WR * WK * 32;
    if constexpr (P == 3) {  // grid 4: (1, 8) with three stages
        if constexpr (RT == 1 && MT <= 2) {
            if (u == 4) return go<F, RT, MT, 4, F32, WR, WK, 3>(cfg, a);
        }
        if constexpr (RT == 1 || MT <= 2) {
            if (u == 2) return go<F, RT, MT, 2, F32, WR, WK, 3>(cfg, a);
        }
    } else if constexpr (NT <= 256) {
        if constexpr (RT == 1 && MT == 1) {
            if (u == 8) return go<F, RT, MT, 8, F32, WR, WK, 2>(cfg, a);
        }
        if constexpr (RT == 1 || MT <= 2) {
            if (u == 4) return go<F, RT, MT, 4, F32, WR, WK, 2>(cfg, a);
        }
        if (u == 2) return go<F, RT, MT, 2, F32, WR, WK, 2>(cfg, a);
    } else if constexpr (NT == 512 && RT == 1 && !F32) {
        if constexpr (MT == 1) {
            if (u == 4) return go<F, RT, MT, 4, F32, WR, WK, 2>(cfg, a);
        }
        if (u == 2) return go<F, RT, MT, 2, F32, WR, WK, 2>(cfg, a);
    } else if constexpr (NT == 1024 && RT == 1 && MT == 1 && !F32) {
        if constexpr (F::NK == 1) {
            if (u == 1) return go<F, RT, MT, 1, F32, WR, WK, 2>(cfg, a);
        }
        if (u == 2) return go<F, RT, MT, 2, F32, WR, WK, 2>(cfg, a);
    }
    return cudaErrorInvalidValue;
}

template <class F, bool F32, int WR, int WK, int P>
static cudaError_t pick_g(cudaLaunchConfig_t* cfg, int rt, int mt, int u, const Args& a) {
    if (rt == 1) {
        if (mt == 1) return pick_u<F, 1, 1, F32, WR, WK, P>(cfg, u, a);
        if (mt == 2) return pick_u<F, 1, 2, F32, WR, WK, P>(cfg, u, a);
        return pick_u<F, 1, 4, F32, WR, WK, P>(cfg, u, a);
    }
    if (mt == 1) return pick_u<F, 2, 1, F32, WR, WK, P>(cfg, u, a);
    if (mt == 2) return pick_u<F, 2, 2, F32, WR, WK, P>(cfg, u, a);
    return pick_u<F, 2, 4, F32, WR, WK, P>(cfg, u, a);
}

template <class F, bool F32>
static cudaError_t pick(cudaLaunchConfig_t* cfg, int g, int rt, int mt, int u, const Args& a) {
    if (g == 0) return pick_g<F, F32, 1, 8, 2>(cfg, rt, mt, u, a);
    if (g == 1) return pick_g<F, F32, 2, 8, 2>(cfg, rt, mt, u, a);
    if (g == 2) return pick_g<F, F32, 2, 16, 2>(cfg, rt, mt, u, a);
    if (g == 3) return pick_g<F, F32, 4, 8, 2>(cfg, rt, mt, u, a);
    if (g == 4) return pick_g<F, F32, 1, 8, 3>(cfg, rt, mt, u, a);
    if (g == 5 || g == 6) return pick_g<F, F32, 1, 4, 2>(cfg, rt, mt, u, a);
    return pick_g<F, F32, 1, 8, 2>(cfg, rt, mt, u, a);
}

static const int GEOM_WR[8] = {1, 2, 2, 4, 1, 1, 1, 1}, GEOM_WK[8] = {8, 8, 16, 8, 8, 4, 4, 8};

// fmt 0: Plain (FP32 output only here: the router), 1: C12, 2: C12W. g: warp grid (see pick_u).
cudaError_t proj_launch(int fmt, bool f32, const void* x, const void* w, const void* exc, void* out, int M,
                        int N, int KX, int KW, int g, int rt, int u, int ksplit, uint32_t basew,
                        cudaStream_t stream) {
    if (g < 0 || g > 7) return cudaErrorInvalidValue;
    const int wr = GEOM_WR[g], wk = GEOM_WK[g], push = g >= 6 && ksplit > 1;
    if (N % (16 * rt * wr) || (fmt > 0 && (f32 || KX != KW)) || (push && ksplit - 1 > wk))
        return cudaErrorInvalidValue;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(N / (16 * rt * wr), ksplit, (M + 31) / 32);
    cfg.blockDim = dim3(wr * wk * 32, 1, 1);
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 1;
    attr[0].val.clusterDim.y = ksplit;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = ksplit > 1 ? 1 : 0;
    const int mt = M <= 8 ? 1 : M <= 16 ? 2 : 4;
    const Args a{static_cast<const __nv_bfloat16*>(x), static_cast<const int4*>(w), static_cast<const uint32_t*>(exc),
                 out, M, N, KX, KW, KW / ksplit, basew, push};
    if (fmt == 1) return pick<C12, false>(&cfg, g, rt, mt, u, a);
    if (fmt == 2) return pick<C12W, false>(&cfg, g, rt, mt, u, a);
    return f32 ? pick<Plain, true>(&cfg, g, rt, mt, u, a) : pick<Plain, false>(&cfg, g, rt, mt, u, a);
}

// Router at few tokens: FP32 weights read as they are, FP32 FMAs, no tensor cores, no split
// across CTAs. One CTA per router row (256 CTAs x 512 threads); each thread issues all 4 of its
// 16-byte weight loads at once, then reads x (bf16, clamped to row m - 1 so no branch) and adds
// M partial dot products; warp shuffles + one shared-memory step finish the row.
template <int M>
__global__ void __launch_bounds__(512) router_scalar_kernel(const float* __restrict__ w,
                                                            const __nv_bfloat16* __restrict__ x,
                                                            float* __restrict__ out, int m) {
    constexpr int K = 8192, NT = 512, PER = K / 4 / NT;
    const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int4* wr = reinterpret_cast<const int4*>(w + (size_t)row * K);
    int4 wv[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) wv[i] = ld_stream(wr + tid + i * NT);
    float acc[M];
#pragma unroll
    for (int t = 0; t < M; ++t) acc[t] = 0.f;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int c = (tid + i * NT) * 4;
        const float4 f = make_float4(__int_as_float(wv[i].x), __int_as_float(wv[i].y), __int_as_float(wv[i].z),
                                     __int_as_float(wv[i].w));
#pragma unroll
        for (int t = 0; t < M; ++t) {
            const uint2 xv = __ldg(reinterpret_cast<const uint2*>(x + (size_t)min(t, m - 1) * K + c));
            const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.x));
            const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.y));
            acc[t] = fmaf(f.w, b.y, fmaf(f.z, b.x, fmaf(f.y, a.y, fmaf(f.x, a.x, acc[t]))));
        }
    }
    __shared__ float sums[M][NT / 32];
#pragma unroll
    for (int t = 0; t < M; ++t) {
        float v = acc[t];
#pragma unroll
        for (int d = 16; d; d >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, d);
        if (lane == 0) sums[t][warp] = v;
    }
    __syncthreads();
    if (warp == 0)
#pragma unroll
        for (int t = 0; t < M; ++t) {
            float v = lane < NT / 32 ? sums[t][lane] : 0.f;
#pragma unroll
            for (int d = 8; d; d >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, d);
            if (lane == 0 && t < m) out[t * 256 + row] = v;
        }
}

cudaError_t router_scalar_launch(const void* w, const void* x, void* out, int m, cudaStream_t stream) {
    auto wp = static_cast<const float*>(w);
    auto xp = static_cast<const __nv_bfloat16*>(x);
    auto op = static_cast<float*>(out);
    if (m == 1) router_scalar_kernel<1><<<256, 512, 0, stream>>>(wp, xp, op, m);
    else if (m <= 4) router_scalar_kernel<4><<<256, 512, 0, stream>>>(wp, xp, op, m);
    else if (m <= 8) router_scalar_kernel<8><<<256, 512, 0, stream>>>(wp, xp, op, m);
    else return cudaErrorInvalidValue;
    return cudaGetLastError();
}
"""

CUDA_TORCH = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

cudaError_t proj_launch(int fmt, bool f32, const void* x, const void* w, const void* exc, void* out, int M,
                        int N, int KX, int KW, int g, int rt, int u, int ksplit, uint32_t basew,
                        cudaStream_t stream);
cudaError_t router_scalar_launch(const void* w, const void* x, void* out, int m, cudaStream_t stream);
cudaError_t diag_launch(const void* x, const void* w, const void* exc, void* out, int M, int N, int K,
                        uint32_t basew, int diag, cudaStream_t stream);

void diag(torch::Tensor x, torch::Tensor w, torch::Tensor exc, torch::Tensor out, int64_t basew, int64_t d) {
    const cudaError_t err = diag_launch(x.data_ptr(), w.data_ptr(), exc.data_ptr(), out.data_ptr(), x.size(0),
                                        out.size(1), x.size(1), (uint32_t)basew, d, at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(err == cudaSuccess, "diag: ", cudaGetErrorString(err));
}

void router_scalar(torch::Tensor x, torch::Tensor w, torch::Tensor out) {
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && out.is_contiguous() && x.size(1) == 8192);
    const cudaError_t err = router_scalar_launch(w.data_ptr(), x.data_ptr(), out.data_ptr(), x.size(0),
                                                 at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(err == cudaSuccess, "router_scalar: ", cudaGetErrorString(err));
}

void proj(torch::Tensor x, torch::Tensor w, torch::Tensor exc, torch::Tensor out, int64_t fmt, int64_t kx,
          int64_t kw, int64_t g, int64_t rt, int64_t u, int64_t ksplit, int64_t basew) {
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && out.is_contiguous());
    const cudaError_t err = proj_launch(
        fmt, out.scalar_type() == at::kFloat, x.data_ptr(), w.data_ptr(), exc.data_ptr(), out.data_ptr(),
        x.size(0), out.size(1), kx, kw, g, rt, u, ksplit, (uint32_t)basew, at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(err == cudaSuccess, "proj: ", cudaGetErrorString(err));
}
"""

CPP_DECLS = """
void proj(torch::Tensor x, torch::Tensor w, torch::Tensor exc, torch::Tensor out, int64_t fmt, int64_t kx,
          int64_t kw, int64_t g, int64_t rt, int64_t u, int64_t ksplit, int64_t basew);
void router_scalar(torch::Tensor x, torch::Tensor w, torch::Tensor out);
void diag(torch::Tensor x, torch::Tensor w, torch::Tensor exc, torch::Tensor out, int64_t basew, int64_t d);
"""

# cuBLASLt with a fixed algorithm, for the m >= 128 shapes (a separate module: if it does not build,
# only the "lt" variants are lost). An algorithm is 9 ints: ID, TILE_ID, SPLITK_NUM, REDUCTION_SCHEME,
# CTA_SWIZZLING, CUSTOM_OPTION, STAGES_ID, INNER_SHAPE_ID, CLUSTER_SHAPE_ID. Layouts of out = x @ W^T:
# 0: W [N, K] as A (transposed), x as B, column-major; 2: W^T [K, N] as A, x as B, column-major;
# 3: row-major x [M, K] @ W^T [K, N]. BF16 in and out, FP32 accumulation.
LT_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <vector>

static cublasLtHandle_t lt_handle() {
    static cublasLtHandle_t h = nullptr;
    if (!h) TORCH_CHECK(cublasLtCreate(&h) == CUBLAS_STATUS_SUCCESS, "cublasLtCreate");
    return h;
}

struct LtMats {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
    LtMats(int layout, int64_t m, int64_t n, int64_t k) {
        cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        const cublasOperation_t ta = layout == 0 ? CUBLAS_OP_T : CUBLAS_OP_N, tb = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
        if (layout == 3) {
            cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, m, k, k);
            cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, k, n, n);
            cublasLtMatrixLayoutCreate(&d, CUDA_R_16BF, m, n, n);
            const cublasLtOrder_t row = CUBLASLT_ORDER_ROW;
            for (auto l : {a, b, d}) cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row));
        } else {
            if (layout == 0) cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, k, n, k);
            else cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, n, k, n);
            cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, k, m, k);
            cublasLtMatrixLayoutCreate(&d, CUDA_R_16BF, n, m, n);
        }
    }
    ~LtMats() {
        if (d) cublasLtMatrixLayoutDestroy(d);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (op) cublasLtMatmulDescDestroy(op);
    }
};

static const cublasLtMatmulAlgoConfigAttributes_t LT_ATTR[9] = {
    CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
    CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
    CUBLASLT_ALGO_CONFIG_STAGES_ID, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID};

// x [M, K]; w = W [N, K] (layout 0) or W^T [K, N] (2, 3); out [M, N]; cfg: the 9 algorithm ints.
void lt_run(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t layout, std::vector<int64_t> cfg,
            torch::Tensor ws) {
    const int64_t m = x.size(0), k = x.size(1), n = out.size(1);
    TORCH_CHECK(cfg.size() == 9 && x.is_contiguous() && w.is_contiguous() && out.is_contiguous());
    LtMats L((int)layout, m, n, k);
    cublasLtMatmulAlgo_t algo;
    TORCH_CHECK(cublasLtMatmulAlgoInit(lt_handle(), CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF,
                                       CUDA_R_16BF, CUDA_R_16BF, (int)cfg[0], &algo) == CUBLAS_STATUS_SUCCESS,
                "lt: algo init");
    for (int i = 1; i < 9; ++i) {
        const uint32_t v32 = (uint32_t)cfg[i];
        const uint16_t v16 = (uint16_t)cfg[i];
        const bool small = i >= 7;  // INNER_SHAPE_ID and CLUSTER_SHAPE_ID are uint16
        TORCH_CHECK(cublasLtMatmulAlgoConfigSetAttribute(&algo, LT_ATTR[i], small ? (const void*)&v16 : (const void*)&v32,
                                                         small ? 2 : 4) == CUBLAS_STATUS_SUCCESS, "lt: config ", i);
    }
    const float alpha = 1.f, beta = 0.f;
    const void* A = layout == 3 ? x.data_ptr() : w.data_ptr();
    const void* B = layout == 3 ? w.data_ptr() : x.data_ptr();
    const cublasStatus_t st = cublasLtMatmul(lt_handle(), L.op, &alpha, A, L.a, B, L.b, &beta, out.data_ptr(), L.d,
                                             out.data_ptr(), L.d, &algo, ws.data_ptr(), ws.numel(),
                                             at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "lt: matmul status ", (int)st);
}

// The heuristic's first `count` algorithms for this problem, 9 ints each (flattened).
std::vector<int64_t> lt_algos(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t layout, torch::Tensor ws,
                              int64_t count) {
    const int64_t m = x.size(0), k = x.size(1), n = out.size(1);
    LtMats L((int)layout, m, n, k);
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    const uint64_t bytes = ws.numel();
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &bytes, sizeof(bytes));
    std::vector<cublasLtMatmulHeuristicResult_t> res(count);
    int got = 0;
    cublasLtMatmulAlgoGetHeuristic(lt_handle(), L.op, L.a, L.b, L.d, L.d, pref, (int)count, res.data(), &got);
    cublasLtMatmulPreferenceDestroy(pref);
    std::vector<int64_t> flat;
    for (int r = 0; r < got; ++r) {
        if (res[r].state != CUBLAS_STATUS_SUCCESS) continue;
        for (int i = 0; i < 9; ++i) {
            uint32_t v32 = 0;
            uint16_t v16 = 0;
            size_t written = 0;
            const bool small = i >= 7;
            cublasLtMatmulAlgoConfigGetAttribute(&res[r].algo, LT_ATTR[i], small ? (void*)&v16 : (void*)&v32,
                                                 small ? 2 : 4, &written);
            flat.push_back(small ? v16 : (int64_t)(i == 0 ? (int32_t)v32 : v32));
        }
    }
    return flat;
}
"""

# Seeds from the public v18 entry's offline search (github.com/NathenFernandes/muse-bf16-proj):
# (layer, m) -> (layout, 9 algorithm ints). Candidates only: the sweep times them with the rest.
LT_SEEDS = {
    ("pooler", 128): [(2, 66, 15, 1, 0, 0, 4, 35, 0, 2)],
    ("pooler", 256): [(3, 66, 18, 1, 0, 0, 0, 35, 0, 2)],
    ("unpooler", 128): [(2, 66, 18, 1, 0, 0, 3, 35, 0, 5)],
    ("unpooler", 256): [(3, 66, 20, 1, 0, 0, 4, 35, 0, 2)],
    ("qkv_gate", 256): [(0, 66, 124, 1, 0, 0, 0, 35, 0, 2)],
}
_LT = None


def _lt_mod():
    """Build the cuBLASLt module once; None if that fails (the "lt" variants are then skipped)."""
    global _LT
    if _LT is None:
        from torch.utils.cpp_extension import load_inline
        t0 = time.time()
        try:
            _LT = load_inline(name="muse_bf16_lt_v1", cpp_sources=LT_SRC, functions=["lt_run", "lt_algos"],
                              with_cuda=True, extra_cflags=["-O3"], extra_ldflags=["-l:libcublasLt.so.13"],
                              verbose=False)
            print(f"cuBLASLt module built in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001 - lose only the "lt" variants, and say why
            print(f"cuBLASLt build failed after {time.time() - t0:.0f}s: {str(e)[-2000:]}", flush=True)
            _LT = False
    return _LT or None


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
                functions=["proj", "router_scalar", "diag"], extra_cuda_cflags=["-O3"], verbose=False)
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
PLAN = {  # check 89e69e97: our kernels and pinned cuBLASLt algorithms where they beat SGLang by 2%
    **{(layer, m): ("torch", None) for layer in LAYER_NAMES for m in BUCKETS},
    ("qkv_gate", 1): ("c12", (2, 2, 4, 6)),
    ("qkv_gate", 8): ("c12", (2, 2, 4, 6)),
    ("qkv_gate", 16): ("cuda", (1, 2, 4, 6)),
    ("qkv_gate", 32): ("mm_t", None),
    ("qkv_gate", 128): ("lt", (2, 66, 17, 3, 2, 0, 0, 35, 0, 4)),
    ("qkv_gate", 256): ("lt", (0, 66, 124, 1, 0, 0, 0, 35, 0, 2)),
    ("o", 1): ("c12w", (2, 2, 1, 0)),
    ("o", 8): ("c12", (2, 2, 1, 0)),
    ("o", 16): ("mm_t", None),
    ("o", 128): ("lt", (3, 66, 18, 1, 0, 0, 3, 35, 0, 5)),
    ("o", 256): ("lt", (3, 66, 20, 1, 0, 0, 1, 35, 0, 3)),
    ("pooler", 1): ("c12", (1, 4, 1, 1)),
    ("pooler", 8): ("c12w", (1, 4, 1, 0)),
    ("pooler", 16): ("c12", (2, 4, 1, 0)),
    ("pooler", 32): ("c12", (2, 2, 1, 0)),
    ("pooler", 128): ("lt", (2, 66, 15, 1, 0, 0, 4, 35, 0, 2)),
    ("pooler", 256): ("lt", (3, 66, 18, 1, 0, 0, 0, 35, 0, 2)),
    ("unpooler", 1): ("cuda", (1, 8, 1, 0)),
    ("unpooler", 8): ("cuda", (1, 8, 1, 0)),
    ("unpooler", 16): ("c12", (2, 4, 1, 5)),
    ("unpooler", 32): ("mm_t", None),
    ("unpooler", 128): ("lt", (3, 66, 18, 1, 0, 0, 3, 35, 0, 5)),
    ("unpooler", 256): ("lt", (3, 66, 20, 1, 0, 0, 4, 35, 0, 2)),
    ("router", 1): ("rs", None),
    ("router", 8): ("cuda", (1, 8, 4, 7)),
    ("router", 16): ("cuda", (2, 4, 8, 0)),
    ("router", 32): ("cuda", (2, 2, 8, 0)),
    ("router", 128): ("mm_st", None),
    ("router", 256): ("mm_st", None),
}


# Warp grids (WR, WK) of ksplit_kernel, index g: 4 has three pipeline stages, 6 and 7 the push
# reduction (see the CUDA comment above pick_u).
GEOMS = ((1, 8), (2, 8), (2, 16), (4, 8), (1, 8), (1, 4), (1, 4), (1, 8))


def _unroll_ok(rt, u, m, g):
    """The (RT, U) instances the CUDA launchers compile for this token count and warp grid."""
    mt, nt = (1 if m <= 8 else 2 if m <= 16 else 4), GEOMS[g][0] * GEOMS[g][1] * 32
    if g == 4:
        return (u == 4 and rt == 1 and mt <= 2) or (u == 2 and (rt == 1 or mt <= 2))
    if nt <= 256:
        return u == 2 or (u == 4 and (rt == 1 or mt <= 2)) or (u == 8 and rt == 1 and mt == 1)
    if nt == 512:
        return rt == 1 and (u == 2 or (u == 4 and mt == 1))
    return rt == 1 and mt == 1 and u in (1, 2)


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


def pack_stream(units, nk, rt, u, ks, g):
    """Units [tiles, n_units, bytes] (a unit holds nk k blocks of a tile) -> the order in which
    ksplit_kernel with (rt, u, ks, warp grid g) reads them: a warp step is rt * (u / nk) units, and
    step s of warp gw = (by * GX + bx) * NW + warp sits at (s * G + gw) * step, so at each step the
    whole grid reads one contiguous region. Warp (bx, by, warp) with wr = warp % WR, wk = warp // WR
    owns tiles (bx * WR + wr) * rt .. + rt - 1 and units (by * WK + wk) * per_warp .. of each."""
    wr_n, wk_n = GEOMS[g]
    tiles, nu, nbytes = units.shape
    vu, per_warp = u // nk, nu // (ks * wk_n)
    shape = (per_warp // vu, ks, tiles // (rt * wr_n), wr_n * wk_n, rt, vu)
    ix = [torch.arange(d, device=units.device).view([-1 if i == j else 1 for j in range(6)])
          for i, d in enumerate(shape)]
    s, by, bx, warp, r, v = ix
    src = ((bx * wr_n + warp % wr_n) * rt + r) * nu + (by * wk_n + warp // wr_n) * per_warp + s * vu + v
    return units.reshape(tiles * nu, nbytes)[src.reshape(-1)].contiguous()


def _stream(state, layer, n, kind, rt, u, ks, g):
    """(stream-ordered weights, per-warp escape lists) for one launch geometry, packed once and cached.
    Raises ValueError if a warp would need more than 32 escapes."""
    key = (layer, n, kind, rt, u, ks, g)
    if key not in state["streams"]:
        if kind == "cuda":
            w = pack_stream(state["units"][layer][: n // 16], 1, rt, u, ks, g)
            state["streams"][key] = (w, state["none"])
        else:
            c = state["c12"][layer]
            units = c["w12" if kind == "c12" else "w12w"][: n // 16]
            esc = pack_escapes(c, n, rt, u, ks, g)
            state["streams"][key] = (pack_stream(units, 2 if kind == "c12w" else 1, rt, u, ks, g), esc)
    return state["streams"][key]


def pack_c12(w):
    """BF16 [N, K] -> lossless 12-bit code (C12 and C12W formats) and its escapes.

    Split each value into its low byte (bf16 bits 0-7: exponent LSB and 7 mantissa bits), its
    7 high exponent bits eh (bits 8-14) and its sign. base = the start of the 8-wide eh window that
    holds the most values (16 exponents). A value keeps its low byte and a nibble: eh - base in
    bits 0-2, plus one sign bit in bit 3 (nibbles 0, 4, 1, 5, 2, 6, 3, 7 carry the signs of values
    0..7 of a row's 8, see c12_row). A value outside the window (tiny values, zeros; about 1e-4 of
    Gaussian weights) is an escape: index 0, so it decodes with eh = base, plus a list entry with
    the XOR base ^ eh (pack_escapes) that the kernel applies to the decoded register before the
    MMA. So the kernel multiplies exactly W.
    Per 16-row tile and 32-wide k block, lane 4g + t holds 16 low bytes (row g then row g + 8,
    k 8t .. 8t + 7) and 8 code bytes, one 32-bit word per row.
      w12:  per k block 768 bytes: the 512 low bytes, then the 256 code bytes;
      w12w: per 2 k blocks 1536 bytes: low bytes of both blocks, then 16 code bytes per lane.
    """
    n, k = w.shape
    tiles, nb = n // 16, k // 32
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    eh, sign = (bits >> 8) & 0x7F, bits >> 15
    window = torch.bincount(eh.flatten(), minlength=128).cumsum(0)
    base = int(torch.argmax(window[7:] - torch.cat([window.new_zeros(1), window[:-8]])))  # window base .. base + 7
    d = eh - base
    exc = (d < 0) | (d >= 8)
    nib = torch.where(exc, 0, d)

    def frag(a):  # [N, K] -> [tile, k block, g, t, row half, 8 values]
        return a.view(tiles, 2, 8, nb, 4, 8).permute(0, 3, 2, 4, 1, 5)

    nib = frag(nib) | (frag(sign)[..., [0, 2, 4, 6, 1, 3, 5, 7]] << 3)  # nibble p: index p, sign of value inv[p]
    lo_b = frag(bits & 0xFF).to(torch.uint8).reshape(tiles, nb, 512)
    ex_b = (nib[..., 0::2] | (nib[..., 1::2] << 4)).to(torch.uint8).reshape(tiles, nb, 32, 8)
    rows, cols = exc.nonzero(as_tuple=True)
    return {
        "w12": torch.cat([lo_b, ex_b.reshape(tiles, nb, 256)], 2).contiguous(),
        "w12w": torch.cat([lo_b.reshape(tiles, nb // 2, 1024),
                           ex_b.reshape(tiles, nb // 2, 2, 32, 8).transpose(2, 3).reshape(tiles, nb // 2, 512)],
                          2).contiguous(),
        "exc_rows": rows.long(), "exc_cols": cols.long(), "exc_xor": (eh[rows, cols] ^ base).long(), "k": k,
        "basew": base,
    }


def pack_escapes(c, n, rt, u, ks, g):
    """Per-warp escape lists for ksplit_kernel with (rt, u, ks, warp grid g) on rows 0 .. n - 1:
    int32 [G * 32], warp gw's entries at gw * 32 .., sorted by seq = (k block within the warp) * rt
    + tile within the warp, padded with -1. Entry: seq << 17 | lane << 12 | register << 9 |
    half << 8 | 7-bit XOR of the high exponent bits. Raises ValueError if a warp needs more than 32."""
    wr_n, wk_n = GEOMS[g]
    keep = c["exc_rows"] < n
    rows, cols, xr = c["exc_rows"][keep], c["exc_cols"][keep], c["exc_xor"][keep]
    gx, nw, per_cta = n // (16 * rt * wr_n), wr_n * wk_n, c["k"] // 32 // ks
    per_warp = per_cta // wk_n
    tile, r, kb, kk = rows // 16, rows % 16, cols // 32, cols % 32
    gw = (kb // per_cta * gx + tile // (rt * wr_n)) * nw + (kb % per_cta) // per_warp * wr_n + (tile // rt) % wr_n
    seq = (kb % per_warp) * rt + tile % rt
    entry = (seq << 17) | ((r % 8 * 4 + kk // 8) << 12) | (((r >= 8).long() * 4 + kk % 8 // 2) << 9) | ((kk % 2) << 8) | xr
    order = torch.argsort(gw * (1 << 20) + seq)
    gw, entry = gw[order], entry[order]
    rank = torch.arange(len(gw), device=gw.device) - torch.searchsorted(gw, gw)
    if len(gw) and int(rank.max()) >= 32:
        raise ValueError("more than 32 escapes in one warp")
    esc = torch.full((gx * ks * nw * 32,), -1, dtype=torch.int64, device=gw.device)
    esc[gw * 32 + rank] = entry
    return esc.to(torch.int32)


def _proj(kind, x, state, layer, n, cfg):
    """Our kernel: kind "cuda" (Plain format), "c12" (C12) or "c12w" (C12W); cfg (rt, u, ksplit, g)."""
    rt, u, ksplit, g = cfg
    out = _out(x, layer, n)
    w, esc = _stream(state, layer, n, kind, rt, u, ksplit, g)
    if kind == "cuda":
        k_x = state["router"].shape[1] if layer == "router" else x.shape[1]
        _MOD.proj(x, w, esc, out, 0, k_x, state["units"][layer].shape[1] * 32, g, rt, u, ksplit, 0)
    else:
        _MOD.proj(x, w, esc, out, 1 if kind == "c12" else 2, x.shape[1], x.shape[1], g, rt, u, ksplit,
                  state["c12"][layer]["basew"])
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
    copy of W; mm_n: W [N, K] as a transposed view. Router: mm_t / mm_n run one [x | x] @ [hi | lo]^T;
    mm_s / mm_st run x @ [hi ; lo]^T ([m, 512], FP32) and add the halves: no copy of x."""
    if layer == "router":
        if kind in ("mm_s", "mm_st"):
            y = torch.mm(x, state["router_hl_t"] if kind == "mm_st" else state["router_hl"].t(),
                         out_dtype=torch.float32)
            return y[:, :256] + y[:, 256:]
        w = state["router_hilo_t"] if kind == "mm_t" else state["router_hilo"].t()
        return torch.mm(torch.cat([x, x], 1), w, out_dtype=torch.float32)
    w = state["wt"][layer][:, :n] if kind == "mm_t" else state[layer][:n].t()
    return torch.mm(x, w)


def _lt(x, state, layer, n, cfg):
    """cuBLASLt with a fixed algorithm; cfg = (layout, 9 algorithm ints), see LT_SRC."""
    out = _out(x, layer, n)
    w = state[layer] if cfg[0] == 0 else state["wt"][layer]
    _lt_mod().lt_run(x, w, out, cfg[0], list(cfg[1:]), state["lt_ws"])
    return out


def _lt_variants(state, layer, m, n):
    """The heuristic's first 16 algorithms for each layout, plus the seeds for (layer, m)."""
    x = torch.zeros((m, state[layer].shape[1]), device=state[layer].device, dtype=torch.bfloat16)
    out, found = _out(x, layer, n), []
    for layout in (0, 2, 3):
        w = state[layer] if layout == 0 else state["wt"][layer]
        flat = _lt_mod().lt_algos(x, w, out, layout, state["lt_ws"], 16)
        found += [(layout, *flat[i:i + 9]) for i in range(0, len(flat), 9)]
    return [("lt", c) for c in dict.fromkeys(found + LT_SEEDS.get((layer, m), []))]


def _run(kind, cfg, x, state, layer, n):
    if kind in ("cuda", "c12", "c12w") and _MOD:
        if not kind.startswith("c12") or state["c12"].get(layer) is not None:
            return _proj(kind, x, state, layer, n, cfg)
    if kind == "rs" and _MOD and x.shape[0] <= 8:  # router, FP32 scalar kernel
        out = _out(x, layer, n)
        _MOD.router_scalar(x, state["router"], out)
        return out
    if kind == "lt" and _lt_mod():
        return _lt(x, state, layer, n, cfg)
    if kind in ("mm_t", "mm_n", "mm_s", "mm_st"):
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
    state["router_hl"] = torch.cat([hi, lo], dim=0).contiguous()  # [512, 8192]: hi rows, then lo rows
    state["router_hl_t"] = state["router_hl"].t().contiguous()
    if BENCH or any(kind in ("mm_t", "lt") for kind, _ in PLAN.values()):
        state["wt"] = {name: state[name].t().contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")}
    if BENCH or any(kind == "lt" for kind, _ in PLAN.values()):
        state["lt_ws"] = torch.empty(32 << 20, device=dev, dtype=torch.uint8)  # cuBLASLt workspace
    state["plan"] = dict(PLAN)
    state["units"] = {name: pack_weight(state[name]) for name in ("qkv_gate", "o", "pooler", "unpooler")}
    state["units"]["router"] = pack_weight(state["router_hilo"])
    state["c12"] = {name: pack_c12(state[name]) for name in ("qkv_gate", "o", "pooler", "unpooler")}
    state["none"] = torch.zeros(1, device=dev, dtype=torch.int32)  # no escape list (Plain format)
    state["streams"] = {}
    for (layer, m), (kind, cfg) in PLAN.items():  # stream order for every planned launch
        if kind in ("cuda", "c12", "c12w"):
            n = 1024 if layer == "unpooler" and m <= 8 else state["units"][layer].shape[0] * 16
            try:
                _stream(state, layer, n, kind, *cfg)
            except ValueError:  # these weights need more escapes per warp: same launch, plain weights
                state["plan"][(layer, m)] = ("cuda", cfg)
                _stream(state, layer, n, "cuda", *cfg)
    if BENCH or any(kind != "torch" for kind, _ in PLAN.values()):
        _cuda()
    if BENCH or any(kind == "lt" for kind, _ in PLAN.values()):
        _lt_mod()
    for (layer, m), (kind, cfg) in list(state["plan"].items()):  # a planned cuBLASLt algorithm must run here
        if kind == "lt":
            try:
                _lt(torch.zeros((m, state[layer].shape[1]), device=dev, dtype=torch.bfloat16), state, layer,
                    state[layer].shape[0], cfg)
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001 - fall back to SGLang's dispatch, and say why
                print(f"cuBLASLt {layer} m={m} {cfg} unusable: {str(e)[:200]}", flush=True)
                state["plan"][(layer, m)] = ("torch", None)
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
    if layer == "router":
        out += [("mm_s", None), ("mm_st", None)] + ([("rs", None)] if m <= 8 else [])
    if PLAN.get((layer, m), ("torch",))[0] != "torch":
        out.append(PLAN[(layer, m)])
    if m >= 32 and layer != "router" and _lt_mod():
        out += _lt_variants(state, layer, m, n)
    if not _MOD or (m > 32 and layer != "router"):  # token groups lost 3-7x to cuBLAS (check 45fd2ba0)
        return out
    has_c12 = state["c12"].get(layer) is not None
    for kind in ["cuda"] + (["c12", "c12w"] if has_c12 else []):
        for g, (wr, wk) in enumerate(GEOMS):
            if g in (1, 2, 3) and layer == "router":  # FP32 output: 256-thread grids only
                continue
            for rt in (1, 2):
                for u in (1, 2, 4, 8):
                    for ks in (1, 2, 4, 8, 16):
                        ctas = n // (16 * rt * wr) * ks
                        if (n % (16 * rt * wr) or k_w % (ks * wk * 32 * u) or not 32 <= ctas <= 2112
                                or not _unroll_ok(rt, u, m, g) or (kind == "c12w" and u % 2)
                                or (g >= 6 and not 1 < ks <= wk + 1)):
                            continue
                        out.append((kind, (rt, u, ks, g)))
    return out


def _c12_probe(state, layer, n):
    """One-hot rows of x at columns that hold exceptions (values below the code's window), and
    W[:n] at those columns: the kernel's output must equal it bit for bit."""
    w = state[layer][:n]
    base = state["c12"][layer]["basew"]
    eh = (w.view(torch.int16).to(torch.int32) >> 8) & 0x7F
    exc = (eh < base) | (eh >= base + 8)
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
            print(f"C12 {layer}: table base {c['basew']}, {n_exc} escapes", flush=True)
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    flush = torch.empty(max(4 * l2, 256 << 20) // 4, device=dev, dtype=torch.float32).fill_(1.0)
    gen = torch.Generator(device=dev).manual_seed(1)
    rows, load_t, probes = [], {}, {}
    if _MOD and state["c12"].get("pooler") is not None:  # where the 12-bit kernel's time goes
        x1 = torch.randn((1, 8192), device=dev).to(torch.bfloat16)
        c, (ws, esc) = state["c12"]["pooler"], _stream(state, "pooler", 4096, "c12", 1, 4, 1, 0)
        o1 = torch.empty((1, 4096), device=dev, dtype=torch.bfloat16)
        res = {"full": _graph_us(lambda: _proj("c12", x1, state, "pooler", 4096, (1, 4, 1, 0)), flush)[0]}
        for d, name in ((1, "no decode"), (2, "no escapes"), (4, "no mma"), (8, "no x"), (15, "loads only")):
            res[name] = _graph_us(lambda: (_MOD.diag(x1, ws, esc, o1, c["basew"], d), o1)[1], flush)[0]
        print("DIAG pooler m=1 c12[1,4,1,0] net us: " + " ".join(f"{k}={v - FIXED_US:.2f}" for k, v in res.items()),
              flush=True)
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
            keep = {(ly, 1024 if ly == "unpooler" and mm <= 8 else state["units"][ly].shape[0] * 16, k, *c)
                    for (ly, mm), (k, c) in state["plan"].items() if k in ("cuda", "c12", "c12w")}
            for key in [key for key in state["streams"] if key not in keep]:  # bounded GPU memory
                del state["streams"][key]
            sgl, best = t_sgl - FIXED_US, best_t - FIXED_US
            load_t.setdefault(m, [0.0, 0.0])
            load_t[m][0] += sgl
            load_t[m][1] += best
            t_torch = next(t for t, k, _ in results if k == "torch") - FIXED_US
            tag = {"cuda": "c", "c12": "z", "c12w": "w", "torch": "t", "mm_t": "T", "mm_n": "N", "mm_s": "S",
                   "mm_st": "U", "rs": "R", "lt": "L"}
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
