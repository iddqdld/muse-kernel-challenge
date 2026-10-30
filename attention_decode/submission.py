# okc: problem=muse_attention_decode
"""Gated attention decode over a paged FP8 E4M3 cache, in two Triton launches per step.

SGLang runs five kernels per layer: FusedQKNorm, store_cache, a q.to(fp8) cast, FA3
(split attention + combine) and rms_norm_gate. Here one split-KV attention kernel does the
norm, RoPE, the FP8 cache write, the FP8 round trip of the block's own new keys and values,
and the online softmax over FP8 cache pages for a group of GQA query heads; a small combine
kernel merges the chunk partials, applies the output RMSNorm and the sigmoid gate, and
writes BF16.

Why it should win: every load is memory-bound (read the FP8 K and V of each KV head once),
so the work is bytes / HBM bandwidth plus launches. We read each KV byte once per logical
position per KV head, keep the 8 query heads that share it in one program, and use two
launches instead of about six. Keys and values are BF16 in registers after the FP8 load
(FA3's FP8 path does the same); probabilities are BF16 in the PV dot and FP32 in the
softmax.
"""

import math
from typing import TypedDict

import torch
import triton
import triton.language as tl

FP8 = torch.float8_e4m3fn
RMS_EPS = 1e-5
NEG = -1.0e38
# Kernel-side constants must be triton constexpr globals, not plain Python ones.
EPS_C = tl.constexpr(1e-5)
NEG_C = tl.constexpr(-1.0e38)


class Config(TypedDict):
    batch: int            # B
    q_len: int            # L: 1 for decode, 8 for a speculative verify block (causal inside it)
    num_q_heads: int      # Hq: 16 per GPU under TP8
    num_kv_heads: int     # Hkv: 2; each serves Hq / Hkv query heads
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers; 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)


class Inputs(TypedDict):
    qkv: torch.Tensor          # bfloat16 [B, L, (2*Hq + 2*Hkv)*64]: q | k | v | gate
    positions: torch.Tensor    # int64 [B*L]
    cache_loc: torch.Tensor    # int64 [B*L]
    cache_seqlens: torch.Tensor  # int32 [B]
    cu_seqlens_q: torch.Tensor   # int32 [B+1]
    scheduler_metadata: torch.Tensor | None
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64]
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64]
    k_scale: torch.Tensor     # float32 []
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages]
    seq_lens: torch.Tensor    # int32 [B]
    config: Config


# --------------------------------------------------------------------------- kernels


@triton.jit
def _attn_kernel(
    QKV, CLOC, SEQLEN, KC, VC, PT, KSC, VSC, INVF, ACC, PM, PL,
    stride_pt,
    L: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
    SM: tl.constexpr, QKSCALE: tl.constexpr, WINDOW: tl.constexpr, ROPE: tl.constexpr,
    PAGE: tl.constexpr, NCHUNK: tl.constexpr, CHUNK: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BP: tl.constexpr,
    SCALED_K: tl.constexpr, SCALED_V: tl.constexpr,
):
    """One (kv head, KV chunk) of the split attention, for all query heads of the group.

    Rows of the score tile are the block's L tokens times the group's query heads. Keys
    come from the FP8 cache for positions below seq_lens[b]; the block's own L new tokens
    are recomputed here and attended by the last chunk only, so no program reads a cache
    row another program writes.
    """
    pid_c = tl.program_id(0)
    b = tl.program_id(1)
    kvh = tl.program_id(2)
    group = HQ // HKV

    d = tl.arange(0, D)
    rows = tl.arange(0, BM)
    jr = rows // group                      # token index inside the block
    hr = rows % group                       # query head inside the GQA group
    rok = rows < L * group
    seq = tl.load(SEQLEN + b)
    if SCALED_K:
        kscale = tl.load(KSC)
    else:
        kscale = 1.0
    if SCALED_V:
        vscale = tl.load(VSC)
    else:
        vscale = 1.0

    # ---- queries: [BM, D] = (token, head) rows for this KV head, normalized and roped
    tok = b * L + jr
    qoff = tok * ((2 * HQ + 2 * HKV) * D) + kvh * group * D + hr * D
    qt = tl.load(QKV + qoff[:, None] + d[None, :], mask=rok[:, None], other=0.0)
    qf = qt.to(tl.float32)
    qf = qf * tl.rsqrt(tl.sum(qf * qf, 1) / D + EPS_C)[:, None] * QKSCALE
    if ROPE:
        ang = (seq + jr).to(tl.float32)[:, None] * tl.load(INVF + tl.arange(0, D // 2))[None, :]
        cs = tl.cos(ang)
        sn = tl.sin(ang)
        even, odd = tl.split(tl.reshape(qf, (BM, D // 2, 2)))
        qf = tl.reshape(tl.join(even * cs - odd * sn, even * sn + odd * cs), (BM, D))
    qb = qf.to(tl.bfloat16)

    # ---- the block's own new keys and values: normalized, roped, rounded to FP8 (round
    # trip in registers) so attention sees exactly what the cache will hold
    r = tl.arange(0, BP)
    rmk = r < L
    koff = (b * L + r) * ((2 * HQ + 2 * HKV) * D) + HQ * D + kvh * D
    voff = koff + HKV * D
    kn = tl.load(QKV + koff[:, None] + d[None, :], mask=rmk[:, None], other=0.0).to(tl.float32)
    vn = tl.load(QKV + voff[:, None] + d[None, :], mask=rmk[:, None], other=0.0).to(tl.float32)
    kn = kn * tl.rsqrt(tl.sum(kn * kn, 1) / D + EPS_C)[:, None]
    if ROPE:
        ang = (seq + r).to(tl.float32)[:, None] * tl.load(INVF + tl.arange(0, D // 2))[None, :]
        cs = tl.cos(ang)
        sn = tl.sin(ang)
        even, odd = tl.split(tl.reshape(kn, (BP, D // 2, 2)))
        kn = tl.reshape(tl.join(even * cs - odd * sn, even * sn + odd * cs), (BP, D))
    kq = (kn / kscale).to(tl.float8e4nv)
    vq = (vn / vscale).to(tl.float8e4nv)
    knb = (kq.to(tl.float32) * kscale).to(tl.bfloat16)
    vnb = (vq.to(tl.float32) * vscale).to(tl.bfloat16)
    if pid_c == 0:
        # The new tokens' rows are disjoint from every row position < seq, so writing them
        # here races with nobody.
        cl = tl.load(CLOC + b * L + r, mask=rmk, other=0).to(tl.int32)
        coff = (cl * HKV + kvh) * D
        tl.store(KC + coff[:, None] + d[None, :], kq, mask=rmk[:, None])
        tl.store(VC + coff[:, None] + d[None, :], vq, mask=rmk[:, None])

    m_i = tl.full([BM], NEG_C, tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    pos_row = seq + jr

    # ---- keys and values already in the cache: positions [lo, hi) of this chunk
    if WINDOW > 0:
        lo0 = tl.maximum(seq - (WINDOW - 1), 0)
    else:
        lo0 = 0
    lo = lo0 + pid_c * CHUNK
    hi = tl.minimum(seq, lo + CHUNK)
    for start in range(lo, hi, BN):
        t = start + tl.arange(0, BN)
        tmk = t < hi
        pg = t // PAGE
        ptab = tl.load(PT + b * stride_pt + pg, mask=tmk, other=0)
        koff2 = (ptab * PAGE + (t % PAGE)) * (HKV * D) + kvh * D
        offs = koff2[:, None] + d[None, :]
        kk = tl.load(KC + offs, mask=tmk[:, None], other=0.0)
        if SCALED_K:
            kb = (kk.to(tl.float32) * kscale).to(tl.bfloat16)
        else:
            kb = kk.to(tl.bfloat16)
        qk = tl.dot(qb, tl.trans(kb)) * SM
        vis = (t[None, :] <= pos_row[:, None]) & tmk[None, :]
        if WINDOW > 0:
            vis = vis & (t[None, :] > pos_row[:, None] - WINDOW)
        qk = tl.where(vis, qk, float("-inf"))
        m_new = tl.maximum(tl.maximum(m_i, tl.max(qk, 1)), NEG_C)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        vv = tl.load(VC + offs, mask=tmk[:, None], other=0.0)
        if SCALED_V:
            vb = (vv.to(tl.float32) * vscale).to(tl.bfloat16)
        else:
            vb = vv.to(tl.bfloat16)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vb)
        m_i = m_new

    # ---- the block's own new tokens, causally masked inside the block, attended once
    if pid_c == NCHUNK - 1:
        qk = tl.dot(qb, tl.trans(knb)) * SM
        vis = (r[None, :] <= jr[:, None]) & rmk[None, :]
        qk = tl.where(vis, qk, float("-inf"))
        m_new = tl.maximum(tl.maximum(m_i, tl.max(qk, 1)), NEG_C)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vnb)
        m_i = m_new

    # ---- partials: [B, L, Hq, NCHUNK, D] accumulators and [B, L, Hq, NCHUNK] softmax stats
    hrow = (b * L + jr) * HQ + (kvh * group + hr)
    abase = hrow * (NCHUNK * D) + pid_c * D
    tl.store(ACC + abase[:, None] + d[None, :], acc, mask=rok[:, None])
    tl.store(PM + hrow * NCHUNK + pid_c, m_i, mask=rok)
    tl.store(PL + hrow * NCHUNK + pid_c, l_i, mask=rok)


@triton.jit
def _combine_kernel(
    QKV, ACC, PM, PL, OUT,
    L: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
    NCHUNK: tl.constexpr, BC: tl.constexpr,
):
    """One (token, query head) per program: merge the chunk partials, RMSNorm, sigmoid gate."""
    pid = tl.program_id(0)
    h = pid % HQ
    j = (pid // HQ) % L
    b = pid // (HQ * L)
    d = tl.arange(0, D)
    hrow = (b * L + j) * HQ + h
    mbase = hrow * NCHUNK

    mx = NEG_C
    for c0 in range(0, NCHUNK, BC):
        c = c0 + tl.arange(0, BC)
        mm = tl.load(PM + mbase + c, mask=c < NCHUNK, other=NEG_C)
        mx = tl.maximum(mx, tl.max(mm))
    o = tl.zeros([D], tl.float32)
    ls = 0.0
    for c0 in range(0, NCHUNK, BC):
        c = c0 + tl.arange(0, BC)
        cmk = c < NCHUNK
        w = tl.exp(tl.load(PM + mbase + c, mask=cmk, other=NEG_C) - mx)
        ll = tl.load(PL + mbase + c, mask=cmk, other=0.0)
        a = tl.load(ACC + hrow * (NCHUNK * D) + c[:, None] * D + d[None, :],
                    mask=cmk[:, None], other=0.0)
        o += tl.sum(a * w[:, None], 0)
        ls += tl.sum(ll * w)
    o = o / ls
    o = o * tl.rsqrt(tl.sum(o * o) / D + EPS_C)
    goff = (b * L + j) * ((2 * HQ + 2 * HKV) * D) + (HQ + 2 * HKV) * D + h * D
    o = o * tl.sigmoid(tl.load(QKV + goff + d).to(tl.float32))
    tl.store(OUT + hrow * D + d, o.to(tl.bfloat16))


# --------------------------------------------------------------------------- host side

_BUFFERS = {}
_INVF = {}


def _next_pow2(n):
    return 1 << int(round(math.log2(max(n, 1))))


# (key block, warps) per shape class. The loop is latency-bound: an indirect
# page-table load must finish before the KV gather can issue. Two resident CTAs per SM
# overlap two such chains, which needs at most 255 registers at 128 threads. L=8 tiles run
# 4 warps with a 128-key block (255 registers, two CTAs); the L=1 full-context tiles take a
# 256-key block at 8 warps, halving the serial latency steps.
_BN = {(1, 0): (256, 8), (1, 1): (128, 4), (8, 0): (128, 4), (8, 1): (128, 4)}


def _plan(batch, length, hkv, span, window):
    """About two programs per SM, plus the key block and warp count for the shape class."""
    bn, warps = _BN[(8 if length > 1 else 1, 0 if not window else 1)]
    floor = max(bn, 256 if window else 1024)
    nchunk = _next_pow2(max(1.0, 264.0 / (batch * hkv)))
    nchunk = max(1, min(nchunk, span // floor))
    return nchunk, -(-span // nchunk), bn, warps


def _setup(data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    page, window = cfg["page_size"], cfg["window"]
    dev = data["qkv"].device
    span = data["page_table"].shape[1] * page
    if window:
        span = min(span, window)
    nchunk, chunk, bn, warps = _plan(batch, length, hkv, span, window)
    key = (batch, length, hq, hkv, d, page, window, span, nchunk)
    bufs = _BUFFERS.get(key)
    if bufs is None:
        hrow = batch * length * hq
        bufs = (
            torch.zeros(hrow * nchunk * d, dtype=torch.float32, device=dev),
            torch.full((hrow * nchunk,), NEG, dtype=torch.float32, device=dev),
            torch.zeros(hrow * nchunk, dtype=torch.float32, device=dev),
        )
        _BUFFERS[key] = bufs
    theta = cfg["rope_theta"]
    invf = data["k_scale"]
    if theta:
        invf = _INVF.get(theta)
        if invf is None:
            invf = 1.0 / (float(theta) ** (torch.arange(0, d, 2, dtype=torch.float32, device=dev) / d))
            _INVF[theta] = invf
    return key, nchunk, chunk, bn, warps, bufs, invf


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    _, nchunk, chunk, bn, warps, bufs, invf = _setup(data)
    acc, pm, pl = bufs
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=data["qkv"].device)
    bm = max(16, length * (hq // hkv))
    bp = max(16, _next_pow2(length))
    _attn_kernel[(nchunk, batch, hkv)](
        data["qkv"], data["cache_loc"], data["seq_lens"], data["k_cache"], data["v_cache"],
        data["page_table"], data["k_scale"], data["v_scale"], invf, acc, pm, pl,
        data["page_table"].stride(0),
        L=length, HQ=hq, HKV=hkv, D=d,
        SM=cfg["sm_scale"], QKSCALE=cfg["qk_scale_factor"], WINDOW=cfg["window"],
        ROPE=bool(cfg["rope_theta"]), PAGE=cfg["page_size"], NCHUNK=nchunk, CHUNK=chunk,
        BM=bm, BN=bn, BP=bp,
        SCALED_K=cfg["kv_scale"] != 1.0, SCALED_V=cfg["kv_scale"] != 1.0,
        num_warps=warps,
    )
    _combine_kernel[(batch * length * hq,)](
        data["qkv"], acc, pm, pl, out,
        L=length, HQ=hq, HKV=hkv, D=d, NCHUNK=nchunk, BC=32,
        num_warps=2,
    )
    return out
