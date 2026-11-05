// SPDX-FileCopyrightText: Copyright (c) 2026 HrMarcussen
// SPDX-License-Identifier: MIT
// Fused single-token (T=1, B=1) streaming kernels for Moshi on RDNA3/RDNA4.
//
// Replaces chains of 20-40 tiny PyTorch ops (each >= ~5 us even inside a HIP
// graph) with one or two kernels:
//   * rope_kv_write:  RoPE on q/k + ring KV-cache write + end_offset += 1
//   * attn_decode_*:  split-K flash-decoding over the ring cache, with Moshi's
//                     exact position / causal-context masking
//   * attn_combine:   merge split-K partials
//   * rmsnorm_bf16:   RMSNorm computed in fp32 (moshi's "rms_norm_f32")
//
// Header-free for the same reason as gemv.hip (no host C++ toolchain needed).

#define KERNEL extern "C" __attribute__((global))
#define DEVICE static __attribute__((device)) inline __attribute__((always_inline))
#define SHARED __attribute__((shared))

typedef unsigned int u32;
typedef unsigned short u16;
typedef long long i64;
typedef unsigned long long u64;
typedef u32 u32x4 __attribute__((ext_vector_type(4)));
typedef u32 u32x2 __attribute__((ext_vector_type(2)));

#define WARP 32
#define NEG_INF (-__builtin_inff())
#define LOG2E 1.4426950408889634f

DEVICE u32 tid() { return __builtin_amdgcn_workitem_id_x(); }
DEVICE u32 bidx() { return __builtin_amdgcn_workgroup_id_x(); }
DEVICE u32 bidy() { return __builtin_amdgcn_workgroup_id_y(); }

DEVICE float as_f(u32 u) { return __builtin_bit_cast(float, u); }
DEVICE u32 as_u(float f) { return __builtin_bit_cast(u32, f); }
DEVICE float bf(u16 h) { return as_f(((u32)h) << 16); }
DEVICE float lo_bf16(u32 u) { return as_f(u << 16); }
DEVICE float hi_bf16(u32 u) { return as_f(u & 0xffff0000u); }
DEVICE u16 to_bf16(float f) {
    u32 u = as_u(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (u16)((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (u16)(u >> 16);
}
DEVICE float fexp2(float x) { return __builtin_amdgcn_exp2f(x); }
DEVICE float fmaxf_(float a, float b) { return a > b ? a : b; }

DEVICE void barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

DEVICE float shfl_xor(float v, int off) {
    const int lane = (int)(tid() & (WARP - 1));
    return as_f((u32)__builtin_amdgcn_ds_bpermute((lane ^ off) << 2, (int)as_u(v)));
}
DEVICE float warp_sum(float v) {
#pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1) v += shfl_xor(v, off);
    return v;
}
DEVICE float warp_max(float v) {
#pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1) v = fmaxf_(v, shfl_xor(v, off));
    return v;
}

// cos/sin of an fp32 angle with exact-ish range reduction (fp64), then the
// hardware v_sin/v_cos (which take revolutions in [-0.5, 0.5]).
DEVICE void cos_sin(float angle, float* c, float* s) {
    const double a = (double)angle;
    const double TWO_PI = 6.283185307179586476925286766559;
    const double r = a - TWO_PI * __builtin_rint(a / TWO_PI);
    const float rev = (float)(r / TWO_PI);
    *c = __builtin_amdgcn_cosf(rev);
    *s = __builtin_amdgcn_sinf(rev);
}

// ---------------------------------------------------------------------------
// RoPE + ring KV-cache write. Single block (so end_offset can be bumped safely).
//   proj:  bf16 [3, H, D]  (q, k, v) for the current token
//   freqs: fp32 [D/2] (rope frequencies, computed by torch) or null (no rope)
//   cache: bf16 [2, H, C, D]
//   q_out: bf16 [H, D] (rotated q)
KERNEL void __attribute__((amdgpu_flat_work_group_size(256, 256)))
rope_kv_write(const u16* __restrict__ proj, const float* __restrict__ freqs,
              const i64* __restrict__ offset, i64* __restrict__ end_offset,
              u16* __restrict__ cache, u16* __restrict__ q_out, int H, int D, int C) {
    const int t = (int)tid();
    const i64 end_old = *end_offset;
    const float pos = (float)(*offset);
    const int slot = (int)(end_old % (i64)C);
    const int half = D / 2;
    const int HD = H * D;
    const u64 cstride = (u64)H * (u64)C * (u64)D;
    for (int p = t; p < H * half; p += 256) {
        const int h = p / half;
        const int i = p - h * half;
        const int e = h * D + 2 * i;
        float qr = bf(proj[e]), qi = bf(proj[e + 1]);
        float kr = bf(proj[HD + e]), ki = bf(proj[HD + e + 1]);
        if (freqs) {
            float c, s;
            cos_sin(freqs[i] * pos, &c, &s);
            const float qor = qr * c - qi * s, qoi = qr * s + qi * c;
            const float kor = kr * c - ki * s, koi = kr * s + ki * c;
            qr = qor; qi = qoi; kr = kor; ki = koi;
        }
        q_out[e] = to_bf16(qr);
        q_out[e + 1] = to_bf16(qi);
        const u64 ci = ((u64)h * (u64)C + (u64)slot) * (u64)D + (u64)(2 * i);
        cache[ci] = to_bf16(kr);
        cache[ci + 1] = to_bf16(ki);
        cache[cstride + ci] = proj[2 * HD + e];
        cache[cstride + ci + 1] = proj[2 * HD + e + 1];
    }
    barrier();  // every thread has read *end_offset
    if (t == 0) *end_offset = end_old + 1;
}

// ---------------------------------------------------------------------------
// Split-K decode attention. grid = (H, S), block = 256.
// Keys of head h live at cache[h, j, :], values at cache[H*C*D + ...].
// end_offset is the value *after* the write (as in RingKVCache.complete),
// offset is the query position (MHA state offset before its increment).
#define MAX_CHUNK 256

template <int D>
DEVICE void attn_decode_impl(const u16* __restrict__ q, const u16* __restrict__ cache,
                             const i64* __restrict__ end_offset, const i64* __restrict__ offset,
                             float* __restrict__ part_o, float* __restrict__ part_ml,
                             u16* __restrict__ out, int H, int C, int S, int chunk, int context,
                             float scale) {
    constexpr int TPK = D / 8;            // threads per key (16-byte loads)
    constexpr int KPI = 256 / TPK;        // keys per iteration
    constexpr int PAIRS = D / 2;          // value dims handled as u32 pairs
    constexpr int NKS = 256 / PAIRS;      // key subsets in the value pass
    SHARED float sc[MAX_CHUNK];
    SHARED float red[NKS * D > 16 ? NKS * D : 16];
    SHARED float wred[8];

    const int t = (int)tid();
    const int lane = t & (WARP - 1);
    const int warp = t / WARP;
    const int h = (int)bidx();
    const int s = (int)bidy();
    const int k0 = s * chunk;
    int len = C - k0;
    len = len < chunk ? len : chunk;
    if (len < 0) len = 0;

    const i64 end_new = *end_offset;
    const i64 pos_q = *offset;
    const i64 end_index = end_new % (i64)C;
    const u16* K = cache + (u64)h * (u64)C * (u64)D;
    const u16* V = cache + (u64)H * (u64)C * (u64)D + (u64)h * (u64)C * (u64)D;

    // --- phase 1: scores (pre-scaled by log2(e) so we can use exp2)
    const int sub = t % TPK;
    const int kk = t / TPK;
    float qf[8];
    {
        const u32x4 qv = *(const u32x4*)(q + h * D + sub * 8);
        const float sl = scale * LOG2E;
        qf[0] = lo_bf16(qv.x) * sl; qf[1] = hi_bf16(qv.x) * sl;
        qf[2] = lo_bf16(qv.y) * sl; qf[3] = hi_bf16(qv.y) * sl;
        qf[4] = lo_bf16(qv.z) * sl; qf[5] = hi_bf16(qv.z) * sl;
        qf[6] = lo_bf16(qv.w) * sl; qf[7] = hi_bf16(qv.w) * sl;
    }
    for (int jj = kk; jj < ((len + KPI - 1) / KPI) * KPI; jj += KPI) {
        float d = 0.f;
        const bool inb = jj < len;
        const int j = k0 + (inb ? jj : 0);
        const u32x4 kv = *(const u32x4*)(K + (u64)j * D + sub * 8);
        d = __builtin_fmaf(qf[0], lo_bf16(kv.x), d); d = __builtin_fmaf(qf[1], hi_bf16(kv.x), d);
        d = __builtin_fmaf(qf[2], lo_bf16(kv.y), d); d = __builtin_fmaf(qf[3], hi_bf16(kv.y), d);
        d = __builtin_fmaf(qf[4], lo_bf16(kv.z), d); d = __builtin_fmaf(qf[5], hi_bf16(kv.z), d);
        d = __builtin_fmaf(qf[6], lo_bf16(kv.w), d); d = __builtin_fmaf(qf[7], hi_bf16(kv.w), d);
#pragma unroll
        for (int off = TPK / 2; off > 0; off >>= 1) d += shfl_xor(d, off);
        if (sub == 0 && inb) {
            // RingKVCache.complete positions + StreamingMultiheadAttention mask
            const i64 idx = (i64)j;
            const i64 delta = idx - end_index;
            i64 pk = delta <= 0 ? end_new + delta : end_new + delta - (i64)C;
            if (idx >= end_new) pk = -1;
            const i64 dq = pos_q - pk;
            const bool valid = pk >= 0 && dq >= 0 && (context < 0 || dq < (i64)context);
            sc[jj] = valid ? d : NEG_INF;
        }
    }
    barrier();

    // --- phase 2: max and sum over this chunk
    float v = t < len ? sc[t] : NEG_INF;
    float m = warp_max(v);
    if (lane == 0) wred[warp] = m;
    barrier();
    m = wred[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) m = fmaxf_(m, wred[w]);
    barrier();
    float p = (t < len && v != NEG_INF) ? fexp2(v - m) : 0.f;
    if (t < len) sc[t] = p;
    float l = warp_sum(p);
    if (lane == 0) wred[warp] = l;
    barrier();
    l = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) l += wred[w];

    // --- phase 3: P @ V
    const int dp = t % PAIRS;
    const int ks = t / PAIRS;
    float a0 = 0.f, a1 = 0.f;
    for (int jj = ks; jj < len; jj += NKS) {
        const float pj = sc[jj];
        if (pj != 0.f) {
            const u32 vv = *(const u32*)(V + (u64)(k0 + jj) * D + 2 * dp);
            a0 = __builtin_fmaf(pj, lo_bf16(vv), a0);
            a1 = __builtin_fmaf(pj, hi_bf16(vv), a1);
        }
    }
    red[ks * D + 2 * dp] = a0;
    red[ks * D + 2 * dp + 1] = a1;
    barrier();
    if (t < D) {
        float o = 0.f;
#pragma unroll
        for (int k = 0; k < NKS; ++k) o += red[k * D + t];
        if (S == 1) {
            out[h * D + t] = to_bf16(l > 0.f ? o / l : 0.f);
        } else {
            part_o[((u64)h * S + s) * D + t] = o;
            if (t == 0) {
                part_ml[((u64)h * S + s) * 2] = m;
                part_ml[((u64)h * S + s) * 2 + 1] = l;
            }
        }
    }
}

#define ATTN_ARGS                                                                               \
    const u16 *__restrict__ q, const u16 *__restrict__ cache, const i64 *__restrict__ end_offset, \
        const i64 *__restrict__ offset, float *__restrict__ part_o, float *__restrict__ part_ml,  \
        u16 *__restrict__ out, int H, int C, int S, int chunk, int context, float scale
#define ATTN_PASS q, cache, end_offset, offset, part_o, part_ml, out, H, C, S, chunk, context, scale

KERNEL void __attribute__((amdgpu_flat_work_group_size(256, 256))) attn_decode_d128(ATTN_ARGS) {
    attn_decode_impl<128>(ATTN_PASS);
}
KERNEL void __attribute__((amdgpu_flat_work_group_size(256, 256))) attn_decode_d64(ATTN_ARGS) {
    attn_decode_impl<64>(ATTN_PASS);
}

// grid = H, block = D (<= 256)
KERNEL void __attribute__((amdgpu_flat_work_group_size(1, 256)))
attn_combine(const float* __restrict__ part_o, const float* __restrict__ part_ml,
             u16* __restrict__ out, int S, int D) {
    const int h = (int)bidx();
    const int d = (int)tid();
    if (d >= D) return;
    float m = NEG_INF;
    for (int s = 0; s < S; ++s) m = fmaxf_(m, part_ml[((u64)h * S + s) * 2]);
    float L = 0.f, o = 0.f;
    if (m != NEG_INF) {
        for (int s = 0; s < S; ++s) {
            const float ms = part_ml[((u64)h * S + s) * 2];
            if (ms == NEG_INF) continue;
            const float w = fexp2(ms - m);
            L += w * part_ml[((u64)h * S + s) * 2 + 1];
            o += w * part_o[((u64)h * S + s) * D + d];
        }
    }
    out[h * D + d] = to_bf16(L > 0.f ? o / L : 0.f);
}

// ---------------------------------------------------------------------------
// RMSNorm for one row: y = x * (alpha * rsqrt(eps + mean(x^2))), fp32 math.
// grid = rows, block = 256. D % 2 == 0.
KERNEL void __attribute__((amdgpu_flat_work_group_size(256, 256)))
rmsnorm_bf16(const u16* __restrict__ x, const u16* __restrict__ alpha, u16* __restrict__ y,
             int D, float eps) {
    SHARED float wred[8];
    const int t = (int)tid();
    const u64 row = bidx();
    const u32* xr = (const u32*)(x + row * (u64)D);
    float ss = 0.f;
    for (int i = t; i < D / 2; i += 256) {
        const u32 v = xr[i];
        const float a = lo_bf16(v), b = hi_bf16(v);
        ss = __builtin_fmaf(a, a, ss);
        ss = __builtin_fmaf(b, b, ss);
    }
    ss = warp_sum(ss);
    if ((t & (WARP - 1)) == 0) wred[t / WARP] = ss;
    barrier();
    float tot = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) tot += wred[w];
    const float r = __builtin_amdgcn_rsqf(eps + tot / (float)D);
    const u32* ar = (const u32*)alpha;
    u32* yr = (u32*)(y + row * (u64)D);
    for (int i = t; i < D / 2; i += 256) {
        const u32 v = xr[i];
        const u32 al = ar[i];
        const u16 lo = to_bf16(lo_bf16(v) * (lo_bf16(al) * r));
        const u16 hi = to_bf16(hi_bf16(v) * (hi_bf16(al) * r));
        yr[i] = (u32)lo | ((u32)hi << 16);
    }
}
