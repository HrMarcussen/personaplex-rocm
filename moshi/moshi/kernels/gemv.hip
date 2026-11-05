// Batch-1 GEMV kernels for RDNA3/RDNA4 (wave32), used by moshi.kernels.fast_linear.
//
//   y[n] = sum_k W[n, k] * x[k]          W: bf16 [N, K] row-major (nn.Linear layout)
//
// Moshi decodes one token per step, so every projection is a matrix-vector
// product that is purely memory bound. The stock rocBLAS/hipBLASLt kernels
// reach only ~150-500 GB/s on several of these shapes on an RX 7900 XTX;
// these kernels stream the weights with 16-byte non-temporal loads and keep
// fp32 accumulators, one wave32 per group of rows.
//
// Deliberately header-free (compiled with -nogpuinc -nogpulib): the HIP
// runtime headers drag in the host C++ standard library, which on Windows
// means MSVC. Only clang AMDGPU builtins are used, so the ROCm SDK's bundled
// clang is all that is needed. Loaded with hipModuleLoadData and launched with
// hipModuleLaunchKernel, so HIP graph capture records it like any kernel.

#define KERNEL extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(256, 256)))
#define DEVICE static __attribute__((device)) inline __attribute__((always_inline))
#define SHARED __attribute__((shared))

typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned long long u64;
typedef u32 u32x4 __attribute__((ext_vector_type(4)));

#define WARP 32
#define BLOCK 256
#define WAVES (BLOCK / WARP)

DEVICE u32 tid() { return __builtin_amdgcn_workitem_id_x(); }
DEVICE u32 bid() { return __builtin_amdgcn_workgroup_id_x(); }

DEVICE float as_f(u32 u) { return __builtin_bit_cast(float, u); }
DEVICE u32 as_u(float f) { return __builtin_bit_cast(u32, f); }
DEVICE float lo_bf16(u32 u) { return as_f(u << 16); }
DEVICE float hi_bf16(u32 u) { return as_f(u & 0xffff0000u); }

DEVICE u16 f32_to_bf16(float f) {
    u32 u = as_u(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (u16)((u >> 16) | 0x40);  // NaN
    u += 0x7fffu + ((u >> 16) & 1u);                                      // RNE
    return (u16)(u >> 16);
}

DEVICE float dot8(u32x4 w, u32x4 x, float acc) {
    acc = __builtin_fmaf(lo_bf16(w.x), lo_bf16(x.x), acc);
    acc = __builtin_fmaf(hi_bf16(w.x), hi_bf16(x.x), acc);
    acc = __builtin_fmaf(lo_bf16(w.y), lo_bf16(x.y), acc);
    acc = __builtin_fmaf(hi_bf16(w.y), hi_bf16(x.y), acc);
    acc = __builtin_fmaf(lo_bf16(w.z), lo_bf16(x.z), acc);
    acc = __builtin_fmaf(hi_bf16(w.z), hi_bf16(x.z), acc);
    acc = __builtin_fmaf(lo_bf16(w.w), lo_bf16(x.w), acc);
    acc = __builtin_fmaf(hi_bf16(w.w), hi_bf16(x.w), acc);
    return acc;
}

DEVICE u32x4 ld_nt(const u32x4* p) { return __builtin_nontemporal_load(p); }

// Butterfly reduction across a wave32 via ds_bpermute (lane index * 4 bytes).
DEVICE float warp_sum(float v) {
    const int lane = (int)(tid() & (WARP - 1));
#pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1) {
        const int src = (lane ^ off) << 2;
        v += as_f((u32)__builtin_amdgcn_ds_bpermute(src, (int)as_u(v)));
    }
    return v;
}

// One wave32 computes ROWS consecutive output rows; 8 waves per block.
// K8 = K / 8 (number of 16-byte chunks per row).
template <int ROWS>
DEVICE void gemv_impl(const u32x4* __restrict__ W, const u32x4* __restrict__ x,
                      u16* __restrict__ y, int N, int K8) {
    const int lane = (int)(tid() & (WARP - 1));
    const int warp = (int)(tid() / WARP);
    const int row0 = ((int)bid() * WAVES + warp) * ROWS;
    if (row0 >= N) return;

    float acc[ROWS];
    const u32x4* wp[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        acc[r] = 0.f;
        const int row = (row0 + r < N) ? row0 + r : N - 1;  // clamp: computed, never stored
        wp[r] = W + (u64)row * (u64)K8;
    }

    int k = lane;
    for (; k + WARP < K8; k += 2 * WARP) {  // x2 unroll keeps more loads in flight
        const u32x4 xa = x[k];
        const u32x4 xb = x[k + WARP];
        u32x4 wa[ROWS], wb[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            wa[r] = ld_nt(wp[r] + k);
            wb[r] = ld_nt(wp[r] + k + WARP);
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            acc[r] = dot8(wa[r], xa, acc[r]);
            acc[r] = dot8(wb[r], xb, acc[r]);
        }
    }
    for (; k < K8; k += WARP) {
        const u32x4 xa = x[k];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[r] = dot8(ld_nt(wp[r] + k), xa, acc[r]);
    }

#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = warp_sum(acc[r]);

    if (lane < ROWS && row0 + lane < N) {
        float v = acc[0];
#pragma unroll
        for (int r = 1; r < ROWS; ++r) v = (lane == r) ? acc[r] : v;
        y[row0 + lane] = f32_to_bf16(v);
    }
}

// Split-K variant for short-and-wide matrices (small N, large K): each block
// owns ROWS rows and all 8 waves split K, then reduce through LDS.
template <int ROWS>
DEVICE void gemv_splitk_impl(const u32x4* __restrict__ W, const u32x4* __restrict__ x,
                             u16* __restrict__ y, int N, int K8, float (*partial)[ROWS]) {
    const int t = (int)tid();
    const int lane = t & (WARP - 1);
    const int warp = t / WARP;
    const int row0 = (int)bid() * ROWS;  // uniform per block: no early exit before barrier issue

    float acc[ROWS];
    const u32x4* wp[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        acc[r] = 0.f;
        const int row = (row0 + r < N) ? row0 + r : N - 1;
        wp[r] = W + (u64)row * (u64)K8;
    }
    for (int k = t; k < K8; k += BLOCK) {
        const u32x4 xa = x[k];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[r] = dot8(ld_nt(wp[r] + k), xa, acc[r]);
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const float s = warp_sum(acc[r]);
        if (lane == 0) partial[warp][r] = s;
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
    if (t < ROWS && row0 + t < N) {
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < WAVES; ++w) v += partial[w][t];
        y[row0 + t] = f32_to_bf16(v);
    }
}

#define ARGS const u32x4* __restrict__ W, const u32x4* __restrict__ x, u16* __restrict__ y, int N, int K8

KERNEL void gemv_bf16_r1(ARGS) { gemv_impl<1>(W, x, y, N, K8); }
KERNEL void gemv_bf16_r2(ARGS) { gemv_impl<2>(W, x, y, N, K8); }
KERNEL void gemv_bf16_r4(ARGS) { gemv_impl<4>(W, x, y, N, K8); }

KERNEL void gemv_bf16_sk2(ARGS) {
    SHARED float partial[WAVES][2];
    gemv_splitk_impl<2>(W, x, y, N, K8, partial);
}
KERNEL void gemv_bf16_sk4(ARGS) {
    SHARED float partial[WAVES][4];
    gemv_splitk_impl<4>(W, x, y, N, K8, partial);
}

// ---------------------------------------------------------------------------
// int8 weight-only GEMV: y[n] = scale[n] * sum_k Wq[n, k] * x[k]
//   Wq: int8 [N, K] row-major, per-row symmetric scale (fp32), x: bf16 [K].
// Halves the bytes streamed per token vs bf16. K16 = K / 16.

DEVICE float dot16_i8(u32x4 w, u32x4 xa, u32x4 xb, float acc) {
    const u32 ws[4] = {w.x, w.y, w.z, w.w};
    const u32 xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int wi = (int)ws[i];
        const float w0 = (float)((wi << 24) >> 24);  // sign-extend byte 0
        const float w1 = (float)((wi << 16) >> 24);
        const float w2 = (float)((wi << 8) >> 24);
        const float w3 = (float)(wi >> 24);
        acc = __builtin_fmaf(w0, lo_bf16(xs[2 * i]), acc);
        acc = __builtin_fmaf(w1, hi_bf16(xs[2 * i]), acc);
        acc = __builtin_fmaf(w2, lo_bf16(xs[2 * i + 1]), acc);
        acc = __builtin_fmaf(w3, hi_bf16(xs[2 * i + 1]), acc);
    }
    return acc;
}

template <int ROWS>
DEVICE void gemv_i8_impl(const u32x4* __restrict__ W, const float* __restrict__ scale,
                         const u32x4* __restrict__ x, u16* __restrict__ y, int N, int K16) {
    const int lane = (int)(tid() & (WARP - 1));
    const int warp = (int)(tid() / WARP);
    const int row0 = ((int)bid() * WAVES + warp) * ROWS;
    if (row0 >= N) return;
    float acc[ROWS];
    const u32x4* wp[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        acc[r] = 0.f;
        const int row = (row0 + r < N) ? row0 + r : N - 1;
        wp[r] = W + (u64)row * (u64)K16;
    }
    for (int k = lane; k < K16; k += WARP) {
        const u32x4 xa = x[2 * k];
        const u32x4 xb = x[2 * k + 1];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[r] = dot16_i8(ld_nt(wp[r] + k), xa, xb, acc[r]);
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = warp_sum(acc[r]);
    if (lane < ROWS && row0 + lane < N) {
        float v = acc[0];
#pragma unroll
        for (int r = 1; r < ROWS; ++r) v = (lane == r) ? acc[r] : v;
        y[row0 + lane] = f32_to_bf16(v * scale[row0 + lane]);
    }
}

template <int ROWS>
DEVICE void gemv_i8_splitk_impl(const u32x4* __restrict__ W, const float* __restrict__ scale,
                                const u32x4* __restrict__ x, u16* __restrict__ y, int N, int K16,
                                float (*partial)[ROWS]) {
    const int t = (int)tid();
    const int lane = t & (WARP - 1);
    const int warp = t / WARP;
    const int row0 = (int)bid() * ROWS;
    float acc[ROWS];
    const u32x4* wp[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        acc[r] = 0.f;
        const int row = (row0 + r < N) ? row0 + r : N - 1;
        wp[r] = W + (u64)row * (u64)K16;
    }
    for (int k = t; k < K16; k += BLOCK) {
        const u32x4 xa = x[2 * k];
        const u32x4 xb = x[2 * k + 1];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[r] = dot16_i8(ld_nt(wp[r] + k), xa, xb, acc[r]);
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const float s = warp_sum(acc[r]);
        if (lane == 0) partial[warp][r] = s;
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
    if (t < ROWS && row0 + t < N) {
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < WAVES; ++w) v += partial[w][t];
        y[row0 + t] = f32_to_bf16(v * scale[row0 + t]);
    }
}

#define ARGS_I8 const u32x4* __restrict__ W, const float* __restrict__ scale, const u32x4* __restrict__ x, \
                u16* __restrict__ y, int N, int K16

KERNEL void gemv_i8_r1(ARGS_I8) { gemv_i8_impl<1>(W, scale, x, y, N, K16); }
KERNEL void gemv_i8_r2(ARGS_I8) { gemv_i8_impl<2>(W, scale, x, y, N, K16); }
KERNEL void gemv_i8_sk2(ARGS_I8) {
    SHARED float partial[WAVES][2];
    gemv_i8_splitk_impl<2>(W, scale, x, y, N, K16, partial);
}
KERNEL void gemv_i8_sk4(ARGS_I8) {
    SHARED float partial[WAVES][4];
    gemv_i8_splitk_impl<4>(W, scale, x, y, N, K16, partial);
}
