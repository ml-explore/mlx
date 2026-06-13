// Shared dequantization utilities for optimized QMM kernels.
// Used by qmv_kernel.hip (GEMV) and qmm_kernel.hip (GEMM).

#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include "mlx/backend/rocm/device/config.h"

namespace mlx::core::rocm {

// --- Compile-time constants ---

// Number of quantized values packed per uint32 word.
// 4-bit: 8 values, 2-bit: 16 values, 8-bit: 4 values.
template <int BITS>
inline constexpr int pack_factor_u32 = 32 / BITS;

// Number of uint32 words each thread loads per K-iteration.
// Chosen so that values_per_thread = 16 for all bit widths.
template <int BITS>
inline constexpr int packs_per_thread = 16 / pack_factor_u32<BITS>;
// 4-bit: 16/8=2, 2-bit: 16/16=1, 8-bit: 16/4=4

// Number of quantized values each thread processes per K-iteration.
template <int BITS>
inline constexpr int values_per_thread = 16;

// Number of K-elements consumed per warp per iteration.
// = values_per_thread * WARP_SIZE = 16 * 32 = 512
inline constexpr int block_size_k = values_per_thread<4> * WARP_SIZE;

// Number of output rows computed per thread block.
inline constexpr int ROWS_PER_BLOCK = 8;

// --- Warp reduction ---

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_xor(val, offset);
  }
  return val;
}

// --- Dequant-and-dot: integer dot product + x-sum accumulation ---
//
// Metal-compatible accumulation: accumulates raw integer dot product and
// x-sum separately. The caller applies scale and bias ONCE per group:
//   result += scale * total_qdot + bias * total_xsum
//
// This matches Metal's qdot() which returns scale * accum + sum * bias,
// where accum and sum span all values_per_thread elements at once.
//
// The naive per-element form `acc += x[i] * (scale * q[i] + bias)` is
// mathematically equivalent but produces different float32 rounding due to
// a different number of scale/bias multiply operations, causing LLM output
// to degenerate into repetitive loops after ~10 tokens.

template <int BITS>
__device__ __forceinline__ void dequant_and_dot(
    uint32_t packed,
    const float* __restrict__ x_local,
    float& qdot_acc,
    float& x_sum) {
  constexpr int pf = pack_factor_u32<BITS>;
  constexpr uint32_t mask = (1u << BITS) - 1u;

#pragma unroll
  for (int i = 0; i < pf; i++) {
    float q = static_cast<float>((packed >> (i * BITS)) & mask);
    qdot_acc += x_local[i] * q;
    x_sum += x_local[i];
  }
}

// --- Vectorized weight load ---
//
// Loads PPT uint32 words in a single wide memory transaction instead of
// PPT scalar loads. For 4-bit (PPT=2), emits global_load_dwordx2 (64-bit).
// For 8-bit (PPT=4), emits global_load_dwordx4 (128-bit).
// Pointer must be naturally aligned (8-byte for uint2, 16-byte for uint4).

template <int BITS>
__device__ __forceinline__ void load_weight_vec(
    const uint32_t* __restrict__ ptr,
    uint32_t (&out)[packs_per_thread<BITS>]) {
  constexpr int PPT = packs_per_thread<BITS>;
  if constexpr (PPT == 2) {
    uint2 v = *reinterpret_cast<const uint2*>(ptr);
    out[0] = v.x;
    out[1] = v.y;
  } else if constexpr (PPT == 4) {
    // Two uint2 loads instead of one uint4. The single-uint4 load
    // (global_load_b128) miscomputes in the 8-bit affine QMV/gather paths
    // (root cause: HIP_vector_type<unsigned int,4> codegen on RDNA 3.5 with
    // hipcc 7.13 / LLVM 23). Two paired global_load_b64 ops yield the same
    // throughput on RDNA 3.5 without the miscompile.
    uint2 v0 = *reinterpret_cast<const uint2*>(ptr);
    uint2 v1 = *reinterpret_cast<const uint2*>(ptr + 2);
    out[0] = v0.x;
    out[1] = v0.y;
    out[2] = v1.x;
    out[3] = v1.y;
  } else {
#pragma unroll
    for (int p = 0; p < PPT; p++) {
      out[p] = ptr[p];
    }
  }
}

// Streaming (non-temporal) weight load for QMV / GEMV (M=1) decode.
//
// In a matrix-vector product every weight is read EXACTLY ONCE — there is no
// weight reuse, so caching the weight stream in L2 only evicts the data that IS
// reused (the shared X activation vector and the scales/biases). On gfx1151 the
// L2 is just 2 MB, so a wide weight stream thrashes it and the effective
// bandwidth oscillates as the L2 hit-rate on X/scales swings.
//
// __builtin_nontemporal_load emits `global_load_* slc` (streaming cache bit) on
// RDNA: the weight bytes flow through without being retained in L2, leaving L2 /
// the 32 MB MALL for the reused X and scales. Used ONLY by the GEMV path; the
// GEMM (M>1) path keeps the normal cached load because there weights ARE reused
// across the M rows.
template <int BITS>
__device__ __forceinline__ void load_weight_vec_streaming(
    const uint32_t* __restrict__ ptr,
    uint32_t (&out)[packs_per_thread<BITS>]) {
  constexpr int PPT = packs_per_thread<BITS>;
#pragma unroll
  for (int p = 0; p < PPT; p++) {
    out[p] = __builtin_nontemporal_load(ptr + p);
  }
}

// --- Type conversion helpers ---

__device__ __forceinline__ float to_float(__half x) {
  return __half2float(x);
}

__device__ __forceinline__ float to_float(hip_bfloat16 x) {
  return static_cast<float>(x);
}

__device__ __forceinline__ float to_float(float x) {
  return x;
}

template <typename T>
__device__ __forceinline__ T from_float(float x);

template <>
__device__ __forceinline__ __half from_float<__half>(float x) {
  return __float2half(x);
}

template <>
__device__ __forceinline__ hip_bfloat16 from_float<hip_bfloat16>(float x) {
  return hip_bfloat16(x);
}

template <>
__device__ __forceinline__ float from_float<float>(float x) {
  return x;
}

} // namespace mlx::core::rocm
