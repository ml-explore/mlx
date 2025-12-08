#include <cuda.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include "mlx/backend/cuda/quantized/quantized_utils.cuh"

namespace mlx::core::cu {

template <typename T>
struct alignas(4 * sizeof(T)) Vector4 {
  T x;
  T y;
  T z;
  T w;
};

using bf16x4 = Vector4<__nv_bfloat16>;
using fp16x4 = Vector4<__half>;
using f32x4 = Vector4<float>;

template <typename T>
__device__ __forceinline__ uint16_t
scale_cvt_Tx4_to_fp4x4_fallback(const Vector4<T> input, const float scale) {
  // Fallback implementation for architectures that do not support the
  // instruction and for cuda versions with no fp4 support (< 12.8) -> scalar
  // conversion __nvfp4e2m1
  uint16_t out_fp4x4 = 0;
  Vector4<float> scaled;
  scaled.x = static_cast<float>(input.x) * scale;
  scaled.y = static_cast<float>(input.y) * scale;
  scaled.z = static_cast<float>(input.z) * scale;
  scaled.w = static_cast<float>(input.w) * scale;
  uint8_t q0 = __nv_fp4_e2m1(scaled.x);
  uint8_t q1 = __nv_fp4_e2m1(scaled.y);
  uint8_t q2 = __nv_fp4_e2m1(scaled.z);
  uint8_t q3 = __nv_fp4_e2m1(scaled.w);
  out_fp4x4 = (static_cast<uint16_t>(q3) << 12) |
      (static_cast<uint16_t>(q2) << 8) | (static_cast<uint16_t>(q1) << 4) |
      static_cast<uint16_t>(q0);
  return out_fp4x4;
}

#ifdef CUDA_FP4_FP8_CVT_PTX_SUPPORT

__device__ __forceinline__ uint16_t
scale_cvt_bf16x4_to_fp4x4_rn(const bf16x4 input_bf16x4, const float2 scale) {
  uint16_t out_fp4x4 = 0;
  asm volatile(
      "{\n"
      ".reg.b16 x0_bf16; \n\t" // first bf16
      ".reg.b16 x1_bf16; \n\t" // second bf16
      ".reg.b16 x2_bf16; \n\t" // third bf16
      ".reg.b16 x3_bf16; \n\t" // fourth bf16
      ".reg.b32 x0; \n\t" // scaled first
      ".reg.b32 x1; \n\t" // scaled second
      ".reg.b32 x2; \n\t" // scaled third
      ".reg.b32 x3; \n\t" // scaled fourth
      ".reg.b64 x01; \n\t" // to do vector mul
      ".reg.b64 x23; \n\t" // to do vector mul
      ".reg.b8 q0; \n\t" // output byte fp4x2 (first pair)
      ".reg.b8 q1; \n\t" // output byte fp4x2 (second pair)
      "mov.b64 {x0_bf16, x1_bf16, x2_bf16, x3_bf16} , %1; \n\t" // unpack bf16
      "cvt.f32.bf16 x0, x0_bf16; \n\t" // convert to f32
      "cvt.f32.bf16 x1, x1_bf16; \n\t" // convert to f32
      "cvt.f32.bf16 x2, x2_bf16; \n\t" // convert to f32
      "cvt.f32.bf16 x3, x3_bf16; \n\t" // convert to f32
      "mov.b64 x01, {x0, x1}; \n\t" // move to 64 bit reg to do vector mul
      "mul.f32x2 x01, x01, %2; \n\t" // scale multiplication
      "mov.b64 x23, {x2, x3}; \n\t" // move to 64 bit reg to do vector mul
      "mul.f32x2 x23, x23, %2; \n\t" // scale multiplication
      "mov.b64 {x0, x1}, x01; \n\t" // unpack back to 2 f32
      "mov.b64 {x2, x3}, x23; \n\t" // unpack back to 2 f32
      "cvt.rn.satfinite.e2m1x2.f32 q0, x0, x1; \n\t" // convert to fp4x2 first
                                                     // pair
      "cvt.rn.satfinite.e2m1x2.f32 q1, x2, x3; \n\t" // convert to fp4x2 second
                                                     // pair
      "mov.b16 %0, {q0, q1}; \n\t" // pack to output
      "}"
      : "=h"(out_fp4x4)
      : "l"(reinterpret_cast<const uint64_t&>(input_bf16x4)),
        "l"(reinterpret_cast<const uint64_t&>(
            scale))); // here cast is needed becuase an asm operand must have
                      // scalar type
  return out_fp4x4;
}

__device__ __forceinline__ uint16_t scale_cvt_bf16x4_to_fp4x4_rs(
    const bf16x4 input_bf16x4,
    const float2 scale,
    uint32_t rbits) {
  uint16_t out_fp4x4 = 0;
  asm volatile(
      "{\n"
      ".reg.b16 x0_bf16; \n\t" // first bf16
      ".reg.b16 x1_bf16; \n\t" // second bf16
      ".reg.b16 x2_bf16; \n\t" // third bf16
      ".reg.b16 x3_bf16; \n\t" // fourth bf16
      ".reg.b32 x0; \n\t" // scaled first
      ".reg.b32 x1; \n\t" // scaled second
      ".reg.b32 x2; \n\t" // scaled third
      ".reg.b32 x3; \n\t" // scaled fourth
      ".reg.b64 x01; \n\t" // to do vector mul
      ".reg.b64 x23; \n\t" // to do vector mul
      ".reg.b16 q0; \n\t" // output byte fp4x4
      "mov.b64 {x0_bf16, x1_bf16, x2_bf16, x3_bf16} , %1; \n\t" // unpack bf16
                                                                // 2x
      "cvt.f32.bf16 x0, x0_bf16; \n\t" // convert to f32
      "cvt.f32.bf16 x1, x1_bf16; \n\t" // convert to f32
      "cvt.f32.bf16 x2, x2_bf16; \n\t" // convert to f32
      "cvt.f32.bf16 x3, x3_bf16; \n\t" // convert to f32
      "mov.b64 x01, {x0, x1}; \n\t" // move to 64 bit reg to do vector mul
      "mul.f32x2 x01, x01, %2; \n\t" // scale multiplication
      "mov.b64 x23, {x2, x3}; \n\t" // move to 64 bit reg to do vector mul
      "mul.f32x2 x23, x23, %2; \n\t" // scale multiplication
      "mov.b64 {x0, x1}, x01; \n\t" // unpack back to 2 f32
      "mov.b64 {x2, x3}, x23; \n\t" // unpack back to 2 f32
      "cvt.rs.satfinite.e2m1x4.f32 q0, {x0, x1, x2, x3}, %3; \n\t" // convert to
                                                                   // fp4x4
      "}"
      : "=h"(out_fp4x4)
      : "l"(reinterpret_cast<const uint64_t&>(input_bf16x4)),
        "l"(reinterpret_cast<const uint64_t&>(scale)),
        "r"(rbits)); // here cast is needed becuase an asm operand must have
                     // scalar type
  return out_fp4x4;
}

__device__ __forceinline__ uint16_t scale_cvt_32x4_to_fp4x4_rn(
    const float2 input_fp32x2_0,
    const float2 input_fp32x2_1,
    const float2 scale) {
  uint16_t out_fp4x4 = 0;
  asm volatile(
      "{\n"
      ".reg.b32 x0; \n\t" // first
      ".reg.b32 x1; \n\t" // second
      ".reg.b32 x2; \n\t" // third
      ".reg.b32 x3; \n\t" // fourth
      ".reg.b64 x01; \n\t" // to do vector mul
      ".reg.b64 x23; \n\t" // to do vector mul
      ".reg.b8 q0; \n\t" // output byte fp4x2 (first pair)
      ".reg.b8 q1; \n\t" // output byte fp4x2 (second pair)
      "mov.b64 x01, {%1, %2}; \n\t" // move input to 64 bit reg
      "mul.f32x2 x01, x01, %5; \n\t" // scale multiplication
      "mov.b64 x23, {%3, %4}; \n\t" // move input to 64 bit reg
      "mul.f32x2 x23, x23, %5; \n\t" // scale multiplication
      "mov.b64 {x0, x1}, x01; \n\t" // unpack back to 2 f32
      "mov.b64 {x2, x3}, x23; \n\t" // unpack back to 2 f32
      "cvt.rn.satfinite.e2m1x2.f32 q0, x0, x1; \n\t" // convert to fp4x2
      "cvt.rn.satfinite.e2m1x2.f32 q1, x2, x3; \n\t" // convert to fp4x2
      "mov.b16 %0, {q0, q1}; \n\t" // pack to output
      "}"
      : "=h"(out_fp4x4)
      : "f"(input_fp32x2_0.x),
        "f"(input_fp32x2_0.y),
        "f"(input_fp32x2_1.x),
        "f"(input_fp32x2_1.y),
        "l"(reinterpret_cast<const uint64_t&>(
            scale))); // here cast is needed becuase an asm operand must have
                      // scalar type
  return out_fp4x4;
}

__device__ __forceinline__ uint16_t scale_cvt_32x4_to_fp4x4_rs(
    const float2 input_fp32x2_0,
    const float2 input_fp32x2_1,
    const float2 scale,
    uint32_t rbits) {
  uint16_t out_fp4x4 = 0;
  asm volatile(
      "{\n"
      ".reg.b32 x0; \n\t" // first
      ".reg.b32 x1; \n\t" // second
      ".reg.b32 x2; \n\t" // third
      ".reg.b32 x3; \n\t" // fourth
      ".reg.b64 x01; \n\t" // to do vector mul
      ".reg.b64 x23; \n\t" // to do vector mul
      ".reg.b16 q0; \n\t" // output byte fp4x4
      "mov.b64 x01, {%1, %2}; \n\t" // move input to 64 bit reg
      "mul.f32x2 x01, x01, %5; \n\t" // scale multiplication
      "mov.b64 x23, {%3, %4}; \n\t" // move input to 64 bit reg
      "mul.f32x2 x23, x23, %5; \n\t" // scale multiplication
      "mov.b64 {x0, x1}, x01; \n\t" // unpack back to 2 f32
      "mov.b64 {x2, x3}, x23; \n\t" // unpack back to 2 f32
      "cvt.rs.satfinite.e2m1x4.f32 q0, {x0, x1, x2, x3}, %6; \n\t" // convert to
                                                                   // fp4x4
      "}"
      : "=h"(out_fp4x4)
      : "f"(input_fp32x2_0.x),
        "f"(input_fp32x2_0.y),
        "f"(input_fp32x2_1.x),
        "f"(input_fp32x2_1.y),
        "l"(reinterpret_cast<const uint64_t&>(scale)),
        "r"(rbits)); // here cast is needed becuase an asm operand must have
                     // scalar type
  return out_fp4x4;
}

__device__ __forceinline__ uint16_t
scale_cvt_fp16x4_to_fp4x4_rn(const uint64_t input_fp16x4, const float2 scale) {
  uint16_t out_fp4x4 = 0;
  asm volatile(
      "{\n"
      ".reg.b16 x0_fp16; \n\t" // first fp16
      ".reg.b16 x1_fp16; \n\t" // second fp16
      ".reg.b16 x2_fp16; \n\t" // third fp16
      ".reg.b16 x3_fp16; \n\t" // fourth fp16
      ".reg.b32 x0; \n\t" // scaled first
      ".reg.b32 x1; \n\t" // scaled second
      ".reg.b32 x2; \n\t" // scaled third
      ".reg.b32 x3; \n\t" // scaled fourth
      ".reg.b64 x01; \n\t" // to do vector mul
      ".reg.b64 x23; \n\t" // to do vector mul
      ".reg.b8 q0; \n\t" // output byte fp4x2 (first pair)
      ".reg.b8 q1; \n\t" // output byte fp4x2 (second pair)
      "mov.b64 {x0_fp16, x1_fp16, x2_fp16, x3_fp16} , %1; \n\t" // unpack fp16
                                                                // 2x
      "cvt.f32.fp16 x0, x0_fp16; \n\t" // convert to f32
      "cvt.f32.fp16 x1, x1_fp16; \n\t" // convert to f32
      "cvt.f32.fp16 x2, x2_fp16; \n\t" // convert to f32
      "cvt.f32.fp16 x3, x3_fp16; \n\t" // convert to f32
      "mov.b64 x01, {x0, x1}; \n\t" // move to 64 bit reg to do vector mul
      "mul.f32x2 x01, x01, %2; \n\t" // scale multiplication
      "mov.b64 x23, {x2, x3}; \n\t" // move to 64 bit reg to do vector mul
      "mul.f32x2 x23, x23, %2; \n\t" // scale multiplication
      "mov.b64 {x0, x1}, x01; \n\t" // unpack back to 2 f32
      "mov.b64 {x2, x3}, x23; \n\t" // unpack back to 2 f32
      "cvt.rn.satfinite.e2m1x2.f32 q0, x0, x1; \n\t" // convert to fp4x2 first
                                                     // pair
      "cvt.rn.satfinite.e2m1x2.f32 q1, x2, x3; \n\t" // convert to fp4x2 second
                                                     // pair
      "mov.b16 %0, {q0, q1}; \n\t" // pack to output
      "}"
      : "=h"(out_fp4x4)
      : "l"(input_fp16x4),
        "l"(reinterpret_cast<const uint64_t&>(
            scale))); // here cast is needed becuase an asm operand must have
                      // scalar type
  return out_fp4x4;
}

__device__ __forceinline__ uint16_t scale_cvt_fp16x4_to_fp4x4_rs(
    const uint64_t input_fp16x4,
    const float2 scale,
    uint32_t rbits) {
  uint16_t out_fp4x4 = 0;
  asm volatile(
      "{\n"
      ".reg.b16 x0_fp16; \n\t" // first fp16
      ".reg.b16 x1_fp16; \n\t" // second fp16
      ".reg.b16 x2_fp16; \n\t" // third fp16
      ".reg.b16 x3_fp16; \n\t" // fourth fp16
      ".reg.b32 x0; \n\t" // scaled first
      ".reg.b32 x1; \n\t" // scaled second
      ".reg.b32 x2; \n\t" // scaled third
      ".reg.b32 x3; \n\t" // scaled fourth
      ".reg.b64 x01; \n\t" // to do vector mul
      ".reg.b64 x23; \n\t" // to do vector mul
      ".reg.b16 q0; \n\t" // output byte fp4x4
      "mov.b64 {x0_fp16, x1_fp16, x2_fp16, x3_fp16} , %1; \n\t" // unpack fp16
                                                                // 2x
      "cvt.f32.fp16 x0, x0_fp16; \n\t" // convert to f32
      "cvt.f32.fp16 x1, x1_fp16; \n\t" // convert to f32
      "cvt.f32.fp16 x2, x2_fp16; \n\t" // convert to f32
      "cvt.f32.fp16 x3, x3_fp16; \n\t" // convert to f32
      "mov.b64 x01, {x0, x1}; \n\t" // move to 64 bit reg to do vector mul
      "mul.f32x2 x01, x01, %2; \n\t" // scale multiplication
      "mov.b64 x23, {x2, x3}; \n\t" // move to 64 bit reg to do vector mul
      "mul.f32x2 x23, x23, %2; \n\t" // scale multiplication
      "mov.b64 {x0, x1}, x01; \n\t" // unpack back to 2 f32
      "mov.b64 {x2, x3}, x23; \n\t" // unpack back to 2 f32
      "cvt.rs.satfinite.e2m1x4.f32 q0, {x0, x1, x2, x3}, %3; \n\t" // convert to
                                                                   // fp4x4
      "}"
      : "=h"(out_fp4x4)
      : "l"(input_fp16x4),
        "l"(reinterpret_cast<const uint64_t&>(scale)),
        "r"(rbits)); // here cast is needed becuase an asm operand must have
                     // scalar type
  return out_fp4x4;
}

template <bool USE_SR>
__device__ __forceinline__ uint16_t scale_cvt_bf16x4_to_fp4x4(
    const bf16x4 input,
    const float scale,
    uint32_t rbits) {
  float2 scale_fp32x2 = make_float2(scale, scale);
  if constexpr (USE_SR) {
    return scale_cvt_bf16x4_to_fp4x4_rs(input, scale_fp32x2, rbits);
  } else {
    return scale_cvt_bf16x4_to_fp4x4_rn(input, scale_fp32x2);
  }
}

template <bool USE_SR>
__device__ __forceinline__ uint16_t scale_cvt_fp16x4_to_fp4x4(
    const fp16x4 input,
    const float scale,
    uint32_t rbits) {
  float2 scale_fp32x2 = make_float2(scale, scale);
  if constexpr (USE_SR) {
    return scale_cvt_fp16x4_to_fp4x4_rs(input, scale_fp32x2, rbits);
  } else {
    return scale_cvt_fp16x4_to_fp4x4_rn(input, scale_fp32x2);
  }
}

template <bool USE_SR>
__device__ __forceinline__ uint16_t
scale_cvt_f32x4_to_fp4x4(const f32x4 input, const float scale, uint32_t rbits) {
  // Stochastic rounding requires PTX support - compile-time error if not
  // available
  float2 scale_fp32x2 = make_float2(scale, scale);
  float2 input_fp32x2_0 = make_float2(input.x, input.y);
  float2 input_fp32x2_1 = make_float2(input.z, input.w);

  if constexpr (USE_SR) {
    return scale_cvt_32x4_to_fp4x4_rs(
        input_fp32x2_0, input_fp32x2_1, scale_fp32x2, rbits);
  } else {
    return scale_cvt_32x4_to_fp4x4_rn(
        input_fp32x2_0, input_fp32x2_1, scale_fp32x2);
  }
}

template <typename T, bool USE_SR>
__device__ __forceinline__ uint16_t scale_cvt_Tx4_to_fp4x4_fast(
    const Vector4<T> input,
    const float scale,
    uint32_t rbits) {
  if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    return scale_cvt_bf16x4_to_fp4x4(input, scale, rbits);
  } else if constexpr (std::is_same<T, __half>::value) {
    return scale_cvt_fp16x4_to_fp4x4(input, scale, rbits);
  } else {
    return scale_cvt_f32x4_to_fp4x4(input, scale, rbits);
  }
}
#endif

template <typename T, bool USE_SR>
__device__ __forceinline__ uint16_t scale_cvt_Tx4_to_fp4x4(
    const Vector4<T> input,
    const float scale,
    uint32_t rbits) {
#ifdef CUDA_FP4_FP8_CVT_PTX_SUPPORT
  return scale_cvt_Tx4_to_fp4x4_fast<T, USE_SR>(input, scale, rbits);
#else
  static_assert(
      !USE_SR,
      "Stochastic rounding (USE_SR=true) requires CUDA_FP4_FP8_CVT_PTX_SUPPORT");
  return scale_cvt_Tx4_to_fp4x4_fallback(input, scale);
#endif
}
} // namespace mlx::core::cu