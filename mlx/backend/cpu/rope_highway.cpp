// Copyright © 2026 Apple Inc.

// Normally this file is compiled directly and Highway emits its runtime
// dispatch targets. Native MSVC builds compile this file once per target with
// MLX_HIGHWAY_MANUAL_TARGET and MLX_HIGHWAY_TARGET_SUFFIX so the same kernels
// are emitted as one manually suffixed specialization.

#include "mlx/backend/cpu/rope_highway.h"

#if !defined(MLX_HIGHWAY_MANUAL_TARGET)
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "mlx/backend/cpu/rope_highway.cpp"
#include "hwy/foreach_target.h" // IWYU pragma: keep
#endif

#include "hwy/highway.h"
#include "mlx/backend/cpu/highway_utils.h"

HWY_BEFORE_NAMESPACE();
namespace mlx::core::fast {
namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;
namespace hu = mlx::core::highway::HWY_NAMESPACE;
using hu::load_interleaved_typed_as_f32;
using hu::load_typed_as_f32;
using hu::store_f32_as_typed;
using hu::store_interleaved_f32_as_typed;

template <typename T, bool forward>
int rope_traditional(
    const T* HWY_RESTRICT x_in,
    T* HWY_RESTRICT x_out,
    const float* HWY_RESTRICT cos_t,
    const float* HWY_RESTRICT sin_t,
    int half_dims) {
  const hn::ScalableTag<float> d;
  using V = hn::Vec<decltype(d)>;

  const int lanes = static_cast<int>(hn::Lanes(d));

  int j = 0;
  HWY_UNROLL(4)
  for (; j + lanes <= half_dims; j += lanes) {
    V x0;
    V x1;
    load_interleaved_typed_as_f32(d, x_in, 2 * j, x0, x1);

    const V c = hn::LoadU(d, cos_t + j);
    const V s = hn::LoadU(d, sin_t + j);

    V out0;
    V out1;
    if constexpr (forward) {
      out0 = hn::Sub(hn::Mul(x0, c), hn::Mul(x1, s));
      out1 = hn::Add(hn::Mul(x0, s), hn::Mul(x1, c));
    } else {
      out0 = hn::Add(hn::Mul(x0, c), hn::Mul(x1, s));
      out1 = hn::Sub(hn::Mul(x1, c), hn::Mul(x0, s));
    }

    store_interleaved_f32_as_typed(d, out0, out1, x_out, 2 * j);
  }
  return j;
}

template <typename T, bool forward>
int rope_non_traditional(
    const T* HWY_RESTRICT x_in,
    T* HWY_RESTRICT x_out,
    const float* HWY_RESTRICT cos_t,
    const float* HWY_RESTRICT sin_t,
    int half_dims) {
  const hn::ScalableTag<float> d;
  using V = hn::Vec<decltype(d)>;

  const int lanes = static_cast<int>(hn::Lanes(d));

  int j = 0;
  HWY_UNROLL(4)
  for (; j + lanes <= half_dims; j += lanes) {
    const V x0 = load_typed_as_f32(d, x_in, j);
    const V x1 = load_typed_as_f32(d, x_in, j + half_dims);
    const V c = hn::LoadU(d, cos_t + j);
    const V s = hn::LoadU(d, sin_t + j);

    V out0;
    V out1;
    if constexpr (forward) {
      out0 = hn::Sub(hn::Mul(x0, c), hn::Mul(x1, s));
      out1 = hn::Add(hn::Mul(x0, s), hn::Mul(x1, c));
    } else {
      out0 = hn::Add(hn::Mul(x0, c), hn::Mul(x1, s));
      out1 = hn::Sub(hn::Mul(x1, c), hn::Mul(x0, s));
    }

    store_f32_as_typed(d, out0, x_out, j);
    store_f32_as_typed(d, out1, x_out, j + half_dims);
  }
  return j;
}

template <bool forward>
int rope_traditional_dispatch(
    const void* HWY_RESTRICT x_in,
    void* HWY_RESTRICT x_out,
    RopeHighwayDType dtype,
    const float* HWY_RESTRICT cos_t,
    const float* HWY_RESTRICT sin_t,
    int half_dims) {
  switch (dtype) {
    case RopeHighwayDType::Float32:
      return rope_traditional<float, forward>(
          static_cast<const float*>(x_in),
          static_cast<float*>(x_out),
          cos_t,
          sin_t,
          half_dims);
    case RopeHighwayDType::Float16:
      return rope_traditional<float16_t, forward>(
          static_cast<const float16_t*>(x_in),
          static_cast<float16_t*>(x_out),
          cos_t,
          sin_t,
          half_dims);
    case RopeHighwayDType::BFloat16:
      return rope_traditional<bfloat16_t, forward>(
          static_cast<const bfloat16_t*>(x_in),
          static_cast<bfloat16_t*>(x_out),
          cos_t,
          sin_t,
          half_dims);
  }
  return 0;
}

template <bool forward>
int rope_non_traditional_dispatch(
    const void* HWY_RESTRICT x_in,
    void* HWY_RESTRICT x_out,
    RopeHighwayDType dtype,
    const float* HWY_RESTRICT cos_t,
    const float* HWY_RESTRICT sin_t,
    int half_dims) {
  switch (dtype) {
    case RopeHighwayDType::Float32:
      return rope_non_traditional<float, forward>(
          static_cast<const float*>(x_in),
          static_cast<float*>(x_out),
          cos_t,
          sin_t,
          half_dims);
    case RopeHighwayDType::Float16:
      return rope_non_traditional<float16_t, forward>(
          static_cast<const float16_t*>(x_in),
          static_cast<float16_t*>(x_out),
          cos_t,
          sin_t,
          half_dims);
    case RopeHighwayDType::BFloat16:
      return rope_non_traditional<bfloat16_t, forward>(
          static_cast<const bfloat16_t*>(x_in),
          static_cast<bfloat16_t*>(x_out),
          cos_t,
          sin_t,
          half_dims);
  }
  return 0;
}

void RopeTraditionalForward(
    const void* HWY_RESTRICT x_in,
    void* HWY_RESTRICT x_out,
    RopeHighwayDType dtype,
    const float* HWY_RESTRICT cos_t,
    const float* HWY_RESTRICT sin_t,
    int half_dims,
    int* HWY_RESTRICT processed) {
  *processed = rope_traditional_dispatch<true>(
      x_in, x_out, dtype, cos_t, sin_t, half_dims);
}

void RopeTraditionalReverse(
    const void* HWY_RESTRICT x_in,
    void* HWY_RESTRICT x_out,
    RopeHighwayDType dtype,
    const float* HWY_RESTRICT cos_t,
    const float* HWY_RESTRICT sin_t,
    int half_dims,
    int* HWY_RESTRICT processed) {
  *processed = rope_traditional_dispatch<false>(
      x_in, x_out, dtype, cos_t, sin_t, half_dims);
}

void RopeNonTraditionalForward(
    const void* HWY_RESTRICT x_in,
    void* HWY_RESTRICT x_out,
    RopeHighwayDType dtype,
    const float* HWY_RESTRICT cos_t,
    const float* HWY_RESTRICT sin_t,
    int half_dims,
    int* HWY_RESTRICT processed) {
  *processed = rope_non_traditional_dispatch<true>(
      x_in, x_out, dtype, cos_t, sin_t, half_dims);
}

void RopeNonTraditionalReverse(
    const void* HWY_RESTRICT x_in,
    void* HWY_RESTRICT x_out,
    RopeHighwayDType dtype,
    const float* HWY_RESTRICT cos_t,
    const float* HWY_RESTRICT sin_t,
    int half_dims,
    int* HWY_RESTRICT processed) {
  *processed = rope_non_traditional_dispatch<false>(
      x_in, x_out, dtype, cos_t, sin_t, half_dims);
}

} // namespace
} // namespace HWY_NAMESPACE
} // namespace mlx::core::fast
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace mlx::core::fast {

#if defined(MLX_HIGHWAY_MANUAL_TARGET)

#ifndef MLX_HIGHWAY_TARGET_SUFFIX
#error "MLX_HIGHWAY_TARGET_SUFFIX must be defined for manual Highway targets"
#endif

#define MLX_HIGHWAY_CONCAT2(a, b) a##b
#define MLX_HIGHWAY_CONCAT(a, b) MLX_HIGHWAY_CONCAT2(a, b)
#define MLX_HIGHWAY_TARGET_FUNC(name) \
  MLX_HIGHWAY_CONCAT(name, MLX_HIGHWAY_TARGET_SUFFIX)

int MLX_HIGHWAY_TARGET_FUNC(rope_traditional_highway_forward)(
    const void* x_in,
    void* x_out,
    RopeHighwayDType dtype,
    const float* cos_t,
    const float* sin_t,
    int half_dims) {
  int processed = 0;
  HWY_STATIC_DISPATCH(RopeTraditionalForward)
  (x_in, x_out, dtype, cos_t, sin_t, half_dims, &processed);
  return processed;
}

int MLX_HIGHWAY_TARGET_FUNC(rope_traditional_highway_reverse)(
    const void* x_in,
    void* x_out,
    RopeHighwayDType dtype,
    const float* cos_t,
    const float* sin_t,
    int half_dims) {
  int processed = 0;
  HWY_STATIC_DISPATCH(RopeTraditionalReverse)
  (x_in, x_out, dtype, cos_t, sin_t, half_dims, &processed);
  return processed;
}

int MLX_HIGHWAY_TARGET_FUNC(rope_non_traditional_highway_forward)(
    const void* x_in,
    void* x_out,
    RopeHighwayDType dtype,
    const float* cos_t,
    const float* sin_t,
    int half_dims) {
  int processed = 0;
  HWY_STATIC_DISPATCH(RopeNonTraditionalForward)
  (x_in, x_out, dtype, cos_t, sin_t, half_dims, &processed);
  return processed;
}

int MLX_HIGHWAY_TARGET_FUNC(rope_non_traditional_highway_reverse)(
    const void* x_in,
    void* x_out,
    RopeHighwayDType dtype,
    const float* cos_t,
    const float* sin_t,
    int half_dims) {
  int processed = 0;
  HWY_STATIC_DISPATCH(RopeNonTraditionalReverse)
  (x_in, x_out, dtype, cos_t, sin_t, half_dims, &processed);
  return processed;
}

#undef MLX_HIGHWAY_TARGET_FUNC
#undef MLX_HIGHWAY_CONCAT
#undef MLX_HIGHWAY_CONCAT2

#else

HWY_EXPORT(RopeTraditionalForward);
HWY_EXPORT(RopeTraditionalReverse);
HWY_EXPORT(RopeNonTraditionalForward);
HWY_EXPORT(RopeNonTraditionalReverse);

int rope_traditional_highway_forward(
    const void* x_in,
    void* x_out,
    RopeHighwayDType dtype,
    const float* cos_t,
    const float* sin_t,
    int half_dims) {
  int processed = 0;
  HWY_DYNAMIC_DISPATCH(RopeTraditionalForward)
  (x_in, x_out, dtype, cos_t, sin_t, half_dims, &processed);
  return processed;
}

int rope_traditional_highway_reverse(
    const void* x_in,
    void* x_out,
    RopeHighwayDType dtype,
    const float* cos_t,
    const float* sin_t,
    int half_dims) {
  int processed = 0;
  HWY_DYNAMIC_DISPATCH(RopeTraditionalReverse)
  (x_in, x_out, dtype, cos_t, sin_t, half_dims, &processed);
  return processed;
}

int rope_non_traditional_highway_forward(
    const void* x_in,
    void* x_out,
    RopeHighwayDType dtype,
    const float* cos_t,
    const float* sin_t,
    int half_dims) {
  int processed = 0;
  HWY_DYNAMIC_DISPATCH(RopeNonTraditionalForward)
  (x_in, x_out, dtype, cos_t, sin_t, half_dims, &processed);
  return processed;
}

int rope_non_traditional_highway_reverse(
    const void* x_in,
    void* x_out,
    RopeHighwayDType dtype,
    const float* cos_t,
    const float* sin_t,
    int half_dims) {
  int processed = 0;
  HWY_DYNAMIC_DISPATCH(RopeNonTraditionalReverse)
  (x_in, x_out, dtype, cos_t, sin_t, half_dims, &processed);
  return processed;
}

#endif

} // namespace mlx::core::fast
#endif // HWY_ONCE
