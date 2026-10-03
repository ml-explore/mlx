// Copyright © 2026 Apple Inc.

#include "mlx/backend/cpu/rope_highway.h"

#include "hwy/highway.h"

namespace mlx::core::fast {

#define MLX_HIGHWAY_CONCAT2(a, b) a##b
#define MLX_HIGHWAY_CONCAT(a, b) MLX_HIGHWAY_CONCAT2(a, b)

using RopeFn = int (*)(
    const void*,
    void*,
    RopeHighwayDType,
    const float*,
    const float*,
    int);
#define MLX_DECLARE_ROPE_TARGET(suffix)                                       \
  int MLX_HIGHWAY_CONCAT(rope_traditional_highway_forward, suffix)(           \
      const void*, void*, RopeHighwayDType, const float*, const float*, int); \
  int MLX_HIGHWAY_CONCAT(rope_traditional_highway_reverse, suffix)(           \
      const void*, void*, RopeHighwayDType, const float*, const float*, int); \
  int MLX_HIGHWAY_CONCAT(rope_non_traditional_highway_forward, suffix)(       \
      const void*, void*, RopeHighwayDType, const float*, const float*, int); \
  int MLX_HIGHWAY_CONCAT(rope_non_traditional_highway_reverse, suffix)(       \
      const void*, void*, RopeHighwayDType, const float*, const float*, int)

MLX_DECLARE_ROPE_TARGET(_avx2);
MLX_DECLARE_ROPE_TARGET(_sse4);
MLX_DECLARE_ROPE_TARGET(_ssse3);
MLX_DECLARE_ROPE_TARGET(_sse2);

#undef MLX_DECLARE_ROPE_TARGET

namespace {

struct RopeHighwayDispatch {
  RopeFn traditional_forward;
  RopeFn traditional_reverse;
  RopeFn non_traditional_forward;
  RopeFn non_traditional_reverse;
};

#define MLX_ROPE_DISPATCH(suffix)                                         \
  RopeHighwayDispatch {                                                   \
    MLX_HIGHWAY_CONCAT(rope_traditional_highway_forward, suffix),         \
        MLX_HIGHWAY_CONCAT(rope_traditional_highway_reverse, suffix),     \
        MLX_HIGHWAY_CONCAT(rope_non_traditional_highway_forward, suffix), \
        MLX_HIGHWAY_CONCAT(rope_non_traditional_highway_reverse, suffix)  \
  }

const RopeHighwayDispatch& rope_dispatch() {
  static const RopeHighwayDispatch dispatch = [] {
    const int64_t targets = hwy::SupportedTargets();
    if (targets & HWY_AVX2) {
      return MLX_ROPE_DISPATCH(_avx2);
    }
    if (targets & HWY_SSE4) {
      return MLX_ROPE_DISPATCH(_sse4);
    }
    if (targets & HWY_SSSE3) {
      return MLX_ROPE_DISPATCH(_ssse3);
    }
    return MLX_ROPE_DISPATCH(_sse2);
  }();
  return dispatch;
}

#undef MLX_ROPE_DISPATCH
#undef MLX_HIGHWAY_CONCAT
#undef MLX_HIGHWAY_CONCAT2

} // namespace

int rope_traditional_highway_forward(
    const void* x_in,
    void* x_out,
    RopeHighwayDType dtype,
    const float* cos_t,
    const float* sin_t,
    int half_dims) {
  return rope_dispatch().traditional_forward(
      x_in, x_out, dtype, cos_t, sin_t, half_dims);
}

int rope_traditional_highway_reverse(
    const void* x_in,
    void* x_out,
    RopeHighwayDType dtype,
    const float* cos_t,
    const float* sin_t,
    int half_dims) {
  return rope_dispatch().traditional_reverse(
      x_in, x_out, dtype, cos_t, sin_t, half_dims);
}

int rope_non_traditional_highway_forward(
    const void* x_in,
    void* x_out,
    RopeHighwayDType dtype,
    const float* cos_t,
    const float* sin_t,
    int half_dims) {
  return rope_dispatch().non_traditional_forward(
      x_in, x_out, dtype, cos_t, sin_t, half_dims);
}

int rope_non_traditional_highway_reverse(
    const void* x_in,
    void* x_out,
    RopeHighwayDType dtype,
    const float* cos_t,
    const float* sin_t,
    int half_dims) {
  return rope_dispatch().non_traditional_reverse(
      x_in, x_out, dtype, cos_t, sin_t, half_dims);
}

} // namespace mlx::core::fast
