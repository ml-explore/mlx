// Copyright © 2026 Apple Inc.

#include "mlx/backend/cpu/conv_highway.h"

#include "hwy/highway.h"

namespace mlx::core::fast {

#define MLX_HIGHWAY_CONCAT2(a, b) a##b
#define MLX_HIGHWAY_CONCAT(a, b) MLX_HIGHWAY_CONCAT2(a, b)

using DepthwiseConv1DKernel4Fn = void (*)(
    void*,
    const void*,
    const void*,
    ConvHighwayDType,
    int,
    int,
    int,
    int);

#define MLX_DECLARE_CONV_TARGET(suffix)                              \
  void MLX_HIGHWAY_CONCAT(depthwise_conv1d_kernel4_highway, suffix)( \
      void*, const void*, const void*, ConvHighwayDType, int, int, int, int)

MLX_DECLARE_CONV_TARGET(_avx2);
MLX_DECLARE_CONV_TARGET(_sse4);
MLX_DECLARE_CONV_TARGET(_ssse3);
MLX_DECLARE_CONV_TARGET(_sse2);

#undef MLX_DECLARE_CONV_TARGET

namespace {

struct ConvHighwayDispatch {
  DepthwiseConv1DKernel4Fn depthwise_conv1d_kernel4;
};

#define MLX_CONV_DISPATCH(suffix)                                \
  ConvHighwayDispatch {                                          \
    MLX_HIGHWAY_CONCAT(depthwise_conv1d_kernel4_highway, suffix) \
  }

const ConvHighwayDispatch& conv_dispatch() {
  static const ConvHighwayDispatch dispatch = [] {
    const int64_t targets = hwy::SupportedTargets();
    if (targets & HWY_AVX2) {
      return MLX_CONV_DISPATCH(_avx2);
    }
    if (targets & HWY_SSE4) {
      return MLX_CONV_DISPATCH(_sse4);
    }
    if (targets & HWY_SSSE3) {
      return MLX_CONV_DISPATCH(_ssse3);
    }
    return MLX_CONV_DISPATCH(_sse2);
  }();
  return dispatch;
}

#undef MLX_CONV_DISPATCH
#undef MLX_HIGHWAY_CONCAT
#undef MLX_HIGHWAY_CONCAT2

} // namespace

void depthwise_conv1d_kernel4_highway(
    void* out,
    const void* in,
    const void* wt,
    ConvHighwayDType dtype,
    int N,
    int iH,
    int C,
    int oH) {
  conv_dispatch().depthwise_conv1d_kernel4(out, in, wt, dtype, N, iH, C, oH);
}

} // namespace mlx::core::fast
