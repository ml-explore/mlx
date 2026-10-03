// Copyright © 2026 Apple Inc.

#include "mlx/backend/cpu/norms_highway.h"

#include "hwy/highway.h"

namespace mlx::core::fast {

#define MLX_HIGHWAY_CONCAT2(a, b) a##b
#define MLX_HIGHWAY_CONCAT(a, b) MLX_HIGHWAY_CONCAT2(a, b)

using RmsNormFn =
    void (*)(const float*, const float*, float*, int, int, float, bool);
using RmsNormF16Fn = void (*)(
    const float16_t*,
    const float16_t*,
    float16_t*,
    int,
    int,
    float,
    bool);
using RmsNormBF16Fn = void (*)(
    const bfloat16_t*,
    const bfloat16_t*,
    bfloat16_t*,
    int,
    int,
    float,
    bool);
using LayerNormFn = void (*)(
    const float*,
    const float*,
    const float*,
    float*,
    int,
    int,
    float,
    bool,
    bool);
using LayerNormF16Fn = void (*)(
    const float16_t*,
    const float16_t*,
    const float16_t*,
    float16_t*,
    int,
    int,
    float,
    bool,
    bool);
using LayerNormBF16Fn = void (*)(
    const bfloat16_t*,
    const bfloat16_t*,
    const bfloat16_t*,
    bfloat16_t*,
    int,
    int,
    float,
    bool,
    bool);
#define MLX_DECLARE_NORMS_TARGET(suffix)                                      \
  void MLX_HIGHWAY_CONCAT(rms_norm_highway_float, suffix)(                    \
      const float*, const float*, float*, int, int, float, bool);             \
  void MLX_HIGHWAY_CONCAT(rms_norm_highway_float16, suffix)(                  \
      const float16_t*, const float16_t*, float16_t*, int, int, float, bool); \
  void MLX_HIGHWAY_CONCAT(rms_norm_highway_bfloat16, suffix)(                 \
      const bfloat16_t*,                                                      \
      const bfloat16_t*,                                                      \
      bfloat16_t*,                                                            \
      int,                                                                    \
      int,                                                                    \
      float,                                                                  \
      bool);                                                                  \
  void MLX_HIGHWAY_CONCAT(layer_norm_highway_float, suffix)(                  \
      const float*,                                                           \
      const float*,                                                           \
      const float*,                                                           \
      float*,                                                                 \
      int,                                                                    \
      int,                                                                    \
      float,                                                                  \
      bool,                                                                   \
      bool);                                                                  \
  void MLX_HIGHWAY_CONCAT(layer_norm_highway_float16, suffix)(                \
      const float16_t*,                                                       \
      const float16_t*,                                                       \
      const float16_t*,                                                       \
      float16_t*,                                                             \
      int,                                                                    \
      int,                                                                    \
      float,                                                                  \
      bool,                                                                   \
      bool);                                                                  \
  void MLX_HIGHWAY_CONCAT(layer_norm_highway_bfloat16, suffix)(               \
      const bfloat16_t*,                                                      \
      const bfloat16_t*,                                                      \
      const bfloat16_t*,                                                      \
      bfloat16_t*,                                                            \
      int,                                                                    \
      int,                                                                    \
      float,                                                                  \
      bool,                                                                   \
      bool)

MLX_DECLARE_NORMS_TARGET(_avx2);
MLX_DECLARE_NORMS_TARGET(_sse4);
MLX_DECLARE_NORMS_TARGET(_ssse3);
MLX_DECLARE_NORMS_TARGET(_sse2);

#undef MLX_DECLARE_NORMS_TARGET

namespace {

struct NormsHighwayDispatch {
  RmsNormFn rms_norm;
  RmsNormF16Fn rms_norm_f16;
  RmsNormBF16Fn rms_norm_bf16;
  LayerNormFn layer_norm;
  LayerNormF16Fn layer_norm_f16;
  LayerNormBF16Fn layer_norm_bf16;
};

#define MLX_NORMS_DISPATCH(suffix)                              \
  NormsHighwayDispatch {                                        \
    MLX_HIGHWAY_CONCAT(rms_norm_highway_float, suffix),         \
        MLX_HIGHWAY_CONCAT(rms_norm_highway_float16, suffix),   \
        MLX_HIGHWAY_CONCAT(rms_norm_highway_bfloat16, suffix),  \
        MLX_HIGHWAY_CONCAT(layer_norm_highway_float, suffix),   \
        MLX_HIGHWAY_CONCAT(layer_norm_highway_float16, suffix), \
        MLX_HIGHWAY_CONCAT(layer_norm_highway_bfloat16, suffix) \
  }

const NormsHighwayDispatch& norms_dispatch() {
  static const NormsHighwayDispatch dispatch = [] {
    const int64_t targets = hwy::SupportedTargets();
    if (targets & HWY_AVX2) {
      return MLX_NORMS_DISPATCH(_avx2);
    }
    if (targets & HWY_SSE4) {
      return MLX_NORMS_DISPATCH(_sse4);
    }
    if (targets & HWY_SSSE3) {
      return MLX_NORMS_DISPATCH(_ssse3);
    }
    return MLX_NORMS_DISPATCH(_sse2);
  }();
  return dispatch;
}

#undef MLX_NORMS_DISPATCH
#undef MLX_HIGHWAY_CONCAT
#undef MLX_HIGHWAY_CONCAT2

} // namespace

void rms_norm_highway_float(
    const float* x,
    const float* weight,
    float* out,
    int width,
    int rows,
    float eps,
    bool has_weight) {
  norms_dispatch().rms_norm(x, weight, out, width, rows, eps, has_weight);
}

void rms_norm_highway_float16(
    const float16_t* x,
    const float16_t* weight,
    float16_t* out,
    int width,
    int rows,
    float eps,
    bool has_weight) {
  norms_dispatch().rms_norm_f16(x, weight, out, width, rows, eps, has_weight);
}

void rms_norm_highway_bfloat16(
    const bfloat16_t* x,
    const bfloat16_t* weight,
    bfloat16_t* out,
    int width,
    int rows,
    float eps,
    bool has_weight) {
  norms_dispatch().rms_norm_bf16(x, weight, out, width, rows, eps, has_weight);
}

void layer_norm_highway_float(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    int width,
    int rows,
    float eps,
    bool has_weight,
    bool has_bias) {
  norms_dispatch().layer_norm(
      x, weight, bias, out, width, rows, eps, has_weight, has_bias);
}

void layer_norm_highway_float16(
    const float16_t* x,
    const float16_t* weight,
    const float16_t* bias,
    float16_t* out,
    int width,
    int rows,
    float eps,
    bool has_weight,
    bool has_bias) {
  norms_dispatch().layer_norm_f16(
      x, weight, bias, out, width, rows, eps, has_weight, has_bias);
}

void layer_norm_highway_bfloat16(
    const bfloat16_t* x,
    const bfloat16_t* weight,
    const bfloat16_t* bias,
    bfloat16_t* out,
    int width,
    int rows,
    float eps,
    bool has_weight,
    bool has_bias) {
  norms_dispatch().layer_norm_bf16(
      x, weight, bias, out, width, rows, eps, has_weight, has_bias);
}

} // namespace mlx::core::fast
