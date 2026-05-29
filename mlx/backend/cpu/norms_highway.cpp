// Copyright © 2026 Apple Inc.

// Normally this file is compiled directly and Highway emits its runtime
// dispatch targets. Native MSVC builds compile this file once per target with
// MLX_HIGHWAY_MANUAL_TARGET and MLX_HIGHWAY_TARGET_SUFFIX so the same kernels
// are emitted as one manually suffixed specialization.

#include <cmath>

#if !defined(MLX_HIGHWAY_MANUAL_TARGET)
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "mlx/backend/cpu/norms_highway.cpp"
#include "hwy/foreach_target.h" // IWYU pragma: keep
#endif

#include "hwy/highway.h"
#include "mlx/backend/cpu/highway_utils.h"
#include "mlx/types/half_types.h"

HWY_BEFORE_NAMESPACE();
namespace mlx::core::fast {
namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;
namespace hu = mlx::core::highway::HWY_NAMESPACE;
using hu::load_typed_as_f32;
using hu::store_f32_as_typed;

void rms_norm_row_f32(
    const float* HWY_RESTRICT x,
    const float* HWY_RESTRICT weight,
    float* HWY_RESTRICT out,
    int width,
    float eps,
    int has_weight) {
  const hn::ScalableTag<float> d;
  using V = hn::Vec<decltype(d)>;

  const int lanes = static_cast<int>(hn::Lanes(d));
  float total = 0.0f;
  int i = 0;
  V sum_sq0 = hn::Zero(d);
  V sum_sq1 = hn::Zero(d);
  V sum_sq2 = hn::Zero(d);
  V sum_sq3 = hn::Zero(d);

  for (; i + 4 * lanes <= width; i += 4 * lanes) {
    const V vals0 = hn::LoadU(d, x + i);
    const V vals1 = hn::LoadU(d, x + i + lanes);
    const V vals2 = hn::LoadU(d, x + i + 2 * lanes);
    const V vals3 = hn::LoadU(d, x + i + 3 * lanes);
    sum_sq0 = hn::MulAdd(vals0, vals0, sum_sq0);
    sum_sq1 = hn::MulAdd(vals1, vals1, sum_sq1);
    sum_sq2 = hn::MulAdd(vals2, vals2, sum_sq2);
    sum_sq3 = hn::MulAdd(vals3, vals3, sum_sq3);
  }
  V sum_sq = hn::Add(hn::Add(sum_sq0, sum_sq1), hn::Add(sum_sq2, sum_sq3));
  for (; i + lanes <= width; i += lanes) {
    const V vals = hn::LoadU(d, x + i);
    sum_sq = hn::MulAdd(vals, vals, sum_sq);
  }

  total = hn::ReduceSum(d, sum_sq);
  for (; i < width; ++i) {
    total += x[i] * x[i];
  }

  const float scale = 1.0f / std::sqrt(total / width + eps);
  const V vscale = hn::Set(d, scale);

  i = 0;
  if (has_weight) {
    for (; i + lanes <= width; i += lanes) {
      const V vals = hn::LoadU(d, x + i);
      const V weights = hn::LoadU(d, weight + i);
      hn::StoreU(hn::Mul(hn::Mul(vals, vscale), weights), d, out + i);
    }
    for (; i < width; ++i) {
      out[i] = x[i] * scale * weight[i];
    }
  } else {
    for (; i + lanes <= width; i += lanes) {
      const V vals = hn::LoadU(d, x + i);
      hn::StoreU(hn::Mul(vals, vscale), d, out + i);
    }
    for (; i < width; ++i) {
      out[i] = x[i] * scale;
    }
  }
}

template <typename T>
void rms_norm_row_half(
    const T* HWY_RESTRICT x,
    const T* HWY_RESTRICT weight,
    T* HWY_RESTRICT out,
    int width,
    float eps,
    int has_weight) {
  const hn::ScalableTag<float> d;
  using V = hn::Vec<decltype(d)>;

  const int lanes = static_cast<int>(hn::Lanes(d));
  float total = 0.0f;
  int i = 0;
  V sum_sq0 = hn::Zero(d);
  V sum_sq1 = hn::Zero(d);
  V sum_sq2 = hn::Zero(d);
  V sum_sq3 = hn::Zero(d);

  for (; i + 4 * lanes <= width; i += 4 * lanes) {
    const V vals0 = load_typed_as_f32(d, x, i);
    const V vals1 = load_typed_as_f32(d, x, i + lanes);
    const V vals2 = load_typed_as_f32(d, x, i + 2 * lanes);
    const V vals3 = load_typed_as_f32(d, x, i + 3 * lanes);
    sum_sq0 = hn::MulAdd(vals0, vals0, sum_sq0);
    sum_sq1 = hn::MulAdd(vals1, vals1, sum_sq1);
    sum_sq2 = hn::MulAdd(vals2, vals2, sum_sq2);
    sum_sq3 = hn::MulAdd(vals3, vals3, sum_sq3);
  }
  V sum_sq = hn::Add(hn::Add(sum_sq0, sum_sq1), hn::Add(sum_sq2, sum_sq3));
  for (; i + lanes <= width; i += lanes) {
    const V vals = load_typed_as_f32(d, x, i);
    sum_sq = hn::MulAdd(vals, vals, sum_sq);
  }

  total = hn::ReduceSum(d, sum_sq);
  for (; i < width; ++i) {
    const float val = static_cast<float>(x[i]);
    total += val * val;
  }

  const float scale = 1.0f / std::sqrt(total / width + eps);
  const V vscale = hn::Set(d, scale);

  i = 0;
  if (has_weight) {
    for (; i + lanes <= width; i += lanes) {
      const V vals = load_typed_as_f32(d, x, i);
      const V weights = load_typed_as_f32(d, weight, i);
      store_f32_as_typed(d, hn::Mul(hn::Mul(vals, vscale), weights), out, i);
    }
    for (; i < width; ++i) {
      out[i] = static_cast<T>(
          static_cast<float>(x[i]) * scale * static_cast<float>(weight[i]));
    }
  } else {
    for (; i + lanes <= width; i += lanes) {
      const V vals = load_typed_as_f32(d, x, i);
      store_f32_as_typed(d, hn::Mul(vals, vscale), out, i);
    }
    for (; i < width; ++i) {
      out[i] = static_cast<T>(static_cast<float>(x[i]) * scale);
    }
  }
}

void layer_norm_row_f32(
    const float* HWY_RESTRICT x,
    const float* HWY_RESTRICT weight,
    const float* HWY_RESTRICT bias,
    float* HWY_RESTRICT out,
    int width,
    float eps,
    int has_weight,
    int has_bias) {
  const hn::ScalableTag<float> d;
  using V = hn::Vec<decltype(d)>;

  const int lanes = static_cast<int>(hn::Lanes(d));
  V sum0 = hn::Zero(d);
  V sum1 = hn::Zero(d);

  int i = 0;
  for (; i + 2 * lanes <= width; i += 2 * lanes) {
    sum0 = hn::Add(sum0, hn::LoadU(d, x + i));
    sum1 = hn::Add(sum1, hn::LoadU(d, x + i + lanes));
  }
  V sum = hn::Add(sum0, sum1);
  for (; i + lanes <= width; i += lanes) {
    sum = hn::Add(sum, hn::LoadU(d, x + i));
  }

  float total = hn::ReduceSum(d, sum);
  for (; i < width; ++i) {
    total += x[i];
  }
  const float mean = total / width;

  const V vmean = hn::Set(d, mean);
  V variance_sum0 = hn::Zero(d);
  V variance_sum1 = hn::Zero(d);

  i = 0;
  for (; i + 2 * lanes <= width; i += 2 * lanes) {
    const V diff0 = hn::Sub(hn::LoadU(d, x + i), vmean);
    const V diff1 = hn::Sub(hn::LoadU(d, x + i + lanes), vmean);
    variance_sum0 = hn::MulAdd(diff0, diff0, variance_sum0);
    variance_sum1 = hn::MulAdd(diff1, diff1, variance_sum1);
  }
  V variance_sum = hn::Add(variance_sum0, variance_sum1);
  for (; i + lanes <= width; i += lanes) {
    const V diff = hn::Sub(hn::LoadU(d, x + i), vmean);
    variance_sum = hn::MulAdd(diff, diff, variance_sum);
  }

  float variance = hn::ReduceSum(d, variance_sum);
  for (; i < width; ++i) {
    const float diff = x[i] - mean;
    variance += diff * diff;
  }
  variance /= width;

  const float scale = 1.0f / std::sqrt(variance + eps);
  const V vscale = hn::Set(d, scale);

  i = 0;
  for (; i + lanes <= width; i += lanes) {
    V result = hn::Mul(hn::Sub(hn::LoadU(d, x + i), vmean), vscale);
    if (has_weight) {
      result = hn::Mul(result, hn::LoadU(d, weight + i));
    }
    if (has_bias) {
      result = hn::Add(result, hn::LoadU(d, bias + i));
    }
    hn::StoreU(result, d, out + i);
  }

  for (; i < width; ++i) {
    float result = (x[i] - mean) * scale;
    if (has_weight) {
      result *= weight[i];
    }
    if (has_bias) {
      result += bias[i];
    }
    out[i] = result;
  }
}

template <typename T>
void layer_norm_row_half(
    const T* HWY_RESTRICT x,
    const T* HWY_RESTRICT weight,
    const T* HWY_RESTRICT bias,
    T* HWY_RESTRICT out,
    int width,
    float eps,
    int has_weight,
    int has_bias) {
  const hn::ScalableTag<float> d;
  using V = hn::Vec<decltype(d)>;

  const int lanes = static_cast<int>(hn::Lanes(d));
  V sum0 = hn::Zero(d);
  V sum1 = hn::Zero(d);

  int i = 0;
  for (; i + 2 * lanes <= width; i += 2 * lanes) {
    sum0 = hn::Add(sum0, load_typed_as_f32(d, x, i));
    sum1 = hn::Add(sum1, load_typed_as_f32(d, x, i + lanes));
  }
  V sum = hn::Add(sum0, sum1);
  for (; i + lanes <= width; i += lanes) {
    sum = hn::Add(sum, load_typed_as_f32(d, x, i));
  }

  float total = hn::ReduceSum(d, sum);
  for (; i < width; ++i) {
    total += static_cast<float>(x[i]);
  }
  const float mean = total / width;

  const V vmean = hn::Set(d, mean);
  V variance_sum0 = hn::Zero(d);
  V variance_sum1 = hn::Zero(d);

  i = 0;
  for (; i + 2 * lanes <= width; i += 2 * lanes) {
    const V diff0 = hn::Sub(load_typed_as_f32(d, x, i), vmean);
    const V diff1 = hn::Sub(load_typed_as_f32(d, x, i + lanes), vmean);
    variance_sum0 = hn::MulAdd(diff0, diff0, variance_sum0);
    variance_sum1 = hn::MulAdd(diff1, diff1, variance_sum1);
  }
  V variance_sum = hn::Add(variance_sum0, variance_sum1);
  for (; i + lanes <= width; i += lanes) {
    const V diff = hn::Sub(load_typed_as_f32(d, x, i), vmean);
    variance_sum = hn::MulAdd(diff, diff, variance_sum);
  }

  float variance = hn::ReduceSum(d, variance_sum);
  for (; i < width; ++i) {
    const float diff = static_cast<float>(x[i]) - mean;
    variance += diff * diff;
  }
  variance /= width;

  const float scale = 1.0f / std::sqrt(variance + eps);
  const V vscale = hn::Set(d, scale);

  i = 0;
  for (; i + lanes <= width; i += lanes) {
    V result = hn::Mul(hn::Sub(load_typed_as_f32(d, x, i), vmean), vscale);
    if (has_weight) {
      result = hn::Mul(result, load_typed_as_f32(d, weight, i));
    }
    if (has_bias) {
      result = hn::Add(result, load_typed_as_f32(d, bias, i));
    }
    store_f32_as_typed(d, result, out, i);
  }

  for (; i < width; ++i) {
    float result = (static_cast<float>(x[i]) - mean) * scale;
    if (has_weight) {
      result *= static_cast<float>(weight[i]);
    }
    if (has_bias) {
      result += static_cast<float>(bias[i]);
    }
    out[i] = static_cast<T>(result);
  }
}

void RmsNormF32(
    const float* HWY_RESTRICT x,
    const float* HWY_RESTRICT weight,
    float* HWY_RESTRICT out,
    int width,
    int rows,
    float eps,
    int has_weight) {
  for (int row = 0; row < rows; ++row) {
    rms_norm_row_f32(
        x + row * width, weight, out + row * width, width, eps, has_weight);
  }
}

template <typename T>
void RmsNormHalf(
    const T* HWY_RESTRICT x,
    const T* HWY_RESTRICT weight,
    T* HWY_RESTRICT out,
    int width,
    int rows,
    float eps,
    int has_weight) {
  for (int row = 0; row < rows; ++row) {
    rms_norm_row_half(
        x + row * width, weight, out + row * width, width, eps, has_weight);
  }
}

void RmsNormF16(
    const float16_t* HWY_RESTRICT x,
    const float16_t* HWY_RESTRICT weight,
    float16_t* HWY_RESTRICT out,
    int width,
    int rows,
    float eps,
    int has_weight) {
  RmsNormHalf(x, weight, out, width, rows, eps, has_weight);
}

void RmsNormBF16(
    const bfloat16_t* HWY_RESTRICT x,
    const bfloat16_t* HWY_RESTRICT weight,
    bfloat16_t* HWY_RESTRICT out,
    int width,
    int rows,
    float eps,
    int has_weight) {
  RmsNormHalf(x, weight, out, width, rows, eps, has_weight);
}

void LayerNormF32(
    const float* HWY_RESTRICT x,
    const float* HWY_RESTRICT weight,
    const float* HWY_RESTRICT bias,
    float* HWY_RESTRICT out,
    int width,
    int rows,
    float eps,
    int has_weight,
    int has_bias) {
  for (int row = 0; row < rows; ++row) {
    layer_norm_row_f32(
        x + row * width,
        weight,
        bias,
        out + row * width,
        width,
        eps,
        has_weight,
        has_bias);
  }
}

template <typename T>
void LayerNormHalf(
    const T* HWY_RESTRICT x,
    const T* HWY_RESTRICT weight,
    const T* HWY_RESTRICT bias,
    T* HWY_RESTRICT out,
    int width,
    int rows,
    float eps,
    int has_weight,
    int has_bias) {
  for (int row = 0; row < rows; ++row) {
    layer_norm_row_half(
        x + row * width,
        weight,
        bias,
        out + row * width,
        width,
        eps,
        has_weight,
        has_bias);
  }
}

void LayerNormF16(
    const float16_t* HWY_RESTRICT x,
    const float16_t* HWY_RESTRICT weight,
    const float16_t* HWY_RESTRICT bias,
    float16_t* HWY_RESTRICT out,
    int width,
    int rows,
    float eps,
    int has_weight,
    int has_bias) {
  LayerNormHalf(x, weight, bias, out, width, rows, eps, has_weight, has_bias);
}

void LayerNormBF16(
    const bfloat16_t* HWY_RESTRICT x,
    const bfloat16_t* HWY_RESTRICT weight,
    const bfloat16_t* HWY_RESTRICT bias,
    bfloat16_t* HWY_RESTRICT out,
    int width,
    int rows,
    float eps,
    int has_weight,
    int has_bias) {
  LayerNormHalf(x, weight, bias, out, width, rows, eps, has_weight, has_bias);
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

void MLX_HIGHWAY_TARGET_FUNC(rms_norm_highway_float)(
    const float* x,
    const float* weight,
    float* out,
    int width,
    int rows,
    float eps,
    bool has_weight) {
  HWY_STATIC_DISPATCH(RmsNormF32)
  (x, weight, out, width, rows, eps, has_weight ? 1 : 0);
}

void MLX_HIGHWAY_TARGET_FUNC(rms_norm_highway_float16)(
    const float16_t* x,
    const float16_t* weight,
    float16_t* out,
    int width,
    int rows,
    float eps,
    bool has_weight) {
  HWY_STATIC_DISPATCH(RmsNormF16)
  (x, weight, out, width, rows, eps, has_weight ? 1 : 0);
}

void MLX_HIGHWAY_TARGET_FUNC(rms_norm_highway_bfloat16)(
    const bfloat16_t* x,
    const bfloat16_t* weight,
    bfloat16_t* out,
    int width,
    int rows,
    float eps,
    bool has_weight) {
  HWY_STATIC_DISPATCH(RmsNormBF16)
  (x, weight, out, width, rows, eps, has_weight ? 1 : 0);
}

void MLX_HIGHWAY_TARGET_FUNC(layer_norm_highway_float)(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    int width,
    int rows,
    float eps,
    bool has_weight,
    bool has_bias) {
  HWY_STATIC_DISPATCH(LayerNormF32)
  (x,
   weight,
   bias,
   out,
   width,
   rows,
   eps,
   has_weight ? 1 : 0,
   has_bias ? 1 : 0);
}

void MLX_HIGHWAY_TARGET_FUNC(layer_norm_highway_float16)(
    const float16_t* x,
    const float16_t* weight,
    const float16_t* bias,
    float16_t* out,
    int width,
    int rows,
    float eps,
    bool has_weight,
    bool has_bias) {
  HWY_STATIC_DISPATCH(LayerNormF16)
  (x,
   weight,
   bias,
   out,
   width,
   rows,
   eps,
   has_weight ? 1 : 0,
   has_bias ? 1 : 0);
}

void MLX_HIGHWAY_TARGET_FUNC(layer_norm_highway_bfloat16)(
    const bfloat16_t* x,
    const bfloat16_t* weight,
    const bfloat16_t* bias,
    bfloat16_t* out,
    int width,
    int rows,
    float eps,
    bool has_weight,
    bool has_bias) {
  HWY_STATIC_DISPATCH(LayerNormBF16)
  (x,
   weight,
   bias,
   out,
   width,
   rows,
   eps,
   has_weight ? 1 : 0,
   has_bias ? 1 : 0);
}

#undef MLX_HIGHWAY_TARGET_FUNC
#undef MLX_HIGHWAY_CONCAT
#undef MLX_HIGHWAY_CONCAT2

#else

HWY_EXPORT(RmsNormF32);
HWY_EXPORT(RmsNormF16);
HWY_EXPORT(RmsNormBF16);
HWY_EXPORT(LayerNormF32);
HWY_EXPORT(LayerNormF16);
HWY_EXPORT(LayerNormBF16);

void rms_norm_highway_float(
    const float* x,
    const float* weight,
    float* out,
    int width,
    int rows,
    float eps,
    bool has_weight) {
  HWY_DYNAMIC_DISPATCH(RmsNormF32)
  (x, weight, out, width, rows, eps, has_weight ? 1 : 0);
}

void rms_norm_highway_float16(
    const float16_t* x,
    const float16_t* weight,
    float16_t* out,
    int width,
    int rows,
    float eps,
    bool has_weight) {
  HWY_DYNAMIC_DISPATCH(RmsNormF16)
  (x, weight, out, width, rows, eps, has_weight ? 1 : 0);
}

void rms_norm_highway_bfloat16(
    const bfloat16_t* x,
    const bfloat16_t* weight,
    bfloat16_t* out,
    int width,
    int rows,
    float eps,
    bool has_weight) {
  HWY_DYNAMIC_DISPATCH(RmsNormBF16)
  (x, weight, out, width, rows, eps, has_weight ? 1 : 0);
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
  HWY_DYNAMIC_DISPATCH(LayerNormF32)
  (x,
   weight,
   bias,
   out,
   width,
   rows,
   eps,
   has_weight ? 1 : 0,
   has_bias ? 1 : 0);
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
  HWY_DYNAMIC_DISPATCH(LayerNormF16)
  (x,
   weight,
   bias,
   out,
   width,
   rows,
   eps,
   has_weight ? 1 : 0,
   has_bias ? 1 : 0);
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
  HWY_DYNAMIC_DISPATCH(LayerNormBF16)
  (x,
   weight,
   bias,
   out,
   width,
   rows,
   eps,
   has_weight ? 1 : 0,
   has_bias ? 1 : 0);
}

#endif

} // namespace mlx::core::fast
#endif // HWY_ONCE
