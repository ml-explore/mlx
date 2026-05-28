// Copyright © 2026 Apple Inc.

#include <cmath>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "mlx/backend/cpu/norms_highway.cpp"
#include "hwy/foreach_target.h" // IWYU pragma: keep

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace mlx::core::fast {
namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

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
  V sum_sq0 = hn::Zero(d);
  V sum_sq1 = hn::Zero(d);

  int i = 0;
  for (; i + 2 * lanes <= width; i += 2 * lanes) {
    const V vals0 = hn::LoadU(d, x + i);
    const V vals1 = hn::LoadU(d, x + i + lanes);
    sum_sq0 = hn::MulAdd(vals0, vals0, sum_sq0);
    sum_sq1 = hn::MulAdd(vals1, vals1, sum_sq1);
  }
  V sum_sq = hn::Add(sum_sq0, sum_sq1);
  for (; i + lanes <= width; i += lanes) {
    const V vals = hn::LoadU(d, x + i);
    sum_sq = hn::MulAdd(vals, vals, sum_sq);
  }

  float total = hn::ReduceSum(d, sum_sq);
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

} // namespace
} // namespace HWY_NAMESPACE
} // namespace mlx::core::fast
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace mlx::core::fast {

HWY_EXPORT(RmsNormF32);
HWY_EXPORT(LayerNormF32);

void rms_norm_highway_float(
    const float* x,
    const float* weight,
    float* out,
    int width,
    int rows,
    float eps,
    bool has_weight) {
  HWY_DYNAMIC_DISPATCH(RmsNormF32)(
      x, weight, out, width, rows, eps, has_weight ? 1 : 0);
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
  HWY_DYNAMIC_DISPATCH(LayerNormF32)(
      x,
      weight,
      bias,
      out,
      width,
      rows,
      eps,
      has_weight ? 1 : 0,
      has_bias ? 1 : 0);
}

} // namespace mlx::core::fast
#endif // HWY_ONCE
