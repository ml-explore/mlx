// Copyright © 2026 Apple Inc.

// Normally this file is compiled directly and Highway emits its runtime
// dispatch targets. Native MSVC builds compile this file once per target with
// MLX_HIGHWAY_MANUAL_TARGET and MLX_HIGHWAY_TARGET_SUFFIX so the same kernels
// are emitted as one manually suffixed specialization.

#include <algorithm>
#include <cmath>
#include <limits>

#include "mlx/backend/cpu/sdpa_highway.h"
#include "mlx/backend/cpu/threading/common.h"
#include "mlx/types/half_types.h"

#if !defined(MLX_HIGHWAY_MANUAL_TARGET)
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "mlx/backend/cpu/sdpa_highway.cpp"
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
using hu::load_typed_as_f32;
using hu::store_f32_as_typed;

template <class DF>
hn::Vec<DF> exp_lanes(DF df, hn::Vec<DF> x) {
  using V = hn::Vec<DF>;
  const hn::Rebind<int32_t, DF> di;

  const V x_init = x;
  x = hn::Mul(x, hn::Set(df, 1.442695f));
  const V ipart = hn::Floor(hn::Add(x, hn::Set(df, 0.5f)));
  const V fpart = hn::Sub(x, ipart);

  V poly = hn::Set(df, 1.535336188319500e-4f);
  poly = hn::MulAdd(poly, fpart, hn::Set(df, 1.339887440266574e-3f));
  poly = hn::MulAdd(poly, fpart, hn::Set(df, 9.618437357674640e-3f));
  poly = hn::MulAdd(poly, fpart, hn::Set(df, 5.550332471162809e-2f));
  poly = hn::MulAdd(poly, fpart, hn::Set(df, 2.402264791363012e-1f));
  poly = hn::MulAdd(poly, fpart, hn::Set(df, 6.931472028550421e-1f));
  poly = hn::MulAdd(poly, fpart, hn::Set(df, 1.000000000000000f));

  const auto biased_exp =
      hn::Add(hn::ConvertTo(di, ipart), hn::Set(di, int32_t{127}));
  V result = hn::Mul(hn::BitCast(df, hn::ShiftLeft<23>(biased_exp)), poly);
  result =
      hn::IfThenElse(hn::Lt(x_init, hn::Set(df, -88.0f)), hn::Zero(df), result);
  result = hn::IfThenElse(
      hn::Gt(x_init, hn::Set(df, 88.0f)),
      hn::Set(df, std::numeric_limits<float>::infinity()),
      result);
  return result;
}

template <typename T, class DF>
float sdpa_dot_one(
    DF df,
    const float* HWY_RESTRICT query,
    const T* HWY_RESTRICT key,
    int head_dim,
    float scale,
    const T* HWY_RESTRICT mask,
    int row) {
  using V = hn::Vec<DF>;
  const int lanes = static_cast<int>(hn::Lanes(df));
  V dot_acc = hn::Zero(df);
  int d = 0;
  for (; d + lanes <= head_dim; d += lanes) {
    dot_acc = hn::MulAdd(
        hn::LoadU(df, query + d), load_typed_as_f32(df, key, d), dot_acc);
  }
  float dot = hn::ReduceSum(df, dot_acc);
  for (; d < head_dim; ++d) {
    dot += query[d] * static_cast<float>(key[d]);
  }
  dot *= scale;
  if (mask) {
    dot += static_cast<float>(mask[row]);
  }
  return dot;
}

template <typename T, class DF>
void sdpa_dot_four(
    DF df,
    const float* HWY_RESTRICT query,
    const T* HWY_RESTRICT keys,
    int head_dim,
    float scale,
    const T* HWY_RESTRICT mask,
    int row,
    float* HWY_RESTRICT dots) {
  using V = hn::Vec<DF>;
  const int lanes = static_cast<int>(hn::Lanes(df));
  const T* key0 = keys + row * head_dim;
  const T* key1 = key0 + head_dim;
  const T* key2 = key1 + head_dim;
  const T* key3 = key2 + head_dim;

  V acc0 = hn::Zero(df);
  V acc1 = hn::Zero(df);
  V acc2 = hn::Zero(df);
  V acc3 = hn::Zero(df);
  int d = 0;
  for (; d + lanes <= head_dim; d += lanes) {
    const V q = hn::LoadU(df, query + d);
    acc0 = hn::MulAdd(q, load_typed_as_f32(df, key0, d), acc0);
    acc1 = hn::MulAdd(q, load_typed_as_f32(df, key1, d), acc1);
    acc2 = hn::MulAdd(q, load_typed_as_f32(df, key2, d), acc2);
    acc3 = hn::MulAdd(q, load_typed_as_f32(df, key3, d), acc3);
  }

  float dot0 = hn::ReduceSum(df, acc0);
  float dot1 = hn::ReduceSum(df, acc1);
  float dot2 = hn::ReduceSum(df, acc2);
  float dot3 = hn::ReduceSum(df, acc3);
  for (; d < head_dim; ++d) {
    const float q = query[d];
    dot0 += q * static_cast<float>(key0[d]);
    dot1 += q * static_cast<float>(key1[d]);
    dot2 += q * static_cast<float>(key2[d]);
    dot3 += q * static_cast<float>(key3[d]);
  }

  dots[0] = dot0 * scale;
  dots[1] = dot1 * scale;
  dots[2] = dot2 * scale;
  dots[3] = dot3 * scale;
  if (mask) {
    dots[0] += static_cast<float>(mask[row]);
    dots[1] += static_cast<float>(mask[row + 1]);
    dots[2] += static_cast<float>(mask[row + 2]);
    dots[3] += static_cast<float>(mask[row + 3]);
  }
}

template <typename T, class DF>
void sdpa_accum_values(
    DF df,
    hn::Vec<DF>* HWY_RESTRICT output,
    int n_acc,
    int lanes,
    int head_dim,
    const T* HWY_RESTRICT values,
    const float* HWY_RESTRICT weights,
    int rows) {
  using V = hn::Vec<DF>;
  for (int i = 0; i < n_acc && i * lanes < head_dim; ++i) {
    const int offset = i * lanes;
    V acc = output[i];
    if (offset + lanes <= head_dim) {
      for (int r = 0; r < rows; ++r) {
        acc = hn::MulAdd(
            hn::Set(df, weights[r]),
            load_typed_as_f32(df, values + r * head_dim, offset),
            acc);
      }
    } else {
      const int remaining = head_dim - offset;
      for (int r = 0; r < rows; ++r) {
        alignas(64) float v_tmp[16] = {};
        for (int j = 0; j < remaining; ++j) {
          v_tmp[j] = static_cast<float>(values[r * head_dim + offset + j]);
        }
        acc = hn::MulAdd(hn::Set(df, weights[r]), hn::LoadU(df, v_tmp), acc);
      }
    }
    output[i] = acc;
  }
}

template <typename T>
void sdpa_single_head(
    T* HWY_RESTRICT output,
    const T* HWY_RESTRICT query,
    const T* HWY_RESTRICT keys,
    const T* HWY_RESTRICT values,
    int head_dim,
    float scale,
    int effective_len,
    const T* HWY_RESTRICT mask,
    float sink_value,
    bool has_sinks) {
  const hn::ScalableTag<float> df;
  using V = hn::Vec<decltype(df)>;

  constexpr int kMaxHeadDim = 256;
  constexpr int kMaxAcc = 64;
  const int lanes = static_cast<int>(hn::Lanes(df));
  const int n_acc = std::min((head_dim + lanes - 1) / lanes, kMaxAcc);
  const float neg_inf = -std::numeric_limits<float>::infinity();

  alignas(64) float q_f[kMaxHeadDim];
  int d = 0;
  for (; d + lanes <= head_dim; d += lanes) {
    hn::StoreU(load_typed_as_f32(df, query, d), df, q_f + d);
  }
  for (; d < head_dim; ++d) {
    q_f[d] = static_cast<float>(query[d]);
  }

  V O_acc[kMaxAcc];
  for (int i = 0; i < n_acc; ++i) {
    O_acc[i] = hn::Zero(df);
  }
  float running_max = neg_inf;
  float running_sum = 0.0f;

  int s = 0;
  for (; s + lanes <= effective_len; s += lanes) {
    alignas(64) float dots[16];
    int b = 0;
    for (; b + 4 <= lanes; b += 4) {
      sdpa_dot_four(df, q_f, keys, head_dim, scale, mask, s + b, dots + b);
    }
    for (; b < lanes; ++b) {
      dots[b] = sdpa_dot_one(
          df, q_f, keys + (s + b) * head_dim, head_dim, scale, mask, s + b);
    }

    const V vdots = hn::LoadU(df, dots);
    const float block_max = hn::ReduceMax(df, vdots);
    if (block_max == neg_inf) {
      continue;
    }
    if (block_max > running_max) {
      const float correction = std::exp(running_max - block_max);
      running_sum *= correction;
      const V vcorr = hn::Set(df, correction);
      for (int i = 0; i < n_acc; ++i) {
        O_acc[i] = hn::Mul(O_acc[i], vcorr);
      }
      running_max = block_max;
    }

    const V vexp = exp_lanes(df, hn::Sub(vdots, hn::Set(df, running_max)));
    running_sum += hn::ReduceSum(df, vexp);

    alignas(64) float exp_arr[16];
    hn::StoreU(vexp, df, exp_arr);
    for (int b = 0; b < lanes; b += 4) {
      sdpa_accum_values(
          df,
          O_acc,
          n_acc,
          lanes,
          head_dim,
          values + (s + b) * head_dim,
          exp_arr + b,
          std::min(4, lanes - b));
    }
  }

  for (; s < effective_len; ++s) {
    const float dot =
        sdpa_dot_one(df, q_f, keys + s * head_dim, head_dim, scale, mask, s);
    if (dot == neg_inf) {
      continue;
    }

    const float new_max = std::max(running_max, dot);
    if (new_max > running_max) {
      const float correction = std::exp(running_max - new_max);
      running_sum *= correction;
      const V vcorr = hn::Set(df, correction);
      for (int i = 0; i < n_acc; ++i) {
        O_acc[i] = hn::Mul(O_acc[i], vcorr);
      }
      running_max = new_max;
    }
    const float exp_dot = std::exp(dot - running_max);
    running_sum += exp_dot;
    sdpa_accum_values(
        df, O_acc, n_acc, lanes, head_dim, values + s * head_dim, &exp_dot, 1);
  }

  if (has_sinks) {
    const float new_max = std::max(running_max, sink_value);
    if (new_max > neg_inf) {
      const float correction = std::exp(running_max - new_max);
      running_sum *= correction;
      const V vcorr = hn::Set(df, correction);
      for (int i = 0; i < n_acc; ++i) {
        O_acc[i] = hn::Mul(O_acc[i], vcorr);
      }
      running_sum += std::exp(sink_value - new_max);
    }
  }

  if (running_sum > 0.0f) {
    const float inv_sum = 1.0f / running_sum;
    const V vinv = hn::Set(df, inv_sum);
    for (int i = 0; i < n_acc && i * lanes < head_dim; ++i) {
      const V result = hn::Mul(O_acc[i], vinv);
      if (i * lanes + lanes <= head_dim) {
        store_f32_as_typed(df, result, output, i * lanes);
      } else {
        alignas(64) float tmp[16];
        hn::StoreU(result, df, tmp);
        const int remaining = head_dim - i * lanes;
        for (int j = 0; j < remaining; ++j) {
          output[i * lanes + j] = static_cast<T>(tmp[j]);
        }
      }
    }
  } else {
    for (int i = 0; i < head_dim; ++i) {
      output[i] = static_cast<T>(0.0f);
    }
  }
}

template <typename T>
void SdpaHighwayForDType(
    T* HWY_RESTRICT output,
    const T* HWY_RESTRICT queries,
    const T* HWY_RESTRICT keys,
    const T* HWY_RESTRICT values,
    int B,
    int n_q_heads,
    int n_kv_heads,
    int M,
    int seq_len,
    int head_dim,
    float scale,
    bool do_causal,
    const T* HWY_RESTRICT mask,
    bool has_mask,
    const T* HWY_RESTRICT sinks,
    bool has_sinks) {
  const int n_repeats = n_q_heads / n_kv_heads;
  const int total_work = B * n_q_heads * M;

  auto& pool = cpu::ThreadPool::instance();
  int nth = std::min(pool.max_threads(), total_work);
  if (total_work < 4) {
    nth = 1;
  }

  auto work = [&](int tid, int num_threads) {
    const int work_per = (total_work + num_threads - 1) / num_threads;
    const int start = work_per * tid;
    const int end = std::min(start + work_per, total_work);

    for (int idx = start; idx < end; ++idx) {
      const int m = idx % M;
      const int rem = idx / M;
      const int h = rem % n_q_heads;
      const int b = rem / n_q_heads;
      const int kv_h = h / n_repeats;

      const T* q = queries + ((b * n_q_heads + h) * M + m) * head_dim;
      const T* k = keys + (b * n_kv_heads + kv_h) * seq_len * head_dim;
      const T* v = values + (b * n_kv_heads + kv_h) * seq_len * head_dim;
      T* out = output + ((b * n_q_heads + h) * M + m) * head_dim;

      const int effective_len =
          do_causal ? std::min(seq_len, seq_len - M + m + 1) : seq_len;
      const T* mask_ptr =
          has_mask ? mask + ((b * n_q_heads + h) * M + m) * seq_len : nullptr;
      const float sink_val = has_sinks ? static_cast<float>(sinks[h]) : 0.0f;

      sdpa_single_head(
          out,
          q,
          k,
          v,
          head_dim,
          scale,
          effective_len,
          mask_ptr,
          sink_val,
          has_sinks);
    }
  };

  if (nth > 1) {
    pool.parallel_for(nth, work);
  } else {
    work(0, 1);
  }
}

void SdpaHighway(
    void* HWY_RESTRICT output,
    const void* HWY_RESTRICT queries,
    const void* HWY_RESTRICT keys,
    const void* HWY_RESTRICT values,
    SdpaHighwayDType dtype,
    int B,
    int n_q_heads,
    int n_kv_heads,
    int M,
    int seq_len,
    int head_dim,
    float scale,
    bool do_causal,
    const void* HWY_RESTRICT mask,
    bool has_mask,
    const void* HWY_RESTRICT sinks,
    bool has_sinks) {
  switch (dtype) {
    case SdpaHighwayDType::Float32:
      SdpaHighwayForDType(
          static_cast<float*>(output),
          static_cast<const float*>(queries),
          static_cast<const float*>(keys),
          static_cast<const float*>(values),
          B,
          n_q_heads,
          n_kv_heads,
          M,
          seq_len,
          head_dim,
          scale,
          do_causal,
          static_cast<const float*>(mask),
          has_mask,
          static_cast<const float*>(sinks),
          has_sinks);
      break;
    case SdpaHighwayDType::Float16:
      SdpaHighwayForDType(
          static_cast<float16_t*>(output),
          static_cast<const float16_t*>(queries),
          static_cast<const float16_t*>(keys),
          static_cast<const float16_t*>(values),
          B,
          n_q_heads,
          n_kv_heads,
          M,
          seq_len,
          head_dim,
          scale,
          do_causal,
          static_cast<const float16_t*>(mask),
          has_mask,
          static_cast<const float16_t*>(sinks),
          has_sinks);
      break;
    case SdpaHighwayDType::BFloat16:
      SdpaHighwayForDType(
          static_cast<bfloat16_t*>(output),
          static_cast<const bfloat16_t*>(queries),
          static_cast<const bfloat16_t*>(keys),
          static_cast<const bfloat16_t*>(values),
          B,
          n_q_heads,
          n_kv_heads,
          M,
          seq_len,
          head_dim,
          scale,
          do_causal,
          static_cast<const bfloat16_t*>(mask),
          has_mask,
          static_cast<const bfloat16_t*>(sinks),
          has_sinks);
      break;
  }
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

void MLX_HIGHWAY_TARGET_FUNC(sdpa_highway)(
    void* output,
    const void* queries,
    const void* keys,
    const void* values,
    SdpaHighwayDType dtype,
    int B,
    int n_q_heads,
    int n_kv_heads,
    int M,
    int seq_len,
    int head_dim,
    float scale,
    bool do_causal,
    const void* mask,
    bool has_mask,
    const void* sinks,
    bool has_sinks) {
  HWY_STATIC_DISPATCH(SdpaHighway)
  (output,
   queries,
   keys,
   values,
   dtype,
   B,
   n_q_heads,
   n_kv_heads,
   M,
   seq_len,
   head_dim,
   scale,
   do_causal,
   mask,
   has_mask,
   sinks,
   has_sinks);
}

#undef MLX_HIGHWAY_TARGET_FUNC
#undef MLX_HIGHWAY_CONCAT
#undef MLX_HIGHWAY_CONCAT2

#else

HWY_EXPORT(SdpaHighway);

void sdpa_highway(
    void* output,
    const void* queries,
    const void* keys,
    const void* values,
    SdpaHighwayDType dtype,
    int B,
    int n_q_heads,
    int n_kv_heads,
    int M,
    int seq_len,
    int head_dim,
    float scale,
    bool do_causal,
    const void* mask,
    bool has_mask,
    const void* sinks,
    bool has_sinks) {
  HWY_DYNAMIC_DISPATCH(SdpaHighway)
  (output,
   queries,
   keys,
   values,
   dtype,
   B,
   n_q_heads,
   n_kv_heads,
   M,
   seq_len,
   head_dim,
   scale,
   do_causal,
   mask,
   has_mask,
   sinks,
   has_sinks);
}

#endif

} // namespace mlx::core::fast
#endif
