// Copyright © 2026 Apple Inc.

#include <cassert>
#include <cmath>

#include "mlx/allocator.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/simd/simd.h"
#include "mlx/backend/cpu/threading/common.h"
#include "mlx/fast_primitives.h"
#include "mlx/transforms.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

namespace {

using namespace mlx::core::simd;

// RMS Norm for a single row (float32): out = x * rsqrt(mean(x^2) + eps) *
// weight
void rms_norm_row_f32(
    const float* x,
    const float* weight,
    float* out,
    int width,
    float eps,
    bool has_weight) {
  constexpr int N = max_size<float>;

  // Step 1: Compute sum of squares
  Simd<float, N> vsum_sq(0.0f);
  const float* ptr = x;
  int s = width;

  while (s >= N) {
    Simd<float, N> vals = load<float, N>(ptr);
    vsum_sq = vsum_sq + vals * vals;
    ptr += N;
    s -= N;
  }

  float sum_sq = sum(vsum_sq);
  while (s-- > 0) {
    float val = *ptr++;
    sum_sq += val * val;
  }

  // Step 2: Compute scale = rsqrt(mean + eps)
  float mean_sq = sum_sq / width;
  float scale = 1.0f / std::sqrt(mean_sq + eps);

  // Step 3: Apply scale and weight
  ptr = x;
  float* optr = out;
  s = width;

  Simd<float, N> vscale(scale);
  if (has_weight) {
    const float* wptr = weight;
    while (s >= N) {
      Simd<float, N> vals = load<float, N>(ptr);
      Simd<float, N> weights = load<float, N>(wptr);
      Simd<float, N> result = vals * vscale * weights;
      store(optr, result);
      ptr += N;
      wptr += N;
      optr += N;
      s -= N;
    }
    while (s-- > 0) {
      *optr++ = *ptr++ * scale * *wptr++;
    }
  } else {
    while (s >= N) {
      Simd<float, N> vals = load<float, N>(ptr);
      Simd<float, N> result = vals * vscale;
      store(optr, result);
      ptr += N;
      optr += N;
      s -= N;
    }
    while (s-- > 0) {
      *optr++ = *ptr++ * scale;
    }
  }
}

// RMS Norm for fp16/bf16 - load as float via SIMD (F16C for float16),
// compute in float32, convert back.
template <typename T>
void rms_norm_row_half(
    const T* x,
    const T* weight,
    T* out,
    int width,
    float eps,
    bool has_weight) {
  // Use max_size<T> so that float16 gets N=8 (F16C), bfloat16 gets N=1 (scalar)
  constexpr int N = max_size<T>;

  // Step 1: Compute sum of squares in float32
  Simd<float, N> vsum_sq(0.0f);
  const T* ptr = x;
  int s = width;

  while (s >= N) {
    Simd<float, N> vals(load<T, N>(ptr));
    vsum_sq = vsum_sq + vals * vals;
    ptr += N;
    s -= N;
  }

  float sum_sq = sum(vsum_sq);
  while (s-- > 0) {
    float val = static_cast<float>(*ptr++);
    sum_sq += val * val;
  }

  // Step 2: Compute scale = rsqrt(mean + eps)
  float mean_sq = sum_sq / width;
  float scale = 1.0f / std::sqrt(mean_sq + eps);

  // Step 3: Apply scale and weight
  ptr = x;
  T* optr = out;
  s = width;

  Simd<float, N> vscale(scale);
  if (has_weight) {
    const T* wptr = weight;
    while (s >= N) {
      Simd<float, N> vals(load<T, N>(ptr));
      Simd<float, N> weights(load<T, N>(wptr));
      Simd<float, N> result = vals * vscale * weights;
      store(optr, Simd<T, N>(result));
      ptr += N;
      wptr += N;
      optr += N;
      s -= N;
    }
    while (s-- > 0) {
      float val = static_cast<float>(*ptr++);
      float w = static_cast<float>(*wptr++);
      *optr++ = static_cast<T>(val * scale * w);
    }
  } else {
    while (s >= N) {
      Simd<float, N> vals(load<T, N>(ptr));
      Simd<float, N> result = vals * vscale;
      store(optr, Simd<T, N>(result));
      ptr += N;
      optr += N;
      s -= N;
    }
    while (s-- > 0) {
      float val = static_cast<float>(*ptr++);
      *optr++ = static_cast<T>(val * scale);
    }
  }
}

// Layer Norm for a single row (float32)
void layer_norm_row_f32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    int width,
    float eps,
    bool has_weight,
    bool has_bias) {
  constexpr int N = max_size<float>;

  // Step 1: Compute mean
  Simd<float, N> vsum(0.0f);
  const float* ptr = x;
  int s = width;

  while (s >= N) {
    Simd<float, N> vals = load<float, N>(ptr);
    vsum = vsum + vals;
    ptr += N;
    s -= N;
  }

  float total = sum(vsum);
  while (s-- > 0) {
    total += *ptr++;
  }
  float mean = total / width;

  // Step 2: Compute variance
  Simd<float, N> vvar(0.0f);
  Simd<float, N> vmean(mean);
  ptr = x;
  s = width;

  while (s >= N) {
    Simd<float, N> vals = load<float, N>(ptr);
    Simd<float, N> diff = vals - vmean;
    vvar = vvar + diff * diff;
    ptr += N;
    s -= N;
  }

  float variance = sum(vvar);
  while (s-- > 0) {
    float diff = *ptr++ - mean;
    variance += diff * diff;
  }
  variance /= width;

  // Step 3: Compute scale = 1 / sqrt(var + eps)
  float scale = 1.0f / std::sqrt(variance + eps);

  // Step 4: Apply normalization, weight, and bias
  ptr = x;
  float* optr = out;
  s = width;

  Simd<float, N> vscale(scale);
  int i = 0;
  while (s >= N) {
    Simd<float, N> vals = load<float, N>(ptr);
    Simd<float, N> result = (vals - vmean) * vscale;

    if (has_weight) {
      Simd<float, N> weights = load<float, N>(weight + i);
      result = result * weights;
    }
    if (has_bias) {
      Simd<float, N> biases = load<float, N>(bias + i);
      result = result + biases;
    }

    store(optr, result);
    ptr += N;
    optr += N;
    s -= N;
    i += N;
  }

  while (s-- > 0) {
    float val = *ptr++;
    float result = (val - mean) * scale;
    if (has_weight) {
      result *= weight[i];
    }
    if (has_bias) {
      result += bias[i];
    }
    *optr++ = result;
    i++;
  }
}

// Layer Norm for fp16/bf16 - load as float via SIMD (F16C for float16),
// compute in float32, convert back.
template <typename T>
void layer_norm_row_half(
    const T* x,
    const T* weight,
    const T* bias,
    T* out,
    int width,
    float eps,
    bool has_weight,
    bool has_bias) {
  constexpr int N = max_size<T>;

  // Step 1: Compute mean
  Simd<float, N> vsum(0.0f);
  const T* ptr = x;
  int s = width;

  while (s >= N) {
    Simd<float, N> vals(load<T, N>(ptr));
    vsum = vsum + vals;
    ptr += N;
    s -= N;
  }

  float total = sum(vsum);
  while (s-- > 0) {
    total += static_cast<float>(*ptr++);
  }
  float mean = total / width;

  // Step 2: Compute variance
  Simd<float, N> vvar(0.0f);
  Simd<float, N> vmean(mean);
  ptr = x;
  s = width;

  while (s >= N) {
    Simd<float, N> vals(load<T, N>(ptr));
    Simd<float, N> diff = vals - vmean;
    vvar = vvar + diff * diff;
    ptr += N;
    s -= N;
  }

  float variance = sum(vvar);
  while (s-- > 0) {
    float diff = static_cast<float>(*ptr++) - mean;
    variance += diff * diff;
  }
  variance /= width;

  // Step 3: Compute scale = 1 / sqrt(var + eps)
  float scale = 1.0f / std::sqrt(variance + eps);

  // Step 4: Apply normalization, weight, and bias
  ptr = x;
  T* optr = out;
  s = width;

  Simd<float, N> vscale(scale);
  int i = 0;
  while (s >= N) {
    Simd<float, N> vals(load<T, N>(ptr));
    Simd<float, N> result = (vals - vmean) * vscale;

    if (has_weight) {
      Simd<float, N> weights(load<T, N>(weight + i));
      result = result * weights;
    }
    if (has_bias) {
      Simd<float, N> biases(load<T, N>(bias + i));
      result = result + biases;
    }

    store(optr, Simd<T, N>(result));
    ptr += N;
    optr += N;
    s -= N;
    i += N;
  }

  while (s-- > 0) {
    float val = static_cast<float>(*ptr++);
    float result = (val - mean) * scale;
    if (has_weight) {
      result *= static_cast<float>(weight[i]);
    }
    if (has_bias) {
      result += static_cast<float>(bias[i]);
    }
    *optr++ = static_cast<T>(result);
    i++;
  }
}

} // namespace

void RMSNorm::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& in_x = inputs[0];
  auto& weight = inputs[1];
  auto& out = outputs[0];

  // Non-contiguous inputs are copied to contiguous layout
  array x = in_x;
  if (!in_x.flags().contiguous || in_x.strides()[in_x.ndim() - 1] != 1) {
    x = contiguous_copy_cpu(in_x, stream());
  }

  // Allocate output with same layout as (contiguous) input
  out.set_data(
      allocator::malloc(x.data_size() * x.itemsize()),
      x.data_size(),
      x.strides(),
      x.flags());

  int width = x.shape(-1);
  int num_rows = x.data_size() / width;
  bool has_weight = weight.ndim() > 0;

  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_input_array(x);
  if (has_weight) {
    encoder.set_input_array(weight);
  }
  encoder.set_output_array(out);

  // Determine threading
  auto& pool = cpu::ThreadPool::instance();
  int nth = std::min(pool.max_threads(), num_rows);
  size_t total_elements = static_cast<size_t>(num_rows) * width;
  if (total_elements < cpu::MIN_TOTAL_ELEMENTS) {
    nth = 1;
  }

  float eps = eps_;

  switch (x.dtype()) {
    case float32: {
      const float* x_ptr = x.data<float>();
      const float* w_ptr = has_weight ? weight.data<float>() : nullptr;
      float* out_ptr = out.data<float>();

      if (nth > 1) {
        encoder.dispatch([=, &pool]() {
          pool.parallel_for(nth, [=](int tid, int num_threads) {
            int rows_per_thread = (num_rows + num_threads - 1) / num_threads;
            int row_start = rows_per_thread * tid;
            int row_end = std::min(row_start + rows_per_thread, num_rows);

            for (int row = row_start; row < row_end; row++) {
              rms_norm_row_f32(
                  x_ptr + row * width,
                  w_ptr,
                  out_ptr + row * width,
                  width,
                  eps,
                  has_weight);
            }
          });
        });
      } else {
        encoder.dispatch([=]() {
          for (int row = 0; row < num_rows; row++) {
            rms_norm_row_f32(
                x_ptr + row * width,
                w_ptr,
                out_ptr + row * width,
                width,
                eps,
                has_weight);
          }
        });
      }
      break;
    }
    case float16: {
      const float16_t* x_ptr = x.data<float16_t>();
      const float16_t* w_ptr = has_weight ? weight.data<float16_t>() : nullptr;
      float16_t* out_ptr = out.data<float16_t>();

      if (nth > 1) {
        encoder.dispatch([=, &pool]() {
          pool.parallel_for(nth, [=](int tid, int num_threads) {
            int rows_per_thread = (num_rows + num_threads - 1) / num_threads;
            int row_start = rows_per_thread * tid;
            int row_end = std::min(row_start + rows_per_thread, num_rows);

            for (int row = row_start; row < row_end; row++) {
              rms_norm_row_half(
                  x_ptr + row * width,
                  w_ptr,
                  out_ptr + row * width,
                  width,
                  eps,
                  has_weight);
            }
          });
        });
      } else {
        encoder.dispatch([=]() {
          for (int row = 0; row < num_rows; row++) {
            rms_norm_row_half(
                x_ptr + row * width,
                w_ptr,
                out_ptr + row * width,
                width,
                eps,
                has_weight);
          }
        });
      }
      break;
    }
    case bfloat16: {
      const bfloat16_t* x_ptr = x.data<bfloat16_t>();
      const bfloat16_t* w_ptr =
          has_weight ? weight.data<bfloat16_t>() : nullptr;
      bfloat16_t* out_ptr = out.data<bfloat16_t>();

      if (nth > 1) {
        encoder.dispatch([=, &pool]() {
          pool.parallel_for(nth, [=](int tid, int num_threads) {
            int rows_per_thread = (num_rows + num_threads - 1) / num_threads;
            int row_start = rows_per_thread * tid;
            int row_end = std::min(row_start + rows_per_thread, num_rows);

            for (int row = row_start; row < row_end; row++) {
              rms_norm_row_half(
                  x_ptr + row * width,
                  w_ptr,
                  out_ptr + row * width,
                  width,
                  eps,
                  has_weight);
            }
          });
        });
      } else {
        encoder.dispatch([=]() {
          for (int row = 0; row < num_rows; row++) {
            rms_norm_row_half(
                x_ptr + row * width,
                w_ptr,
                out_ptr + row * width,
                width,
                eps,
                has_weight);
          }
        });
      }
      break;
    }
    default:
      throw std::runtime_error("[RMSNorm::eval_cpu] Unsupported type");
  }
}

void RMSNormVJP::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // Use fallback for VJP - it's less performance critical
  // The fallback returns unevaluated arrays, need to eval them
  auto results = fallback_(inputs);
  eval(results);
  for (size_t i = 0; i < outputs.size(); i++) {
    outputs[i].copy_shared_buffer(results[i]);
  }
}

void LayerNorm::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& in_x = inputs[0];
  auto& weight = inputs[1];
  auto& bias = inputs[2];
  auto& out = outputs[0];

  // Non-contiguous inputs are copied to contiguous layout
  array x = in_x;
  if (!in_x.flags().contiguous || in_x.strides()[in_x.ndim() - 1] != 1) {
    x = contiguous_copy_cpu(in_x, stream());
  }

  // Allocate output with same layout as (contiguous) input
  out.set_data(
      allocator::malloc(x.data_size() * x.itemsize()),
      x.data_size(),
      x.strides(),
      x.flags());

  int width = x.shape(-1);
  int num_rows = x.data_size() / width;
  bool has_weight = weight.ndim() > 0;
  bool has_bias = bias.ndim() > 0;

  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_input_array(x);
  if (has_weight) {
    encoder.set_input_array(weight);
  }
  if (has_bias) {
    encoder.set_input_array(bias);
  }
  encoder.set_output_array(out);

  // Determine threading
  auto& pool = cpu::ThreadPool::instance();
  int nth = std::min(pool.max_threads(), num_rows);
  size_t total_elements = static_cast<size_t>(num_rows) * width;
  if (total_elements < cpu::MIN_TOTAL_ELEMENTS) {
    nth = 1;
  }

  float eps = eps_;

  switch (x.dtype()) {
    case float32: {
      const float* x_ptr = x.data<float>();
      const float* w_ptr = has_weight ? weight.data<float>() : nullptr;
      const float* b_ptr = has_bias ? bias.data<float>() : nullptr;
      float* out_ptr = out.data<float>();

      if (nth > 1) {
        encoder.dispatch([=, &pool]() {
          pool.parallel_for(nth, [=](int tid, int num_threads) {
            int rows_per_thread = (num_rows + num_threads - 1) / num_threads;
            int row_start = rows_per_thread * tid;
            int row_end = std::min(row_start + rows_per_thread, num_rows);

            for (int row = row_start; row < row_end; row++) {
              layer_norm_row_f32(
                  x_ptr + row * width,
                  w_ptr,
                  b_ptr,
                  out_ptr + row * width,
                  width,
                  eps,
                  has_weight,
                  has_bias);
            }
          });
        });
      } else {
        encoder.dispatch([=]() {
          for (int row = 0; row < num_rows; row++) {
            layer_norm_row_f32(
                x_ptr + row * width,
                w_ptr,
                b_ptr,
                out_ptr + row * width,
                width,
                eps,
                has_weight,
                has_bias);
          }
        });
      }
      break;
    }
    case float16: {
      const float16_t* x_ptr = x.data<float16_t>();
      const float16_t* w_ptr = has_weight ? weight.data<float16_t>() : nullptr;
      const float16_t* b_ptr = has_bias ? bias.data<float16_t>() : nullptr;
      float16_t* out_ptr = out.data<float16_t>();

      if (nth > 1) {
        encoder.dispatch([=, &pool]() {
          pool.parallel_for(nth, [=](int tid, int num_threads) {
            int rows_per_thread = (num_rows + num_threads - 1) / num_threads;
            int row_start = rows_per_thread * tid;
            int row_end = std::min(row_start + rows_per_thread, num_rows);

            for (int row = row_start; row < row_end; row++) {
              layer_norm_row_half(
                  x_ptr + row * width,
                  w_ptr,
                  b_ptr,
                  out_ptr + row * width,
                  width,
                  eps,
                  has_weight,
                  has_bias);
            }
          });
        });
      } else {
        encoder.dispatch([=]() {
          for (int row = 0; row < num_rows; row++) {
            layer_norm_row_half(
                x_ptr + row * width,
                w_ptr,
                b_ptr,
                out_ptr + row * width,
                width,
                eps,
                has_weight,
                has_bias);
          }
        });
      }
      break;
    }
    case bfloat16: {
      const bfloat16_t* x_ptr = x.data<bfloat16_t>();
      const bfloat16_t* w_ptr =
          has_weight ? weight.data<bfloat16_t>() : nullptr;
      const bfloat16_t* b_ptr = has_bias ? bias.data<bfloat16_t>() : nullptr;
      bfloat16_t* out_ptr = out.data<bfloat16_t>();

      if (nth > 1) {
        encoder.dispatch([=, &pool]() {
          pool.parallel_for(nth, [=](int tid, int num_threads) {
            int rows_per_thread = (num_rows + num_threads - 1) / num_threads;
            int row_start = rows_per_thread * tid;
            int row_end = std::min(row_start + rows_per_thread, num_rows);

            for (int row = row_start; row < row_end; row++) {
              layer_norm_row_half(
                  x_ptr + row * width,
                  w_ptr,
                  b_ptr,
                  out_ptr + row * width,
                  width,
                  eps,
                  has_weight,
                  has_bias);
            }
          });
        });
      } else {
        encoder.dispatch([=]() {
          for (int row = 0; row < num_rows; row++) {
            layer_norm_row_half(
                x_ptr + row * width,
                w_ptr,
                b_ptr,
                out_ptr + row * width,
                width,
                eps,
                has_weight,
                has_bias);
          }
        });
      }
      break;
    }
    default:
      throw std::runtime_error("[LayerNorm::eval_cpu] Unsupported type");
  }
}

void LayerNormVJP::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // Use fallback for VJP - it's less performance critical
  // The fallback returns unevaluated arrays, need to eval them
  auto results = fallback_(inputs);
  eval(results);
  for (size_t i = 0; i < outputs.size(); i++) {
    outputs[i].copy_shared_buffer(results[i]);
  }
}

} // namespace mlx::core::fast
