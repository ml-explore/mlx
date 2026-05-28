// Copyright © 2026 Apple Inc.
//
// Highway implementations for quantized CPU helpers.

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include "mlx/backend/cpu/quantized.h"
#include "mlx/backend/cpu/threading/common.h"
#include "mlx/types/half_types.h"
#include "mlx/utils.h"

namespace mlx::core {

constexpr bool has_simd_qmm = true;

enum class QuantizedHighwayDType : uint8_t {
  Float32,
  Float16,
  BFloat16,
};

template <typename T>
constexpr QuantizedHighwayDType quantized_highway_dtype() {
  if constexpr (std::is_same_v<T, float>) {
    return QuantizedHighwayDType::Float32;
  } else if constexpr (std::is_same_v<T, float16_t>) {
    return QuantizedHighwayDType::Float16;
  } else if constexpr (std::is_same_v<T, bfloat16_t>) {
    return QuantizedHighwayDType::BFloat16;
  } else {
    static_assert(
        std::is_same_v<T, float> || std::is_same_v<T, float16_t> ||
            std::is_same_v<T, bfloat16_t>,
        "Unsupported Highway quantized dtype");
  }
}

template <>
struct LoadAsFloat<float16_t, 8> {
  static inline simd::Simd<float, 8> apply(const float16_t* ptr) {
    return simd::Simd<float, 8>(simd::load<float16_t, 8>(ptr));
  }
};

template <>
struct LoadAsFloat<bfloat16_t, 8> {
  static inline simd::Simd<float, 8> apply(const bfloat16_t* ptr) {
    return simd::Simd<float, 8>(simd::load<bfloat16_t, 8>(ptr));
  }
};

void dequant_row_highway_4bit(
    const uint32_t* w_row,
    const float* scales_row,
    const float* biases_row,
    float* out,
    int group_size,
    int K);

void dequant_row_highway_8bit(
    const uint32_t* w_row,
    const float* scales_row,
    const float* biases_row,
    float* out,
    int group_size,
    int K);

void quantize_activation_int8_highway(
    const void* x,
    QuantizedHighwayDType dtype,
    int K,
    int group_size,
    int8_t* x_q,
    float* x_scales,
    float* x_group_sums);

void qmm_t_int8_highway_row(
    float* result,
    const int8_t* x_q,
    const float* x_scales,
    const float* x_group_sums,
    const uint32_t* w,
    const void* scales,
    const void* biases,
    QuantizedHighwayDType dtype,
    int bits,
    int group_size,
    int n_start,
    int n_end,
    int K);

void fp_qmm_t_highway_row(
    void* result,
    const void* x,
    const uint32_t* w,
    const uint8_t* scales,
    QuantizedHighwayDType dtype,
    int bits,
    int group_size,
    int n_start,
    int n_end,
    int K,
    float scale_factor,
    const float* fp4_lut,
    const float* fp8_lut);

template <int bits, int group_size>
void _dequant_row_highway(
    const uint32_t* w_row,
    const float* scales_row,
    const float* biases_row,
    float* out,
    int K) {
  if constexpr (bits == 4) {
    dequant_row_highway_4bit(w_row, scales_row, biases_row, out, group_size, K);
  } else if constexpr (bits == 8) {
    dequant_row_highway_8bit(w_row, scales_row, biases_row, out, group_size, K);
  }
}

template <typename T, int bits, int group_size, int NC>
int try_batch_extract_multi_col(
    const T*&,
    const uint32_t* [NC],
    simd::Simd<float, simd::max_size<float>>[NC],
    int) {
  return 0;
}

template <typename T, int bits, int group_size>
bool try_int8_simd_row(
    T* result,
    const T* x,
    const uint32_t* w,
    const T* scales,
    const T* biases,
    int n_start,
    int n_end,
    int K,
    int groups_per_col,
    int,
    float* x_group_sums,
    const PreqAct* preq) {
  constexpr int INT8_MAX_K = 16384;
  constexpr int STACK_GROUPS = 128;

  if constexpr (!((bits == 4 || bits == 8) && group_size >= 32)) {
    return false;
  } else {
    if (!env::enable_tf32()) {
      return false;
    }
    if (K > INT8_MAX_K) {
      return false;
    }

    const int n_cols = n_end - n_start;
    if (!preq && n_cols < K) {
      return false;
    }

    alignas(64) int8_t x_q_local[INT8_MAX_K];
    float x_scales_stack[STACK_GROUPS];
    std::unique_ptr<float[]> x_scales_heap;

    const int8_t* x_q = nullptr;
    const float* x_scales = nullptr;
    if (preq) {
      x_q = static_cast<const int8_t*>(preq->x_q);
      x_scales = preq->x_scales;
    } else {
      float* x_scales_w = x_scales_stack;
      if (groups_per_col > STACK_GROUPS) {
        x_scales_heap.reset(new float[groups_per_col]);
        x_scales_w = x_scales_heap.get();
      }
      quantize_activation_int8_highway(
          x,
          quantized_highway_dtype<T>(),
          K,
          group_size,
          x_q_local,
          x_scales_w,
          x_group_sums);
      x_q = x_q_local;
      x_scales = x_scales_w;
    }

    constexpr int STACK_COLS = 128;
    float result_stack[STACK_COLS];
    std::unique_ptr<float[]> result_heap;
    float* result_f32 = result_stack;
    if (n_cols > STACK_COLS) {
      result_heap.reset(new float[n_cols]);
      result_f32 = result_heap.get();
    }

    qmm_t_int8_highway_row(
        result_f32,
        x_q,
        x_scales,
        x_group_sums,
        w,
        scales,
        biases,
        quantized_highway_dtype<T>(),
        bits,
        group_size,
        n_start,
        n_end,
        K);

    for (int n = 0; n < n_cols; ++n) {
      result[n] = static_cast<T>(result_f32[n]);
    }
    return true;
  }
}

template <typename T, int bits, int group_size>
bool try_int8_preq_parallel(
    T* result,
    const T* x,
    const uint32_t* w,
    const T* scales,
    const T* biases,
    int N,
    int K,
    int n_threads,
    int n_chunks,
    std::atomic<int>& steal_counter,
    int CHUNK_COLS) {
  constexpr int INT8_MAX_K = 16384;
  constexpr int STACK_GROUPS = 128;

  if constexpr (!((bits == 4 || bits == 8) && group_size >= 32)) {
    return false;
  } else {
    if (!env::enable_tf32()) {
      return false;
    }
    if (K > INT8_MAX_K) {
      return false;
    }

    const int groups_per_col = K / group_size;
    alignas(64) int8_t x_q[INT8_MAX_K];
    float x_scales_stack[STACK_GROUPS];
    float x_group_sums_stack[STACK_GROUPS];
    std::unique_ptr<float[]> x_scales_heap;
    std::unique_ptr<float[]> x_group_sums_heap;
    float* x_scales = x_scales_stack;
    float* x_group_sums = x_group_sums_stack;
    if (groups_per_col > STACK_GROUPS) {
      x_scales_heap.reset(new float[groups_per_col]);
      x_group_sums_heap.reset(new float[groups_per_col]);
      x_scales = x_scales_heap.get();
      x_group_sums = x_group_sums_heap.get();
    }

    quantize_activation_int8_highway(
        x,
        quantized_highway_dtype<T>(),
        K,
        group_size,
        x_q,
        x_scales,
        x_group_sums);

    auto& pool = cpu::ThreadPool::instance();
    PreqAct preq{x_q, x_scales, x_group_sums};
    steal_counter.store(n_threads, std::memory_order_relaxed);
    pool.parallel_for(n_threads, [&](int tid, int /*nth*/) {
      int my_chunk = tid;
      while (my_chunk < n_chunks) {
        int n_start = std::min(my_chunk * CHUNK_COLS, N);
        int n_end = std::min(n_start + CHUNK_COLS, N);
        if (n_start < n_end) {
          _qmm_t_simd_row<T, bits, group_size>(
              result + n_start, x, w, scales, biases, n_start, n_end, K, &preq);
        }
        my_chunk = steal_counter.fetch_add(1, std::memory_order_relaxed);
      }
    });
    return true;
  }
}

} // namespace mlx::core
