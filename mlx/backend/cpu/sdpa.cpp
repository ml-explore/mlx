// Copyright © 2026 Apple Inc.

#include <cassert>
#include <cmath>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/simd/simd.h"
#include "mlx/backend/cpu/threading/common.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

namespace {

using namespace mlx::core::simd;

// SDPA kernel for a single query position (float32), prompt path.
// Uses online softmax with scalar exp: minimal register pressure, best for
// prompt where SDPA is a small fraction of total time.
void sdpa_single_head_f32_prompt(
    float* output,
    const float* query,
    const float* keys,
    const float* values,
    int seq_len,
    int head_dim,
    float scale,
    int effective_len,
    const float* mask,
    float sink_value,
    bool has_sinks) {
  constexpr int N = max_size<float>;
  constexpr int MAX_ACC = 256 / N;
  int n_full_acc = std::min(head_dim / N, (int)MAX_ACC);
  int d_simd = n_full_acc * N;
  int tail_len = head_dim - d_simd;
  const float neg_inf = -std::numeric_limits<float>::infinity();

  Simd<float, N> O_acc[MAX_ACC];
  for (int i = 0; i < n_full_acc; i++) {
    O_acc[i] = Simd<float, N>(0.0f);
  }
  float O_tail[N] = {};
  float running_max = neg_inf;
  float running_sum = 0.0f;

  for (int s = 0; s < effective_len; s++) {
    // Q * K[s] dot product
    const float* k_row = keys + s * head_dim;
    Simd<float, N> dot_acc(0.0f);
    int d = 0;
    for (; d + N <= head_dim; d += N) {
      dot_acc =
          fma(load<float, N>(query + d), load<float, N>(k_row + d), dot_acc);
    }
    float dot = sum(dot_acc);
    for (; d < head_dim; d++) {
      dot += query[d] * k_row[d];
    }
    dot *= scale;
    if (mask) {
      dot += mask[s];
    }
    if (dot == neg_inf) {
      continue;
    }

    // Online softmax update
    float new_max = std::max(running_max, dot);
    if (new_max > running_max) {
      float correction = std::exp(running_max - new_max);
      running_sum *= correction;
      Simd<float, N> vcorr(correction);
      for (int i = 0; i < n_full_acc; i++) {
        O_acc[i] = O_acc[i] * vcorr;
      }
      for (int j = 0; j < tail_len; j++) {
        O_tail[j] *= correction;
      }
      running_max = new_max;
    }

    float exp_dot = std::exp(dot - running_max);
    running_sum += exp_dot;

    // Accumulate weighted V
    const float* v_row = values + s * head_dim;
    Simd<float, N> vexp(exp_dot);
    for (int i = 0; i < n_full_acc; i++) {
      O_acc[i] = fma(vexp, load<float, N>(v_row + i * N), O_acc[i]);
    }
    for (int j = 0; j < tail_len; j++) {
      O_tail[j] += exp_dot * v_row[d_simd + j];
    }
  }

  // Attention sinks
  if (has_sinks) {
    float new_max = std::max(running_max, sink_value);
    if (new_max > neg_inf) {
      float correction = std::exp(running_max - new_max);
      running_sum *= correction;
      Simd<float, N> vcorr(correction);
      for (int i = 0; i < n_full_acc; i++) {
        O_acc[i] = O_acc[i] * vcorr;
      }
      for (int j = 0; j < tail_len; j++) {
        O_tail[j] *= correction;
      }
      running_sum += std::exp(sink_value - new_max);
    }
  }

  // Normalize and store
  if (running_sum > 0.0f) {
    float inv_sum = 1.0f / running_sum;
    Simd<float, N> vinv(inv_sum);
    for (int i = 0; i < n_full_acc; i++) {
      store(output + i * N, O_acc[i] * vinv);
    }
    for (int j = 0; j < tail_len; j++) {
      output[d_simd + j] = O_tail[j] * inv_sum;
    }
  } else {
    std::fill(output, output + head_dim, 0.0f);
  }
}

// SDPA kernel for a single query position (float32), generation path.
// Uses online softmax with SIMD-batched exp: processes N KV positions at a
// time, amortizing exp() cost. Best for generation where SDPA is ~24% of
// per-token time and KV caches are long.
void sdpa_single_head_f32_gen(
    float* output,
    const float* query,
    const float* keys,
    const float* values,
    int seq_len,
    int head_dim,
    float scale,
    int effective_len,
    const float* mask,
    float sink_value,
    bool has_sinks) {
  constexpr int N = max_size<float>;
  constexpr int MAX_ACC = 256 / N;
  int n_full_acc = std::min(head_dim / N, (int)MAX_ACC);
  int d_simd = n_full_acc * N;
  int tail_len = head_dim - d_simd;
  const float neg_inf = -std::numeric_limits<float>::infinity();

  Simd<float, N> O_acc[MAX_ACC];
  for (int i = 0; i < n_full_acc; i++) {
    O_acc[i] = Simd<float, N>(0.0f);
  }
  float O_tail[N] = {};
  float running_max = neg_inf;
  float running_sum = 0.0f;

  // Main loop: N positions per iteration for SIMD exp
  int s = 0;
  for (; s + N <= effective_len; s += N) {
    // Compute N dot products
    alignas(32) float dots[N];
    for (int b = 0; b < N; b++) {
      const float* k_row = keys + (s + b) * head_dim;
      Simd<float, N> dot_acc(0.0f);
      int d = 0;
      for (; d + N <= head_dim; d += N) {
        dot_acc =
            fma(load<float, N>(query + d), load<float, N>(k_row + d), dot_acc);
      }
      float dot = sum(dot_acc);
      for (; d < head_dim; d++) {
        dot += query[d] * k_row[d];
      }
      dot *= scale;
      if (mask) {
        dot += mask[s + b];
      }
      dots[b] = dot;
    }

    Simd<float, N> vdots = load<float, N>(dots);
    float block_max = max(vdots);
    if (block_max == neg_inf) {
      continue;
    }
    if (block_max > running_max) {
      float correction = std::exp(running_max - block_max);
      running_sum *= correction;
      Simd<float, N> vcorr(correction);
      for (int i = 0; i < n_full_acc; i++) {
        O_acc[i] = O_acc[i] * vcorr;
      }
      for (int j = 0; j < tail_len; j++) {
        O_tail[j] *= correction;
      }
      running_max = block_max;
    }

    // SIMD exp of N scores at once
    Simd<float, N> vexp = exp(vdots - Simd<float, N>(running_max));
    running_sum += sum(vexp);

    alignas(32) float exp_arr[N];
    store(exp_arr, vexp);
    for (int b = 0; b < N; b++) {
      if (exp_arr[b] == 0.0f) {
        continue;
      }
      const float* v_row = values + (s + b) * head_dim;
      Simd<float, N> vw(exp_arr[b]);
      for (int i = 0; i < n_full_acc; i++) {
        O_acc[i] = fma(vw, load<float, N>(v_row + i * N), O_acc[i]);
      }
      for (int j = 0; j < tail_len; j++) {
        O_tail[j] += exp_arr[b] * v_row[d_simd + j];
      }
    }
  }

  // Scalar tail for remaining < N positions
  for (; s < effective_len; s++) {
    const float* k_row = keys + s * head_dim;
    Simd<float, N> dot_acc(0.0f);
    int d = 0;
    for (; d + N <= head_dim; d += N) {
      dot_acc =
          fma(load<float, N>(query + d), load<float, N>(k_row + d), dot_acc);
    }
    float dot = sum(dot_acc);
    for (; d < head_dim; d++) {
      dot += query[d] * k_row[d];
    }
    dot *= scale;
    if (mask) {
      dot += mask[s];
    }
    if (dot == neg_inf) {
      continue;
    }
    float new_max = std::max(running_max, dot);
    if (new_max > running_max) {
      float correction = std::exp(running_max - new_max);
      running_sum *= correction;
      Simd<float, N> vcorr(correction);
      for (int i = 0; i < n_full_acc; i++) {
        O_acc[i] = O_acc[i] * vcorr;
      }
      for (int j = 0; j < tail_len; j++) {
        O_tail[j] *= correction;
      }
      running_max = new_max;
    }
    float exp_dot = std::exp(dot - running_max);
    running_sum += exp_dot;
    const float* v_row = values + s * head_dim;
    Simd<float, N> vw(exp_dot);
    for (int i = 0; i < n_full_acc; i++) {
      O_acc[i] = fma(vw, load<float, N>(v_row + i * N), O_acc[i]);
    }
    for (int j = 0; j < tail_len; j++) {
      O_tail[j] += exp_dot * v_row[d_simd + j];
    }
  }

  // Attention sinks
  if (has_sinks) {
    float new_max = std::max(running_max, sink_value);
    if (new_max > neg_inf) {
      float correction = std::exp(running_max - new_max);
      running_sum *= correction;
      Simd<float, N> vcorr(correction);
      for (int i = 0; i < n_full_acc; i++) {
        O_acc[i] = O_acc[i] * vcorr;
      }
      for (int j = 0; j < tail_len; j++) {
        O_tail[j] *= correction;
      }
      running_sum += std::exp(sink_value - new_max);
    }
  }

  // Normalize and store
  if (running_sum > 0.0f) {
    float inv_sum = 1.0f / running_sum;
    Simd<float, N> vinv(inv_sum);
    for (int i = 0; i < n_full_acc; i++) {
      store(output + i * N, O_acc[i] * vinv);
    }
    for (int j = 0; j < tail_len; j++) {
      output[d_simd + j] = O_tail[j] * inv_sum;
    }
  } else {
    std::fill(output, output + head_dim, 0.0f);
  }
}

// Load N elements of half type T as Simd<float, N>.
// Uses F16C SIMD conversion when max_size<T> >= N, scalar fallback otherwise.
template <typename T, int N>
inline Simd<float, N> load_half_as_float(const T* ptr) {
  constexpr int NT = max_size<T>;
  if constexpr (NT >= N) {
    return Simd<float, N>(load<T, NT>(ptr));
  } else {
    alignas(32) float tmp[N];
    for (int j = 0; j < N; j++) {
      tmp[j] = static_cast<float>(ptr[j]);
    }
    return load<float, N>(tmp);
  }
}

// Store N float elements as half type T.
// Uses F16C SIMD conversion when max_size<T> >= N, scalar fallback otherwise.
template <typename T, int N>
inline void store_float_as_half(T* ptr, Simd<float, N> v) {
  constexpr int NT = max_size<T>;
  if constexpr (NT >= N) {
    store(ptr, Simd<T, NT>(v));
  } else {
    alignas(32) float tmp[N];
    store(tmp, v);
    for (int j = 0; j < N; j++) {
      ptr[j] = static_cast<T>(tmp[j]);
    }
  }
}

// SDPA kernel for half types (float16_t / bfloat16_t), prompt path.
// Online softmax with scalar exp. Converts to float32 via SIMD for compute.
template <typename T>
void sdpa_single_head_half_prompt(
    T* output,
    const T* query,
    const T* keys,
    const T* values,
    int seq_len,
    int head_dim,
    float scale,
    int effective_len,
    const T* mask,
    float sink_value,
    bool has_sinks) {
  constexpr int N = max_size<float>;
  constexpr int MAX_ACC = 256 / N;
  int n_acc = std::min((head_dim + N - 1) / N, (int)MAX_ACC);
  const float neg_inf = -std::numeric_limits<float>::infinity();

  alignas(32) float q_f[256];
  {
    int d = 0;
    for (; d + N <= head_dim; d += N) {
      store(&q_f[d], load_half_as_float<T, N>(query + d));
    }
    for (; d < head_dim; d++) {
      q_f[d] = static_cast<float>(query[d]);
    }
  }

  Simd<float, N> O_acc[MAX_ACC];
  for (int i = 0; i < n_acc; i++) {
    O_acc[i] = Simd<float, N>(0.0f);
  }
  float running_max = neg_inf;
  float running_sum = 0.0f;

  for (int s = 0; s < effective_len; s++) {
    const T* k_row = keys + s * head_dim;
    Simd<float, N> dot_acc(0.0f);
    int d = 0;
    for (; d + N <= head_dim; d += N) {
      dot_acc =
          fma(load<float, N>(&q_f[d]),
              load_half_as_float<T, N>(k_row + d),
              dot_acc);
    }
    float dot = sum(dot_acc);
    for (; d < head_dim; d++) {
      dot += q_f[d] * static_cast<float>(k_row[d]);
    }
    dot *= scale;
    if (mask) {
      dot += static_cast<float>(mask[s]);
    }
    if (dot == neg_inf) {
      continue;
    }

    float new_max = std::max(running_max, dot);
    if (new_max > running_max) {
      float correction = std::exp(running_max - new_max);
      running_sum *= correction;
      Simd<float, N> vcorr(correction);
      for (int i = 0; i < n_acc; i++) {
        O_acc[i] = O_acc[i] * vcorr;
      }
      running_max = new_max;
    }

    float exp_dot = std::exp(dot - running_max);
    running_sum += exp_dot;

    const T* v_row = values + s * head_dim;
    Simd<float, N> vw(exp_dot);
    for (int i = 0; i < n_acc && i * N < head_dim; i++) {
      if (i * N + N <= head_dim) {
        O_acc[i] = fma(vw, load_half_as_float<T, N>(v_row + i * N), O_acc[i]);
      } else {
        alignas(32) float v_tmp[N] = {};
        int remaining = head_dim - i * N;
        for (int j = 0; j < remaining; j++) {
          v_tmp[j] = static_cast<float>(v_row[i * N + j]);
        }
        O_acc[i] = fma(vw, load<float, N>(v_tmp), O_acc[i]);
      }
    }
  }

  if (has_sinks) {
    float new_max = std::max(running_max, sink_value);
    if (new_max > neg_inf) {
      float correction = std::exp(running_max - new_max);
      running_sum *= correction;
      Simd<float, N> vcorr(correction);
      for (int i = 0; i < n_acc; i++) {
        O_acc[i] = O_acc[i] * vcorr;
      }
      running_sum += std::exp(sink_value - new_max);
    }
  }

  if (running_sum > 0.0f) {
    float inv_sum = 1.0f / running_sum;
    Simd<float, N> vinv(inv_sum);
    for (int i = 0; i < n_acc && i * N < head_dim; i++) {
      if (i * N + N <= head_dim) {
        store_float_as_half<T, N>(output + i * N, O_acc[i] * vinv);
      } else {
        alignas(32) float tmp[N];
        store(tmp, O_acc[i] * vinv);
        int remaining = head_dim - i * N;
        for (int j = 0; j < remaining; j++) {
          output[i * N + j] = static_cast<T>(tmp[j]);
        }
      }
    }
  } else {
    for (int d = 0; d < head_dim; d++) {
      output[d] = static_cast<T>(0.0f);
    }
  }
}

// SDPA kernel for half types, generation path.
// Online softmax with SIMD-batched exp over N KV positions at a time.
template <typename T>
void sdpa_single_head_half_gen(
    T* output,
    const T* query,
    const T* keys,
    const T* values,
    int seq_len,
    int head_dim,
    float scale,
    int effective_len,
    const T* mask,
    float sink_value,
    bool has_sinks) {
  constexpr int N = max_size<float>;
  constexpr int MAX_ACC = 256 / N;
  int n_acc = std::min((head_dim + N - 1) / N, (int)MAX_ACC);
  const float neg_inf = -std::numeric_limits<float>::infinity();

  alignas(32) float q_f[256];
  {
    int d = 0;
    for (; d + N <= head_dim; d += N) {
      store(&q_f[d], load_half_as_float<T, N>(query + d));
    }
    for (; d < head_dim; d++) {
      q_f[d] = static_cast<float>(query[d]);
    }
  }

  Simd<float, N> O_acc[MAX_ACC];
  for (int i = 0; i < n_acc; i++) {
    O_acc[i] = Simd<float, N>(0.0f);
  }
  float running_max = neg_inf;
  float running_sum = 0.0f;

  // Helper: accumulate weighted V (kept as lambda -- only ~32 calls/SDPA for
  // generation, so call overhead is negligible)
  auto accum_v = [&](const T* v_row, float w) {
    Simd<float, N> vw(w);
    for (int i = 0; i < n_acc && i * N < head_dim; i++) {
      if (i * N + N <= head_dim) {
        O_acc[i] = fma(vw, load_half_as_float<T, N>(v_row + i * N), O_acc[i]);
      } else {
        alignas(32) float v_tmp[N] = {};
        int remaining = head_dim - i * N;
        for (int j = 0; j < remaining; j++) {
          v_tmp[j] = static_cast<float>(v_row[i * N + j]);
        }
        O_acc[i] = fma(vw, load<float, N>(v_tmp), O_acc[i]);
      }
    }
  };

  // Main loop: N positions per iteration
  int s = 0;
  for (; s + N <= effective_len; s += N) {
    alignas(32) float dots[N];
    for (int b = 0; b < N; b++) {
      const T* k_row = keys + (s + b) * head_dim;
      Simd<float, N> dot_acc(0.0f);
      int d = 0;
      for (; d + N <= head_dim; d += N) {
        dot_acc =
            fma(load<float, N>(&q_f[d]),
                load_half_as_float<T, N>(k_row + d),
                dot_acc);
      }
      float dot = sum(dot_acc);
      for (; d < head_dim; d++) {
        dot += q_f[d] * static_cast<float>(k_row[d]);
      }
      dot *= scale;
      if (mask) {
        dot += static_cast<float>(mask[s + b]);
      }
      dots[b] = dot;
    }

    Simd<float, N> vdots = load<float, N>(dots);
    float block_max = max(vdots);
    if (block_max == neg_inf) {
      continue;
    }
    if (block_max > running_max) {
      float correction = std::exp(running_max - block_max);
      running_sum *= correction;
      Simd<float, N> vcorr(correction);
      for (int i = 0; i < n_acc; i++) {
        O_acc[i] = O_acc[i] * vcorr;
      }
      running_max = block_max;
    }

    Simd<float, N> vexp = exp(vdots - Simd<float, N>(running_max));
    running_sum += sum(vexp);

    alignas(32) float exp_arr[N];
    store(exp_arr, vexp);
    for (int b = 0; b < N; b++) {
      if (exp_arr[b] == 0.0f) {
        continue;
      }
      accum_v(values + (s + b) * head_dim, exp_arr[b]);
    }
  }

  // Scalar tail
  for (; s < effective_len; s++) {
    const T* k_row = keys + s * head_dim;
    Simd<float, N> dot_acc(0.0f);
    int d = 0;
    for (; d + N <= head_dim; d += N) {
      dot_acc =
          fma(load<float, N>(&q_f[d]),
              load_half_as_float<T, N>(k_row + d),
              dot_acc);
    }
    float dot = sum(dot_acc);
    for (; d < head_dim; d++) {
      dot += q_f[d] * static_cast<float>(k_row[d]);
    }
    dot *= scale;
    if (mask) {
      dot += static_cast<float>(mask[s]);
    }
    if (dot == neg_inf) {
      continue;
    }
    float new_max = std::max(running_max, dot);
    if (new_max > running_max) {
      float correction = std::exp(running_max - new_max);
      running_sum *= correction;
      Simd<float, N> vcorr(correction);
      for (int i = 0; i < n_acc; i++) {
        O_acc[i] = O_acc[i] * vcorr;
      }
      running_max = new_max;
    }
    float exp_dot = std::exp(dot - running_max);
    running_sum += exp_dot;
    accum_v(values + s * head_dim, exp_dot);
  }

  if (has_sinks) {
    float new_max = std::max(running_max, sink_value);
    if (new_max > neg_inf) {
      float correction = std::exp(running_max - new_max);
      running_sum *= correction;
      Simd<float, N> vcorr(correction);
      for (int i = 0; i < n_acc; i++) {
        O_acc[i] = O_acc[i] * vcorr;
      }
      running_sum += std::exp(sink_value - new_max);
    }
  }

  if (running_sum > 0.0f) {
    float inv_sum = 1.0f / running_sum;
    Simd<float, N> vinv(inv_sum);
    for (int i = 0; i < n_acc && i * N < head_dim; i++) {
      if (i * N + N <= head_dim) {
        store_float_as_half<T, N>(output + i * N, O_acc[i] * vinv);
      } else {
        alignas(32) float tmp[N];
        store(tmp, O_acc[i] * vinv);
        int remaining = head_dim - i * N;
        for (int j = 0; j < remaining; j++) {
          output[i * N + j] = static_cast<T>(tmp[j]);
        }
      }
    }
  } else {
    for (int d = 0; d < head_dim; d++) {
      output[d] = static_cast<T>(0.0f);
    }
  }
}

// Kernel dispatch function pointer types
using f32_kernel_fn = void (*)(
    float*,
    const float*,
    const float*,
    const float*,
    int,
    int,
    float,
    int,
    const float*,
    float,
    bool);

template <typename T>
using half_kernel_fn = void (*)(
    T*,
    const T*,
    const T*,
    const T*,
    int,
    int,
    float,
    int,
    const T*,
    float,
    bool);

template <typename T>
void sdpa_impl(
    T* output,
    const T* queries,
    const T* keys,
    const T* values,
    int B,
    int n_q_heads,
    int n_kv_heads,
    int M,
    int seq_len,
    int head_dim,
    float scale,
    bool do_causal,
    const T* mask,
    bool has_mask,
    const T* sinks,
    bool has_sinks) {
  int n_repeats = n_q_heads / n_kv_heads;
  int total_work = B * n_q_heads * M;

  auto& pool = cpu::ThreadPool::instance();
  int nth = std::min(pool.max_threads(), total_work);
  if (total_work < 4) {
    nth = 1;
  }

  // Select kernel: SIMD-batched exp for generation (M=1), scalar for prompt.
  // Resolved once here via function pointer -- no branch in hot loop.
  auto get_kernel = [&]() {
    if constexpr (std::is_same_v<T, float>) {
      return (M == 1) ? &sdpa_single_head_f32_gen
                      : &sdpa_single_head_f32_prompt;
    } else {
      return (M == 1) ? &sdpa_single_head_half_gen<T>
                      : &sdpa_single_head_half_prompt<T>;
    }
  };
  auto kernel = get_kernel();

  auto work = [&](int tid, int num_threads) {
    int work_per = (total_work + num_threads - 1) / num_threads;
    int start = work_per * tid;
    int end = std::min(start + work_per, total_work);

    for (int idx = start; idx < end; idx++) {
      int m = idx % M;
      int rem = idx / M;
      int h = rem % n_q_heads;
      int b = rem / n_q_heads;
      int kv_h = h / n_repeats;

      const T* q = queries + ((b * n_q_heads + h) * M + m) * head_dim;
      const T* k = keys + (b * n_kv_heads + kv_h) * seq_len * head_dim;
      const T* v = values + (b * n_kv_heads + kv_h) * seq_len * head_dim;
      T* out = output + ((b * n_q_heads + h) * M + m) * head_dim;

      int effective_len =
          do_causal ? std::min(seq_len, seq_len - M + m + 1) : seq_len;

      const T* m_ptr =
          has_mask ? mask + ((b * n_q_heads + h) * M + m) * seq_len : nullptr;

      float sink_val = has_sinks ? static_cast<float>(sinks[h]) : 0.0f;

      kernel(
          out,
          q,
          k,
          v,
          seq_len,
          head_dim,
          scale,
          effective_len,
          m_ptr,
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

} // namespace

void ScaledDotProductAttention::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (output_logsumexp_) {
    throw std::runtime_error(
        "[ScaledDotProductAttention::eval_cpu] logsumexp output not supported "
        "on CPU. This should have been caught by use_fallback.");
  }

  auto& q_in = inputs[0]; // (B, n_q_heads, M, head_dim)
  auto& k_in = inputs[1]; // (B, n_kv_heads, S, head_dim)
  auto& v_in = inputs[2]; // (B, n_kv_heads, S, head_dim)
  auto& out = outputs[0]; // (B, n_q_heads, M, head_dim)

  // Determine which optional inputs are present.
  // Input layout: {q, k, v, [mask], [sinks]}
  // - No mask, no sinks: size=3
  // - Mask only: size=4, mask=inputs[3]
  // - Sinks only: size=4, sinks=inputs[3] (has_sinks_ is true)
  // - Both: size=5, mask=inputs[3], sinks=inputs[4]
  bool has_arr_mask =
      (!has_sinks_ && inputs.size() > 3) || (has_sinks_ && inputs.size() > 4);

  // Ensure contiguous with stride-1 last dim
  auto ensure_contiguous = [this](const array& arr) -> array {
    if (arr.flags().row_contiguous) {
      return arr;
    }
    return contiguous_copy_cpu(arr, stream());
  };

  array q = ensure_contiguous(q_in);
  array k = ensure_contiguous(k_in);
  array v = ensure_contiguous(v_in);

  // Make mask contiguous if present
  auto mask_arr = has_arr_mask
      ? std::optional<array>(ensure_contiguous(inputs[3]))
      : std::nullopt;

  // Sinks is 1D (n_q_heads,), always contiguous
  auto sinks_arr =
      has_sinks_ ? std::optional<array>(inputs.back()) : std::nullopt;

  int B = q.shape(0);
  int n_q_heads = q.shape(1);
  int n_kv_heads = k.shape(1);
  int M = q.shape(2);
  int seq_len = k.shape(2);
  int head_dim = q.shape(3);

  // Allocate output
  out.set_data(allocator::malloc(out.nbytes()));

  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_input_array(q);
  encoder.set_input_array(k);
  encoder.set_input_array(v);
  if (has_arr_mask) {
    encoder.set_input_array(*mask_arr);
  }
  if (has_sinks_) {
    encoder.set_input_array(*sinks_arr);
  }
  encoder.set_output_array(out);

  // Keep temporaries alive if we made copies
  if (!q_in.flags().row_contiguous) {
    encoder.add_temporary(q);
  }
  if (!k_in.flags().row_contiguous) {
    encoder.add_temporary(k);
  }
  if (!v_in.flags().row_contiguous) {
    encoder.add_temporary(v);
  }
  if (has_arr_mask && !inputs[3].flags().row_contiguous) {
    encoder.add_temporary(*mask_arr);
  }

  float scale = scale_;
  bool do_causal = do_causal_;
  bool has_sinks = has_sinks_;

  switch (q.dtype()) {
    case float32: {
      const float* q_ptr = q.data<float>();
      const float* k_ptr = k.data<float>();
      const float* v_ptr = v.data<float>();
      float* out_ptr = out.data<float>();
      const float* mask_ptr = has_arr_mask ? mask_arr->data<float>() : nullptr;
      const float* sinks_ptr = has_sinks ? sinks_arr->data<float>() : nullptr;

      encoder.dispatch([=]() {
        sdpa_impl(
            out_ptr,
            q_ptr,
            k_ptr,
            v_ptr,
            B,
            n_q_heads,
            n_kv_heads,
            M,
            seq_len,
            head_dim,
            scale,
            do_causal,
            mask_ptr,
            has_arr_mask,
            sinks_ptr,
            has_sinks);
      });
      break;
    }
    case float16: {
      const float16_t* q_ptr = q.data<float16_t>();
      const float16_t* k_ptr = k.data<float16_t>();
      const float16_t* v_ptr = v.data<float16_t>();
      float16_t* out_ptr = out.data<float16_t>();
      const float16_t* mask_ptr =
          has_arr_mask ? mask_arr->data<float16_t>() : nullptr;
      const float16_t* sinks_ptr =
          has_sinks ? sinks_arr->data<float16_t>() : nullptr;

      encoder.dispatch([=]() {
        sdpa_impl(
            out_ptr,
            q_ptr,
            k_ptr,
            v_ptr,
            B,
            n_q_heads,
            n_kv_heads,
            M,
            seq_len,
            head_dim,
            scale,
            do_causal,
            mask_ptr,
            has_arr_mask,
            sinks_ptr,
            has_sinks);
      });
      break;
    }
    case bfloat16: {
      const bfloat16_t* q_ptr = q.data<bfloat16_t>();
      const bfloat16_t* k_ptr = k.data<bfloat16_t>();
      const bfloat16_t* v_ptr = v.data<bfloat16_t>();
      bfloat16_t* out_ptr = out.data<bfloat16_t>();
      const bfloat16_t* mask_ptr =
          has_arr_mask ? mask_arr->data<bfloat16_t>() : nullptr;
      const bfloat16_t* sinks_ptr =
          has_sinks ? sinks_arr->data<bfloat16_t>() : nullptr;

      encoder.dispatch([=]() {
        sdpa_impl(
            out_ptr,
            q_ptr,
            k_ptr,
            v_ptr,
            B,
            n_q_heads,
            n_kv_heads,
            M,
            seq_len,
            head_dim,
            scale,
            do_causal,
            mask_ptr,
            has_arr_mask,
            sinks_ptr,
            has_sinks);
      });
      break;
    }
    default:
      throw std::runtime_error(
          "[ScaledDotProductAttention::eval_cpu] Unsupported type");
  }
}

} // namespace mlx::core::fast
