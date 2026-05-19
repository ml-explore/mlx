// Copyright © 2025 Apple Inc.
#pragma once

#include <immintrin.h>
#include <algorithm>
#include <cstring>

#include "mlx/backend/cpu/gemms/aligned_buffer.h"
#include "mlx/backend/cpu/gemms/avx_gemm_simd.h"

namespace mlx::core {

// Output-dim block: 4096 fp32 = 16KB, fits in L1 alongside B rows.
constexpr int GEMV_NC_BLOCK = 4096;

// acc[0:width] += sum_k vec[k] * mat[k * mat_stride + 0:width]
template <typename T>
static void gemv_outer_product(
    const T* vec,
    const T* mat,
    float* acc,
    int K,
    int width,
    int mat_stride) {
  constexpr int sw = 8;

  for (int jc = 0; jc < width; jc += GEMV_NC_BLOCK) {
    int nc = std::min(GEMV_NC_BLOCK, width - jc);
    float* acc_block = acc + jc;

    for (int k = 0; k < K; k++) {
      float v = static_cast<float>(vec[k]);
      detail::Simd<float, 8> v_bcast(v);
      const T* mat_row = mat + k * mat_stride + jc;

      // Prefetch start of next row for this block
      if (k + 1 < K) {
        _mm_prefetch(
            reinterpret_cast<const char*>(mat + (k + 1) * mat_stride + jc),
            _MM_HINT_T0);
      }

      int j = 0;
      for (; j + sw <= nc; j += sw) {
        detail::Simd<float, 8> m = detail::load_convert_to_float<T>(mat_row + j);
        detail::Simd<float, 8> c = detail::load<float, sw>(acc_block + j);
        detail::store<float, sw>(
            acc_block + j, detail::fma<float, sw>(v_bcast, m, c));
      }
      for (; j < nc; j++) {
        acc_block[j] += v * static_cast<float>(mat_row[j]);
      }
    }
  }
}

// acc[i] += dot(mat[i*mat_stride : +K], vec[0:K]); 4-row unroll to share vec
// loads.
template <typename T>
static void gemv_dot_product(
    const T* mat,
    const T* vec,
    float* acc,
    int n_outputs,
    int K,
    int mat_stride) {
  constexpr int sw = 8;
  constexpr int UNROLL = 4;

  int i = 0;
  for (; i + UNROLL <= n_outputs; i += UNROLL) {
    detail::Simd<float, 8> s0, s1, s2, s3;

    const T* r0 = mat + (i + 0) * mat_stride;
    const T* r1 = mat + (i + 1) * mat_stride;
    const T* r2 = mat + (i + 2) * mat_stride;
    const T* r3 = mat + (i + 3) * mat_stride;

    int k = 0;
    for (; k + sw <= K; k += sw) {
      detail::Simd<float, 8> v = detail::load_convert_to_float<T>(vec + k);
      s0 = detail::fma<float, sw>(detail::load_convert_to_float<T>(r0 + k), v, s0);
      s1 = detail::fma<float, sw>(detail::load_convert_to_float<T>(r1 + k), v, s1);
      s2 = detail::fma<float, sw>(detail::load_convert_to_float<T>(r2 + k), v, s2);
      s3 = detail::fma<float, sw>(detail::load_convert_to_float<T>(r3 + k), v, s3);
    }

    float d0 = detail::sum(s0);
    float d1 = detail::sum(s1);
    float d2 = detail::sum(s2);
    float d3 = detail::sum(s3);

    for (; k < K; k++) {
      float vk = static_cast<float>(vec[k]);
      d0 += vk * static_cast<float>(r0[k]);
      d1 += vk * static_cast<float>(r1[k]);
      d2 += vk * static_cast<float>(r2[k]);
      d3 += vk * static_cast<float>(r3[k]);
    }

    acc[i + 0] += d0;
    acc[i + 1] += d1;
    acc[i + 2] += d2;
    acc[i + 3] += d3;
  }

  for (; i < n_outputs; i++) {
    detail::Simd<float, 8> s;
    const T* row = mat + i * mat_stride;

    int k = 0;
    for (; k + sw <= K; k += sw) {
      detail::Simd<float, 8> v = detail::load_convert_to_float<T>(vec + k);
      s = detail::fma<float, sw>(detail::load_convert_to_float<T>(row + k), v, s);
    }

    float d = detail::sum(s);
    for (; k < K; k++) {
      d += static_cast<float>(vec[k]) * static_cast<float>(row[k]);
    }
    acc[i] += d;
  }
}

// C = alpha * op(A) * op(B) + beta * C, for M=1 or N=1.
// Dispatches to outer-product or dot-product core based on shape and transpose.
template <typename T>
void simd_gemv(
    const T* a,
    const T* b,
    T* c,
    bool a_trans,
    bool b_trans,
    int M,
    int N,
    int K,
    int ldA,
    int ldB,
    int ldC,
    float alpha,
    float beta) {
  int out_len = (M == 1) ? N : M;

  // Thread-local fp32 accumulator (grow-only).
  thread_local aligned_unique_ptr<float> acc_buf(1);
  acc_buf.reset(out_len);
  float* acc = acc_buf.get();

  constexpr int sw = 8;
  std::memset(acc, 0, out_len * sizeof(float));

  // acc += op(A) * op(B)
  if (M == 1) {
    if (!b_trans) {
      gemv_outer_product(a, b, acc, K, N, ldB);
    } else {
      gemv_dot_product(b, a, acc, N, K, ldB);
    }
  } else {
    if (!a_trans) {
      gemv_dot_product(a, b, acc, M, K, ldA);
    } else {
      gemv_outer_product(b, a, acc, K, M, ldA);
    }
  }

  // Writeback: C = alpha * acc + beta * C (convert fp32 → T)
  bool apply_alpha = (alpha != 1.0f);
  bool apply_beta = (beta != 0.0f);
  detail::Simd<float, 8> alpha_vec(alpha);
  detail::Simd<float, 8> beta_vec(beta);
  int j = 0;
  for (; j + sw <= out_len; j += sw) {
    detail::Simd<float, 8> val = detail::load<float, sw>(acc + j);
    if (apply_alpha)
      val = alpha_vec * val;
    if (apply_beta) {
      detail::Simd<float, 8> cv = detail::load_convert_to_float<T>(c + j);
      val = val + beta_vec * cv;
    }
    detail::store_convert_from_float<T>(c + j, val);
  }
  for (; j < out_len; j++) {
    float val = acc[j];
    if (apply_alpha)
      val *= alpha;
    if (apply_beta)
      val += beta * static_cast<float>(c[j]);
    c[j] = static_cast<T>(val);
  }
}

} // namespace mlx::core
