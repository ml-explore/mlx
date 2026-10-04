// Copyright © 2026 Apple Inc.

#pragma once

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/gemm.h"
#include "mlx/backend/cpu/gemms/simd_gemm.h"
#include "mlx/backend/cpu/lapack.h"
#include "mlx/backend/cpu/scratch.h"
#include "mlx/backend/cpu/simd/simd.h"
#include "mlx/backend/cpu/threading/common.h"

#if defined(MLX_USE_HIGHWAY_KERNELS)
#include "mlx/backend/cpu/quantized_highway.h"
#endif

#include <algorithm>
#include <cstring>
#include <vector>

namespace mlx::core::detail {

constexpr int LOWP_MIN_BATCHES_PER_THREAD = 4;

// Maximum M routed through the few-row fused GEMV. Above this the
// convert-to-f32 + BLAS path wins on cache blocking (the fused kernel's
// cost grows ~M/8 while BLAS approaches compute-bound). Measured on Zen 4
// at N=K=4096: fused holds ~250-280 GFLOP/s through M=256 but bf16 degrades
// past that, while convert+sgemm reaches ~450 GFLOP/s at M=512.
constexpr int LOWP_FEWROW_MAX_M = 192;

#if defined(MLX_USE_HIGHWAY_KERNELS)
// Few-row x @ B^T via the runtime-dispatched Highway kernel: streams B in
// its native precision (half traffic vs f32 materialization, native F16C
// conversion) with N split across the thread pool.
template <typename T>
void lowp_fewrow_gemm_threaded(
    const T* a,
    const T* b,
    T* out,
    size_t M,
    size_t N,
    size_t K,
    size_t lda,
    size_t ldb,
    size_t ldc,
    float alpha,
    float beta) {
  auto& pool = cpu::ThreadPool::instance();
  int n_threads =
      std::min(pool.max_threads(), std::max(1, static_cast<int>(N / 64)));
  auto dtype = quantized_highway_dtype<T>();
  if (n_threads > 1) {
    pool.parallel_for(n_threads, [&](int tid, int nth) {
      size_t chunk = (N + nth - 1) / nth;
      size_t n_start = chunk * tid;
      size_t n_end = std::min(n_start + chunk, N);
      if (n_start < n_end) {
        lowp_gemv_fewrow_highway(
            out,
            a,
            b,
            dtype,
            static_cast<int>(M),
            static_cast<int>(n_start),
            static_cast<int>(n_end),
            static_cast<int>(K),
            lda,
            ldb,
            ldc,
            alpha,
            beta);
      }
    });
  } else {
    lowp_gemv_fewrow_highway(
        out,
        a,
        b,
        dtype,
        static_cast<int>(M),
        0,
        static_cast<int>(N),
        static_cast<int>(K),
        lda,
        ldb,
        ldc,
        alpha,
        beta);
  }
}

template <typename BT>
void mixed_lowp_fewrow_gemm_threaded(
    const float* a,
    const BT* b,
    float* out,
    size_t M,
    size_t N,
    size_t K,
    size_t lda,
    size_t ldb,
    size_t ldc,
    float alpha,
    float beta) {
  auto& pool = cpu::ThreadPool::instance();
  int n_threads =
      std::min(pool.max_threads(), std::max(1, static_cast<int>(N / 64)));
  auto b_dtype = quantized_highway_dtype<BT>();
  auto run = [&](size_t begin, size_t end) {
    mixed_lowp_gemv_fewrow_highway(
        out,
        a,
        b,
        b_dtype,
        static_cast<int>(M),
        static_cast<int>(begin),
        static_cast<int>(end),
        static_cast<int>(K),
        lda,
        ldb,
        ldc,
        alpha,
        beta);
  };
  cpu::parallel_for_range(n_threads, N, run);
}
#endif

template <typename T>
void lowp_to_f32(const T* src, float* dst, size_t n) {
  constexpr int N = simd::max_size<T>;
  size_t i = 0;
  for (; i + N <= n; i += N) {
    simd::store(dst + i, simd::Simd<float, N>(simd::load<T, N>(src + i)));
  }
  for (; i < n; i++) {
    dst[i] = static_cast<float>(src[i]);
  }
}

template <typename T>
void f32_to_lowp(const float* src, T* dst, size_t n) {
  constexpr int N = simd::max_size<T>;
  size_t i = 0;
  for (; i + N <= n; i += N) {
    simd::store(dst + i, simd::Simd<T, N>(simd::load<float, N>(src + i)));
  }
  for (; i < n; i++) {
    dst[i] = static_cast<T>(src[i]);
  }
}

// Fused low-precision GEMV: reads weights directly, converts in-register, and
// accumulates in f32. This avoids the extra memory traffic of materializing
// both low-precision inputs as f32 before calling BLAS.
template <typename T>
void lowp_gemv(
    const T* a,
    const T* b,
    T* out,
    bool b_transposed,
    size_t N,
    size_t K,
    size_t ldb,
    float alpha,
    float beta) {
  constexpr int W = simd::max_size<T>;

  if (b_transposed) {
    // B is [N][K]: each row n is a K-element weight vector for output n.
    for (size_t n = 0; n < N; n++) {
      const T* b_row = b + n * ldb;
      simd::Simd<float, W> acc0(0.0f);
      simd::Simd<float, W> acc1(0.0f);
      simd::Simd<float, W> acc2(0.0f);
      simd::Simd<float, W> acc3(0.0f);
      size_t k = 0;
      for (; k + 4 * W <= K; k += 4 * W) {
        auto a0 = simd::Simd<float, W>(simd::load<T, W>(a + k));
        auto b0 = simd::Simd<float, W>(simd::load<T, W>(b_row + k));
        acc0 = acc0 + a0 * b0;
        auto a1 = simd::Simd<float, W>(simd::load<T, W>(a + k + W));
        auto b1 = simd::Simd<float, W>(simd::load<T, W>(b_row + k + W));
        acc1 = acc1 + a1 * b1;
        auto a2 = simd::Simd<float, W>(simd::load<T, W>(a + k + 2 * W));
        auto b2 = simd::Simd<float, W>(simd::load<T, W>(b_row + k + 2 * W));
        acc2 = acc2 + a2 * b2;
        auto a3 = simd::Simd<float, W>(simd::load<T, W>(a + k + 3 * W));
        auto b3 = simd::Simd<float, W>(simd::load<T, W>(b_row + k + 3 * W));
        acc3 = acc3 + a3 * b3;
      }
      for (; k + W <= K; k += W) {
        auto av = simd::Simd<float, W>(simd::load<T, W>(a + k));
        auto bv = simd::Simd<float, W>(simd::load<T, W>(b_row + k));
        acc0 = acc0 + av * bv;
      }
      float acc = simd::sum(acc0 + acc1 + acc2 + acc3);
      for (; k < K; k++) {
        acc += static_cast<float>(a[k]) * static_cast<float>(b_row[k]);
      }
      if (beta != 0) {
        out[n] =
            static_cast<T>(alpha * acc + beta * static_cast<float>(out[n]));
      } else {
        out[n] = static_cast<T>(alpha * acc);
      }
    }
  } else {
    // B is [K][N]: accumulate outer products a[k] * B[k][:].
    float* accum = cpu::scratch_buffer(N);
    if (beta != 0) {
      for (size_t n = 0; n < N; n++) {
        accum[n] = beta * static_cast<float>(out[n]);
      }
    } else {
      std::memset(accum, 0, N * sizeof(float));
    }
    for (size_t k = 0; k < K; k++) {
      float a_val = alpha * static_cast<float>(a[k]);
      auto a_broadcast = simd::Simd<float, W>(a_val);
      const T* b_row = b + k * ldb;
      size_t n = 0;
      for (; n + W <= N; n += W) {
        auto bv = simd::Simd<float, W>(simd::load<T, W>(b_row + n));
        auto cv = simd::load<float, W>(accum + n);
        simd::store(accum + n, cv + a_broadcast * bv);
      }
      for (; n < N; n++) {
        accum[n] += a_val * static_cast<float>(b_row[n]);
      }
    }
    for (size_t n = 0; n < N; n++) {
      out[n] = static_cast<T>(accum[n]);
    }
  }
}

template <typename T>
void lowp_gemv_threaded(
    const T* a,
    const T* b,
    T* out,
    bool b_transposed,
    size_t N,
    size_t K,
    size_t ldb,
    float alpha,
    float beta) {
  auto& pool = cpu::ThreadPool::instance();
  int n_threads =
      std::min(pool.max_threads(), std::max(1, static_cast<int>(N / 128)));

  if (n_threads > 1 && b_transposed) {
    pool.parallel_for(n_threads, [&](int tid, int nth) {
      size_t chunk = (N + nth - 1) / nth;
      size_t n_start = chunk * tid;
      size_t n_end = std::min(n_start + chunk, N);
      if (n_start < n_end) {
        lowp_gemv(
            a,
            b + n_start * ldb,
            out + n_start,
            true,
            n_end - n_start,
            K,
            ldb,
            alpha,
            beta);
      }
    });
  } else if (n_threads > 1 && !b_transposed) {
    // Split N into ranges. Each thread accumulates its own output slice while
    // iterating over all K rows of B.
    pool.parallel_for(n_threads, [&](int tid, int nth) {
      size_t chunk = (N + nth - 1) / nth;
      size_t n_start = chunk * tid;
      size_t n_end = std::min(n_start + chunk, N);
      if (n_start >= n_end) {
        return;
      }

      constexpr int W = simd::max_size<T>;
      size_t n_len = n_end - n_start;
      float* accum = cpu::scratch_buffer(n_len);
      if (beta != 0) {
        for (size_t n = 0; n < n_len; n++) {
          accum[n] = beta * static_cast<float>(out[n_start + n]);
        }
      } else {
        std::memset(accum, 0, n_len * sizeof(float));
      }
      for (size_t k = 0; k < K; k++) {
        float a_val = alpha * static_cast<float>(a[k]);
        auto a_broadcast = simd::Simd<float, W>(a_val);
        const T* b_row = b + k * ldb + n_start;
        size_t n = 0;
        for (; n + W <= n_len; n += W) {
          auto bv = simd::Simd<float, W>(simd::load<T, W>(b_row + n));
          auto cv = simd::load<float, W>(accum + n);
          simd::store(accum + n, cv + a_broadcast * bv);
        }
        for (; n < n_len; n++) {
          accum[n] += a_val * static_cast<float>(b_row[n]);
        }
      }
      for (size_t n = 0; n < n_len; n++) {
        out[n_start + n] = static_cast<T>(accum[n]);
      }
    });
  } else {
    lowp_gemv(a, b, out, b_transposed, N, K, ldb, alpha, beta);
  }
}

template <typename T>
void matmul_lowp(
    const T* a,
    const T* b,
    T* out,
    bool a_transposed,
    bool b_transposed,
    size_t lda,
    size_t ldb,
    size_t ldc,
    float alpha,
    float beta,
    size_t batch_size,
    const Shape& a_shape,
    const Strides& a_strides,
    const Shape& b_shape,
    const Strides& b_strides) {
  auto ndim = a_shape.size();
  size_t M = a_shape[ndim - 2];
  size_t N = b_shape[ndim - 1];
  size_t K = a_shape[ndim - 1];

  if (batch_size > 1) {
    size_t a_rows = a_transposed ? K : M;
    size_t b_rows = b_transposed ? N : K;
    size_t a_elems = a_rows * lda;
    size_t b_elems = b_rows * ldb;
    size_t out_elems = M * ldc;

#if defined(MLX_USE_HIGHWAY_KERNELS)
    if (!a_transposed && M <= LOWP_FEWROW_MAX_M && N * K >= 65536 &&
        (b_transposed || M == 1)) {
      auto run_fast_batches = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
          const auto* a_ptr = a + elem_to_loc(M * K * i, a_shape, a_strides);
          const auto* b_ptr = b + elem_to_loc(K * N * i, b_shape, b_strides);
          auto* out_ptr = out + M * N * i;

          if (b_transposed) {
            lowp_fewrow_gemm_threaded(
                a_ptr, b_ptr, out_ptr, M, N, K, lda, ldb, ldc, alpha, beta);
          } else {
            lowp_gemv_threaded(
                a_ptr, b_ptr, out_ptr, b_transposed, N, K, ldb, alpha, beta);
          }
        }
      };

      auto& pool = cpu::ThreadPool::instance();
      int n_threads = std::min(
          pool.max_threads(),
          static_cast<int>(batch_size / LOWP_MIN_BATCHES_PER_THREAD));
      if (n_threads > 1) {
        cpu::parallel_for_range(n_threads, batch_size, run_fast_batches);
      } else {
        run_fast_batches(0, batch_size);
      }
      return;
    }
#endif

    auto run_batches = [&](size_t start,
                           size_t end,
                           float* a_f32,
                           float* b_f32,
                           float* out_f32) {
      for (size_t i = start; i < end; ++i) {
        const auto* a_ptr = a + elem_to_loc(M * K * i, a_shape, a_strides);
        const auto* b_ptr = b + elem_to_loc(K * N * i, b_shape, b_strides);
        auto* out_ptr = out + M * N * i;

        lowp_to_f32(a_ptr, a_f32, a_elems);
        lowp_to_f32(b_ptr, b_f32, b_elems);
        if (beta != 0) {
          lowp_to_f32(out_ptr, out_f32, out_elems);
        }

        cblas_sgemm(
            CblasRowMajor,
            a_transposed ? CblasTrans : CblasNoTrans,
            b_transposed ? CblasTrans : CblasNoTrans,
            M,
            N,
            K,
            alpha,
            a_f32,
            lda,
            b_f32,
            ldb,
            beta,
            out_f32,
            ldc);

        f32_to_lowp(out_f32, out_ptr, out_elems);
      }
    };

    auto& pool = cpu::ThreadPool::instance();
    int n_threads = std::min(
        pool.max_threads(),
        static_cast<int>(batch_size / LOWP_MIN_BATCHES_PER_THREAD));

    if (n_threads > 1) {
      size_t buf_size = a_elems + b_elems + out_elems;

      pool.parallel_for(n_threads, [&](int tid, int nth) {
        float* base = cpu::scratch_buffer(buf_size);
        float* a_f32 = base;
        float* b_f32 = base + a_elems;
        float* out_f32 = base + a_elems + b_elems;

        size_t chunk = (batch_size + nth - 1) / nth;
        size_t start = chunk * tid;
        size_t end = std::min(start + chunk, batch_size);
        run_batches(start, end, a_f32, b_f32, out_f32);
      });
    } else {
      float* scratch = cpu::scratch_buffer(a_elems + b_elems + out_elems);
      run_batches(
          0,
          batch_size,
          scratch,
          scratch + a_elems,
          scratch + a_elems + b_elems);
    }
  } else {
    for (size_t i = 0; i < batch_size; ++i) {
      const auto* a_ptr = a + elem_to_loc(M * K * i, a_shape, a_strides);
      const auto* b_ptr = b + elem_to_loc(K * N * i, b_shape, b_strides);
      auto* out_ptr = out + M * N * i;

#if defined(MLX_USE_HIGHWAY_KERNELS)
      if (!a_transposed && b_transposed && M <= LOWP_FEWROW_MAX_M &&
          N * K >= 65536) {
        lowp_fewrow_gemm_threaded(
            a_ptr, b_ptr, out_ptr, M, N, K, lda, ldb, ldc, alpha, beta);
        continue;
      }
#endif
      if (M == 1 && N * K >= 65536) {
        lowp_gemv_threaded(
            a_ptr, b_ptr, out_ptr, b_transposed, N, K, ldb, alpha, beta);
      } else if (M * N * K >= 65536) {
        size_t a_rows = a_transposed ? K : M;
        size_t b_rows = b_transposed ? N : K;
        size_t a_elems = a_rows * lda;
        size_t b_elems = b_rows * ldb;
        size_t out_elems = M * ldc;

        float* a_f32 = cpu::scratch_buffer(a_elems + b_elems + out_elems);
        float* b_f32 = a_f32 + a_elems;
        float* out_f32 = b_f32 + b_elems;

        lowp_to_f32(a_ptr, a_f32, a_elems);
        lowp_to_f32(b_ptr, b_f32, b_elems);
        if (beta != 0) {
          lowp_to_f32(out_ptr, out_f32, out_elems);
        }

        auto& pool = cpu::ThreadPool::instance();
        int m_threads = 1;
        if (M >= 16) {
          m_threads = std::min(
              pool.max_threads(), std::max(1, static_cast<int>(M / 8)));
        }

        if (m_threads > 1) {
          pool.parallel_for(m_threads, [&](int tid, int nth) {
            size_t m_chunk = (M + nth - 1) / nth;
            size_t m_start = m_chunk * tid;
            size_t m_end = std::min(m_start + m_chunk, M);
            if (m_start < m_end) {
              size_t a_offset = a_transposed ? m_start : m_start * lda;
              cblas_sgemm(
                  CblasRowMajor,
                  a_transposed ? CblasTrans : CblasNoTrans,
                  b_transposed ? CblasTrans : CblasNoTrans,
                  m_end - m_start,
                  N,
                  K,
                  alpha,
                  a_f32 + a_offset,
                  lda,
                  b_f32,
                  ldb,
                  beta,
                  out_f32 + m_start * ldc,
                  ldc);
            }
          });
        } else {
          cblas_sgemm(
              CblasRowMajor,
              a_transposed ? CblasTrans : CblasNoTrans,
              b_transposed ? CblasTrans : CblasNoTrans,
              M,
              N,
              K,
              alpha,
              a_f32,
              lda,
              b_f32,
              ldb,
              beta,
              out_f32,
              ldc);
        }

        f32_to_lowp(out_f32, out_ptr, out_elems);
      } else {
        simd_gemm<T, float>(
            a_ptr,
            b_ptr,
            out_ptr,
            a_transposed,
            b_transposed,
            M,
            N,
            K,
            alpha,
            beta);
      }
    }
  }
}

} // namespace mlx::core::detail
