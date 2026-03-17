// Copyright © 2025-2026 Apple Inc.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/gemm.h"
#include "mlx/backend/cpu/gemms/simd_gemm.h"
#include "mlx/backend/cpu/lapack.h"
#include "mlx/backend/cpu/simd/simd.h"
#include "mlx/backend/cpu/threading/common.h"

#include <cstring>
#include <vector>

namespace mlx::core {

// Minimum batches per thread before parallelizing the BLAS path.
constexpr int FP16_MIN_BATCHES_PER_THREAD = 4;

namespace {

// Thread-local scratch buffer for f16->f32 conversion.
// Avoids mmap/munmap cycles and page faults from repeated large allocations.
float* get_scratch(size_t n) {
  thread_local std::vector<float> buf;
  if (buf.size() < n) {
    buf.resize(n);
  }
  return buf.data();
}

void f16_to_f32(const float16_t* src, float* dst, size_t n) {
  constexpr int N = simd::max_size<float16_t>;
  size_t i = 0;
  for (; i + N <= n; i += N) {
    simd::store(
        dst + i, simd::Simd<float, N>(simd::load<float16_t, N>(src + i)));
  }
  for (; i < n; i++) {
    dst[i] = static_cast<float>(src[i]);
  }
}

void f32_to_f16(const float* src, float16_t* dst, size_t n) {
  constexpr int N = simd::max_size<float16_t>;
  size_t i = 0;
  for (; i + N <= n; i += N) {
    simd::store(
        dst + i, simd::Simd<float16_t, N>(simd::load<float, N>(src + i)));
  }
  for (; i < n; i++) {
    dst[i] = static_cast<float16_t>(src[i]);
  }
}

// Fused f16 GEMV: reads f16 weights directly, converts in-register via F16C,
// FMAs. Eliminates 3x memory traffic of separate f16->f32 conversion + BLAS.
// b_transposed=true: B is [N][K] (row = output), most common for LLM linear
// layers. b_transposed=false: B is [K][N] (row = input).
void f16_gemv(
    const float16_t* a,
    const float16_t* b,
    float16_t* out,
    bool b_transposed,
    size_t N,
    size_t K,
    size_t ldb,
    float alpha,
    float beta) {
  constexpr int W = simd::max_size<float16_t>; // 8 for AVX2

  if (b_transposed) {
    // B is [N][K]: each row n is a K-element weight vector for output n.
    // Compute: out[n] = alpha * dot(A, B[n]) + beta * out[n]
    // A stays in cache, stream B rows sequentially.
    for (size_t n = 0; n < N; n++) {
      const float16_t* b_row = b + n * ldb;
      simd::Simd<float, W> acc0(0.0f);
      simd::Simd<float, W> acc1(0.0f);
      simd::Simd<float, W> acc2(0.0f);
      simd::Simd<float, W> acc3(0.0f);
      size_t k = 0;
      for (; k + 4 * W <= K; k += 4 * W) {
        auto a0 = simd::Simd<float, W>(simd::load<float16_t, W>(a + k));
        auto b0 = simd::Simd<float, W>(simd::load<float16_t, W>(b_row + k));
        acc0 = acc0 + a0 * b0;
        auto a1 = simd::Simd<float, W>(simd::load<float16_t, W>(a + k + W));
        auto b1 = simd::Simd<float, W>(simd::load<float16_t, W>(b_row + k + W));
        acc1 = acc1 + a1 * b1;
        auto a2 = simd::Simd<float, W>(simd::load<float16_t, W>(a + k + 2 * W));
        auto b2 =
            simd::Simd<float, W>(simd::load<float16_t, W>(b_row + k + 2 * W));
        acc2 = acc2 + a2 * b2;
        auto a3 = simd::Simd<float, W>(simd::load<float16_t, W>(a + k + 3 * W));
        auto b3 =
            simd::Simd<float, W>(simd::load<float16_t, W>(b_row + k + 3 * W));
        acc3 = acc3 + a3 * b3;
      }
      for (; k + W <= K; k += W) {
        auto av = simd::Simd<float, W>(simd::load<float16_t, W>(a + k));
        auto bv = simd::Simd<float, W>(simd::load<float16_t, W>(b_row + k));
        acc0 = acc0 + av * bv;
      }
      float acc = simd::sum(acc0 + acc1 + acc2 + acc3);
      for (; k < K; k++) {
        acc += static_cast<float>(a[k]) * static_cast<float>(b_row[k]);
      }
      if (beta != 0) {
        out[n] = static_cast<float16_t>(
            alpha * acc + beta * static_cast<float>(out[n]));
      } else {
        out[n] = static_cast<float16_t>(alpha * acc);
      }
    }
  } else {
    // B is [K][N]: accumulate outer products a[k] * B[k][:]
    float* accum = get_scratch(N);
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
      const float16_t* b_row = b + k * ldb;
      size_t n = 0;
      for (; n + W <= N; n += W) {
        auto bv = simd::Simd<float, W>(simd::load<float16_t, W>(b_row + n));
        auto cv = simd::load<float, W>(accum + n);
        simd::store(accum + n, cv + a_broadcast * bv);
      }
      for (; n < N; n++) {
        accum[n] += a_val * static_cast<float>(b_row[n]);
      }
    }
    for (size_t n = 0; n < N; n++) {
      out[n] = static_cast<float16_t>(accum[n]);
    }
  }
}

// Multi-threaded wrapper for f16_gemv: splits N across threads.
void f16_gemv_threaded(
    const float16_t* a,
    const float16_t* b,
    float16_t* out,
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
    // For b_transposed, each output n is independent -- trivially parallel.
    pool.parallel_for(n_threads, [&](int tid, int nth) {
      size_t chunk = (N + nth - 1) / nth;
      size_t n_start = chunk * tid;
      size_t n_end = std::min(n_start + chunk, N);
      if (n_start < n_end) {
        f16_gemv(
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
    // For not-transposed, split N into ranges. Each thread accumulates its own
    // slice of accumulators, iterating over all K rows of B.
    pool.parallel_for(n_threads, [&](int tid, int nth) {
      size_t chunk = (N + nth - 1) / nth;
      size_t n_start = chunk * tid;
      size_t n_end = std::min(n_start + chunk, N);
      if (n_start >= n_end)
        return;

      constexpr int W = simd::max_size<float16_t>;
      size_t n_len = n_end - n_start;
      float* accum = get_scratch(n_len);
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
        const float16_t* b_row = b + k * ldb + n_start;
        size_t n = 0;
        for (; n + W <= n_len; n += W) {
          auto bv = simd::Simd<float, W>(simd::load<float16_t, W>(b_row + n));
          auto cv = simd::load<float, W>(accum + n);
          simd::store(accum + n, cv + a_broadcast * bv);
        }
        for (; n < n_len; n++) {
          accum[n] += a_val * static_cast<float>(b_row[n]);
        }
      }
      for (size_t n = 0; n < n_len; n++) {
        out[n_start + n] = static_cast<float16_t>(accum[n]);
      }
    });
  } else {
    f16_gemv(a, b, out, b_transposed, N, K, ldb, alpha, beta);
  }
}

} // namespace

template <>
void matmul<float16_t>(
    const float16_t* a,
    const float16_t* b,
    float16_t* out,
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

  // For batched operations (e.g. multi-head attention with 32+ heads),
  // cast to float32 and use threaded BLAS -- much faster than naive simd_gemm.
  // For single-batch ops (e.g. linear projections with huge weight matrices),
  // use simd_gemm to avoid copying the entire weight matrix to float32.
  if (batch_size > 1) {
    size_t a_rows = a_transposed ? K : M;
    size_t b_rows = b_transposed ? N : K;
    size_t a_elems = a_rows * lda;
    size_t b_elems = b_rows * ldb;
    size_t out_elems = M * ldc;

    auto run_batches = [&](size_t start,
                           size_t end,
                           float* a_f32,
                           float* b_f32,
                           float* out_f32) {
      for (size_t i = start; i < end; ++i) {
        const auto* a_ptr = a + elem_to_loc(M * K * i, a_shape, a_strides);
        const auto* b_ptr = b + elem_to_loc(K * N * i, b_shape, b_strides);
        auto* out_ptr = out + M * N * i;

        f16_to_f32(a_ptr, a_f32, a_elems);
        f16_to_f32(b_ptr, b_f32, b_elems);
        if (beta != 0) {
          f16_to_f32(out_ptr, out_f32, out_elems);
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

        f32_to_f16(out_f32, out_ptr, out_elems);
      }
    };

    auto& pool = cpu::ThreadPool::instance();
    int n_threads = std::min(
        pool.max_threads(),
        static_cast<int>(batch_size / FP16_MIN_BATCHES_PER_THREAD));

    if (n_threads > 1) {
      size_t buf_size = a_elems + b_elems + out_elems;

      pool.parallel_for(n_threads, [&](int tid, int nth) {
        float* base = get_scratch(buf_size);
        float* a_f32 = base;
        float* b_f32 = base + a_elems;
        float* out_f32 = base + a_elems + b_elems;

        size_t chunk = (batch_size + nth - 1) / nth;
        size_t start = chunk * tid;
        size_t end = std::min(start + chunk, batch_size);
        run_batches(start, end, a_f32, b_f32, out_f32);
      });
    } else {
      float* scratch = get_scratch(a_elems + b_elems + out_elems);
      run_batches(
          0,
          batch_size,
          scratch,
          scratch + a_elems,
          scratch + a_elems + b_elems);
    }
  } else {
    // Single batch
    for (size_t i = 0; i < batch_size; ++i) {
      const auto* a_ptr = a + elem_to_loc(M * K * i, a_shape, a_strides);
      const auto* b_ptr = b + elem_to_loc(K * N * i, b_shape, b_strides);
      auto* out_ptr = out + M * N * i;

      if (M == 1 && N * K >= 65536) {
        // M=1 GEMV: fused f16 read + F16C convert + FMA in one pass.
        f16_gemv_threaded(
            a_ptr, b_ptr, out_ptr, b_transposed, N, K, ldb, alpha, beta);
      } else if (M * N * K >= 65536) {
        // Larger GEMM: cast to f32 + use BLAS
        auto& pool = cpu::ThreadPool::instance();
        int m_threads = 1;
        if (M >= 16) {
          m_threads = std::min(
              pool.max_threads(), std::max(1, static_cast<int>(M / 8)));
        }

        size_t a_rows = a_transposed ? K : M;
        size_t b_rows = b_transposed ? N : K;
        size_t a_elems = a_rows * lda;
        size_t b_elems = b_rows * ldb;
        size_t out_elems = M * ldc;

        float* a_f32 = get_scratch(a_elems + b_elems + out_elems);
        float* b_f32 = a_f32 + a_elems;
        float* out_f32 = b_f32 + b_elems;

        f16_to_f32(a_ptr, a_f32, a_elems);
        f16_to_f32(b_ptr, b_f32, b_elems);
        if (beta != 0) {
          f16_to_f32(out_ptr, out_f32, out_elems);
        }

        if (m_threads > 1) {
          pool.parallel_for(m_threads, [&](int tid, int nth) {
            size_t m_chunk = (M + nth - 1) / nth;
            size_t m_start = m_chunk * tid;
            size_t m_end = std::min(m_start + m_chunk, M);
            if (m_start < m_end) {
              cblas_sgemm(
                  CblasRowMajor,
                  a_transposed ? CblasTrans : CblasNoTrans,
                  b_transposed ? CblasTrans : CblasNoTrans,
                  m_end - m_start,
                  N,
                  K,
                  alpha,
                  a_f32 + m_start * lda,
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

        f32_to_f16(out_f32, out_ptr, out_elems);
      } else {
        // Small GEMM: use simd_gemm directly (no cast overhead)
        simd_gemm<float16_t, float>(
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

} // namespace mlx::core
