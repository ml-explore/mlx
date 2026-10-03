// Copyright © 2025-2026 Apple Inc.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/gemm.h"
#include "mlx/backend/cpu/lapack.h"
#include "mlx/backend/cpu/threading/common.h"

#if defined(MLX_USE_HIGHWAY_KERNELS)
#include "mlx/backend/cpu/gemms/simd_low_precision_gemm.h"
#endif

namespace mlx::core {

namespace {

// Minimum batches per thread before parallelizing batch matmul
// Higher threshold than unary/binary ops because BLAS calls are heavier
constexpr int MIN_BATCHES_PER_THREAD = 4;

// Maximum M routed through the few-row fused f32 GEMV. Measured on Zen 4 at
// N=K=4096: the fused kernel holds ~280 GFLOP/s through M=256 while
// single/M-split sgemm only overtakes it beyond that.
constexpr size_t F32_FEWROW_MAX_M = 256;

// Shared batched-GEMM driver. BLAS is pinned to a single thread for pool
// coordination, so parallelism comes from the pool: across batches when
// there are many, otherwise by splitting M rows of each GEMM.
// `try_fast_path` may fully handle one batch item (few-row kernels);
// `gemm` runs one M-slice of one batch item.
template <typename T, typename GemmFn, typename FastPathFn>
void batched_gemm(
    GemmFn gemm,
    FastPathFn try_fast_path,
    const T* a,
    const T* b,
    T* out,
    size_t batch_size,
    const Shape& a_shape,
    const Strides& a_strides,
    const Shape& b_shape,
    const Strides& b_strides) {
  auto ndim = a_shape.size();
  size_t M = a_shape[ndim - 2];
  size_t N = b_shape[ndim - 1];
  size_t K = a_shape[ndim - 1];

  auto& pool = cpu::ThreadPool::instance();
  int n_threads = std::min(
      pool.max_threads(),
      static_cast<int>(batch_size / MIN_BATCHES_PER_THREAD));

  if (n_threads > 1) {
    cpu::parallel_for_range(
        n_threads, batch_size, [&](size_t begin, size_t end) {
          for (size_t i = begin; i < end; ++i) {
            const T* a_ptr = a + elem_to_loc(M * K * i, a_shape, a_strides);
            const T* b_ptr = b + elem_to_loc(K * N * i, b_shape, b_strides);
            T* out_ptr = out + M * N * i;
            if (try_fast_path(a_ptr, b_ptr, out_ptr)) {
              continue;
            }
            gemm(a_ptr, b_ptr, out_ptr, 0, M);
          }
        });
    return;
  }

  for (size_t i = 0; i < batch_size; ++i) {
    const T* a_ptr = a + elem_to_loc(M * K * i, a_shape, a_strides);
    const T* b_ptr = b + elem_to_loc(K * N * i, b_shape, b_strides);
    T* out_ptr = out + M * N * i;
    if (try_fast_path(a_ptr, b_ptr, out_ptr)) {
      continue;
    }
    int m_threads = 1;
    if (M >= 16 && M * N * K >= 65536) {
      m_threads =
          std::min(pool.max_threads(), std::max(1, static_cast<int>(M / 8)));
    }
    cpu::parallel_for_range(m_threads, M, [&](size_t m_start, size_t m_end) {
      gemm(a_ptr, b_ptr, out_ptr, m_start, m_end - m_start);
    });
  }
}

} // namespace

template <>
void matmul<float>(
    const float* a,
    const float* b,
    float* out,
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

  auto gemm = [&](const float* a_ptr,
                  const float* b_ptr,
                  float* out_ptr,
                  size_t m_start,
                  size_t m_rows) {
    // When a_transposed: A is stored KxM row-major, so an M slice offsets
    // columns of stored A rather than rows.
    size_t a_offset = a_transposed ? m_start : m_start * lda;
    cblas_sgemm(
        CblasRowMajor,
        a_transposed ? CblasTrans : CblasNoTrans,
        b_transposed ? CblasTrans : CblasNoTrans,
        m_rows,
        N,
        K,
        alpha,
        a_ptr + a_offset,
        lda,
        b_ptr,
        ldb,
        beta,
        out_ptr + m_start * ldc,
        ldc);
  };
  auto fast_path = [&](const float* a_ptr, const float* b_ptr, float* out_ptr) {
#if defined(MLX_USE_HIGHWAY_KERNELS)
    // Few-row x @ B^T: stream B once per register block instead of a
    // single-threaded sgemm (the M split cannot engage below 16 rows).
    if (!a_transposed && b_transposed && M <= F32_FEWROW_MAX_M &&
        N * K >= 65536) {
      detail::lowp_fewrow_gemm_threaded(
          a_ptr, b_ptr, out_ptr, M, N, K, lda, ldb, ldc, alpha, beta);
      return true;
    }
#endif
    (void)a_ptr;
    (void)b_ptr;
    (void)out_ptr;
    return false;
  };
  batched_gemm<float>(
      gemm,
      fast_path,
      a,
      b,
      out,
      batch_size,
      a_shape,
      a_strides,
      b_shape,
      b_strides);
}

template <>
void matmul<double>(
    const double* a,
    const double* b,
    double* out,
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
  size_t N = b_shape[ndim - 1];
  size_t K = a_shape[ndim - 1];

  auto gemm = [&](const double* a_ptr,
                  const double* b_ptr,
                  double* out_ptr,
                  size_t m_start,
                  size_t m_rows) {
    size_t a_offset = a_transposed ? m_start : m_start * lda;
    cblas_dgemm(
        CblasRowMajor,
        a_transposed ? CblasTrans : CblasNoTrans,
        b_transposed ? CblasTrans : CblasNoTrans,
        m_rows,
        N,
        K,
        alpha,
        a_ptr + a_offset,
        lda,
        b_ptr,
        ldb,
        beta,
        out_ptr + m_start * ldc,
        ldc);
  };
  auto no_fast_path = [](const double*, const double*, double*) {
    return false;
  };
  batched_gemm<double>(
      gemm,
      no_fast_path,
      a,
      b,
      out,
      batch_size,
      a_shape,
      a_strides,
      b_shape,
      b_strides);
}

template <>
void matmul<complex64_t>(
    const complex64_t* a,
    const complex64_t* b,
    complex64_t* out,
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
  size_t N = b_shape[ndim - 1];
  size_t K = a_shape[ndim - 1];
  auto calpha = static_cast<complex64_t>(alpha);
  auto cbeta = static_cast<complex64_t>(beta);

  auto gemm = [&](const complex64_t* a_ptr,
                  const complex64_t* b_ptr,
                  complex64_t* out_ptr,
                  size_t m_start,
                  size_t m_rows) {
    size_t a_offset = a_transposed ? m_start : m_start * lda;
    cblas_cgemm(
        CblasRowMajor,
        a_transposed ? CblasTrans : CblasNoTrans,
        b_transposed ? CblasTrans : CblasNoTrans,
        m_rows,
        N,
        K,
        &calpha,
        a_ptr + a_offset,
        lda,
        b_ptr,
        ldb,
        &cbeta,
        out_ptr + m_start * ldc,
        ldc);
  };
  auto no_fast_path = [](const complex64_t*, const complex64_t*, complex64_t*) {
    return false;
  };
  batched_gemm<complex64_t>(
      gemm,
      no_fast_path,
      a,
      b,
      out,
      batch_size,
      a_shape,
      a_strides,
      b_shape,
      b_strides);
}

} // namespace mlx::core
