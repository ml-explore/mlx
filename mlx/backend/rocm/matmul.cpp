// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/matmul.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/gemms/gemv.h"
#include "mlx/backend/rocm/gemms/hipblaslt_gemm.h"
#include "mlx/backend/rocm/gemms/naive_gemm.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/utils.h"
#include "mlx/primitives.h"
#include "mlx/types/half_types.h"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

namespace mlx::core {

namespace {

std::tuple<bool, int64_t, array>
check_transpose(rocm::CommandEncoder& enc, const Stream& s, const array& arr) {
  auto stx = arr.strides()[arr.ndim() - 2];
  auto sty = arr.strides()[arr.ndim() - 1];
  if (sty == 1 && stx == arr.shape(-1)) {
    return std::make_tuple(false, stx, arr);
  } else if (stx == 1 && sty == arr.shape(-2)) {
    return std::make_tuple(true, sty, arr);
  } else {
    array arr_copy = contiguous_copy_gpu(arr, s);
    enc.add_temporary(arr_copy);
    return std::make_tuple(false, arr.shape(-1), arr_copy);
  }
}

std::tuple<bool, int64_t, array> ensure_batch_contiguous(
    const array& x,
    rocm::CommandEncoder& encoder,
    Stream s) {
  if (x.flags().row_contiguous) {
    return std::make_tuple(false, x.strides(-2), x);
  }

  bool rc = true;
  for (int i = 0; i < x.ndim() - 3; i++) {
    rc &= (x.strides(i + 1) * x.shape(i)) == x.strides(i);
  }
  if (rc) {
    return check_transpose(encoder, s, x);
  }

  array x_copy = contiguous_copy_gpu(x, s);
  encoder.add_temporary(x_copy);
  return std::make_tuple(false, x_copy.strides(-2), x_copy);
}

std::pair<bool, int64_t> get_uniform_batch_stride(
    const Shape& batch_shape,
    const Strides& batch_strides) {
  if (batch_shape.empty() || batch_shape.size() != batch_strides.size()) {
    return {false, 0};
  }

  if (batch_shape.size() == 1) {
    return {true, batch_strides.back()};
  }

  for (int i = batch_shape.size() - 2; i >= 0; --i) {
    int64_t cur = batch_strides[i];
    int64_t next = batch_strides[i + 1];
    if (cur == 0 && next == 0) {
      continue;
    }
    if (cur != next * batch_shape[i + 1]) {
      return {false, 0};
    }
  }

  return {true, batch_strides.back()};
}

int parse_non_negative_int_env(const char* env_name, int default_value) {
  const char* raw = std::getenv(env_name);
  if (raw == nullptr || *raw == '\0') {
    return default_value;
  }

  char* end = nullptr;
  long value = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || value < 0) {
    return default_value;
  }
  return static_cast<int>(value);
}

int gemm_solution_index_f32(bool batched) {
  static int single_index =
      parse_non_negative_int_env("MLX_ROCM_GEMM_F32_SOLUTION_INDEX", 0);
  static int batched_index = parse_non_negative_int_env(
      "MLX_ROCM_GEMM_F32_BATCHED_SOLUTION_INDEX", -1);
  if (!batched) {
    return single_index;
  }
  return batched_index >= 0 ? batched_index : single_index;
}

int gemm_solution_index_bf16(bool batched) {
  static int single_index =
      parse_non_negative_int_env("MLX_ROCM_GEMM_BF16_SOLUTION_INDEX", 0);
  static int batched_index = parse_non_negative_int_env(
      "MLX_ROCM_GEMM_BF16_BATCHED_SOLUTION_INDEX", -1);
  if (!batched) {
    return single_index;
  }
  return batched_index >= 0 ? batched_index : single_index;
}

void gemm_rocblas(
    rocm::CommandEncoder& encoder,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    bool b_transposed,
    int64_t ldb,
    array& out,
    const array& a,
    const array& b,
    float alpha = 1.0f,
    float beta = 0.0f) {
  // Try hipBLASLt for bf16/fp16 GEMMs -- it often picks faster kernels than
  // rocBLAS for half-precision on RDNA 3/3.5/4 and CDNA GPUs.
  if ((a.dtype() == bfloat16 || a.dtype() == float16) &&
      rocm::is_hipblaslt_available()) {
    try {
      rocm::hipblaslt_gemm(
          encoder,
          a_transposed,
          b_transposed,
          M,
          N,
          K,
          alpha,
          a,
          lda,
          b,
          ldb,
          beta,
          out,
          N, // ldc = N for row-major output
          a.dtype());
      return;
    } catch (...) {
      // hipBLASLt failed (unsupported config, etc.) -- fall through to rocBLAS.
    }
  }

  auto& device = encoder.device();
  rocblas_handle handle = device.get_rocblas_handle();

  // rocBLAS uses column-major, so we swap A and B and compute B^T * A^T = (A *
  // B)^T But since we want row-major output, we compute C = A * B by doing C^T
  // = B^T * A^T
  rocblas_operation trans_a =
      b_transposed ? rocblas_operation_transpose : rocblas_operation_none;
  rocblas_operation trans_b =
      a_transposed ? rocblas_operation_transpose : rocblas_operation_none;

  // We pass B then A (swapped) to compute C^T = B^T * A^T. The leading
  // dimensions come directly from check_transpose() for each operand.
  const int64_t ld_b = ldb;
  const int64_t ld_a = lda;
  const void* a_ptr = gpu_ptr<void>(a);
  const void* b_ptr = gpu_ptr<void>(b);
  void* out_ptr = gpu_ptr<void>(out);

  encoder.launch_kernel([&, a_ptr, b_ptr, out_ptr](hipStream_t stream) {
    encoder.device().set_rocblas_stream(stream);

    switch (a.dtype()) {
      case float32: {
        float alpha_f = alpha;
        float beta_f = beta;
        int solution_index = gemm_solution_index_f32(false);
        static std::atomic<bool> solution_valid{true};

        if (solution_index > 0 &&
            solution_valid.load(std::memory_order_relaxed)) {
          rocblas_status status = rocblas_gemm_ex(
              handle,
              trans_a,
              trans_b,
              N,
              M,
              K,
              &alpha_f,
              b_ptr,
              rocblas_datatype_f32_r,
              ld_b,
              a_ptr,
              rocblas_datatype_f32_r,
              ld_a,
              &beta_f,
              out_ptr,
              rocblas_datatype_f32_r,
              N,
              out_ptr,
              rocblas_datatype_f32_r,
              N,
              rocblas_datatype_f32_r,
              rocblas_gemm_algo_solution_index,
              solution_index,
              0);
          if (status != rocblas_status_success) {
            solution_valid.store(false, std::memory_order_relaxed);
            rocblas_sgemm(
                handle,
                trans_a,
                trans_b,
                N,
                M,
                K,
                &alpha_f,
                static_cast<const float*>(b_ptr),
                ld_b,
                static_cast<const float*>(a_ptr),
                ld_a,
                &beta_f,
                static_cast<float*>(out_ptr),
                N);
          }
        } else {
          rocblas_sgemm(
              handle,
              trans_a,
              trans_b,
              N,
              M,
              K,
              &alpha_f,
              static_cast<const float*>(b_ptr),
              ld_b,
              static_cast<const float*>(a_ptr),
              ld_a,
              &beta_f,
              static_cast<float*>(out_ptr),
              N);
        }
        break;
      }
      case float64: {
        double alpha_d = static_cast<double>(alpha);
        double beta_d = static_cast<double>(beta);
        rocblas_dgemm(
            handle,
            trans_a,
            trans_b,
            N,
            M,
            K,
            &alpha_d,
            static_cast<const double*>(b_ptr),
            ld_b,
            static_cast<const double*>(a_ptr),
            ld_a,
            &beta_d,
            static_cast<double*>(out_ptr),
            N);
        break;
      }
      case float16: {
        rocblas_half alpha_h, beta_h;
        // Convert float to rocblas_half using memcpy
        float16_t alpha_f16 = static_cast<float16_t>(alpha);
        float16_t beta_f16 = static_cast<float16_t>(beta);
        std::memcpy(&alpha_h, &alpha_f16, sizeof(rocblas_half));
        std::memcpy(&beta_h, &beta_f16, sizeof(rocblas_half));
        rocblas_hgemm(
            handle,
            trans_a,
            trans_b,
            N,
            M,
            K,
            &alpha_h,
            reinterpret_cast<const rocblas_half*>(
                static_cast<const float16_t*>(b_ptr)),
            ld_b,
            reinterpret_cast<const rocblas_half*>(
                static_cast<const float16_t*>(a_ptr)),
            ld_a,
            &beta_h,
            reinterpret_cast<rocblas_half*>(static_cast<float16_t*>(out_ptr)),
            N);
        break;
      }
      case bfloat16: {
        float alpha_f = alpha;
        float beta_f = beta;
        int solution_index = gemm_solution_index_bf16(false);
        static std::atomic<bool> solution_valid{true};

        rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
        if (solution_index > 0 &&
            solution_valid.load(std::memory_order_relaxed)) {
          algo = rocblas_gemm_algo_solution_index;
        } else {
          solution_index = 0;
        }

        rocblas_status status = rocblas_gemm_ex(
            handle,
            trans_a,
            trans_b,
            N,
            M,
            K,
            &alpha_f,
            static_cast<const bfloat16_t*>(b_ptr),
            rocblas_datatype_bf16_r,
            ld_b,
            static_cast<const bfloat16_t*>(a_ptr),
            rocblas_datatype_bf16_r,
            ld_a,
            &beta_f,
            static_cast<bfloat16_t*>(out_ptr),
            rocblas_datatype_bf16_r,
            N,
            static_cast<bfloat16_t*>(out_ptr),
            rocblas_datatype_bf16_r,
            N,
            rocblas_datatype_f32_r,
            algo,
            solution_index,
            0);
        if (status != rocblas_status_success &&
            algo == rocblas_gemm_algo_solution_index) {
          solution_valid.store(false, std::memory_order_relaxed);
          rocblas_gemm_ex(
              handle,
              trans_a,
              trans_b,
              N,
              M,
              K,
              &alpha_f,
              static_cast<const bfloat16_t*>(b_ptr),
              rocblas_datatype_bf16_r,
              ld_b,
              static_cast<const bfloat16_t*>(a_ptr),
              rocblas_datatype_bf16_r,
              ld_a,
              &beta_f,
              static_cast<bfloat16_t*>(out_ptr),
              rocblas_datatype_bf16_r,
              N,
              static_cast<bfloat16_t*>(out_ptr),
              rocblas_datatype_bf16_r,
              N,
              rocblas_datatype_f32_r,
              rocblas_gemm_algo_standard,
              0,
              0);
        }
        break;
      }
      default:
        throw std::runtime_error("Unsupported dtype for matmul on ROCm");
    }
  });
}

void gemm_strided_batched_rocblas(
    rocm::CommandEncoder& encoder,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    int64_t stride_a,
    bool b_transposed,
    int64_t ldb,
    int64_t stride_b,
    int64_t stride_c,
    int batch_count,
    array& out,
    const array& a,
    const array& b,
    float alpha = 1.0f,
    float beta = 0.0f) {
  // Try hipBLASLt for bf16/fp16 batched GEMMs.
  if ((a.dtype() == bfloat16 || a.dtype() == float16) &&
      rocm::is_hipblaslt_available()) {
    try {
      rocm::hipblaslt_gemm_batched(
          encoder,
          a_transposed,
          b_transposed,
          M,
          N,
          K,
          alpha,
          a,
          lda,
          stride_a,
          b,
          ldb,
          stride_b,
          beta,
          out,
          N, // ldc = N for row-major output
          stride_c,
          batch_count,
          a.dtype());
      return;
    } catch (...) {
      // hipBLASLt failed -- fall through to rocBLAS.
    }
  }

  auto& device = encoder.device();
  rocblas_handle handle = device.get_rocblas_handle();

  rocblas_operation trans_a =
      b_transposed ? rocblas_operation_transpose : rocblas_operation_none;
  rocblas_operation trans_b =
      a_transposed ? rocblas_operation_transpose : rocblas_operation_none;

  const int64_t ld_b = ldb;
  const int64_t ld_a = lda;
  const void* a_ptr = gpu_ptr<void>(a);
  const void* b_ptr = gpu_ptr<void>(b);
  void* out_ptr = gpu_ptr<void>(out);

  encoder.launch_kernel([&, a_ptr, b_ptr, out_ptr](hipStream_t stream) {
    encoder.device().set_rocblas_stream(stream);

    switch (a.dtype()) {
      case float32: {
        float alpha_f = alpha;
        float beta_f = beta;
        int solution_index = gemm_solution_index_f32(true);
        static std::atomic<bool> solution_valid{true};

        if (solution_index > 0 &&
            solution_valid.load(std::memory_order_relaxed)) {
          rocblas_status status = rocblas_gemm_strided_batched_ex(
              handle,
              trans_a,
              trans_b,
              N,
              M,
              K,
              &alpha_f,
              b_ptr,
              rocblas_datatype_f32_r,
              ld_b,
              stride_b,
              a_ptr,
              rocblas_datatype_f32_r,
              ld_a,
              stride_a,
              &beta_f,
              out_ptr,
              rocblas_datatype_f32_r,
              N,
              stride_c,
              out_ptr,
              rocblas_datatype_f32_r,
              N,
              stride_c,
              batch_count,
              rocblas_datatype_f32_r,
              rocblas_gemm_algo_solution_index,
              solution_index,
              0);
          if (status != rocblas_status_success) {
            solution_valid.store(false, std::memory_order_relaxed);
            rocblas_sgemm_strided_batched(
                handle,
                trans_a,
                trans_b,
                N,
                M,
                K,
                &alpha_f,
                static_cast<const float*>(b_ptr),
                ld_b,
                stride_b,
                static_cast<const float*>(a_ptr),
                ld_a,
                stride_a,
                &beta_f,
                static_cast<float*>(out_ptr),
                N,
                stride_c,
                batch_count);
          }
        } else {
          rocblas_sgemm_strided_batched(
              handle,
              trans_a,
              trans_b,
              N,
              M,
              K,
              &alpha_f,
              static_cast<const float*>(b_ptr),
              ld_b,
              stride_b,
              static_cast<const float*>(a_ptr),
              ld_a,
              stride_a,
              &beta_f,
              static_cast<float*>(out_ptr),
              N,
              stride_c,
              batch_count);
        }
        break;
      }
      case float64: {
        double alpha_d = static_cast<double>(alpha);
        double beta_d = static_cast<double>(beta);
        rocblas_dgemm_strided_batched(
            handle,
            trans_a,
            trans_b,
            N,
            M,
            K,
            &alpha_d,
            static_cast<const double*>(b_ptr),
            ld_b,
            stride_b,
            static_cast<const double*>(a_ptr),
            ld_a,
            stride_a,
            &beta_d,
            static_cast<double*>(out_ptr),
            N,
            stride_c,
            batch_count);
        break;
      }
      case float16: {
        rocblas_half alpha_h, beta_h;
        float16_t alpha_f16 = static_cast<float16_t>(alpha);
        float16_t beta_f16 = static_cast<float16_t>(beta);
        std::memcpy(&alpha_h, &alpha_f16, sizeof(rocblas_half));
        std::memcpy(&beta_h, &beta_f16, sizeof(rocblas_half));
        rocblas_hgemm_strided_batched(
            handle,
            trans_a,
            trans_b,
            N,
            M,
            K,
            &alpha_h,
            reinterpret_cast<const rocblas_half*>(
                static_cast<const float16_t*>(b_ptr)),
            ld_b,
            stride_b,
            reinterpret_cast<const rocblas_half*>(
                static_cast<const float16_t*>(a_ptr)),
            ld_a,
            stride_a,
            &beta_h,
            reinterpret_cast<rocblas_half*>(static_cast<float16_t*>(out_ptr)),
            N,
            stride_c,
            batch_count);
        break;
      }
      case bfloat16: {
        float alpha_f = alpha;
        float beta_f = beta;
        int solution_index = gemm_solution_index_bf16(true);
        static std::atomic<bool> solution_valid{true};

        rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
        if (solution_index > 0 &&
            solution_valid.load(std::memory_order_relaxed)) {
          algo = rocblas_gemm_algo_solution_index;
        } else {
          solution_index = 0;
        }

        rocblas_status status = rocblas_gemm_strided_batched_ex(
            handle,
            trans_a,
            trans_b,
            N,
            M,
            K,
            &alpha_f,
            static_cast<const bfloat16_t*>(b_ptr),
            rocblas_datatype_bf16_r,
            ld_b,
            stride_b,
            static_cast<const bfloat16_t*>(a_ptr),
            rocblas_datatype_bf16_r,
            ld_a,
            stride_a,
            &beta_f,
            static_cast<bfloat16_t*>(out_ptr),
            rocblas_datatype_bf16_r,
            N,
            stride_c,
            static_cast<bfloat16_t*>(out_ptr),
            rocblas_datatype_bf16_r,
            N,
            stride_c,
            batch_count,
            rocblas_datatype_f32_r,
            algo,
            solution_index,
            0);
        if (status != rocblas_status_success &&
            algo == rocblas_gemm_algo_solution_index) {
          solution_valid.store(false, std::memory_order_relaxed);
          rocblas_gemm_strided_batched_ex(
              handle,
              trans_a,
              trans_b,
              N,
              M,
              K,
              &alpha_f,
              static_cast<const bfloat16_t*>(b_ptr),
              rocblas_datatype_bf16_r,
              ld_b,
              stride_b,
              static_cast<const bfloat16_t*>(a_ptr),
              rocblas_datatype_bf16_r,
              ld_a,
              stride_a,
              &beta_f,
              static_cast<bfloat16_t*>(out_ptr),
              rocblas_datatype_bf16_r,
              N,
              stride_c,
              static_cast<bfloat16_t*>(out_ptr),
              rocblas_datatype_bf16_r,
              N,
              stride_c,
              batch_count,
              rocblas_datatype_f32_r,
              rocblas_gemm_algo_standard,
              0,
              0);
        }
        break;
      }
      default:
        throw std::runtime_error(
            "Unsupported dtype for batched matmul on ROCm");
    }
  });
}

void gemm_and_bias(
    rocm::CommandEncoder& encoder,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    bool b_transposed,
    int64_t ldb,
    array& out,
    const array& a,
    const array& b,
    float alpha = 1.0f,
    float beta = 0.0f) {
  // Check and collapse batch dimensions
  auto [batch_shape, a_batch_strides, b_batch_strides] = collapse_batches(a, b);

  auto batch_count = out.size() / (M * N);

  // Collapse batches into M if needed
  if (batch_count > 1 && !a_transposed && batch_shape.size() == 1 &&
      a.strides()[a.ndim() - 2] == K && a_batch_strides.back() == M * K &&
      b_batch_strides.back() == 0) {
    M *= batch_shape.back();
    batch_count = 1;

    a_batch_strides = {0};
    b_batch_strides = {0};
    batch_shape = {1};
  }

  // Use GEMV when possible
  if (rocm::can_use_gemv(M, N, K, a_transposed, b_transposed)) {
    rocm::gemv(
        a,
        b,
        out,
        M,
        N,
        K,
        batch_count,
        batch_shape,
        a_batch_strides,
        b_batch_strides,
        encoder);
    return;
  }

  // Check if rocBLAS is available
  bool use_rocblas = encoder.device().is_rocblas_available();
  auto [a_uniform_batch, a_uniform_stride] =
      get_uniform_batch_stride(batch_shape, a_batch_strides);
  auto [b_uniform_batch, b_uniform_stride] =
      get_uniform_batch_stride(batch_shape, b_batch_strides);

  if (batch_count == 1) {
    // Simple single GEMM
    if (use_rocblas) {
      gemm_rocblas(
          encoder,
          M,
          N,
          K,
          a_transposed,
          lda,
          b_transposed,
          ldb,
          out,
          a,
          b,
          alpha,
          beta);
    } else {
      // Use naive GEMM fallback
      rocm::naive_gemm(
          encoder,
          a,
          b,
          out,
          M,
          N,
          K,
          a_transposed,
          lda,
          b_transposed,
          ldb,
          alpha,
          beta);
    }
  } else if (a_uniform_batch && b_uniform_batch) {
    // Use strided batched GEMM for uniform batches
    if (use_rocblas) {
      gemm_strided_batched_rocblas(
          encoder,
          M,
          N,
          K,
          a_transposed,
          lda,
          a_uniform_stride,
          b_transposed,
          ldb,
          b_uniform_stride,
          M * N,
          batch_count,
          out,
          a,
          b,
          alpha,
          beta);
    } else {
      // Use naive batched GEMM fallback
      rocm::naive_gemm_batched(
          encoder,
          a,
          b,
          out,
          M,
          N,
          K,
          a_transposed,
          lda,
          a_uniform_stride,
          b_transposed,
          ldb,
          b_uniform_stride,
          M * N,
          batch_count,
          alpha,
          beta);
    }
  } else {
    // Fallback: loop over batches for non-uniform strides
    if (use_rocblas) {
      const void* a_ptr_base = gpu_ptr<void>(a);
      const void* b_ptr_base = gpu_ptr<void>(b);
      void* out_ptr_base = gpu_ptr<void>(out);
      for (int64_t batch = 0; batch < batch_count; ++batch) {
        int64_t a_offset = 0, b_offset = 0;
        int64_t batch_idx = batch;
        for (int i = batch_shape.size() - 1; i >= 0; --i) {
          int64_t idx = batch_idx % batch_shape[i];
          batch_idx /= batch_shape[i];
          a_offset += idx * a_batch_strides[i];
          b_offset += idx * b_batch_strides[i];
        }

        encoder.launch_kernel([&,
                               a_offset,
                               b_offset,
                               batch,
                               a_ptr_base,
                               b_ptr_base,
                               out_ptr_base](hipStream_t stream) {
          auto& device = encoder.device();
          device.set_rocblas_stream(stream);
          rocblas_handle handle = device.get_rocblas_handle();

          rocblas_operation trans_a = b_transposed ? rocblas_operation_transpose
                                                   : rocblas_operation_none;
          rocblas_operation trans_b = a_transposed ? rocblas_operation_transpose
                                                   : rocblas_operation_none;

          const int64_t ld_b = ldb;
          const int64_t ld_a = lda;

          switch (a.dtype()) {
            case float32: {
              float alpha_f = alpha, beta_f = beta;
              rocblas_sgemm(
                  handle,
                  trans_a,
                  trans_b,
                  N,
                  M,
                  K,
                  &alpha_f,
                  static_cast<const float*>(b_ptr_base) + b_offset,
                  ld_b,
                  static_cast<const float*>(a_ptr_base) + a_offset,
                  ld_a,
                  &beta_f,
                  static_cast<float*>(out_ptr_base) + batch * M * N,
                  N);
              break;
            }
            case float64: {
              double alpha_d = static_cast<double>(alpha);
              double beta_d = static_cast<double>(beta);
              rocblas_dgemm(
                  handle,
                  trans_a,
                  trans_b,
                  N,
                  M,
                  K,
                  &alpha_d,
                  static_cast<const double*>(b_ptr_base) + b_offset,
                  ld_b,
                  static_cast<const double*>(a_ptr_base) + a_offset,
                  ld_a,
                  &beta_d,
                  static_cast<double*>(out_ptr_base) + batch * M * N,
                  N);
              break;
            }
            case float16: {
              rocblas_half alpha_h, beta_h;
              float16_t alpha_f16 = static_cast<float16_t>(alpha);
              float16_t beta_f16 = static_cast<float16_t>(beta);
              std::memcpy(&alpha_h, &alpha_f16, sizeof(rocblas_half));
              std::memcpy(&beta_h, &beta_f16, sizeof(rocblas_half));
              rocblas_hgemm(
                  handle,
                  trans_a,
                  trans_b,
                  N,
                  M,
                  K,
                  &alpha_h,
                  reinterpret_cast<const rocblas_half*>(
                      static_cast<const float16_t*>(b_ptr_base) + b_offset),
                  ld_b,
                  reinterpret_cast<const rocblas_half*>(
                      static_cast<const float16_t*>(a_ptr_base) + a_offset),
                  ld_a,
                  &beta_h,
                  reinterpret_cast<rocblas_half*>(
                      static_cast<float16_t*>(out_ptr_base) + batch * M * N),
                  N);
              break;
            }
            case bfloat16: {
              float alpha_f = alpha;
              float beta_f = beta;
              auto* out_ptr =
                  static_cast<bfloat16_t*>(out_ptr_base) + batch * M * N;
              rocblas_gemm_ex(
                  handle,
                  trans_a,
                  trans_b,
                  N,
                  M,
                  K,
                  &alpha_f,
                  static_cast<const bfloat16_t*>(b_ptr_base) + b_offset,
                  rocblas_datatype_bf16_r,
                  ld_b,
                  static_cast<const bfloat16_t*>(a_ptr_base) + a_offset,
                  rocblas_datatype_bf16_r,
                  ld_a,
                  &beta_f,
                  out_ptr,
                  rocblas_datatype_bf16_r,
                  N,
                  out_ptr,
                  rocblas_datatype_bf16_r,
                  N,
                  rocblas_datatype_f32_r,
                  rocblas_gemm_algo_standard,
                  0,
                  0);
              break;
            }
            default:
              throw std::runtime_error(
                  "Unsupported dtype for non-uniform batched matmul on ROCm");
          }
        });
      }
    } else {
      // Use naive GEMM for each batch when rocBLAS is not available
      // This is less efficient but provides correctness
      for (int64_t batch = 0; batch < batch_count; ++batch) {
        int64_t a_offset = 0, b_offset = 0;
        int64_t batch_idx = batch;
        for (int i = batch_shape.size() - 1; i >= 0; --i) {
          int64_t idx = batch_idx % batch_shape[i];
          batch_idx /= batch_shape[i];
          a_offset += idx * a_batch_strides[i];
          b_offset += idx * b_batch_strides[i];
        }

        // Use naive GEMM with explicit offsets
        rocm::naive_gemm_with_offset(
            encoder,
            a,
            b,
            out,
            M,
            N,
            K,
            a_transposed,
            lda,
            a_offset,
            b_transposed,
            ldb,
            b_offset,
            batch * M * N,
            alpha,
            beta);
      }
    }
  }
}

} // namespace

void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  assert(inputs.size() == 2);
  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];

  // Return 0s if either input is empty.
  if (a_pre.size() == 0 || b_pre.size() == 0) {
    array zero(0, a_pre.dtype());
    encoder.add_temporary(zero);
    fill_gpu(zero, out, s);
    return;
  }

  out.set_data(mlx::core::rocm::malloc_async(out.nbytes(), encoder));

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int K = a_pre.shape(-1);

  auto [a_transposed, lda, a] = check_transpose(encoder, s, a_pre);
  auto [b_transposed, ldb, b] = check_transpose(encoder, s, b_pre);

  gemm_and_bias(
      encoder, M, N, K, a_transposed, lda, b_transposed, ldb, out, a, b);
}

void AddMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  assert(inputs.size() == 3);
  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];
  auto c = inputs[2];

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int K = a_pre.shape(-1);

  auto [a_transposed, lda, a] = check_transpose(encoder, s, a_pre);
  auto [b_transposed, ldb, b] = check_transpose(encoder, s, b_pre);

  // Copy C into out only when beta uses it.
  if (beta_ != 0.0f) {
    copy_gpu(c, out, CopyType::General, s);
  } else {
    out.set_data(mlx::core::rocm::malloc_async(out.nbytes(), encoder));
  }

  // Check if rocBLAS is available
  if (encoder.device().is_rocblas_available()) {
    // Do GEMM with alpha and beta
    gemm_rocblas(
        encoder,
        M,
        N,
        K,
        a_transposed,
        lda,
        b_transposed,
        ldb,
        out,
        a,
        b,
        alpha_,
        beta_);
  } else {
    // Use naive GEMM fallback
    rocm::naive_gemm(
        encoder,
        a,
        b,
        out,
        M,
        N,
        K,
        a_transposed,
        lda,
        b_transposed,
        ldb,
        alpha_,
        beta_);
  }
}

// Sorted-rhs gather_mm (M==1): collapse consecutive tokens that share an expert
// into dense GEMMs. Matches CUDA gather_mm_rhs intent for MoE prefill/train.
static bool try_sorted_rhs_gather_mm(
    rocm::CommandEncoder& encoder,
    const array& a,
    const array& b,
    const array& rhs_indices,
    array& out,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    bool b_transposed,
    int64_t ldb) {
  (void)lda;
  if (a_transposed) {
    return false;
  }
  if (!rocm::is_hipblaslt_available()) {
    return false;
  }

  const int batch = static_cast<int>(out.size() / static_cast<size_t>(N));
  if (batch <= 0) {
    return false;
  }

  // Require a to be a contiguous batch of 1xK rows: shape (..., 1, K).
  if (a.shape(-2) != 1 || a.shape(-1) != K) {
    return false;
  }
  if (a.strides()[a.ndim() - 1] != 1 || a.strides()[a.ndim() - 2] != K) {
    return false;
  }

  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(rhs_indices);
  encoder.set_output_array(out);

  std::vector<uint32_t> rhs(static_cast<size_t>(batch));
  CHECK_HIP_ERROR(hipMemcpy(
      rhs.data(),
      gpu_ptr<const uint32_t>(rhs_indices),
      rhs.size() * sizeof(uint32_t),
      hipMemcpyDeviceToHost));

  mlx::core::Shape b_batch_shape{b.shape().begin(), b.shape().end() - 2};
  mlx::core::Strides b_batch_strides{b.strides().begin(), b.strides().end() - 2};

  auto expert_offset = [&](uint32_t expert) -> int64_t {
    if (b_batch_shape.empty()) {
      return 0;
    }
    if (b_batch_shape.size() == 1) {
      return static_cast<int64_t>(expert) * b_batch_strides[0];
    }
    int64_t off = 0;
    int64_t idx = static_cast<int64_t>(expert);
    for (int d = static_cast<int>(b_batch_shape.size()) - 1; d >= 0; --d) {
      int64_t coord = idx % b_batch_shape[d];
      idx /= b_batch_shape[d];
      off += coord * b_batch_strides[d];
    }
    return off;
  };

  const size_t esize = size_of(a.dtype());
  const char* a_base = static_cast<const char*>(gpu_ptr<const void>(a));
  const char* b_base = static_cast<const char*>(gpu_ptr<const void>(b));
  char* out_base = static_cast<char*>(gpu_ptr<void>(out));

  int start = 0;
  while (start < batch) {
    uint32_t e = rhs[static_cast<size_t>(start)];
    int end = start + 1;
    while (end < batch && rhs[static_cast<size_t>(end)] == e) {
      ++end;
    }
    int Mseg = end - start;
    const void* a_ptr =
        a_base + static_cast<size_t>(start) * static_cast<size_t>(K) * esize;
    const void* b_ptr =
        b_base + static_cast<size_t>(expert_offset(e)) * esize;
    void* c_ptr =
        out_base + static_cast<size_t>(start) * static_cast<size_t>(N) * esize;

    rocm::hipblaslt_gemm_ptrs(
        encoder,
        /*transpose_a=*/false,
        b_transposed,
        Mseg,
        N,
        K,
        1.0f,
        a_ptr,
        /*lda=*/K,
        b_ptr,
        static_cast<int>(ldb),
        0.0f,
        c_ptr,
        N,
        a.dtype());
    start = end;
  }
  return true;
}

void GatherMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  assert(inputs.size() == 4);
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto& lhs_indices = inputs[2];
  auto& rhs_indices = inputs[3];

  // Return 0s if either input is empty.
  if (a.size() == 0 || b.size() == 0) {
    array zero(0, a.dtype());
    encoder.add_temporary(zero);
    fill_gpu(zero, out, s);
    return;
  }

  out.set_data(mlx::core::rocm::malloc_async(out.nbytes(), encoder));

  // Extract shapes from inputs.
  int M = a.shape(-2);
  int N = b.shape(-1);
  int K = a.shape(-1);

  auto [transposed_a, lda, a_] = check_transpose(encoder, s, a);
  auto [transposed_b, ldb, b_] = check_transpose(encoder, s, b);

  // Prefer sorted-rhs dense segments (MoE) over per-token gemv.
  if (M == 1 && right_sorted_) {
    if (try_sorted_rhs_gather_mm(
            encoder,
            a_,
            b_,
            rhs_indices,
            out,
            N,
            K,
            transposed_a,
            lda,
            transposed_b,
            ldb)) {
      return;
    }
  }

  auto use_gemv = rocm::can_use_gemv(M, N, K, transposed_a, transposed_b);

  if (M == 1 && use_gemv) {
    rocm::gather_mv(b_, a_, rhs_indices, lhs_indices, out, N, K, encoder);
    return;
  }

  if (N == 1 && use_gemv) {
    rocm::gather_mv(a_, b_, lhs_indices, rhs_indices, out, M, K, encoder);
    return;
  }

  // Keep gather indices on device and resolve per-batch matrix offsets inside
  // the kernel to avoid host synchronization.
  rocm::naive_gemm_gather(
      encoder,
      a_,
      b_,
      lhs_indices,
      rhs_indices,
      out,
      M,
      N,
      K,
      transposed_a,
      lda,
      transposed_b,
      ldb,
      1.0f,
      0.0f);
}

// SegmentedMM: out[i] = A[:, k0:k1] @ B[k0:k1, :] for each segment [k0, k1).
// Used by gather_mm weight VJP when indices are sorted (MoE training).
void SegmentedMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  assert(inputs.size() == 3);
  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];
  auto& segments_pre = inputs[2];

  if (out.size() == 0 || a_pre.size() == 0 || b_pre.size() == 0) {
    array zero(0, a_pre.dtype());
    encoder.add_temporary(zero);
    fill_gpu(zero, out, s);
    return;
  }

  out.set_data(mlx::core::rocm::malloc_async(out.nbytes(), encoder));

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int num_segments = static_cast<int>(segments_pre.size() / 2);

  auto [a_transposed, lda, a] = check_transpose(encoder, s, a_pre);
  auto [b_transposed, ldb, b] = check_transpose(encoder, s, b_pre);

  array segments = segments_pre;
  if (!segments_pre.flags().row_contiguous) {
    segments = contiguous_copy_gpu(segments_pre, s);
    encoder.add_temporary(segments);
  }

  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(segments);
  encoder.set_output_array(out);

  // Segments are small (typically #experts). Host-side split lets us dispatch
  // one tuned hipBLASLt GEMM per segment instead of a naive gather kernel.
  std::vector<uint32_t> segs(static_cast<size_t>(num_segments) * 2);
  CHECK_HIP_ERROR(hipMemcpy(
      segs.data(),
      gpu_ptr<const uint32_t>(segments),
      segs.size() * sizeof(uint32_t),
      hipMemcpyDeviceToHost));

  const size_t esize = size_of(a.dtype());
  const char* a_base = static_cast<const char*>(gpu_ptr<const void>(a));
  const char* b_base = static_cast<const char*>(gpu_ptr<const void>(b));
  char* out_base = static_cast<char*>(gpu_ptr<void>(out));

  const int64_t a_k_stride = a.strides()[a.ndim() - 1];
  const int64_t b_k_stride = b.strides()[b.ndim() - 2];
  const int64_t out_stride = static_cast<int64_t>(M) * N;

  const bool use_lt = rocm::is_hipblaslt_available();

  for (int i = 0; i < num_segments; ++i) {
    uint32_t k0 = segs[static_cast<size_t>(2 * i)];
    uint32_t k1 = segs[static_cast<size_t>(2 * i + 1)];
    void* c_ptr = out_base + static_cast<size_t>(i) * out_stride * esize;
    if (k1 <= k0) {
      size_t bytes = static_cast<size_t>(out_stride) * esize;
      encoder.launch_kernel([c_ptr, bytes](hipStream_t stream) {
        (void)hipMemsetAsync(c_ptr, 0, bytes, stream);
      });
      continue;
    }
    int Kseg = static_cast<int>(k1 - k0);
    const void* a_ptr = a_base +
        static_cast<size_t>(k0) * static_cast<size_t>(a_k_stride) * esize;
    const void* b_ptr = b_base +
        static_cast<size_t>(k0) * static_cast<size_t>(b_k_stride) * esize;

    if (use_lt) {
      rocm::hipblaslt_gemm_ptrs(
          encoder,
          a_transposed,
          b_transposed,
          M,
          N,
          Kseg,
          1.0f,
          a_ptr,
          static_cast<int>(lda),
          b_ptr,
          static_cast<int>(ldb),
          0.0f,
          c_ptr,
          N,
          a.dtype());
    } else {
      rocm::naive_gemm_with_offset(
          encoder,
          a,
          b,
          out,
          M,
          N,
          Kseg,
          a_transposed,
          lda,
          static_cast<int64_t>(k0) * a_k_stride,
          b_transposed,
          ldb,
          static_cast<int64_t>(k0) * b_k_stride,
          static_cast<int64_t>(i) * out_stride,
          1.0f,
          0.0f);
    }
  }
}

} // namespace mlx::core
