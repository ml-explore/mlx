// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/gemms/rocblas_gemm.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/gemms/naive_gemm.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/types/half_types.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace mlx::core::rocm {

namespace {

rocblas_operation to_rocblas_op(bool transpose) {
  return transpose ? rocblas_operation_transpose : rocblas_operation_none;
}

rocblas_datatype to_rocblas_dtype(Dtype dtype) {
  switch (dtype) {
    case float32:
      return rocblas_datatype_f32_r;
    case float16:
      return rocblas_datatype_f16_r;
    case bfloat16:
      return rocblas_datatype_bf16_r;
    default:
      throw std::runtime_error("Unsupported dtype for rocBLAS GEMM");
  }
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

} // namespace

void rocblas_gemm(
    CommandEncoder& encoder,
    bool transpose_a,
    bool transpose_b,
    int M,
    int N,
    int K,
    float alpha,
    const array& a,
    int lda,
    const array& b,
    int ldb,
    float beta,
    array& c,
    int ldc,
    Dtype dtype) {
  // Check if rocBLAS is available
  if (!encoder.device().is_rocblas_available()) {
    // Use naive GEMM fallback
    naive_gemm(
        encoder,
        a,
        b,
        c,
        M,
        N,
        K,
        transpose_a,
        lda,
        transpose_b,
        ldb,
        alpha,
        beta);
    return;
  }

  const void* a_ptr = gpu_ptr<void>(a);
  const void* b_ptr = gpu_ptr<void>(b);
  void* c_ptr = gpu_ptr<void>(c);

  encoder.launch_kernel([&, a_ptr, b_ptr, c_ptr](hipStream_t stream) {
    encoder.device().set_rocblas_stream(stream);
    rocblas_handle handle = encoder.device().get_rocblas_handle();

    rocblas_operation op_a = to_rocblas_op(transpose_a);
    rocblas_operation op_b = to_rocblas_op(transpose_b);

    switch (dtype) {
      case float32: {
        float alpha_f = alpha;
        float beta_f = beta;
        int solution_index = gemm_solution_index_f32(false);
        static std::atomic<bool> solution_valid{true};

        if (solution_index > 0 &&
            solution_valid.load(std::memory_order_relaxed)) {
          rocblas_status status = rocblas_gemm_ex(
              handle,
              op_b,
              op_a,
              N,
              M,
              K,
              &alpha_f,
              b_ptr,
              rocblas_datatype_f32_r,
              ldb,
              a_ptr,
              rocblas_datatype_f32_r,
              lda,
              &beta_f,
              c_ptr,
              rocblas_datatype_f32_r,
              ldc,
              c_ptr,
              rocblas_datatype_f32_r,
              ldc,
              rocblas_datatype_f32_r,
              rocblas_gemm_algo_solution_index,
              solution_index,
              0);
          if (status != rocblas_status_success) {
            solution_valid.store(false, std::memory_order_relaxed);
            rocblas_sgemm(
                handle,
                op_b,
                op_a,
                N,
                M,
                K,
                &alpha_f,
                static_cast<const float*>(b_ptr),
                ldb,
                static_cast<const float*>(a_ptr),
                lda,
                &beta_f,
                static_cast<float*>(c_ptr),
                ldc);
          }
        } else {
          rocblas_sgemm(
              handle,
              op_b,
              op_a,
              N,
              M,
              K,
              &alpha_f,
              static_cast<const float*>(b_ptr),
              ldb,
              static_cast<const float*>(a_ptr),
              lda,
              &beta_f,
              static_cast<float*>(c_ptr),
              ldc);
        }
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
            op_b,
            op_a,
            N,
            M,
            K,
            &alpha_h,
            reinterpret_cast<const rocblas_half*>(
                static_cast<const uint16_t*>(b_ptr)),
            ldb,
            reinterpret_cast<const rocblas_half*>(
                static_cast<const uint16_t*>(a_ptr)),
            lda,
            &beta_h,
            reinterpret_cast<rocblas_half*>(static_cast<uint16_t*>(c_ptr)),
            ldc);
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
            op_b,
            op_a,
            N,
            M,
            K,
            &alpha_f,
            b_ptr,
            rocblas_datatype_bf16_r,
            ldb,
            a_ptr,
            rocblas_datatype_bf16_r,
            lda,
            &beta_f,
            c_ptr,
            rocblas_datatype_bf16_r,
            ldc,
            c_ptr,
            rocblas_datatype_bf16_r,
            ldc,
            rocblas_datatype_f32_r,
            algo,
            solution_index,
            0);
        if (status != rocblas_status_success &&
            algo == rocblas_gemm_algo_solution_index) {
          solution_valid.store(false, std::memory_order_relaxed);
          rocblas_gemm_ex(
              handle,
              op_b,
              op_a,
              N,
              M,
              K,
              &alpha_f,
              b_ptr,
              rocblas_datatype_bf16_r,
              ldb,
              a_ptr,
              rocblas_datatype_bf16_r,
              lda,
              &beta_f,
              c_ptr,
              rocblas_datatype_bf16_r,
              ldc,
              c_ptr,
              rocblas_datatype_bf16_r,
              ldc,
              rocblas_datatype_f32_r,
              rocblas_gemm_algo_standard,
              0,
              0);
        }
        break;
      }
      default:
        throw std::runtime_error("Unsupported dtype for rocBLAS GEMM");
    }
  });
}

void rocblas_gemm_batched(
    CommandEncoder& encoder,
    bool transpose_a,
    bool transpose_b,
    int M,
    int N,
    int K,
    float alpha,
    const array& a,
    int lda,
    int64_t stride_a,
    const array& b,
    int ldb,
    int64_t stride_b,
    float beta,
    array& c,
    int ldc,
    int64_t stride_c,
    int batch_count,
    Dtype dtype) {
  // Check if rocBLAS is available
  if (!encoder.device().is_rocblas_available()) {
    // Use naive batched GEMM fallback
    naive_gemm_batched(
        encoder,
        a,
        b,
        c,
        M,
        N,
        K,
        transpose_a,
        lda,
        stride_a,
        transpose_b,
        ldb,
        stride_b,
        stride_c,
        batch_count,
        alpha,
        beta);
    return;
  }

  const void* a_ptr = gpu_ptr<void>(a);
  const void* b_ptr = gpu_ptr<void>(b);
  void* c_ptr = gpu_ptr<void>(c);

  encoder.launch_kernel([&, a_ptr, b_ptr, c_ptr](hipStream_t stream) {
    encoder.device().set_rocblas_stream(stream);
    rocblas_handle handle = encoder.device().get_rocblas_handle();

    rocblas_operation op_a = to_rocblas_op(transpose_a);
    rocblas_operation op_b = to_rocblas_op(transpose_b);

    switch (dtype) {
      case float32: {
        float alpha_f = alpha;
        float beta_f = beta;
        int solution_index = gemm_solution_index_f32(true);
        static std::atomic<bool> solution_valid{true};

        if (solution_index > 0 &&
            solution_valid.load(std::memory_order_relaxed)) {
          rocblas_status status = rocblas_gemm_strided_batched_ex(
              handle,
              op_b,
              op_a,
              N,
              M,
              K,
              &alpha_f,
              b_ptr,
              rocblas_datatype_f32_r,
              ldb,
              stride_b,
              a_ptr,
              rocblas_datatype_f32_r,
              lda,
              stride_a,
              &beta_f,
              c_ptr,
              rocblas_datatype_f32_r,
              ldc,
              stride_c,
              c_ptr,
              rocblas_datatype_f32_r,
              ldc,
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
                op_b,
                op_a,
                N,
                M,
                K,
                &alpha_f,
                static_cast<const float*>(b_ptr),
                ldb,
                stride_b,
                static_cast<const float*>(a_ptr),
                lda,
                stride_a,
                &beta_f,
                static_cast<float*>(c_ptr),
                ldc,
                stride_c,
                batch_count);
          }
        } else {
          rocblas_sgemm_strided_batched(
              handle,
              op_b,
              op_a,
              N,
              M,
              K,
              &alpha_f,
              static_cast<const float*>(b_ptr),
              ldb,
              stride_b,
              static_cast<const float*>(a_ptr),
              lda,
              stride_a,
              &beta_f,
              static_cast<float*>(c_ptr),
              ldc,
              stride_c,
              batch_count);
        }
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
            op_b,
            op_a,
            N,
            M,
            K,
            &alpha_h,
            reinterpret_cast<const rocblas_half*>(
                static_cast<const uint16_t*>(b_ptr)),
            ldb,
            stride_b,
            reinterpret_cast<const rocblas_half*>(
                static_cast<const uint16_t*>(a_ptr)),
            lda,
            stride_a,
            &beta_h,
            reinterpret_cast<rocblas_half*>(static_cast<uint16_t*>(c_ptr)),
            ldc,
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
            op_b,
            op_a,
            N,
            M,
            K,
            &alpha_f,
            b_ptr,
            rocblas_datatype_bf16_r,
            ldb,
            stride_b,
            a_ptr,
            rocblas_datatype_bf16_r,
            lda,
            stride_a,
            &beta_f,
            c_ptr,
            rocblas_datatype_bf16_r,
            ldc,
            stride_c,
            c_ptr,
            rocblas_datatype_bf16_r,
            ldc,
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
              op_b,
              op_a,
              N,
              M,
              K,
              &alpha_f,
              b_ptr,
              rocblas_datatype_bf16_r,
              ldb,
              stride_b,
              a_ptr,
              rocblas_datatype_bf16_r,
              lda,
              stride_a,
              &beta_f,
              c_ptr,
              rocblas_datatype_bf16_r,
              ldc,
              stride_c,
              c_ptr,
              rocblas_datatype_bf16_r,
              ldc,
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
        throw std::runtime_error("Unsupported dtype for rocBLAS batched GEMM");
    }
  });
}

} // namespace mlx::core::rocm
