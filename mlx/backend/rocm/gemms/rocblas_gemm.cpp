// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/gemms/rocblas_gemm.h"
#include "mlx/backend/rocm/device.h"

#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

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
  
  encoder.launch_kernel([&](hipStream_t stream) {
    rocblas_handle handle = encoder.device().get_rocblas_handle();
    rocblas_set_stream(handle, stream);
    
    rocblas_operation op_a = to_rocblas_op(transpose_a);
    rocblas_operation op_b = to_rocblas_op(transpose_b);
    
    switch (dtype) {
      case float32: {
        float alpha_f = alpha;
        float beta_f = beta;
        rocblas_sgemm(
            handle,
            op_b,  // Note: rocBLAS uses column-major, so we swap a and b
            op_a,
            N, M, K,
            &alpha_f,
            b.data<float>(), ldb,
            a.data<float>(), lda,
            &beta_f,
            c.data<float>(), ldc);
        break;
      }
      case float16: {
        rocblas_half alpha_h;
        rocblas_half beta_h;
        // Convert float to half
        alpha_h = rocblas_half(alpha);
        beta_h = rocblas_half(beta);
        rocblas_hgemm(
            handle,
            op_b,
            op_a,
            N, M, K,
            &alpha_h,
            reinterpret_cast<const rocblas_half*>(b.data<uint16_t>()), ldb,
            reinterpret_cast<const rocblas_half*>(a.data<uint16_t>()), lda,
            &beta_h,
            reinterpret_cast<rocblas_half*>(c.data<uint16_t>()), ldc);
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
  
  encoder.launch_kernel([&](hipStream_t stream) {
    rocblas_handle handle = encoder.device().get_rocblas_handle();
    rocblas_set_stream(handle, stream);
    
    rocblas_operation op_a = to_rocblas_op(transpose_a);
    rocblas_operation op_b = to_rocblas_op(transpose_b);
    
    switch (dtype) {
      case float32: {
        float alpha_f = alpha;
        float beta_f = beta;
        rocblas_sgemm_strided_batched(
            handle,
            op_b,
            op_a,
            N, M, K,
            &alpha_f,
            b.data<float>(), ldb, stride_b,
            a.data<float>(), lda, stride_a,
            &beta_f,
            c.data<float>(), ldc, stride_c,
            batch_count);
        break;
      }
      case float16: {
        rocblas_half alpha_h;
        rocblas_half beta_h;
        alpha_h = rocblas_half(alpha);
        beta_h = rocblas_half(beta);
        rocblas_hgemm_strided_batched(
            handle,
            op_b,
            op_a,
            N, M, K,
            &alpha_h,
            reinterpret_cast<const rocblas_half*>(b.data<uint16_t>()), ldb, stride_b,
            reinterpret_cast<const rocblas_half*>(a.data<uint16_t>()), lda, stride_a,
            &beta_h,
            reinterpret_cast<rocblas_half*>(c.data<uint16_t>()), ldc, stride_c,
            batch_count);
        break;
      }
      default:
        throw std::runtime_error("Unsupported dtype for rocBLAS batched GEMM");
    }
  });
}

} // namespace mlx::core::rocm
