// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/quantized/cublas_qqmm.h"

#include <fmt/format.h>
#include "mlx/backend/cuda/cublas_utils.h"

#include "mlx/backend/cuda/device.h"
#include "mlx/dtype_utils.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

// Currently cublas supports only mxfp8 and nvfp4
// quantization modes for block scaled quantization
cudaDataType_t qmode_to_cublas_scale_dtype(std::string mode) {
  if (mode == "mxfp8") {
    return CUDA_R_8F_UE8M0;
  } else if (mode == "nvfp4") {
    return CUDA_R_8F_UE4M3;
  } else {
    throw std::runtime_error(
        fmt::format("Unsupported quantization mode in CublasQQMM: {}.", mode));
  }
}

cudaDataType_t qmode_to_cublas_dtype(std::string mode) {
  if (mode == "mxfp8") {
    return CUDA_R_8F_E4M3;
  } else if (mode == "nvfp4") {
    return CUDA_R_4F_E2M1;
  } else {
    throw std::runtime_error(
        fmt::format("Unsupported quantization mode in CublasQQMM: {}.", mode));
  }
}

cublasLtMatmulMatrixScale_t qmode_to_cublas_scale_mode(std::string mode) {
  if (mode == "mxfp8") {
    return CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
  } else if (mode == "nvfp4") {
    return CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  } else {
    throw std::runtime_error(
        fmt::format("Unsupported quantization mode in CublasQQMM: {}.", mode));
  }
}

} // namespace

CublasQQMM::CublasQQMM(
    cu::Device& device,
    bool a_transposed,
    uint64_t a_rows,
    uint64_t a_cols,
    int64_t lda,
    bool b_transposed,
    uint64_t b_rows,
    uint64_t b_cols,
    int64_t ldb,
    int32_t batch_count,
    int64_t a_batch_stride,
    int64_t b_batch_stride,
    Dtype out_dtype,
    std::string qmode) {
  // The compute type must be CUBLAS_COMPUTE_32F.
  // The scale type must be CUDA_R_32F.
  cudaDataType_t scale_type = CUDA_R_32F;
  cublasComputeType_t gemm_compute_type = CUBLAS_COMPUTE_32F;
  cudaDataType_t output_type =
      cublas_utils::dtype_to_cublas_type(out_dtype, "CublasQQMM");
  cudaDataType_t data_type = qmode_to_cublas_dtype(qmode);
  quantization_mode_ = std::string(qmode);

  init_base(
      device,
      scale_type,
      gemm_compute_type,
      data_type,
      output_type,
      a_transposed,
      a_rows,
      a_cols,
      lda,
      b_transposed,
      b_rows,
      b_cols,
      ldb,
      batch_count,
      a_batch_stride,
      b_batch_stride);

  a_scale_mode_ = qmode_to_cublas_scale_mode(qmode);
  b_scale_mode_ = qmode_to_cublas_scale_mode(qmode);

  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
      &a_scale_mode_,
      sizeof(a_scale_mode_)));
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
      &b_scale_mode_,
      sizeof(b_scale_mode_)));
}

CublasQQMM::CublasQQMM(
    cu::Device& device,
    bool a_transposed,
    uint64_t a_rows,
    uint64_t a_cols,
    int64_t lda,
    bool b_transposed,
    uint64_t b_rows,
    uint64_t b_cols,
    int64_t ldb,
    int64_t ldc,
    int32_t batch_count,
    int64_t a_batch_stride,
    int64_t b_batch_stride,
    int64_t c_batch_stride,
    Dtype out_dtype,
    std::string qmode)
    : CublasQQMM(
          device,
          a_transposed,
          a_rows,
          a_cols,
          lda,
          b_transposed,
          b_rows,
          b_cols,
          ldb,
          batch_count,
          a_batch_stride,
          b_batch_stride,
          out_dtype,
          qmode) {
  auto type = cublas_utils::dtype_to_cublas_type(
      out_dtype, "CublasQQMM"); // must match the output type
  c_desc_ = cublas_utils::create_matrix_layout(
      type,
      b_transposed ? b_rows : b_cols,
      a_transposed ? a_cols : a_rows,
      false,
      ldc,
      batch_count,
      c_batch_stride);
}

void CublasQQMM::run(
    cu::CommandEncoder& encoder,
    array& out,
    const array& a,
    const array& b,
    const array& a_scale,
    const array& b_scale,
    float alpha) {
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(a_scale);
  encoder.set_input_array(b_scale);
  encoder.set_output_array(out);

  execute(
      encoder,
      gpu_ptr<void>(out),
      gpu_ptr<void>(a),
      gpu_ptr<void>(b),
      gpu_ptr<void>(a_scale),
      gpu_ptr<void>(b_scale),
      nullptr,
      alpha);
}

void CublasQQMM::execute(
    cu::CommandEncoder& encoder,
    void* out,
    const void* a,
    const void* b,
    const void* a_scale,
    const void* b_scale,
    const void* c,
    float alpha /* = 1 */,
    float beta /* = 0 */) {
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
      &b_scale,
      sizeof(b_scale)));
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
      &a_scale,
      sizeof(a_scale)));

  const void* alpha_ptr = &alpha;
  const void* beta_ptr = &beta;

  execute_matmul(encoder, out, a, b, c, alpha_ptr, beta_ptr);
}

} // namespace mlx::core
