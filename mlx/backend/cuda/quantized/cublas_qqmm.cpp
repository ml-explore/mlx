// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/quantized/cublas_qqmm.h"

#include <fmt/format.h>
#include "mlx/backend/cuda/cublas_utils.h"

#include "mlx/backend/cuda/device.h"
#include "mlx/dtype_utils.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

struct QuantModeConfig {
  cudaDataType_t data_type;
  cudaDataType_t scale_dtype;
  cublasLtMatmulMatrixScale_t scale_mode;
};

QuantModeConfig get_quant_mode_config(const std::string& mode) {
  if (mode == "mxfp8") {
    return {
        CUDA_R_8F_E4M3,
        CUDA_R_8F_UE8M0,
        CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0};
  } else if (mode == "nvfp4") {
    return {
        CUDA_R_4F_E2M1,
        CUDA_R_8F_UE4M3,
        CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3};
  }
  throw std::runtime_error(
      fmt::format("Unsupported quantization mode in CublasQQMM: {}.", mode));
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
    const std::string& qmode) {
  auto config = get_quant_mode_config(qmode);

  // The compute type must be CUBLAS_COMPUTE_32F.
  // The scale type must be CUDA_R_32F.
  cudaDataType_t scale_type = CUDA_R_32F;
  cublasComputeType_t gemm_compute_type = CUBLAS_COMPUTE_32F;
  cudaDataType_t output_type =
      cublas_utils::dtype_to_cublas_type(out_dtype, "CublasQQMM");

  init_base(
      device,
      scale_type,
      gemm_compute_type,
      config.data_type,
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

  a_scale_mode_ = config.scale_mode;
  b_scale_mode_ = config.scale_mode;

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
    const std::string& qmode)
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
// Supported overloads:
// alpha float
// alpha device ptr

void CublasQQMM::run(
    cu::CommandEncoder& encoder,
    array& out,
    const array& a,
    const array& b,
    const array& a_scale,
    const array& b_scale,
    const array& alpha,
    const array& beta) {
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(a_scale);
  encoder.set_input_array(b_scale);
  encoder.set_input_array(alpha);
  encoder.set_input_array(beta);
  encoder.set_output_array(out);

  execute(
      encoder,
      gpu_ptr<void>(out),
      gpu_ptr<void>(a),
      gpu_ptr<void>(b),
      gpu_ptr<void>(a_scale),
      gpu_ptr<void>(b_scale),
      nullptr,
      gpu_ptr<void>(alpha),
      gpu_ptr<void>(beta));
}

void CublasQQMM::run(
    cu::CommandEncoder& encoder,
    array& out,
    const array& a,
    const array& b,
    const array& a_scale,
    const array& b_scale) {
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
      nullptr);
}

void CublasQQMM::set_scales_ptrs(
    cu::CommandEncoder& encoder,
    const void* a_scale,
    const void* b_scale) {
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
}

void CublasQQMM::execute(
    cu::CommandEncoder& encoder,
    void* out,
    const void* a,
    const void* b,
    const void* a_scale,
    const void* b_scale,
    const void* c,
    const void* alpha,
    const void* beta) {
  set_scales_ptrs(encoder, a_scale, b_scale);
  // alpha and beta are both should be device pointers for nvfp4
  // by default cublas uses host pointers
  // https://docs.nvidia.com/cuda/cublas/#cublasltpointermode-t
  cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_POINTER_MODE,
      &pointer_mode,
      sizeof(pointer_mode)));
  execute_matmul(encoder, out, a, b, c, alpha, beta);
}

void CublasQQMM::execute(
    cu::CommandEncoder& encoder,
    void* out,
    const void* a,
    const void* b,
    const void* a_scale,
    const void* b_scale,
    const void* c,
    const float alpha /* = 1 */,
    const float beta /* = 0 */) {
  set_scales_ptrs(encoder, a_scale, b_scale);
  // alpha and beta are both should be host pointers
  cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_HOST;
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_POINTER_MODE,
      &pointer_mode,
      sizeof(pointer_mode)));

  const void* alpha_ptr = &alpha;
  const void* beta_ptr = &beta;

  execute_matmul(encoder, out, a, b, c, alpha_ptr, beta_ptr);
}

} // namespace mlx::core
