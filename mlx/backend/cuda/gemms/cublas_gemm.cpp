// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/gemms/cublas_gemm.h"
#include "mlx/backend/cuda/cublas_utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/dtype_utils.h"
#include "mlx/utils.h"

#include <fmt/format.h>

namespace mlx::core {

namespace {

cublasComputeType_t dtype_to_compute_type(Dtype dtype) {
  switch (dtype) {
    case float16:
      return CUBLAS_COMPUTE_32F;
    case bfloat16:
      return CUBLAS_COMPUTE_32F;
    case float32:
      return mlx::core::env::enable_tf32() ? CUBLAS_COMPUTE_32F_FAST_TF32
                                           : CUBLAS_COMPUTE_32F;
    case float64:
      return CUBLAS_COMPUTE_64F;
    case complex64:
      return mlx::core::env::enable_tf32() ? CUBLAS_COMPUTE_32F_FAST_TF32
                                           : CUBLAS_COMPUTE_32F;
    default:
      throw std::runtime_error(fmt::format(
          "Unsupported dtype in CublasGemm: {}.", dtype_to_string(dtype)));
  }
}

} // namespace

CublasGemm::CublasGemm(
    cu::Device& device,
    Dtype dtype,
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
    int64_t b_batch_stride) {
  scale_type_ = cublas_utils::dtype_to_cublas_type(dtype, "CublasGemm");
  if (dtype == bfloat16 || dtype == float16) {
    scale_type_ = CUDA_R_32F;
  }
  cudaDataType_t cublas_dtype =
      cublas_utils::dtype_to_cublas_type(dtype, "CublasGemm");

  init_base(
      device,
      scale_type_,
      dtype_to_compute_type(dtype),
      cublas_dtype,
      cublas_dtype,
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
}

CublasGemm::CublasGemm(
    cu::Device& device,
    Dtype dtype,
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
    int64_t c_batch_stride)
    : CublasGemm(
          device,
          dtype,
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
          b_batch_stride) {
  auto type = cublas_utils::dtype_to_cublas_type(dtype, "CublasGemm");
  c_desc_ = cublas_utils::create_matrix_layout(
      type, b_cols, a_rows, false, ldc, batch_count, c_batch_stride);
}

void CublasGemm::set_out(
    Dtype dtype,
    bool transposed,
    uint64_t rows,
    uint64_t cols,
    int64_t ld,
    int32_t batch_count,
    int64_t batch_stride) {
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(out_desc_));
  out_desc_ = cublas_utils::create_matrix_layout(
      cublas_utils::dtype_to_cublas_type(dtype, "CublasGemm"),
      cols,
      rows,
      transposed,
      ld,
      batch_count,
      batch_stride);
}

void CublasGemm::run(
    cu::CommandEncoder& encoder,
    array& out,
    const array& a,
    const array& b,
    const Shape& batch_shape,
    const Strides& a_batch_strides,
    const Strides& b_batch_strides,
    float alpha) {
  int batch_count = out.size() / (M_ * N_);
  if (batch_count / batch_shape.back() > 1) {
    run_batched(
        encoder,
        out,
        a,
        b,
        batch_shape,
        a_batch_strides,
        b_batch_strides,
        alpha);
    return;
  }

  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);

  execute(
      encoder,
      gpu_ptr<void>(out),
      gpu_ptr<void>(a),
      gpu_ptr<void>(b),
      nullptr,
      alpha);
}

void CublasGemm::run(
    cu::CommandEncoder& encoder,
    array& out,
    const array& a,
    const array& b,
    const array& c,
    const Shape& batch_shape,
    const Strides& a_batch_strides,
    const Strides& b_batch_strides,
    const Strides& c_batch_strides,
    float alpha,
    float beta) {
  int batch_count = out.size() / (M_ * N_);
  if (batch_count / batch_shape.back() > 1) {
    run_batched(
        encoder,
        out,
        a,
        b,
        c,
        batch_shape,
        a_batch_strides,
        b_batch_strides,
        c_batch_strides,
        alpha,
        beta);
    return;
  }

  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(c);
  encoder.set_output_array(out);

  execute(
      encoder,
      gpu_ptr<void>(out),
      gpu_ptr<void>(a),
      gpu_ptr<void>(b),
      gpu_ptr<void>(c),
      alpha,
      beta);
}

void CublasGemm::execute(
    cu::CommandEncoder& encoder,
    void* out,
    const void* a,
    const void* b,
    const void* c,
    float alpha /* = 1 */,
    float beta /* = 0 */) {
  const void* alpha_ptr = &alpha;
  const void* beta_ptr = &beta;
  complex64_t alpha_c, beta_c;
  if (scale_type_ == CUDA_C_32F) {
    alpha_c = complex64_t{alpha, 0.0f};
    beta_c = complex64_t{beta, 0.0f};
    alpha_ptr = &alpha_c;
    beta_ptr = &beta_c;
  }

  execute_matmul(encoder, out, a, b, c, alpha_ptr, beta_ptr);
}

} // namespace mlx::core
