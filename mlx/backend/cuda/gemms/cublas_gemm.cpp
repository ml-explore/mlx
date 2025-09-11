// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/gemms/cublas_gemm.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/dtype_utils.h"
#include "mlx/utils.h"

#include <fmt/format.h>

namespace mlx::core {

namespace {

struct CublasPreference {
  CublasPreference(cu::Device& device) {
    // The recommended cublas workspace size is 4 MiB for pre-Hopper and 32 MiB
    // for Hopper+:
    // https://docs.nvidia.com/cuda/cublas/#cublassetworkspace
    uint64_t MiB = 1024 * 1024;
    uint64_t workspace_size =
        device.compute_capability_major() >= 9 ? 32 * MiB : 4 * MiB;

    CHECK_CUBLAS_ERROR(cublasLtMatmulPreferenceCreate(&pref_));
    CHECK_CUBLAS_ERROR(cublasLtMatmulPreferenceSetAttribute(
        pref_,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size,
        sizeof(uint64_t)));
  }

  ~CublasPreference() {
    CHECK_CUBLAS_ERROR(cublasLtMatmulPreferenceDestroy(pref_));
  }

  cublasLtMatmulPreference_t pref_{nullptr};
};

cublasLtMatmulPreference_t cublas_preference(cu::Device& device) {
  static CublasPreference pref(device);
  return pref.pref_;
}

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
    case complex64:
      return CUBLAS_COMPUTE_64F;
    default:
      throw std::runtime_error(fmt::format(
          "Unsupported dtype in CublasGemm: {}.", dtype_to_string(dtype)));
  }
}

cudaDataType_t dtype_to_cublas_type(Dtype dtype) {
  switch (dtype) {
    case float16:
      return CUDA_R_16F;
    case bfloat16:
      return CUDA_R_16BF;
    case float32:
      return CUDA_R_32F;
    case float64:
      return CUDA_R_64F;
    case complex64:
      return CUDA_C_32F;
    default:
      throw std::runtime_error(fmt::format(
          "Unsupported dtype in CublasGemm: {}.", dtype_to_string(dtype)));
  }
}

cublasLtMatrixLayout_t create_matrix_layout(
    cudaDataType_t type,
    uint64_t rows,
    uint64_t cols,
    bool transposed,
    int64_t ld,
    int32_t batch_count,
    int64_t batch_stride) {
  cublasLtMatrixLayout_t desc;
  if (transposed) {
    std::swap(rows, cols);
  }
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutCreate(&desc, type, rows, cols, ld));
  if (batch_count > 1) {
    CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutSetAttribute(
        desc,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_count,
        sizeof(int32_t)));
    CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutSetAttribute(
        desc,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batch_stride,
        sizeof(int64_t)));
  }
  return desc;
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
    int64_t b_batch_stride)
    : handle_(device.lt_handle()),
      pref_(cublas_preference(device)),
      M_(a_rows),
      N_(b_cols) {
  heuristic_.state = CUBLAS_STATUS_NOT_INITIALIZED;

  auto scale_type = dtype_to_cublas_type(dtype);
  if (dtype == bfloat16 || dtype == float16) {
    scale_type = CUDA_R_32F;
  }
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescCreate(
      &matmul_desc_, dtype_to_compute_type(dtype), scale_type));
  int32_t pointer_mode = CUBLASLT_POINTER_MODE_HOST;
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_POINTER_MODE,
      &pointer_mode,
      sizeof(int32_t)));

  // In cublasLt matrices use column-major layout, while it is possible to use
  // the CUBLASLT_ORDER_ROW option to switch to row-major layout, the bias
  // epilogue does not work with the option. So instead we swap A and B to make
  // cublasLt return the row-major result, which works because:
  // - the data of a matrix in row-major layout is identical to its transpose in
  //   column-major layout
  // - C^T = (A @ B)^T = B^T @ A^T
  cublasOperation_t a_op = b_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_TRANSA,
      &a_op,
      sizeof(cublasOperation_t)));
  cublasOperation_t b_op = a_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_TRANSB,
      &b_op,
      sizeof(cublasOperation_t)));

  auto type = dtype_to_cublas_type(dtype);
  a_desc_ = create_matrix_layout(
      type, b_cols, b_rows, b_transposed, ldb, batch_count, b_batch_stride);
  b_desc_ = create_matrix_layout(
      type, a_cols, a_rows, a_transposed, lda, batch_count, a_batch_stride);
  out_desc_ = create_matrix_layout(
      type, b_cols, a_rows, false, b_cols, batch_count, a_rows * b_cols);
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
  auto type = dtype_to_cublas_type(dtype);
  c_desc_ = create_matrix_layout(
      type, b_cols, a_rows, false, ldc, batch_count, c_batch_stride);
}

CublasGemm::~CublasGemm() {
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(a_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(b_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(c_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(out_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescDestroy(matmul_desc_));
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
  out_desc_ = create_matrix_layout(
      dtype_to_cublas_type(dtype),
      cols,
      rows,
      transposed,
      ld,
      batch_count,
      batch_stride);
}

void CublasGemm::set_bias(cu::CommandEncoder& encoder, const array& bias) {
  encoder.set_input_array(bias);
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_EPILOGUE,
      &epilogue,
      sizeof(epilogue)));
  auto* bias_ptr = bias.data<void>();
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_BIAS_POINTER,
      &bias_ptr,
      sizeof(bias_ptr)));
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
      out.data<void>(),
      a.data<void>(),
      b.data<void>(),
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
      out.data<void>(),
      a.data<void>(),
      b.data<void>(),
      c.data<void>(),
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
  if (heuristic_.state != CUBLAS_STATUS_SUCCESS) {
    int ret = 0;
    CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoGetHeuristic(
        handle_,
        matmul_desc_,
        a_desc_,
        b_desc_,
        c ? c_desc_ : out_desc_,
        out_desc_,
        pref_,
        1,
        &heuristic_,
        &ret));
    if (ret == 0) {
      throw std::runtime_error("Can not find algorithm for matmul.");
    }
  }

  void* workspace_ptr = nullptr;
  if (heuristic_.workspaceSize > 0) {
    // Ensure workspace is 256-byte aligned
    int nbytes = cuda::ceil_div(heuristic_.workspaceSize, 256) * 256;
    array workspace(
        allocator::malloc(nbytes),
        {static_cast<int>(heuristic_.workspaceSize)},
        int8);
    encoder.add_temporary(workspace);
    workspace_ptr = workspace.data<void>();
  }

  auto capture = encoder.capture_context();
  CHECK_CUBLAS_ERROR(cublasLtMatmul(
      handle_,
      matmul_desc_,
      &alpha,
      b, // a and b are swapped
      a_desc_,
      a,
      b_desc_,
      &beta,
      c ? c : out,
      c ? c_desc_ : out_desc_,
      out,
      out_desc_,
      &heuristic_.algo,
      workspace_ptr,
      heuristic_.workspaceSize,
      encoder.stream()));
}

} // namespace mlx::core
