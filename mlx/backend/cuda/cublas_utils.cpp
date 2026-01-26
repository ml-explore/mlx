// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/cublas_utils.h"
#include "mlx/backend/cuda/cuda.h"
#include "mlx/utils.h"

namespace mlx::core {
namespace cublas_utils {

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

} // namespace

cublasLtMatmulPreference_t get_preference(cu::Device& device) {
  static CublasPreference pref(device);
  return pref.pref_;
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

} // namespace cublas_utils

CublasMatmulBase::~CublasMatmulBase() {
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(a_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(b_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(c_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(out_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescDestroy(matmul_desc_));
}

void CublasMatmulBase::init_base(
    cu::Device& device,
    cudaDataType_t scale_type,
    cublasComputeType_t compute_type,
    cudaDataType_t data_type,
    cudaDataType_t output_type,
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
  M_ = a_rows;
  N_ = b_cols;
  scale_type_ = scale_type;
  handle_ = device.lt_handle();
  pref_ = cublas_utils::get_preference(device);
  heuristic_.state = CUBLAS_STATUS_NOT_INITIALIZED;

  CHECK_CUBLAS_ERROR(
      cublasLtMatmulDescCreate(&matmul_desc_, compute_type, scale_type));

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

  a_desc_ = cublas_utils::create_matrix_layout(
      data_type,
      b_cols,
      b_rows,
      b_transposed,
      ldb,
      batch_count,
      b_batch_stride);
  b_desc_ = cublas_utils::create_matrix_layout(
      data_type,
      a_cols,
      a_rows,
      a_transposed,
      lda,
      batch_count,
      a_batch_stride);
  out_desc_ = cublas_utils::create_matrix_layout(
      output_type, b_cols, a_rows, false, b_cols, batch_count, b_cols * a_rows);
}

void CublasMatmulBase::execute_matmul(
    cu::CommandEncoder& encoder,
    void* out,
    const void* a,
    const void* b,
    const void* c,
    const void* alpha_ptr,
    const void* beta_ptr) {
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

  void* workspace_ptr = allocate_workspace(encoder, heuristic_.workspaceSize);

  // Execute matmul
  auto capture = encoder.capture_context();
  CHECK_CUBLAS_ERROR(cublasLtMatmul(
      handle_,
      matmul_desc_,
      alpha_ptr,
      b, // a and b are swapped for row-major layout
      a_desc_,
      a,
      b_desc_,
      beta_ptr,
      c ? c : out,
      c ? c_desc_ : out_desc_,
      out,
      out_desc_,
      &heuristic_.algo,
      workspace_ptr,
      heuristic_.workspaceSize,
      encoder.stream()));
}

void CublasMatmulBase::set_bias(
    cu::CommandEncoder& encoder,
    const array& bias) {
  encoder.set_input_array(bias);
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_EPILOGUE,
      &epilogue,
      sizeof(epilogue)));
  auto* bias_ptr = gpu_ptr<void>(bias);
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_BIAS_POINTER,
      &bias_ptr,
      sizeof(bias_ptr)));
}

} // namespace mlx::core
