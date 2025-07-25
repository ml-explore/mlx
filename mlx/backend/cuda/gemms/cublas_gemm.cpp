// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/gemms/cublas_gemm.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/dtype_utils.h"
#include "mlx/utils.h"

#include <fmt/format.h>

namespace mlx::core::cu {

struct CublasPreference {
  CublasPreference(Device& device) {
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

cublasLtMatmulPreference_t cublas_preference(Device& device) {
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
          "Unsupported dtype in Matmul: {}.", dtype_to_string(dtype)));
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
          "Unsupported dtype in Matmul: {}.", dtype_to_string(dtype)));
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
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutCreate(&desc, type, rows, cols, ld));
  cublasLtOrder_t order = transposed ? CUBLASLT_ORDER_COL : CUBLASLT_ORDER_ROW;
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutSetAttribute(
      desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(cublasLtOrder_t)));
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

Matmul::Matmul(
    Device& device,
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
  cublasOperation_t op = CUBLAS_OP_N;
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_TRANSA,
      &op,
      sizeof(cublasOperation_t)));
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_TRANSB,
      &op,
      sizeof(cublasOperation_t)));

  auto type = dtype_to_cublas_type(dtype);
  a_desc_ = create_matrix_layout(
      type, a_rows, a_cols, a_transposed, lda, batch_count, a_batch_stride);
  b_desc_ = create_matrix_layout(
      type, b_rows, b_cols, b_transposed, ldb, batch_count, b_batch_stride);
  out_desc_ = create_matrix_layout(
      type, a_rows, b_cols, false, b_cols, batch_count, a_rows * b_cols);
}

Matmul::Matmul(
    Device& device,
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
    : Matmul(
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
      type, a_rows, b_cols, false, ldc, batch_count, c_batch_stride);
}

Matmul::~Matmul() {
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(a_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(b_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(c_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(out_desc_));
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescDestroy(matmul_desc_));
}

void Matmul::run_impl(
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
        out_desc_, // TODO should that be c_desc is it's set?
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
    array workspace(
        allocator::malloc(heuristic_.workspaceSize),
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
      a,
      a_desc_,
      b,
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

void Matmul::run(
    cu::CommandEncoder& encoder,
    array& out,
    const array& a,
    const array& b,
    const std::optional<array>& c /* = std::nullopt */,
    float alpha /* = 1 */,
    float beta /* = 0 */) {
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  if (c) {
    encoder.set_input_array(*c);
  }
  encoder.set_output_array(out);

  run_impl(
      encoder,
      out.data<void>(),
      a.data<void>(),
      b.data<void>(),
      c ? c->data<void>() : nullptr,
      alpha,
      beta);
}

} // namespace mlx::core::cu
