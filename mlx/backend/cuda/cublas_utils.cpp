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

void* allocate_workspace(cu::CommandEncoder& encoder, size_t workspace_size) {
  if (workspace_size == 0) {
    return nullptr;
  }

  // Ensure workspace is 256-byte aligned
  int nbytes = cuda::ceil_div(workspace_size, 256) * 256;
  array workspace(
      cu::malloc_async(nbytes, encoder.stream()),
      {static_cast<int>(workspace_size)},
      int8);
  encoder.add_temporary(workspace);
  return gpu_ptr<void>(workspace);
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

void execute_matmul(
    cu::CommandEncoder& encoder,
    cublasLtHandle_t handle,
    cublasLtMatmulDesc_t matmul_desc,
    cublasLtMatrixLayout_t a_desc,
    cublasLtMatrixLayout_t b_desc,
    cublasLtMatrixLayout_t c_desc,
    cublasLtMatrixLayout_t out_desc,
    cublasLtMatmulHeuristicResult_t& heuristic,
    cublasLtMatmulPreference_t pref,
    void* out,
    const void* a,
    const void* b,
    const void* c,
    const void* alpha_ptr,
    const void* beta_ptr) {
  if (heuristic.state != CUBLAS_STATUS_SUCCESS) {
    int ret = 0;
    CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoGetHeuristic(
        handle,
        matmul_desc,
        a_desc,
        b_desc,
        c ? c_desc : out_desc,
        out_desc,
        pref,
        1,
        &heuristic,
        &ret));
    if (ret == 0) {
      throw std::runtime_error("Can not find algorithm for matmul.");
    }
  }

  void* workspace_ptr = allocate_workspace(encoder, heuristic.workspaceSize);

  // Execute matmul
  auto capture = encoder.capture_context();
  CHECK_CUBLAS_ERROR(cublasLtMatmul(
      handle,
      matmul_desc,
      alpha_ptr,
      b, // a and b are swapped for row-major layout
      a_desc,
      a,
      b_desc,
      beta_ptr,
      c ? c : out,
      c ? c_desc : out_desc,
      out,
      out_desc,
      &heuristic.algo,
      workspace_ptr,
      heuristic.workspaceSize,
      encoder.stream()));
}

void set_bias(
    cu::CommandEncoder& encoder,
    cublasLtMatmulDesc_t matmul_desc,
    const array& bias) {
  encoder.set_input_array(bias);
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
  auto* bias_ptr = gpu_ptr<void>(bias);
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc,
      CUBLASLT_MATMUL_DESC_BIAS_POINTER,
      &bias_ptr,
      sizeof(bias_ptr)));
}

} // namespace cublas_utils
} // namespace mlx::core
