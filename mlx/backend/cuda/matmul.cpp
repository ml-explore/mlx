// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/matmul.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/gemv.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

#include <fmt/format.h>
#include <nvtx3/nvtx3.hpp>

#include <numeric>

namespace mlx::core {

namespace cu {

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

class MatMul {
 public:
  MatMul(
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
      : handle_(device.lt_handle()), pref_(cublas_preference(device)) {
    heuristic_.state = CUBLAS_STATUS_NOT_INITIALIZED;

    auto scale_type = dtype_to_cuda_type(dtype);
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

    auto type = dtype_to_cuda_type(dtype);
    a_desc_ = create_matrix_layout(
        type, a_rows, a_cols, a_transposed, lda, batch_count, a_batch_stride);
    b_desc_ = create_matrix_layout(
        type, b_rows, b_cols, b_transposed, ldb, batch_count, b_batch_stride);
    out_desc_ = create_matrix_layout(
        type, a_rows, b_cols, false, b_cols, batch_count, a_rows * b_cols);
  }

  MatMul(
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
      : MatMul(
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
    auto type = dtype_to_cuda_type(dtype);
    c_desc_ = create_matrix_layout(
        type, a_rows, b_cols, false, ldc, batch_count, c_batch_stride);
  }

  ~MatMul() {
    CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(a_desc_));
    CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(b_desc_));
    CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(c_desc_));
    CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(out_desc_));
    CHECK_CUBLAS_ERROR(cublasLtMatmulDescDestroy(matmul_desc_));
  }

  void run(
      cu::CommandEncoder& encoder,
      void* out,
      void* a,
      void* b,
      void* c = nullptr,
      float alpha = 1,
      float beta = 0) {
    if (heuristic_.state != CUBLAS_STATUS_SUCCESS) {
      int ret = 0;
      CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoGetHeuristic(
          handle_,
          matmul_desc_,
          a_desc_,
          b_desc_,
          out_desc_,
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

 private:
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
            "Unsupported dtype in MatMul: {}.", dtype_to_string(dtype)));
    }
  }

  cudaDataType_t dtype_to_cuda_type(Dtype dtype) {
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
            "Unsupported dtype in MatMul: {}.", dtype_to_string(dtype)));
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
    cublasLtOrder_t order =
        transposed ? CUBLASLT_ORDER_COL : CUBLASLT_ORDER_ROW;
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

  cublasLtMatmulPreference_t pref_{nullptr};
  cublasLtHandle_t handle_{nullptr};
  cublasLtMatmulDesc_t matmul_desc_{nullptr};
  cublasLtMatrixLayout_t a_desc_{nullptr};
  cublasLtMatrixLayout_t b_desc_{nullptr};
  cublasLtMatrixLayout_t c_desc_{nullptr};
  cublasLtMatrixLayout_t out_desc_{nullptr};
  cublasLtMatmulHeuristicResult_t heuristic_;
};

} // namespace cu

namespace {

std::tuple<bool, int64_t, array>
check_transpose(cu::CommandEncoder& enc, const Stream& s, const array& arr) {
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

} // namespace

void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Matmul::eval_gpu");
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

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

  out.set_data(allocator::malloc(out.nbytes()));

  /////////////////////////////////////////////////////////////////////////////
  // Init checks and prep

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int K = a_pre.shape(-1);

  // Keep a vector with copies to be cleared in the completed buffer to release
  // the arrays
  auto [a_transposed, lda, a] = check_transpose(encoder, s, a_pre);
  auto [b_transposed, ldb, b] = check_transpose(encoder, s, b_pre);

  /////////////////////////////////////////////////////////////////////////////
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

  if (cu::can_use_gemv(M, N, K, a_transposed, b_transposed)) {
    cu::gemv(
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

  /////////////////////////////////////////////////////////////////////////////
  // Invoke cublasLt

  cu::MatMul matmul(
      cu::device(s.device),
      a.dtype(),
      a_transposed,
      M,
      K,
      lda,
      b_transposed,
      K,
      N,
      ldb,
      batch_shape.back(),
      a_batch_strides.back(),
      b_batch_strides.back());

  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);
  auto nbatch = batch_count / batch_shape.back();
  if (nbatch == 1) {
    matmul.run(encoder, out.data<int8_t>(), a.data<int8_t>(), b.data<int8_t>());
    return;
  }

  ContiguousIterator a_it(batch_shape, a_batch_strides, batch_shape.size() - 1);
  ContiguousIterator b_it(batch_shape, b_batch_strides, batch_shape.size() - 1);
  auto concurrent = encoder.concurrent_context();
  for (size_t i = 0; i < nbatch; ++i) {
    matmul.run(
        encoder,
        out.data<int8_t>() + out.itemsize() * i * batch_shape.back() * M * N,
        a.data<int8_t>() + a.itemsize() * a_it.loc,
        b.data<int8_t>() + b.itemsize() * b_it.loc);
    a_it.step();
    b_it.step();
  }
}

void AddMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("AddMM::eval_gpu");
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  assert(inputs.size() == 3);
  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];
  auto c = inputs[2];

  /////////////////////////////////////////////////////////////////////////////
  // Init checks and prep

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int K = a_pre.shape(-1);

  // Keep a vector with copies to be cleared in the completed buffer to release
  // the arrays
  auto [a_transposed, lda, a] = check_transpose(encoder, s, a_pre);
  auto [b_transposed, ldb, b] = check_transpose(encoder, s, b_pre);

  int64_t ldc;
  {
    auto stx = c.strides()[c.ndim() - 2];
    auto sty = c.strides()[c.ndim() - 1];
    if (sty == 1 && stx == c.shape(-1)) {
      ldc = stx;
      out.set_data(allocator::malloc(out.nbytes()));
    } else if (sty == 1 && stx == 0) {
      ldc = 0;
      out.set_data(allocator::malloc(out.nbytes()));
    } else {
      // Copy C into out and set C to out
      ldc = c.shape(-1);
      copy_gpu(c, out, CopyType::General, s);
      c = out;
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // Check and collapse batch dimensions

  auto [batch_shape, a_batch_strides, b_batch_strides, c_batch_strides] =
      collapse_batches(a, b, c);

  auto batch_count = out.size() / (M * N);

  // Collapse batches into M if needed
  if (batch_count > 1 && !a_transposed && batch_shape.size() == 1 &&
      a.strides()[a.ndim() - 2] == K && a_batch_strides.back() == M * K &&
      c_batch_strides.back() == M * c.strides()[c.ndim() - 2] &&
      b_batch_strides.back() == 0) {
    M *= batch_shape.back();
    batch_count = 1;

    a_batch_strides = {0};
    b_batch_strides = {0};
    c_batch_strides = {0};
    batch_shape = {1};
  }

  /////////////////////////////////////////////////////////////////////////////
  // Invoke cublasLt

  cu::MatMul matmul(
      cu::device(s.device),
      a.dtype(),
      a_transposed,
      M,
      K,
      lda,
      b_transposed,
      K,
      N,
      ldb,
      ldc,
      batch_shape.back(),
      a_batch_strides.back(),
      b_batch_strides.back(),
      c_batch_strides.back());

  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(c);
  encoder.set_output_array(out);

  auto nbatch = batch_count / batch_shape.back();
  if (nbatch == 1) {
    matmul.run(
        encoder,
        out.data<int8_t>(),
        a.data<int8_t>(),
        b.data<int8_t>(),
        c.data<int8_t>(),
        alpha_,
        beta_);
    return;
  }

  ContiguousIterator a_it(batch_shape, a_batch_strides, batch_shape.size() - 1);
  ContiguousIterator b_it(batch_shape, b_batch_strides, batch_shape.size() - 1);
  ContiguousIterator c_it(batch_shape, c_batch_strides, batch_shape.size() - 1);
  auto concurrent = encoder.concurrent_context();
  for (size_t i = 0; i < nbatch; ++i) {
    matmul.run(
        encoder,
        out.data<int8_t>() + out.itemsize() * i * batch_shape.back() * M * N,
        a.data<int8_t>() + a.itemsize() * a_it.loc,
        b.data<int8_t>() + b.itemsize() * b_it.loc,
        c.data<int8_t>() + c.itemsize() * c_it.loc,
        alpha_,
        beta_);
    a_it.step();
    b_it.step();
    c_it.step();
  }
}

} // namespace mlx::core
