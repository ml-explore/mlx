// Copyright © 2025 Apple Inc.

#include "mlx/array.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/dtype_utils.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

#include <cublasLt.h>
#include <fmt/format.h>
#include <nvtx3/nvtx3.hpp>

#include <numeric>
#include <sstream>

namespace mlx::core {

namespace {

auto collapse_batches(const array& a, const array& b) {
  // Get and check the shape for the batched dims
  Shape A_bshape{a.shape().begin(), a.shape().end() - 2};
  Shape B_bshape{b.shape().begin(), b.shape().end() - 2};
  if (A_bshape != B_bshape) {
    std::ostringstream msg;
    msg << "[matmul] Got matrices with incorrectly broadcasted shapes: " << "A "
        << a.shape() << ", B " << b.shape() << ".";
    throw std::runtime_error(msg.str());
  }

  Strides A_bstride{a.strides().begin(), a.strides().end() - 2};
  Strides B_bstride{b.strides().begin(), b.strides().end() - 2};

  auto [batch_shape, batch_strides] =
      collapse_contiguous_dims(A_bshape, std::vector{A_bstride, B_bstride});

  auto a_batch_strides = batch_strides[0];
  auto b_batch_strides = batch_strides[1];

  if (batch_shape.empty()) {
    batch_shape.push_back(1);
    a_batch_strides.push_back(0);
    b_batch_strides.push_back(0);
  }

  return std::make_tuple(batch_shape, a_batch_strides, b_batch_strides);
}

std::tuple<bool, int64_t, array>
check_transpose(std::vector<array>& copies, const Stream& s, const array& arr) {
  auto stx = arr.strides()[arr.ndim() - 2];
  auto sty = arr.strides()[arr.ndim() - 1];
  if (sty == 1 && stx == arr.shape(-1)) {
    return std::make_tuple(false, stx, arr);
  } else if (stx == 1 && sty == arr.shape(-2)) {
    return std::make_tuple(true, sty, arr);
  } else {
    array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
    copy_gpu(arr, arr_copy, CopyType::General, s);
    copies.push_back(arr_copy);
    return std::make_tuple(false, arr.shape(-1), arr_copy);
  }
}

#define CHECK_CUBLAS_ERROR(cmd) check_cublas_error(#cmd, (cmd))

void check_cublas_error(const char* name, cublasStatus_t err) {
  if (err != CUBLAS_STATUS_SUCCESS) {
    // TODO: Use cublasGetStatusString when it is available.
    throw std::runtime_error(
        fmt::format("{} failed: {}", name, static_cast<int>(err)));
  }
}

// TODO: Set workspace size depending on architecture.
uint64_t workspace_size = 4 * 1024 * 1024;

class CudaMatMul {
 public:
  CudaMatMul(
      cu::CommandEncoder& encoder,
      Dtype ab_dtype,
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
    auto type = dtype_to_cuda_type(ab_dtype);
    CHECK_CUBLAS_ERROR(cublasLtMatmulDescCreate(
        &matmul_desc_, dtype_to_compute_type(ab_dtype), type));
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

    a_desc_ = create_matrix_layout(
        type, a_rows, a_cols, a_transposed, lda, batch_count, a_batch_stride);
    b_desc_ = create_matrix_layout(
        type, b_rows, b_cols, b_transposed, ldb, batch_count, b_batch_stride);
    out_desc_ = create_matrix_layout(
        type, a_rows, b_cols, false, b_cols, batch_count, a_rows * b_cols);

    CHECK_CUBLAS_ERROR(cublasLtMatmulPreferenceCreate(&pref_));
    CHECK_CUBLAS_ERROR(cublasLtMatmulPreferenceSetAttribute(
        pref_,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size,
        sizeof(uint64_t)));

    int ret = 0;
    CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoGetHeuristic(
        encoder.device().lt_handle(),
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

  ~CudaMatMul() {
    if (a_desc_) {
      cublasLtMatrixLayoutDestroy(a_desc_);
    }
    if (b_desc_) {
      cublasLtMatrixLayoutDestroy(b_desc_);
    }
    if (out_desc_) {
      cublasLtMatrixLayoutDestroy(out_desc_);
    }
    if (matmul_desc_) {
      cublasLtMatmulDescDestroy(matmul_desc_);
    }
  }

  template <typename T>
  void Run(cu::CommandEncoder& encoder, array& workspace, T* out, T* a, T* b) {
    // TODO: Allocate alpha/beta in temporary array.
    float alpha = 1;
    float beta = 0;
    encoder.launch_kernel([&](cudaStream_t stream) {
      CHECK_CUBLAS_ERROR(cublasLtMatmul(
          encoder.device().lt_handle(),
          matmul_desc_,
          &alpha,
          a,
          a_desc_,
          b,
          b_desc_,
          &beta,
          out,
          out_desc_,
          out,
          out_desc_,
          &heuristic_.algo,
          workspace.data<void>(),
          workspace.nbytes(),
          stream));
    });
  }

 private:
  cublasComputeType_t dtype_to_compute_type(Dtype dtype) {
    switch (dtype) {
      case uint8:
      case uint16:
      case int8:
      case int16:
      case int32:
        return CUBLAS_COMPUTE_32I;
      case float16:
      case bfloat16:
        return CUBLAS_COMPUTE_16F;
      case float32:
        return CUBLAS_COMPUTE_32F;
      case float64:
      case complex64:
        return CUBLAS_COMPUTE_64F;
      default:
        throw std::runtime_error(fmt::format(
            "Unsupported dtype in CudaMatMul: {}", dtype_to_string(dtype)));
    }
  }

  cudaDataType_t dtype_to_cuda_type(Dtype dtype) {
    switch (dtype) {
      case uint8:
        return CUDA_R_8U;
      case uint16:
        return CUDA_R_16U;
      case int8:
        return CUDA_R_8I;
      case int16:
        return CUDA_R_16I;
      case int32:
        return CUDA_R_32I;
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
            "Unsupported dtype in CudaMatMul: {}", dtype_to_string(dtype)));
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

  cublasLtMatmulDesc_t matmul_desc_{nullptr};
  cublasLtMatmulPreference_t pref_{nullptr};
  cublasLtMatrixLayout_t a_desc_{nullptr};
  cublasLtMatrixLayout_t b_desc_{nullptr};
  cublasLtMatrixLayout_t out_desc_{nullptr};
  cublasLtMatmulHeuristicResult_t heuristic_;
};

} // namespace

void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Matmul::eval_gpu");
  assert(inputs.size() == 2);
  if (!issubdtype(out.dtype(), floating)) {
    throw std::runtime_error(
        "[matmul] Does not yet support non-floating point types.");
  }
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];
  // Return 0s if either input is empty.
  if (a_pre.size() == 0 || b_pre.size() == 0) {
    array zero = array(0, a_pre.dtype());
    fill_gpu(zero, out, s);
    encoder.add_temporary(std::move(zero));
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
  std::vector<array> copies;
  auto [a_transposed, lda, a] = check_transpose(copies, s, a_pre);
  auto [b_transposed, ldb, b] = check_transpose(copies, s, b_pre);

  for (auto& temp : copies) {
    encoder.add_temporary(temp);
  }

  /////////////////////////////////////////////////////////////////////////////
  // Check and collapse batch dimensions

  auto [batch_shape, a_batch_strides, b_batch_strides] = collapse_batches(a, b);

  auto batch_count = out.size() / (size_t(M) * size_t(N));

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

  /////////////////////////////////////////////////////////////////////////////
  // Invoke cublasLt

  if (batch_shape.size() > 1) {
    // TODO: Implement by looping the matmul
    throw std::runtime_error(
        "Non-contiguous batch gemm is not implemented in CUDA backend.");
  }

  array workspace(
      allocator::malloc(workspace_size),
      {static_cast<int>(workspace_size)},
      uint8);
  encoder.add_temporary(workspace);

  CudaMatMul matmul(
      encoder,
      a.dtype(),
      a_transposed,
      M,
      K,
      lda,
      b_transposed,
      K,
      N,
      ldb,
      batch_count,
      a_batch_strides[0],
      b_batch_strides[0]);
  MLX_SWITCH_FLOAT_TYPES_CHECKED(a.dtype(), "matmul", CTYPE, {
    using ABType = cuda_type_t<CTYPE>;
    matmul.Run(
        encoder,
        workspace,
        out.data<ABType>(),
        a.data<ABType>(),
        b.data<ABType>());
  });
}

} // namespace mlx::core
