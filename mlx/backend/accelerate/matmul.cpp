// Copyright © 2023 Apple Inc.

#include <cassert>

#include <vecLib/BNNS/bnns.h>
#include <vecLib/cblas_new.h>

#include "mlx/backend/accelerate/utils.h"
#include "mlx/backend/common/copy.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

std::tuple<bool, size_t, array> check_transpose(const array& arr) {
  auto stx = arr.strides()[arr.ndim() - 2];
  auto sty = arr.strides()[arr.ndim() - 1];
  if (stx == arr.shape(-1) && sty == 1) {
    return std::make_tuple(false, stx, arr);
  } else if (stx == 1 && sty == arr.shape(-2)) {
    return std::make_tuple(true, sty, arr);
  } else {
    array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
    copy(arr, arr_copy, CopyType::General);
    size_t stx = arr.shape(-1);
    return std::make_tuple(false, stx, arr_copy);
  }
}

inline void matmul_cblas_general(
    const array& a_pre,
    const array& b_pre,
    array& out,
    float alpha = 1.0f,
    float beta = 0.0f) {
  if (out.dtype() != float32) {
    throw std::runtime_error(
        "[matmul_cblas] on CPU currently only supports float32");
  }

  auto [a_transposed, lda, a] = check_transpose(a_pre);
  auto [b_transposed, ldb, b] = check_transpose(b_pre);
  size_t M = a.shape(-2);
  size_t N = b.shape(-1);
  size_t K = a.shape(-1);

  for (int i = 0; i < (a.size() / (M * K)); ++i) {
    cblas_sgemm(
        CblasRowMajor,
        a_transposed ? CblasTrans : CblasNoTrans, // transA
        b_transposed ? CblasTrans : CblasNoTrans, // transB
        M,
        N,
        K,
        alpha, // alpha
        a.data<float>() + elem_to_loc(M * K * i, a.shape(), a.strides()),
        lda,
        b.data<float>() + elem_to_loc(K * N * i, b.shape(), b.strides()),
        ldb,
        beta, // beta
        out.data<float>() + M * N * i,
        out.shape(-1) // ldc
    );
  }
}

inline void matmul_cblas(const array& a_pre, const array& b_pre, array& out) {
  if (out.dtype() != float32) {
    throw std::runtime_error(
        "[matmul_cblas] on CPU currently only supports float32");
  }
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  return matmul_cblas_general(a_pre, b_pre, out);
}

inline void matmul_bnns_general(
    const array& a_pre,
    const array& b_pre,
    array& out,
    float alpha = 1.0f,
    float beta = 0.0f) {
  // TODO: Update to utilize BNNS broadcasting

  auto [a_transposed, lda, a] = check_transpose(a_pre);
  auto [b_transposed, ldb, b] = check_transpose(b_pre);
  size_t M = a.shape(-2);
  size_t N = b.shape(-1);
  size_t K = a.shape(-1);

  BNNSDataType bnns_dtype = to_bnns_dtype(out.dtype());

  const BNNSLayerParametersBroadcastMatMul gemm_params{
      /* float alpha = */ alpha,
      /* float beta = */ beta,
      /* bool transA = */ a_transposed,
      /* bool transB = */ b_transposed,
      /* bool quadratic = */ false,
      /* bool a_is_weights = */ false,
      /* bool b_is_weights = */ false,
      /* BNNSNDArrayDescriptor iA_desc = */
      BNNSNDArrayDescriptor{
          /* BNNSNDArrayFlags flags = */ BNNSNDArrayFlagBackpropSet,
          /* BNNSDataLayout layout = */ BNNSDataLayoutRowMajorMatrix,

          /* size_t size[BNNS_MAX_TENSOR_DIMENSION] = */
          {lda, (M * K) / lda, 0, 0, 0, 0, 0, 0},
          /* size_t stride[BNNS_MAX_TENSOR_DIMENSION] = */
          {1, lda, 0, 0, 0, 0, 0, 0},

          /* void * _Nullable data = */ nullptr,
          /* BNNSDataType data_type = */ bnns_dtype,

          /* void * _Nullable table_data = */ nullptr,
          /* BNNSDataType table_data_type = */ bnns_dtype,

          /* float data_scale = */ 1.0,
          /* float data_bias = */ 0.0,
      },
      /* BNNSNDArrayDescriptor iB_desc = */
      BNNSNDArrayDescriptor{
          /* BNNSNDArrayFlags flags = */ BNNSNDArrayFlagBackpropSet,
          /* BNNSDataLayout layout = */ BNNSDataLayoutRowMajorMatrix,

          /* size_t size[BNNS_MAX_TENSOR_DIMENSION] = */
          {ldb, (K * N) / ldb, 0, 0, 0, 0, 0, 0},
          /* size_t stride[BNNS_MAX_TENSOR_DIMENSION] = */
          {1, ldb, 0, 0, 0, 0, 0, 0},

          /* void * _Nullable data = */ nullptr,
          /* BNNSDataType data_type = */ bnns_dtype,

          /* void * _Nullable table_data = */ nullptr,
          /* BNNSDataType table_data_type = */ bnns_dtype,

          /* float data_scale = */ 1.0,
          /* float data_bias = */ 0.0,
      },
      /* BNNSNDArrayDescriptor o_desc = */
      BNNSNDArrayDescriptor{
          /* BNNSNDArrayFlags flags = */ BNNSNDArrayFlagBackpropSet,
          /* BNNSDataLayout layout = */ BNNSDataLayoutRowMajorMatrix,

          /* size_t size[BNNS_MAX_TENSOR_DIMENSION] = */
          {N, M, 0, 0, 0, 0, 0, 0},
          /* size_t stride[BNNS_MAX_TENSOR_DIMENSION] = */
          {1, N, 0, 0, 0, 0, 0, 0},

          /* void * _Nullable data = */ nullptr,
          /* BNNSDataType data_type = */ bnns_dtype,

          /* void * _Nullable table_data = */ nullptr,
          /* BNNSDataType table_data_type = */ bnns_dtype,

          /* float data_scale = */ 1.0,
          /* float data_bias = */ 0.0,
      },
  };

  auto bnns_filter =
      BNNSFilterCreateLayerBroadcastMatMul(&gemm_params, nullptr);

  for (int i = 0; i < (a.size() / (M * K)); ++i) {
    BNNSFilterApplyTwoInput(
        bnns_filter,
        a.data<uint8_t>() +
            elem_to_loc(M * K * i, a.shape(), a.strides()) * a.itemsize(),
        b.data<uint8_t>() +
            elem_to_loc(K * N * i, b.shape(), b.strides()) * b.itemsize(),
        out.data<uint8_t>() + M * N * i * out.itemsize());
  }

  BNNSFilterDestroy(bnns_filter);
}

inline void matmul_bnns(const array& a_pre, const array& b_pre, array& out) {
  // TODO: Update to utilize BNNS broadcasting
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  return matmul_bnns_general(a_pre, b_pre, out);
}

} // namespace

void Matmul::eval_cpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == float32) {
    return matmul_cblas(inputs[0], inputs[1], out);
  }
  return matmul_bnns(inputs[0], inputs[1], out);
}

void AddMM::eval_cpu(const std::vector<array>& inputs, array& out) {
  // Fill output with C
  auto& c = inputs[2];
  CopyType ctype = c.data_size() == 1 ? CopyType::Scalar : CopyType::General;
  copy(c, out, ctype);

  if (out.dtype() == float32) {
    return matmul_cblas_general(inputs[0], inputs[1], out, alpha_, beta_);
  }
  return matmul_bnns_general(inputs[0], inputs[1], out, alpha_, beta_);
}

} // namespace mlx::core