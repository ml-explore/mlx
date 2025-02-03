// Copyright © 2023-2024 Apple Inc.

#include <Accelerate/Accelerate.h>

#include "mlx/array.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/gemm.h"
#include "mlx/dtype.h"

namespace mlx::core {

BNNSDataType to_bnns_dtype(Dtype mlx_dtype) {
  uint32_t size_bits = size_of(mlx_dtype) * 8;
  switch (kindof(mlx_dtype)) {
    case Dtype::Kind::b:
      return BNNSDataTypeBoolean;
    case Dtype::Kind::u:
      return BNNSDataType(BNNSDataTypeUIntBit | size_bits);
    case Dtype::Kind::i:
      return BNNSDataType(BNNSDataTypeIntBit | size_bits);
    case Dtype::Kind::f:
      return BNNSDataType(BNNSDataTypeFloatBit | size_bits);
    case Dtype::Kind::V:
      return BNNSDataTypeBFloat16;
    case Dtype::Kind::c:
      throw std::invalid_argument("BNNS does not support complex types");
  }
}

void matmul_bnns(
    const array& a,
    const array& b,
    array& out,
    bool a_transposed,
    bool b_transposed,
    size_t lda,
    size_t ldb,
    float alpha,
    float beta) {
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

template <>
void matmul<float16_t>(
    const array& a,
    const array& b,
    array& out,
    bool a_transposed,
    bool b_transposed,
    size_t lda,
    size_t ldb,
    float alpha,
    float beta) {
  matmul_bnns(a, b, out, a_transposed, b_transposed, lda, ldb, alpha, beta);
}

template <>
void matmul<bfloat16_t>(
    const array& a,
    const array& b,
    array& out,
    bool a_transposed,
    bool b_transposed,
    size_t lda,
    size_t ldb,
    float alpha,
    float beta) {
  matmul_bnns(a, b, out, a_transposed, b_transposed, lda, ldb, alpha, beta);
}

} // namespace mlx::core
