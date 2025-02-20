// Copyright © 2023-2024 Apple Inc.

#include <Accelerate/Accelerate.h>

#include "mlx/array.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/gemm.h"
#include "mlx/dtype.h"

namespace mlx::core {

template <typename T>
constexpr BNNSDataType to_bnns_dtype();

template <>
constexpr BNNSDataType to_bnns_dtype<float>() {
  return BNNSDataType(BNNSDataTypeFloatBit | 32);
}
template <>
constexpr BNNSDataType to_bnns_dtype<float16_t>() {
  return BNNSDataType(BNNSDataTypeFloatBit | 16);
}

template <>
constexpr BNNSDataType to_bnns_dtype<bfloat16_t>() {
  return BNNSDataTypeBFloat16;
}

template <typename T>
void matmul_bnns(
    const T* a,
    const T* b,
    T* out,
    bool a_transposed,
    bool b_transposed,
    size_t lda,
    size_t ldb,
    size_t ldc,
    float alpha,
    float beta,
    size_t batch_size,
    const Shape& a_shape,
    const Strides& a_strides,
    const Shape& b_shape,
    const Strides& b_strides) {
  auto ndim = a_shape.size();
  size_t M = a_shape[ndim - 2];
  size_t N = b_shape[ndim - 1];
  size_t K = a_shape[ndim - 1];

  BNNSDataType bnns_dtype = to_bnns_dtype<T>();

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
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

  for (int i = 0; i < batch_size; ++i) {
    BNNSFilterApplyTwoInput(
        bnns_filter,
        reinterpret_cast<const uint8_t*>(
            a + elem_to_loc(M * K * i, a_shape, a_strides)),
        reinterpret_cast<const uint8_t*>(
            b + elem_to_loc(K * N * i, b_shape, b_strides)),
        reinterpret_cast<uint8_t*>(out + M * N * i));
  }

  BNNSFilterDestroy(bnns_filter);
#pragma GCC diagnostic pop
}

template <>
void matmul<float16_t>(
    const float16_t* a,
    const float16_t* b,
    float16_t* out,
    bool a_transposed,
    bool b_transposed,
    size_t lda,
    size_t ldb,
    size_t ldc,
    float alpha,
    float beta,
    size_t batch_size,
    const Shape& a_shape,
    const Strides& a_strides,
    const Shape& b_shape,
    const Strides& b_strides) {
  matmul_bnns(
      a,
      b,
      out,
      a_transposed,
      b_transposed,
      lda,
      ldb,
      ldc,
      alpha,
      beta,
      batch_size,
      a_shape,
      a_strides,
      b_shape,
      b_strides);
}

template <>
void matmul<bfloat16_t>(
    const bfloat16_t* a,
    const bfloat16_t* b,
    bfloat16_t* out,
    bool a_transposed,
    bool b_transposed,
    size_t lda,
    size_t ldb,
    size_t ldc,
    float alpha,
    float beta,
    size_t batch_size,
    const Shape& a_shape,
    const Strides& a_strides,
    const Shape& b_shape,
    const Strides& b_strides) {
  matmul_bnns(
      a,
      b,
      out,
      a_transposed,
      b_transposed,
      lda,
      ldb,
      ldc,
      alpha,
      beta,
      batch_size,
      a_shape,
      a_strides,
      b_shape,
      b_strides);
}

} // namespace mlx::core
