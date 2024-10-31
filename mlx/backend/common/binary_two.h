// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/backend/common/binary.h"
#include "mlx/backend/common/utils.h"

namespace mlx::core {

namespace {

template <typename T, typename U, typename Op, int D>
void binary_op_dims(
    const T* a,
    const T* b,
    U* out_a,
    U* out_b,
    Op op,
    const std::vector<int>& shape,
    const std::vector<size_t>& a_strides,
    const std::vector<size_t>& b_strides,
    const std::vector<size_t>& out_strides,
    int axis) {
  auto stride_a = a_strides[axis];
  auto stride_b = b_strides[axis];
  auto stride_out = out_strides[axis];
  auto N = shape[axis];

  for (int i = 0; i < N; i++) {
    if constexpr (D > 1) {
      binary_op_dims<T, U, Op, D - 1>(
          a,
          b,
          out_a,
          out_b,
          op,
          shape,
          a_strides,
          b_strides,
          out_strides,
          axis + 1);
    } else {
      std::tie(*out_a, *out_b) = op(*a, *b);
    }
    a += stride_a;
    b += stride_b;
    out_a += stride_out;
    out_b += stride_out;
  }
}

template <typename T, typename U, typename Op>
void binary_op_dispatch_dims(
    const array& a,
    const array& b,
    array& out_a,
    array& out_b,
    Op op) {
  auto [shape, strides] = collapse_contiguous_dims(
      a.shape(), {a.strides(), b.strides(), out_a.strides()});
  const auto& a_strides = strides[0];
  const auto& b_strides = strides[1];
  const auto& out_strides = strides[2];
  const T* a_ptr = a.data<T>();
  const T* b_ptr = b.data<T>();
  U* out_a_ptr = out_a.data<U>();
  U* out_b_ptr = out_b.data<U>();

  int ndim = shape.size();
  switch (ndim) {
    case 1:
      binary_op_dims<T, U, Op, 1>(
          a_ptr,
          b_ptr,
          out_a_ptr,
          out_b_ptr,
          op,
          shape,
          a_strides,
          b_strides,
          out_strides,
          0);
      return;
    case 2:
      binary_op_dims<T, U, Op, 2>(
          a_ptr,
          b_ptr,
          out_a_ptr,
          out_b_ptr,
          op,
          shape,
          a_strides,
          b_strides,
          out_strides,
          0);
      return;
  }

  ContiguousIterator<size_t> a_it(shape, a_strides, ndim - 2);
  ContiguousIterator<size_t> b_it(shape, b_strides, ndim - 2);
  size_t stride = out_strides[ndim - 3];
  for (size_t elem = 0; elem < a.size(); elem += stride) {
    binary_op_dims<T, U, Op, 2>(
        a_ptr + a_it.loc,
        b_ptr + b_it.loc,
        out_a_ptr + elem,
        out_b_ptr + elem,
        op,
        shape,
        a_strides,
        b_strides,
        out_strides,
        ndim - 2);
    a_it.step();
    b_it.step();
  }
}

template <typename T, typename U = T, typename Op>
void binary_op(
    const array& a,
    const array& b,
    std::vector<array>& outputs,
    Op op) {
  auto bopt = get_binary_op_type(a, b);
  auto& out_a = outputs[0];
  auto& out_b = outputs[1];
  set_binary_op_output_data(a, b, out_a, bopt);
  set_binary_op_output_data(a, b, out_b, bopt);

  // The full computation is scalar scalar so call the base op once
  if (bopt == BinaryOpType::General) {
    binary_op_dispatch_dims<T, U, Op>(a, b, out_a, out_b, op);
    return;
  }

  auto a_ptr = a.data<T>();
  auto b_ptr = b.data<T>();
  auto out_a_ptr = out_a.data<U>();
  auto out_b_ptr = out_b.data<U>();
  if (bopt == BinaryOpType::ScalarScalar) {
    std::tie(*out_a_ptr, *out_b_ptr) = op(*a_ptr, *b_ptr);
  } else if (bopt == BinaryOpType::ScalarVector) {
    for (size_t i = 0; i < b.size(); ++i) {
      std::tie(*out_a_ptr, *out_b_ptr) = op(*a_ptr, *b_ptr);
      out_a_ptr++;
      out_b_ptr++;
      b_ptr++;
    }
  } else if (bopt == BinaryOpType::VectorScalar) {
    for (size_t i = 0; i < a.size(); ++i) {
      std::tie(*out_a_ptr, *out_b_ptr) = op(*a_ptr, *b_ptr);
      out_a_ptr++;
      out_b_ptr++;
      a_ptr++;
    }
  } else { // VectorVector
    for (size_t i = 0; i < a.size(); ++i) {
      std::tie(*out_a_ptr, *out_b_ptr) = op(*a_ptr, *b_ptr);
      out_a_ptr++;
      out_b_ptr++;
      a_ptr++;
      b_ptr++;
    }
  }
}

template <typename Op>
void binary(
    const array& a,
    const array& b,
    std::vector<array>& outputs,
    Op op) {
  switch (outputs[0].dtype()) {
    case bool_:
      binary_op<bool>(a, b, outputs, op);
      break;
    case uint8:
      binary_op<uint8_t>(a, b, outputs, op);
      break;
    case uint16:
      binary_op<uint16_t>(a, b, outputs, op);
      break;
    case uint32:
      binary_op<uint32_t>(a, b, outputs, op);
      break;
    case uint64:
      binary_op<uint64_t>(a, b, outputs, op);
      break;
    case int8:
      binary_op<int8_t>(a, b, outputs, op);
      break;
    case int16:
      binary_op<int16_t>(a, b, outputs, op);
      break;
    case int32:
      binary_op<int32_t>(a, b, outputs, op);
      break;
    case int64:
      binary_op<int64_t>(a, b, outputs, op);
      break;
    case float16:
      binary_op<float16_t>(a, b, outputs, op);
      break;
    case float32:
      binary_op<float>(a, b, outputs, op);
      break;
    case bfloat16:
      binary_op<bfloat16_t>(a, b, outputs, op);
      break;
    case complex64:
      binary_op<complex64_t>(a, b, outputs, op);
      break;
  }
}

} // namespace

} // namespace mlx::core
