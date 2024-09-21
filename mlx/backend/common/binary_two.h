// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/backend/common/binary.h"
#include "mlx/backend/common/utils.h"

namespace mlx::core {

namespace {

template <typename T, typename U, typename Op>
void binary_op_dispatch_dims(
    const array& a,
    const array& b,
    array& out_a,
    array& out_b,
    Op op,
    int dim,
    int stride) {
  // Number of dimensions to loop over for vectorized ops
  switch (dim) {
    case 1:
      binary_op_dims1<T, U, Op>(a, b, out_a, out_b, op, stride);
      return;
    case 2:
      binary_op_dims2<T, U, Op>(a, b, out_a, out_b, op, stride);
      return;
  }

  const T* a_ptr = a.data<T>();
  const T* b_ptr = b.data<T>();
  U* dst_a = out_a.data<U>();
  U* dst_b = out_b.data<U>();
  for (size_t i = 0; i < out_a.size(); i += stride) {
    int a_idx = elem_to_loc(i, a.shape(), a.strides());
    int b_idx = elem_to_loc(i, b.shape(), b.strides());
    op(a_ptr + a_idx, b_ptr + b_idx, dst_a, dst_b, stride);
    dst_a += stride;
    dst_b += stride;
  }
}

template <typename T, typename U, typename Op, int D>
void binary_op_dims(
    const array& a,
    const array& b,
    array& out_a,
    array& out_b,
    Op op,
    const std::vector<int>& shape,
    const std::vector<size_t>& a_strides,
    const std::vector<size_t>& b_strides,
    const std::vector<size_t>& out_strides,
    size_t a_offset,
    size_t b_offset,
    size_t o_offset) {
  int axis = shape.size() - D;
  auto stride_a = a_strides[axis];
  auto stride_b = b_strides[axis];
  auto stride_out = out_strides[axis];
  auto N = shape[axis];

  if constexpr (D > 1) {
    for (int i = 0; i < N; i++) {
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
          a_offset,
          b_offset,
          o_offset);
      a_offset += stride_a;
      b_offset += stride_b;
      o_offset += stride_out;
    }
  } else {
    const T* a_ptr = a.data<T>() + a_offset;
    const T* b_ptr = b.data<T>() + b_offset;
    U* out_a_ptr = out_a.data<U>() + o_offset;
    U* out_b_ptr = out_b.data<U>() + o_offset;
    for (int i = 0; i < N; i++) {
      std::tie(*out_a_ptr, *out_b_ptr) = op(*a_ptr, *b_ptr);
      a_ptr += stride_a;
      b_ptr += stride_b;
      out_a_ptr += stride_out;
      out_b_ptr += stride_out;
    }
  }
}

template <typename T, typename U, typename Op>
void binary_op_dispatch_dims(
    const array& a,
    const array& b,
    array& out_a,
    array& out_b,
    Op op) {
  auto [new_shape, new_strides] = collapse_contiguous_dims(
      a.shape(), {a.strides(), b.strides(), out_a.strides()});
  const auto& a_strides = new_strides[0];
  const auto& b_strides = new_strides[1];
  const auto& out_strides = new_strides[2];

  switch (new_shape.size()) {
    case 1:
      binary_op_dims<T, U, Op, 1>(
          a,
          b,
          out_a,
          out_b,
          op,
          new_shape,
          a_strides,
          b_strides,
          out_strides,
          0,
          0,
          0);
      return;
    case 2:
      binary_op_dims<T, U, Op, 2>(
          a,
          b,
          out_a,
          out_b,
          op,
          new_shape,
          a_strides,
          b_strides,
          out_strides,
          0,
          0,
          0);
      return;
    case 3:
      binary_op_dims<T, U, Op, 3>(
          a,
          b,
          out_a,
          out_b,
          op,
          new_shape,
          a_strides,
          b_strides,
          out_strides,
          0,
          0,
          0);
      return;
    case 4:
      binary_op_dims<T, U, Op, 4>(
          a,
          b,
          out_a,
          out_b,
          op,
          new_shape,
          a_strides,
          b_strides,
          out_strides,
          0,
          0,
          0);
      return;
  }

  int size = std::accumulate(
      new_shape.end() - 4, new_shape.end(), 1, std::multiplies<int>());
  for (int i = 0; i < a.size(); i += size) {
    auto a_offset = elem_to_loc(i, new_shape, a_strides);
    auto b_offset = elem_to_loc(i, new_shape, b_strides);
    binary_op_dims<T, U, Op, 4>(
        a,
        b,
        out_a,
        out_b,
        op,
        new_shape,
        a_strides,
        b_strides,
        out_strides,
        a_offset,
        b_offset,
        i);
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
