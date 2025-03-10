// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/binary.h"

namespace mlx::core {

namespace {

template <typename T, typename U, typename Op, int D>
void binary_op_dims(
    const T* a,
    const T* b,
    U* out_a,
    U* out_b,
    Op op,
    const Shape& shape,
    const Strides& a_strides,
    const Strides& b_strides,
    const Strides& out_strides,
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
  const T* a_ptr = a.data<T>();
  const T* b_ptr = b.data<T>();
  U* out_a_ptr = out_a.data<U>();
  U* out_b_ptr = out_b.data<U>();

  const auto& a_strides = strides[0];
  const auto& b_strides = strides[1];
  const auto& out_strides = strides[2];
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

  ContiguousIterator a_it(shape, a_strides, ndim - 2);
  ContiguousIterator b_it(shape, b_strides, ndim - 2);
  auto stride = out_strides[ndim - 3];
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
    array& out_a,
    array& out_b,
    Op op,
    BinaryOpType bopt) {
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
    for (size_t i = 0; i < b.data_size(); ++i) {
      std::tie(*out_a_ptr, *out_b_ptr) = op(*a_ptr, *b_ptr);
      out_a_ptr++;
      out_b_ptr++;
      b_ptr++;
    }
  } else if (bopt == BinaryOpType::VectorScalar) {
    for (size_t i = 0; i < a.data_size(); ++i) {
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

} // namespace

} // namespace mlx::core
