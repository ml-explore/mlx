// Copyright © 2023 Apple Inc.

#pragma once
#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/common/ops.h"
#include "mlx/backend/common/utils.h"
namespace mlx::core {

namespace {

// TODO: Add support for more combinations of input types.
enum class TernaryOpType {
  ScalarScalarScalar,
  General,
};

TernaryOpType
get_ternary_op_type(const array& a, const array& b, const array& c) {
  TernaryOpType topt;
  if (a.data_size() == 1 && b.data_size() == 1 && c.data_size() == 1) {
    topt = TernaryOpType::ScalarScalarScalar;
  } else {
    topt = TernaryOpType::General;
  }
  return topt;
}

void set_ternary_op_output_data(
    const array& a,
    const array& b,
    const array& c,
    array& out,
    TernaryOpType topt,
    bool donate_with_move = false) {
  switch (topt) {
    case TernaryOpType::ScalarScalarScalar:
      out.set_data(
          allocator::malloc_or_wait(out.itemsize()), 1, b.strides(), b.flags());
      break;
    case TernaryOpType::General:
      out.set_data(allocator::malloc_or_wait(out.nbytes()));
      break;
  }
}

template <typename T1, typename T2, typename T3, typename U, typename Op>
void ternary_op_dims1(
    const array& a,
    const array& b,
    const array& c,
    array& out,
    Op op) {
  const T1* a_ptr = a.data<T1>();
  const T2* b_ptr = b.data<T2>();
  const T3* c_ptr = c.data<T3>();

  U* dst = out.data<U>();
  size_t a_idx = 0;
  size_t b_idx = 0;
  size_t c_idx = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    dst[i] = op(a_ptr[a_idx], b_ptr[b_idx], c_ptr[c_idx]);
    a_idx += a.strides()[0];
    b_idx += b.strides()[0];
    c_idx += c.strides()[0];
  }
}

template <typename T1, typename T2, typename T3, typename U, typename Op>
void ternary_op_dims2(
    const array& a,
    const array& b,
    const array& c,
    array& out,
    Op op) {
  const T1* a_ptr = a.data<T1>();
  const T2* b_ptr = b.data<T2>();
  const T3* c_ptr = c.data<T3>();

  U* dst = out.data<U>();
  size_t a_idx = 0;
  size_t b_idx = 0;
  size_t c_idx = 0;
  size_t out_idx = 0;
  for (size_t i = 0; i < a.shape()[0]; ++i) {
    for (size_t j = 0; j < a.shape()[1]; ++j) {
      dst[out_idx++] = op(a_ptr[a_idx], b_ptr[b_idx], c_ptr[c_idx]);
      a_idx += a.strides()[1];
      b_idx += b.strides()[1];
      c_idx += c.strides()[1];
    }
    a_idx += a.strides()[0] - a.strides()[1] * a.shape()[1];
    b_idx += b.strides()[0] - b.strides()[1] * b.shape()[1];
    c_idx += c.strides()[0] - c.strides()[1] * c.shape()[1];
  }
}

template <typename T1, typename T2, typename T3, typename U, typename Op>
void ternary_op_dims3(
    const array& a,
    const array& b,
    const array& c,
    array& out,
    Op op) {
  const T1* a_ptr = a.data<T1>();
  const T2* b_ptr = b.data<T2>();
  const T3* c_ptr = c.data<T3>();
  U* dst = out.data<U>();
  size_t a_idx = 0;
  size_t b_idx = 0;
  size_t c_idx = 0;
  size_t out_idx = 0;
  for (size_t i = 0; i < a.shape()[0]; ++i) {
    for (size_t j = 0; j < a.shape()[1]; ++j) {
      for (size_t k = 0; k < a.shape()[2]; ++k) {
        dst[out_idx++] = op(a_ptr[a_idx], b_ptr[b_idx], c_ptr[c_idx]);
        a_idx += a.strides()[2];
        b_idx += b.strides()[2];
        c_idx += c.strides()[2];
      }
      a_idx += a.strides()[1] - a.strides()[2] * a.shape()[2];
      b_idx += b.strides()[1] - b.strides()[2] * b.shape()[2];
      c_idx += c.strides()[1] - c.strides()[2] * c.shape()[2];
    }
    a_idx += a.strides()[0] - a.strides()[1] * a.shape()[1];
    b_idx += b.strides()[0] - b.strides()[1] * b.shape()[1];
    c_idx += c.strides()[0] - c.strides()[1] * c.shape()[1];
  }
}

template <typename T1, typename T2, typename T3, typename U, typename Op>
void ternary_op_dims4(
    const array& a,
    const array& b,
    const array& c,
    array& out,
    Op op) {
  const T1* a_ptr = a.data<T1>();
  const T2* b_ptr = b.data<T2>();
  const T3* c_ptr = c.data<T3>();

  U* dst = out.data<U>();
  size_t a_idx = 0;
  size_t b_idx = 0;
  size_t c_idx = 0;
  size_t out_idx = 0;
  for (size_t i = 0; i < a.shape()[0]; ++i) {
    for (size_t j = 0; j < a.shape()[1]; ++j) {
      for (size_t k = 0; k < a.shape()[2]; ++k) {
        for (size_t ii = 0; ii < a.shape()[3]; ++ii) {
          dst[out_idx++] = op(a_ptr[a_idx], b_ptr[b_idx], c_ptr[c_idx]);
          a_idx += a.strides()[3];
          b_idx += b.strides()[3];
          c_idx += c.strides()[3];
        }
        a_idx += a.strides()[2] - a.strides()[3] * a.shape()[3];
        b_idx += b.strides()[2] - b.strides()[3] * b.shape()[3];
        c_idx += c.strides()[2] - c.strides()[3] * c.shape()[3];
      }
      a_idx += a.strides()[1] - a.strides()[2] * a.shape()[2];
      b_idx += b.strides()[1] - b.strides()[2] * b.shape()[2];
      c_idx += c.strides()[1] - c.strides()[2] * c.shape()[2];
    }
    a_idx += a.strides()[0] - a.strides()[1] * a.shape()[1];
    b_idx += b.strides()[0] - b.strides()[1] * b.shape()[1];
    c_idx += c.strides()[0] - c.strides()[1] * c.shape()[1];
  }
}

template <typename T1, typename T2, typename T3, typename U, typename Op>
void ternary_op_dispatch_dims(
    const array& a,
    const array& b,
    const array& c,
    array& out,
    Op op) {
  switch (out.ndim()) {
    case 1:
      ternary_op_dims1<T1, T2, T3, U, Op>(a, b, c, out, op);
      return;
    case 2:
      ternary_op_dims2<T1, T2, T3, U, Op>(a, b, c, out, op);
      return;
    case 3:
      ternary_op_dims3<T1, T2, T3, U, Op>(a, b, c, out, op);
      return;
    case 4:
      ternary_op_dims4<T1, T2, T3, U, Op>(a, b, c, out, op);
      return;
  }

  const T1* a_ptr = a.data<T1>();
  const T2* b_ptr = b.data<T2>();
  const T3* c_ptr = c.data<T3>();
  U* dst = out.data<U>();
  for (size_t i = 0; i < out.size(); i++) {
    int a_idx = elem_to_loc(i, a.shape(), a.strides());
    int b_idx = elem_to_loc(i, b.shape(), b.strides());
    int c_idx = elem_to_loc(i, c.shape(), c.strides());
    dst[i] = op(a_ptr[a_idx], b_ptr[b_idx], c_ptr[c_idx]);
  }
}

template <typename T1, typename T2, typename T3, typename U, typename Op>
void ternary_op(
    const array& a,
    const array& b,
    const array& c,
    array& out,
    Op op) {
  TernaryOpType topt = get_ternary_op_type(a, b, c);
  set_ternary_op_output_data(a, b, c, out, topt);

  // The full computation is scalar-scalar-scalar so we call the base op once.
  if (topt == TernaryOpType::ScalarScalarScalar) {
    *(out.data<U>()) = op(*a.data<T1>(), *b.data<T2>(), *c.data<T3>());
    return;
  }

  ternary_op_dispatch_dims<T1, T2, T3, U>(a, b, c, out, op);
}

} // namespace

} // namespace mlx::core
