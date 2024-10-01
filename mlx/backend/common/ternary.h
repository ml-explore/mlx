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
  VectorVectorVector,
  General,
};

TernaryOpType
get_ternary_op_type(const array& a, const array& b, const array& c) {
  TernaryOpType topt;
  if (a.data_size() == 1 && b.data_size() == 1 && c.data_size() == 1) {
    topt = TernaryOpType::ScalarScalarScalar;
  } else if (
      (a.flags().row_contiguous && b.flags().row_contiguous &&
       c.flags().row_contiguous) ||
      (a.flags().col_contiguous && b.flags().col_contiguous &&
       c.flags().col_contiguous)) {
    topt = TernaryOpType::VectorVectorVector;
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
  auto maybe_donate = [&out, donate_with_move](const array& x) {
    if (is_donatable(x, out)) {
      if (donate_with_move) {
        out.move_shared_buffer(x);
      } else {
        out.copy_shared_buffer(x);
      }
      return true;
    }
    return false;
  };

  switch (topt) {
    case TernaryOpType::ScalarScalarScalar:
      out.set_data(
          allocator::malloc_or_wait(out.itemsize()), 1, b.strides(), b.flags());
      break;
    case TernaryOpType::VectorVectorVector:
      if (!(maybe_donate(a) || maybe_donate(b) || maybe_donate(c))) {
        out.set_data(
            allocator::malloc_or_wait(out.itemsize() * b.data_size()),
            b.data_size(),
            b.strides(),
            b.flags());
      }
      break;
    case TernaryOpType::General:
      out.set_data(allocator::malloc_or_wait(out.nbytes()));
      break;
  }
}
template <typename T1, typename T2, typename T3, typename U, typename Op, int D>
void ternary_op_dims(
    const T1* a,
    const T2* b,
    const T3* c,
    U* out,
    Op op,
    const std::vector<int>& shape,
    const std::vector<size_t>& a_strides,
    const std::vector<size_t>& b_strides,
    const std::vector<size_t>& c_strides,
    const std::vector<size_t>& out_strides,
    int axis) {
  auto stride_a = a_strides[axis];
  auto stride_b = b_strides[axis];
  auto stride_c = c_strides[axis];
  auto stride_out = out_strides[axis];
  auto N = shape[axis];

  for (int i = 0; i < N; i++) {
    if constexpr (D > 1) {
      ternary_op_dims<T1, T2, T3, U, Op, D - 1>(
          a,
          b,
          c,
          out,
          op,
          shape,
          a_strides,
          b_strides,
          c_strides,
          out_strides,
          axis + 1);
    } else {
      *out = op(*a, *b, *c);
    }
    a += stride_a;
    b += stride_b;
    c += stride_c;
    out += stride_out;
  }
}

template <typename T1, typename T2, typename T3, typename U, typename Op>
void ternary_op_dispatch_dims(
    const array& a,
    const array& b,
    const array& c,
    array& out,
    Op op) {
  auto [shape, strides] = collapse_contiguous_dims(
      a.shape(), {a.strides(), b.strides(), c.strides(), out.strides()});
  const auto& a_strides = strides[0];
  const auto& b_strides = strides[1];
  const auto& c_strides = strides[2];
  const auto& out_strides = strides[3];

  const T1* a_ptr = a.data<T1>();
  const T2* b_ptr = b.data<T2>();
  const T3* c_ptr = c.data<T3>();
  U* out_ptr = out.data<T3>();
  int ndim = shape.size();
  switch (ndim) {
    case 1:
      ternary_op_dims<T1, T2, T3, U, Op, 1>(
          a_ptr,
          b_ptr,
          c_ptr,
          out_ptr,
          op,
          shape,
          a_strides,
          b_strides,
          c_strides,
          out_strides,
          0);
      return;
    case 2:
      ternary_op_dims<T1, T2, T3, U, Op, 2>(
          a_ptr,
          b_ptr,
          c_ptr,
          out_ptr,
          op,
          shape,
          a_strides,
          b_strides,
          c_strides,
          out_strides,
          0);
      return;
  }

  ContiguousIterator<size_t> a_it(shape, a_strides, ndim - 2);
  ContiguousIterator<size_t> b_it(shape, b_strides, ndim - 2);
  ContiguousIterator<size_t> c_it(shape, c_strides, ndim - 2);
  size_t stride = out_strides[ndim - 3];
  for (size_t elem = 0; elem < a.size(); elem += stride) {
    ternary_op_dims<T1, T2, T3, U, Op, 2>(
        a_ptr + a_it.loc,
        b_ptr + b_it.loc,
        c_ptr + c_it.loc,
        out_ptr + elem,
        op,
        shape,
        a_strides,
        b_strides,
        c_strides,
        out_strides,
        ndim - 2);
    a_it.step();
    b_it.step();
    c_it.step();
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
  } else if (topt == TernaryOpType::VectorVectorVector) {
    const T1* a_ptr = a.data<T1>();
    const T2* b_ptr = b.data<T2>();
    const T3* c_ptr = c.data<T3>();
    U* out_ptr = out.data<U>();
    for (size_t i = 0; i < out.size(); ++i) {
      *out_ptr = op(*a_ptr, *b_ptr, *c_ptr);
      a_ptr++;
      b_ptr++;
      c_ptr++;
      out_ptr++;
    }
  } else {
    ternary_op_dispatch_dims<T1, T2, T3, U>(a, b, c, out, op);
  }
}

} // namespace

} // namespace mlx::core
