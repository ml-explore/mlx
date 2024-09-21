// Copyright © 2023 Apple Inc.

#pragma once
#include <numeric>
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
    const array& a,
    const array& b,
    const array& c,
    array& out,
    Op op,
    const std::vector<int>& shape,
    const std::vector<size_t>& a_strides,
    const std::vector<size_t>& b_strides,
    const std::vector<size_t>& c_strides,
    const std::vector<size_t>& out_strides,
    size_t a_offset,
    size_t b_offset,
    size_t c_offset,
    size_t o_offset) {
  int axis = shape.size() - D;
  auto stride_a = a_strides[axis];
  auto stride_b = b_strides[axis];
  auto stride_c = c_strides[axis];
  auto stride_out = out_strides[axis];
  auto N = shape[axis];

  if constexpr (D > 1) {
    for (int i = 0; i < N; i++) {
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
          a_offset,
          b_offset,
          c_offset,
          o_offset);
      a_offset += stride_a;
      b_offset += stride_b;
      c_offset += stride_c;
      o_offset += stride_out;
    }
  } else {
    const T1* a_ptr = a.data<T1>() + a_offset;
    const T2* b_ptr = b.data<T2>() + b_offset;
    const T3* c_ptr = c.data<T3>() + c_offset;
    U* out_ptr = out.data<U>() + o_offset;
    for (int i = 0; i < N; i++) {
      *out_ptr = op(*a_ptr, *b_ptr, *c_ptr);
      a_ptr += stride_a;
      b_ptr += stride_b;
      c_ptr += stride_c;
      out_ptr += stride_out;
    }
  }
}

template <typename T1, typename T2, typename T3, typename U, typename Op>
void ternary_op_dispatch_dims(
    const array& a,
    const array& b,
    const array& c,
    array& out,
    Op op) {
  auto [new_shape, new_strides] = collapse_contiguous_dims(
      a.shape(), {a.strides(), b.strides(), c.strides(), out.strides()});
  const auto& a_strides = new_strides[0];
  const auto& b_strides = new_strides[1];
  const auto& c_strides = new_strides[2];
  const auto& out_strides = new_strides[3];

  switch (new_shape.size()) {
    case 1:
      ternary_op_dims<T1, T2, T3, U, Op, 1>(
          a,
          b,
          c,
          out,
          op,
          new_shape,
          a_strides,
          b_strides,
          c_strides,
          out_strides,
          0,
          0,
          0,
          0);
      return;
    case 2:
      ternary_op_dims<T1, T2, T3, U, Op, 2>(
          a,
          b,
          c,
          out,
          op,
          new_shape,
          a_strides,
          b_strides,
          c_strides,
          out_strides,
          0,
          0,
          0,
          0);
      return;
    case 3:
      ternary_op_dims<T1, T2, T3, U, Op, 3>(
          a,
          b,
          c,
          out,
          op,
          new_shape,
          a_strides,
          b_strides,
          c_strides,
          out_strides,
          0,
          0,
          0,
          0);
      return;
    case 4:
      ternary_op_dims<T1, T2, T3, U, Op, 4>(
          a,
          b,
          c,
          out,
          op,
          new_shape,
          a_strides,
          b_strides,
          c_strides,
          out_strides,
          0,
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
    auto c_offset = elem_to_loc(i, new_shape, c_strides);
    ternary_op_dims<T1, T2, T3, U, Op, 4>(
        a,
        b,
        c,
        out,
        op,
        new_shape,
        a_strides,
        b_strides,
        c_strides,
        out_strides,
        a_offset,
        b_offset,
        c_offset,
        i);
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
