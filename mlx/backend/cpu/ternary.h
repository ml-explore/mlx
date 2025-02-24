// Copyright © 2023 Apple Inc.

#pragma once
#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/common/ternary.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/primitives.h"

namespace mlx::core {

template <typename T1, typename T2, typename T3, typename U, typename Op, int D>
void ternary_op_dims(
    const T1* a,
    const T2* b,
    const T3* c,
    U* out,
    Op op,
    const Shape& shape,
    const Strides& a_strides,
    const Strides& b_strides,
    const Strides& c_strides,
    const Strides& out_strides,
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
    const T1* a_ptr,
    const T2* b_ptr,
    const T3* c_ptr,
    U* out_ptr,
    Op op,
    size_t size,
    Shape& shape,
    std::vector<Strides>& strides) {
  const auto& a_strides = strides[0];
  const auto& b_strides = strides[1];
  const auto& c_strides = strides[2];
  const auto& out_strides = strides[3];
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

  ContiguousIterator a_it(shape, a_strides, ndim - 2);
  ContiguousIterator b_it(shape, b_strides, ndim - 2);
  ContiguousIterator c_it(shape, c_strides, ndim - 2);
  auto stride = out_strides[ndim - 3];
  for (size_t elem = 0; elem < size; elem += stride) {
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

  auto& encoder = cpu::get_command_encoder(out.primitive().stream());
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(c);
  encoder.set_output_array(out);

  const T1* a_ptr = a.data<T1>();
  const T2* b_ptr = b.data<T2>();
  const T3* c_ptr = c.data<T3>();
  U* out_ptr = out.data<U>();

  if (topt == TernaryOpType::ScalarScalarScalar) {
    encoder.dispatch(
        [a_ptr, b_ptr, c_ptr, out_ptr, op = std::move(op)]() mutable {
          *out_ptr = op(*a_ptr, *b_ptr, *c_ptr);
        });
  } else if (topt == TernaryOpType::VectorVectorVector) {
    encoder.dispatch([a_ptr,
                      b_ptr,
                      c_ptr,
                      out_ptr,
                      op = std::move(op),
                      size = out.size()]() mutable {
      for (size_t i = 0; i < size; ++i) {
        *out_ptr = op(*a_ptr, *b_ptr, *c_ptr);
        a_ptr++;
        b_ptr++;
        c_ptr++;
        out_ptr++;
      }
    });
  } else {
    auto [shape, strides] = collapse_contiguous_dims(
        a.shape(), {a.strides(), b.strides(), c.strides(), out.strides()});
    encoder.dispatch(

        [a_ptr,
         b_ptr,
         c_ptr,
         out_ptr,
         op = std::move(op),
         size = out.size(),
         shape = std::move(shape),
         strides = std::move(strides)]() mutable {
          ternary_op_dispatch_dims<T1, T2, T3, U>(
              a_ptr, b_ptr, c_ptr, out_ptr, op, size, shape, strides);
        });
  }
}

} // namespace mlx::core
