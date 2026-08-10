// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/common/utils.h"

namespace mlx::core {

enum class BinaryOpType {
  ScalarScalar,
  ScalarVector,
  VectorScalar,
  VectorVector,
  General,
};

inline BinaryOpType get_binary_op_type(const array& a, const array& b) {
  BinaryOpType bopt;
  if (a.data_size() == 1 && b.data_size() == 1) {
    bopt = BinaryOpType::ScalarScalar;
  } else if (a.data_size() == 1 && b.flags().contiguous) {
    bopt = BinaryOpType::ScalarVector;
  } else if (b.data_size() == 1 && a.flags().contiguous) {
    bopt = BinaryOpType::VectorScalar;
  } else if (
      (a.flags().row_contiguous && b.flags().row_contiguous) ||
      (a.flags().col_contiguous && b.flags().col_contiguous)) {
    bopt = BinaryOpType::VectorVector;
  } else {
    bopt = BinaryOpType::General;
  }
  return bopt;
}

inline void set_binary_op_output_data(
    const array& a,
    const array& b,
    array& out,
    BinaryOpType bopt,
    std::function<allocator::Buffer(size_t)> mallocfn = allocator::malloc) {
  bool b_donatable = is_donatable(b, out);
  bool a_donatable = is_donatable(a, out);
  switch (bopt) {
    case BinaryOpType::ScalarScalar:
      out.set_data(mallocfn(out.itemsize()), 1, a.strides(), a.flags());
      break;
    case BinaryOpType::ScalarVector:
    case BinaryOpType::VectorScalar: {
      const auto& x = bopt == BinaryOpType::ScalarVector ? b : a;
      bool x_donatable =
          bopt == BinaryOpType::ScalarVector ? b_donatable : a_donatable;
      if (x_donatable) {
        out.copy_shared_buffer(x);
      } else {
        out.set_data(
            mallocfn(x.data_size() * out.itemsize()),
            x.data_size(),
            x.strides(),
            x.flags());
      }
      break;
    }
    case BinaryOpType::VectorVector:
      if (a_donatable) {
        out.copy_shared_buffer(a);
      } else if (b_donatable) {
        out.copy_shared_buffer(b);
      } else {
        out.set_data(
            mallocfn(a.data_size() * out.itemsize()),
            a.data_size(),
            a.strides(),
            a.flags());
      }
      break;
    case BinaryOpType::General:
      if (a_donatable && a.flags().row_contiguous && a.size() == out.size()) {
        out.copy_shared_buffer(a);
      } else if (
          b_donatable && b.flags().row_contiguous && b.size() == out.size()) {
        out.copy_shared_buffer(b);
      } else {
        out.set_data(mallocfn(out.nbytes()));
      }
      break;
  }
}

} // namespace mlx::core
