// Copyright © 2023 Apple Inc.

#pragma once
#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/common/utils.h"

namespace mlx::core {

// TODO: Add support for more combinations of input types.
enum class TernaryOpType {
  ScalarScalarScalar,
  VectorVectorVector,
  General,
};

inline TernaryOpType
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

inline void set_ternary_op_output_data(
    const array& a,
    const array& b,
    const array& c,
    array& out,
    TernaryOpType topt) {
  auto maybe_donate = [&out](const array& x) {
    if (is_donatable(x, out)) {
      out.copy_shared_buffer(x);
      return true;
    }
    return false;
  };

  switch (topt) {
    case TernaryOpType::ScalarScalarScalar:
      out.set_data(
          allocator::malloc(out.itemsize()), 1, b.strides(), b.flags());
      break;
    case TernaryOpType::VectorVectorVector:
      if (!(maybe_donate(a) || maybe_donate(b) || maybe_donate(c))) {
        out.set_data(
            allocator::malloc(out.itemsize() * b.data_size()),
            b.data_size(),
            b.strides(),
            b.flags());
      }
      break;
    case TernaryOpType::General:
      // Try to donate an input which is row_contiguous
      if (!((a.flags().row_contiguous && maybe_donate(a)) ||
            (b.flags().row_contiguous && maybe_donate(b)) ||
            (c.flags().row_contiguous && maybe_donate(c)))) {
        out.set_data(allocator::malloc(out.nbytes()));
      }
      break;
  }
}

} // namespace mlx::core
