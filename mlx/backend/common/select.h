// Copyright © 2023 Apple Inc.

#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>

#include "mlx/allocator.h"
#include "mlx/backend/common/binary.h"

namespace mlx::core {

namespace {

// Returns the binary operation type that will be used in 2 binary op calls in
// 'select'. Once with (condition, a) and another with (condition, b).
BinaryOpType get_select_binary_op_type(
    const array& condition,
    const array& a,
    const array& b) {
  BinaryOpType bopt;

  auto are_fully_contiguous = [](const array& x, const array& y) {
    return (x.flags().row_contiguous && y.flags().row_contiguous) ||
        (x.flags().col_contiguous && y.flags().col_contiguous);
  };

  if (condition.data_size() == 1 && a.data_size() == 1 && b.data_size() == 1) {
    bopt = ScalarScalar;
  } else if (condition.data_size() == 1 && are_fully_contiguous(a, b)) {
    bopt = ScalarVector;
  } else if (
      condition.flags().contiguous && a.data_size() == 1 &&
      b.data_size() == 1) {
    bopt = VectorScalar;
  } else if (
      are_fully_contiguous(condition, a) &&
      are_fully_contiguous(condition, b)) {
    bopt = VectorVector;
  } else {
    bopt = General;
  }
  return bopt;
}

void set_select_binary_op_output_data(
    const array& condition,
    const array& a,
    const array& b,
    array& out,
    BinaryOpType bopt,
    bool donate_with_move = false) {
  switch (bopt) {
    case ScalarScalar:
      out.set_data(
          allocator::malloc_or_wait(out.itemsize()), 1, a.strides(), a.flags());
    case ScalarVector:
    case VectorVector:
      if (a.is_donatable() && a.itemsize() == out.itemsize()) {
        if (donate_with_move) {
          out.move_shared_buffer(a);
        } else {
          out.copy_shared_buffer(a);
        }
      } else if (b.is_donatable() && b.itemsize() == out.itemsize()) {
        if (donate_with_move) {
          out.move_shared_buffer(b);
        } else {
          out.copy_shared_buffer(b);
        }
      } else {
        out.set_data(allocator::malloc_or_wait(out.nbytes()));
      }
      break;
    case VectorScalar:
      // The buffer of `condition` can't be donated because it's called twice in
      // the 'select' op.
      out.set_data(
          allocator::malloc_or_wait(condition.data_size() * out.itemsize()),
          condition.data_size(),
          condition.strides(),
          condition.flags());
      break;
    case General:
      out.set_data(allocator::malloc_or_wait(out.nbytes()));
      break;
  }
}

template <typename T>
struct SelectScalarScalar {
  bool invert_predicate;

  SelectScalarScalar(bool invert_predicate_)
      : invert_predicate(invert_predicate_) {}
  void operator()(const bool* a, const T* b, T* dst) {
    if ((*a && !invert_predicate) || (!*a && invert_predicate)) {
      *dst = *b;
    }
  }
};

template <typename T>
struct SelectVectorScalar {
  bool invert_predicate;

  SelectVectorScalar(bool invert_predicate_)
      : invert_predicate(invert_predicate_) {}
  void operator()(const bool* a, const T* b, T* dst, int size) {
    T scalar = *b;
    while (size-- > 0) {
      if ((*a && !invert_predicate) || (!*a && invert_predicate)) {
        *dst = scalar;
      }
      dst++;
      a++;
    }
  }
};

template <typename T>
struct SelectScalarVector {
  bool invert_predicate;

  SelectScalarVector(bool invert_predicate_)
      : invert_predicate(invert_predicate_) {}
  void operator()(const bool* a, const T* b, T* dst, int size) {
    bool scalar = *a;

    while (size-- > 0) {
      if ((scalar && !invert_predicate) || (!scalar && invert_predicate)) {
        *dst = *b;
      }
      dst++;
      b++;
    }
  }
};

template <typename T>
struct SelectVectorVector {
  bool invert_predicate;

  SelectVectorVector(bool invert_predicate_)
      : invert_predicate(invert_predicate_) {}
  void operator()(const bool* a, const T* b, T* dst, int size) {
    while (size-- > 0) {
      if ((*a && !invert_predicate) || (!*a && invert_predicate)) {
        *dst = *b;
      }
      dst++;
      a++;
      b++;
    }
  }
};

} // namespace

} // namespace mlx::core
