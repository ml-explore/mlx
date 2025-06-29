// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include <sstream>

namespace mlx::core {

inline std::tuple<Shape, Strides, Strides> collapse_batches(
    const array& a,
    const array& b) {
  if (a.ndim() == 2) {
    return {{1}, {0}, {0}};
  }

  Shape A_bshape{a.shape().begin(), a.shape().end() - 2};
  Strides A_bstride{a.strides().begin(), a.strides().end() - 2};
  Strides B_bstride{b.strides().begin(), b.strides().end() - 2};

  auto [batch_shape, batch_strides] =
      collapse_contiguous_dims(A_bshape, std::vector{A_bstride, B_bstride});

  auto a_batch_strides = batch_strides[0];
  auto b_batch_strides = batch_strides[1];

  if (batch_shape.empty()) {
    batch_shape.push_back(1);
    a_batch_strides.push_back(0);
    b_batch_strides.push_back(0);
  }

  return std::make_tuple(batch_shape, a_batch_strides, b_batch_strides);
}

inline std::tuple<Shape, Strides, Strides, Strides>
collapse_batches(const array& a, const array& b, const array& c) {
  if (a.ndim() == 2) {
    return {{1}, {0}, {0}, {0}};
  }

  Shape A_bshape{a.shape().begin(), a.shape().end() - 2};
  Strides A_bstride{a.strides().begin(), a.strides().end() - 2};
  Strides B_bstride{b.strides().begin(), b.strides().end() - 2};
  Strides C_bstride{c.strides().begin(), c.strides().end() - 2};

  auto [batch_shape, batch_strides] = collapse_contiguous_dims(
      A_bshape, std::vector{A_bstride, B_bstride, C_bstride});

  auto A_batch_stride = batch_strides[0];
  auto B_batch_stride = batch_strides[1];
  auto C_batch_stride = batch_strides[2];

  if (batch_shape.empty()) {
    batch_shape.push_back(1);
    A_batch_stride.push_back(0);
    B_batch_stride.push_back(0);
    C_batch_stride.push_back(0);
  }

  return std::make_tuple(
      batch_shape, A_batch_stride, B_batch_stride, C_batch_stride);
}

} // namespace mlx::core
