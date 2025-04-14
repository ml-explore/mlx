// Copyright © 2025 Apple Inc.

#pragma once

#include <thrust/iterator/transform_iterator.h>

namespace mlx::core::cu {

template <typename T, typename U>
struct CastOp {
  __device__ U operator()(T x) {
    return static_cast<U>(x);
  }
};

// Return an iterator that static_cast the value to T.
template <typename T, typename Iterator>
auto make_cast_iterator(Iterator it) {
  return thrust::make_transform_iterator(
      it, CastOp<typename Iterator::value_type, T>{});
}

} // namespace mlx::core::cu
