// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>
#include <type_traits>

namespace mlx::core::rocm {

// Convert an absolute index to positions in a 3d grid, assuming the index is
// calculated with:
// index = x * dim1 * dim2 + y * dim2 + z
template <typename T>
inline __host__ __device__ void
index_to_dims(T index, T dim1, T dim2, T& x, T& y, T& z) {
  x = index / (dim1 * dim2);
  y = (index % (dim1 * dim2)) / dim2;
  z = index % dim2;
}

// Get absolute index from possible negative index.
template <typename IdxT>
inline __host__ __device__ auto absolute_index(IdxT idx, int32_t size) {
  if constexpr (std::is_unsigned_v<IdxT>) {
    return idx;
  } else {
    return static_cast<int32_t>(idx < 0 ? idx + size : idx);
  }
}

} // namespace mlx::core::rocm
