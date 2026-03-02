// Copyright © 2025 Apple Inc.

#include <cuda/std/tuple>
#include <cuda/std/type_traits>

namespace mlx::core::cu {

// Convert an absolute index to positions in a 3d grid, assuming the index is
// calculated with:
// index = x * dim1 * dim2 + y * dim2 + z
template <typename T>
inline __host__ __device__ cuda::std::tuple<T, T, T>
index_to_dims(T index, T dim1, T dim2) {
  T x = index / (dim1 * dim2);
  T y = (index % (dim1 * dim2)) / dim2;
  T z = index % dim2;
  return cuda::std::make_tuple(x, y, z);
}

// Get absolute index from possible negative index.
template <typename IdxT>
inline __host__ __device__ auto absolute_index(IdxT idx, int32_t size) {
  if constexpr (cuda::std::is_unsigned_v<IdxT>) {
    return idx;
  } else {
    return static_cast<int32_t>(idx < 0 ? idx + size : idx);
  }
}

} // namespace mlx::core::cu
