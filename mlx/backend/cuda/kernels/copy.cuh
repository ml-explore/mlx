// Copyright © 2025 Apple Inc.

#include <cooperative_groups.h>

namespace mlx::core::mxcuda {

namespace cg = cooperative_groups;

template <typename T, typename U>
__global__ void copy_s(const T* src, U* dst, uint32_t size) {
  uint32_t index = cg::this_grid().thread_rank();
  if (index < size) {
    dst[index] = static_cast<U>(src[0]);
  }
}

template <typename T, typename U>
__global__ void copy_v(const T* src, U* dst, uint32_t size) {
  uint32_t index = cg::this_grid().thread_rank();
  if (index < size) {
    dst[index] = static_cast<U>(src[index]);
  }
}

} // namespace mlx::core::mxcuda
