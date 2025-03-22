// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/kernels/binary_ops.cuh"

#include <cooperative_groups.h>

namespace mlx::core::mxcuda {

namespace cg = cooperative_groups;

template <typename Op, typename T, typename U>
__global__ void binary_ss(const T* a, const T* b, U* c, uint32_t size) {
  uint32_t index = cg::this_grid().thread_rank();
  if (index < size) {
    c[index] = Op()(a[0], b[0]);
  }
}

template <typename Op, typename T, typename U>
__global__ void binary_sv(const T* a, const T* b, U* c, uint32_t size) {
  uint32_t index = cg::this_grid().thread_rank();
  if (index < size) {
    c[index] = Op()(a[0], b[index]);
  }
}

template <typename Op, typename T, typename U>
__global__ void binary_vs(const T* a, const T* b, U* c, uint32_t size) {
  uint32_t index = cg::this_grid().thread_rank();
  if (index < size) {
    c[index] = Op()(a[index], b[0]);
  }
}

template <typename Op, typename T, typename U>
__global__ void binary_vv(const T* a, const T* b, U* c, uint32_t size) {
  uint32_t index = cg::this_grid().thread_rank();
  if (index < size) {
    c[index] = Op()(a[index], b[index]);
  }
}

} // namespace mlx::core::mxcuda
