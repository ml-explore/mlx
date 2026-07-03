// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device/indexing.cuh"
#include "mlx/backend/cuda/device/utils.cuh"

namespace mlx::core::cu {

template <typename T, typename IdxT, typename LocT, int N>
__global__ void gather_front(
    const T* src,
    const IdxT* indices,
    T* out,
    int64_t stride,
    int32_t size) {
  LocT row = blockIdx.x;
  int n_vec = (stride + N - 1) / N;
  int vec = blockIdx.y * blockDim.x + threadIdx.x;
  if (vec >= n_vec) {
    return;
  }

  auto idx = absolute_index(indices[row], size);
  const T* src_row = src + static_cast<LocT>(idx) * static_cast<LocT>(stride);
  T* out_row = out + static_cast<LocT>(row) * static_cast<LocT>(stride);

  auto v = load_vector<N>(src_row, vec, stride, static_cast<T>(0));
  store_vector<N>(out_row, vec, v, stride);
}

} // namespace mlx::core::cu
