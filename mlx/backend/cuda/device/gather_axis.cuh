// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device/indexing.cuh"
#include "mlx/backend/cuda/device/utils.cuh"

#include <cooperative_groups.h>

namespace mlx::core::cu {

namespace cg = cooperative_groups;

template <
    typename T,
    typename IdxT,
    int NDIM,
    bool SrcC,
    bool IdxC,
    typename LocT>
__global__ void gather_axis(
    const T* src,
    const IdxT* indices,
    T* out,
    LocT idx_size_pre,
    LocT idx_size_axis,
    LocT idx_size_post,
    const __grid_constant__ cuda::std::array<int32_t, NDIM> shape,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> src_strides,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> idx_strides,
    int32_t axis,
    int32_t axis_size,
    int64_t src_stride_axis,
    int64_t idx_stride_axis) {
  LocT index = cg::this_grid().thread_rank();
  if (index >= idx_size_pre * idx_size_axis * idx_size_post) {
    return;
  }

  auto [x, y, z] = index_to_dims(index, idx_size_axis, idx_size_pre);

  LocT elem_idx = z * idx_size_post;

  LocT idx_loc = y * idx_stride_axis;
  if constexpr (IdxC) {
    idx_loc += elem_idx * idx_size_axis + x;
  } else {
    idx_loc +=
        elem_to_loc_nd<NDIM>(elem_idx + x, shape.data(), idx_strides.data());
  }

  auto idx_val = absolute_index(indices[idx_loc], axis_size);

  LocT src_loc = idx_val * src_stride_axis;
  if constexpr (SrcC) {
    src_loc += elem_idx * axis_size + x;
  } else {
    src_loc +=
        elem_to_loc_nd<NDIM>(elem_idx + x, shape.data(), src_strides.data());
  }

  LocT out_idx = y * idx_size_post + elem_idx * idx_size_axis + x;

  out[out_idx] = src[src_loc];
}

} // namespace mlx::core::cu
