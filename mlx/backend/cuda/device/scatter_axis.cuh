// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device/indexing.cuh"
#include "mlx/backend/cuda/device/scatter_ops.cuh"
#include "mlx/backend/cuda/device/utils.cuh"

#include <cooperative_groups.h>

namespace mlx::core::cu {

namespace cg = cooperative_groups;

template <
    typename T,
    typename IdxT,
    typename Op,
    int NDIM,
    bool UpdC,
    bool IdxC,
    typename LocT>
__global__ void scatter_axis(
    const T* upd,
    const IdxT* indices,
    T* out,
    LocT idx_size_pre,
    LocT idx_size_axis,
    LocT idx_size_post,
    const __grid_constant__ cuda::std::array<int32_t, NDIM> shape,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> upd_strides,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> idx_strides,
    int32_t axis,
    int32_t axis_size,
    int64_t upd_stride_axis,
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

  LocT upd_loc = y * upd_stride_axis;
  if constexpr (UpdC) {
    upd_loc += elem_idx * idx_size_axis + x;
  } else {
    upd_loc +=
        elem_to_loc_nd<NDIM>(elem_idx + x, shape.data(), upd_strides.data());
  }

  LocT out_idx = idx_val * idx_size_post + elem_idx * axis_size + x;

  Op{}(out + out_idx, upd[upd_loc]);
}

} // namespace mlx::core::cu
