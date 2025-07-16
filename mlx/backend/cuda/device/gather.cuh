// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device/indexing.cuh"
#include "mlx/backend/cuda/device/utils.cuh"

#include <cooperative_groups.h>

namespace mlx::core::cu {

namespace cg = cooperative_groups;

template <typename T, typename IdxT, int NIDX, int IDX_NDIM, typename LocT>
__global__ void gather(
    const T* src,
    T* out,
    LocT size,
    const __grid_constant__ Shape src_shape,
    const __grid_constant__ Strides src_strides,
    int32_t src_ndim,
    const __grid_constant__ Shape slice_sizes,
    uint32_t slice_size,
    const __grid_constant__ cuda::std::array<int32_t, NIDX> axes,
    const __grid_constant__ cuda::std::array<IdxT*, NIDX> indices,
    const __grid_constant__ cuda::std::array<int32_t, NIDX * IDX_NDIM>
        indices_shape,
    const __grid_constant__ cuda::std::array<int64_t, NIDX * IDX_NDIM>
        indices_strides) {
  LocT out_idx = cg::this_grid().thread_rank();
  if (out_idx >= size) {
    return;
  }

  LocT src_elem = out_idx % slice_size;
  LocT idx_elem = out_idx / slice_size;

  LocT src_loc =
      elem_to_loc(src_elem, slice_sizes.data(), src_strides.data(), src_ndim);

#pragma unroll
  for (int i = 0; i < NIDX; ++i) {
    LocT idx_loc = elem_to_loc_nd<IDX_NDIM>(
        idx_elem,
        indices_shape.data() + i * IDX_NDIM,
        indices_strides.data() + i * IDX_NDIM);
    int32_t axis = axes[i];
    LocT idx_val = absolute_index(indices[i][idx_loc], src_shape[axis]);
    src_loc += idx_val * src_strides[axis];
  }

  out[out_idx] = src[src_loc];
}

} // namespace mlx::core::cu
