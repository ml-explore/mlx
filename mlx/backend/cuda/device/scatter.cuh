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
    int NIDX,
    int IDX_NDIM,
    typename LocT>
__global__ void scatter(
    const T* upd,
    T* out,
    LocT size,
    const __grid_constant__ Shape upd_shape,
    const __grid_constant__ Strides upd_strides,
    int32_t upd_ndim,
    LocT upd_post_idx_size,
    const __grid_constant__ Shape out_shape,
    const __grid_constant__ Strides out_strides,
    int32_t out_ndim,
    const __grid_constant__ cuda::std::array<int32_t, NIDX> axes,
    const __grid_constant__ cuda::std::array<IdxT*, NIDX> indices,
    const __grid_constant__ cuda::std::array<int32_t, NIDX * IDX_NDIM>
        indices_shape,
    const __grid_constant__ cuda::std::array<int64_t, NIDX * IDX_NDIM>
        indices_strides) {
  LocT upd_idx = cg::this_grid().thread_rank();
  if (upd_idx >= size) {
    return;
  }

  LocT out_elem = upd_idx % upd_post_idx_size;
  LocT idx_elem = upd_idx / upd_post_idx_size;

  LocT out_idx = elem_to_loc(
      out_elem, upd_shape.data() + IDX_NDIM, out_strides.data(), out_ndim);

#pragma unroll
  for (int i = 0; i < NIDX; ++i) {
    LocT idx_loc = elem_to_loc_nd<IDX_NDIM>(
        idx_elem,
        indices_shape.data() + i * IDX_NDIM,
        indices_strides.data() + i * IDX_NDIM);
    int32_t axis = axes[i];
    LocT idx_val = absolute_index(indices[i][idx_loc], out_shape[axis]);
    out_idx += idx_val * out_strides[axis];
  }

  LocT upd_loc = elem_to_loc(
      out_elem + idx_elem * upd_post_idx_size,
      upd_shape.data(),
      upd_strides.data(),
      upd_ndim);

  Op{}(out + out_idx, upd[upd_loc]);
}

template <typename T, bool SrcContiguous, bool DstContiguous, typename IdxT>
__global__ void masked_scatter_fused(
    const T* dst,
    const bool* mask,
    const int32_t* scatter_offsets,
    const T* src,
    T* out,
    IdxT size,
    IdxT src_batch_size,
    IdxT mask_batch_size,
    const __grid_constant__ Shape dst_shape,
    const __grid_constant__ Strides dst_strides,
    int32_t dst_ndim,
    const __grid_constant__ Shape src_shape,
    const __grid_constant__ Strides src_strides,
    int32_t src_ndim) {
  IdxT index = cg::this_grid().thread_rank();
  if (index >= size) {
    return;
  }

  T dst_val;
  if constexpr (DstContiguous) {
    dst_val = dst[index];
  } else {
    IdxT dst_loc =
        elem_to_loc(index, dst_shape.data(), dst_strides.data(), dst_ndim);
    dst_val = dst[dst_loc];
  }

  if (mask[index]) {
    IdxT src_index = static_cast<IdxT>(scatter_offsets[index]);
    if (src_index < src_batch_size) {
      IdxT batch_idx = index / mask_batch_size;
      if constexpr (SrcContiguous) {
        out[index] = src[batch_idx * src_batch_size + src_index];
      } else {
        IdxT src_elem = batch_idx * src_batch_size + src_index;
        IdxT src_loc = elem_to_loc(
            src_elem, src_shape.data(), src_strides.data(), src_ndim);
        out[index] = src[src_loc];
      }
      return;
    }
  }

  out[index] = dst_val;
}

template <typename IdxT, int ITEMS_PER_THREAD>
__global__ void masked_scatter_tile_count(
    const bool* mask,
    int32_t* tile_counts,
    IdxT mask_batch_size,
    int32_t num_tiles_per_batch) {
  IdxT tile = cg::this_grid().block_rank();
  IdxT batch_idx = tile / num_tiles_per_batch;
  IdxT tile_in_batch = tile - batch_idx * num_tiles_per_batch;
  IdxT tile_items = static_cast<IdxT>(blockDim.x) * ITEMS_PER_THREAD;
  IdxT tile_start = batch_idx * mask_batch_size + tile_in_batch * tile_items;
  IdxT batch_end = (batch_idx + 1) * mask_batch_size;
  IdxT tile_end = tile_start + tile_items;
  if (tile_end > batch_end) {
    tile_end = batch_end;
  }

  int32_t local_count = 0;
  IdxT index = tile_start + threadIdx.x;
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    if (index < tile_end) {
      local_count += static_cast<int32_t>(mask[index]);
    }
    index += blockDim.x;
  }

  int lane = threadIdx.x & (WARP_SIZE - 1);
  int warp = threadIdx.x / WARP_SIZE;
  int nwarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

  unsigned int active = __activemask();
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    local_count += __shfl_down_sync(active, local_count, offset);
  }

  __shared__ int32_t warp_sums[WARP_SIZE];
  if (lane == 0) {
    warp_sums[warp] = local_count;
  }
  __syncthreads();

  if (warp == 0) {
    int32_t block_sum = (lane < nwarps) ? warp_sums[lane] : 0;
    unsigned int warp0_active = __activemask();
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      block_sum += __shfl_down_sync(warp0_active, block_sum, offset);
    }
    if (lane == 0) {
      tile_counts[tile] = block_sum;
    }
  }
}

template <typename T, typename IdxT, int ITEMS_PER_THREAD>
__global__ void masked_scatter_fused_vec_contiguous(
    const T* dst,
    const bool* mask,
    const int32_t* tile_offsets,
    const T* src,
    T* out,
    IdxT src_batch_size,
    IdxT mask_batch_size,
    int32_t num_tiles_per_batch) {
  IdxT tile = cg::this_grid().block_rank();
  IdxT batch_idx = tile / num_tiles_per_batch;
  IdxT tile_in_batch = tile - batch_idx * num_tiles_per_batch;
  IdxT tile_items = static_cast<IdxT>(blockDim.x) * ITEMS_PER_THREAD;
  IdxT tile_start = batch_idx * mask_batch_size + tile_in_batch * tile_items;
  IdxT batch_end = (batch_idx + 1) * mask_batch_size;
  IdxT tile_end = tile_start + tile_items;
  if (tile_end > batch_end) {
    tile_end = batch_end;
  }

  IdxT src_base = batch_idx * src_batch_size;
  IdxT tile_prefix = static_cast<IdxT>(tile_offsets[tile]);
  IdxT iter_prefix = 0;

  int lane = threadIdx.x & (WARP_SIZE - 1);
  int warp = threadIdx.x / WARP_SIZE;
  int nwarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

  __shared__ int32_t warp_counts[WARP_SIZE];
  __shared__ int32_t warp_offsets[WARP_SIZE];
  __shared__ int32_t iter_count;

  IdxT index = tile_start + threadIdx.x;

#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    bool active = index < tile_end;
    bool mask_value = active ? mask[index] : false;
    T out_value = active ? dst[index] : static_cast<T>(0);

    unsigned int active_mask = __activemask();
    unsigned int ballots = __ballot_sync(active_mask, mask_value);
    unsigned int lane_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);
    int32_t warp_exclusive = __popc(ballots & lane_mask);
    int32_t warp_count = __popc(ballots);

    if (lane == 0) {
      warp_counts[warp] = warp_count;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      int32_t offset = 0;
      for (int w = 0; w < nwarps; ++w) {
        warp_offsets[w] = offset;
        offset += warp_counts[w];
      }
      iter_count = offset;
    }
    __syncthreads();

    if (active && mask_value) {
      IdxT src_index = tile_prefix + iter_prefix +
          static_cast<IdxT>(warp_offsets[warp] + warp_exclusive);
      if (src_index < src_batch_size) {
        out_value = src[src_base + src_index];
      }
    }

    if (active) {
      out[index] = out_value;
    }

    iter_prefix += static_cast<IdxT>(iter_count);
    index += blockDim.x;
    __syncthreads();
  }
}

} // namespace mlx::core::cu
