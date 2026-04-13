// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device/binary_ops.cuh"
#include "mlx/backend/cuda/device/utils.cuh"

#include <cooperative_groups.h>

namespace mlx::core::cu {

namespace cg = cooperative_groups;

template <
    typename T,
    typename IdxT,
    typename Op,
    bool OUT_ROW_CONTIG,
    bool UPD_ROW_CONTIG,
    bool UPD_SCALAR,
    int NWORK>
__global__ void slice_update_op(
    const T* updates,
    T* out,
    int64_t update_size,
    const __grid_constant__ Shape update_shape,
    const __grid_constant__ Strides update_strides,
    int32_t update_ndim,
    const __grid_constant__ Strides output_strides,
    int64_t output_offset) {
  Op op;

  IdxT idx = cg::this_grid().thread_rank() * NWORK;
  IdxT out_idx;
  IdxT update_idx;

  if constexpr (OUT_ROW_CONTIG) {
    out_idx = idx;
  } else {
    out_idx = elem_to_loc<IdxT>(
        idx, update_shape.data(), output_strides.data(), update_ndim);
  }

  if constexpr (!UPD_SCALAR) {
    if constexpr (UPD_ROW_CONTIG) {
      update_idx = idx;
    } else {
      update_idx = elem_to_loc<IdxT>(
          idx, update_shape.data(), update_strides.data(), update_ndim);
    }
  } else {
    update_idx = 0;
  }

  out += output_offset;

  for (int j = 0; j < NWORK && idx < update_size; j++) {
    out[out_idx] = op(out[out_idx], updates[update_idx]);
    idx++;

    if constexpr (OUT_ROW_CONTIG) {
      out_idx = idx;
    } else {
      out_idx += output_strides[update_ndim - 1];
    }

    if constexpr (UPD_ROW_CONTIG) {
      update_idx = idx;
    } else if constexpr (!UPD_SCALAR) {
      update_idx += update_strides[update_ndim - 1];
    }
  }
}

} // namespace mlx::core::cu
