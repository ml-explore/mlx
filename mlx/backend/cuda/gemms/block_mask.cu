// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/utils.cuh"
#include "mlx/backend/cuda/gemms/block_mask.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"

namespace mlx::core::cu {

template <typename T, typename MaskT>
__global__ void block_mask_matrix(
    T* data,
    const MaskT* mask,
    int block_size,
    int rows,
    int cols,
    int64_t data_batch_stride,
    const __grid_constant__ Shape mask_shape,
    const __grid_constant__ Strides mask_strides,
    int mask_ndim,
    int64_t mask_row_stride,
    int64_t mask_col_stride,
    int mask_mat_size,
    int batch_count) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = int64_t(batch_count) * rows * cols;
  if (idx >= total)
    return;

  int mat_size = rows * cols;
  int batch = idx / mat_size;
  int within = idx % mat_size;
  int row = within / cols;
  int col = within % cols;
  int mask_row = row / block_size;
  int mask_col = col / block_size;

  // Compute mask batch offset (handles broadcasting via stride=0).
  int64_t mask_batch_offset = elem_to_loc(
      int64_t(batch) * mask_mat_size,
      mask_shape.data(),
      mask_strides.data(),
      mask_ndim);
  MaskT mask_val = mask
      [mask_batch_offset + mask_row * mask_row_stride +
       mask_col * mask_col_stride];

  int64_t data_offset = int64_t(batch) * data_batch_stride + within;
  if constexpr (std::is_same_v<MaskT, bool>) {
    if (!mask_val) {
      data[data_offset] = T(0);
    }
  } else {
    data[data_offset] *= T(mask_val);
  }
}

} // namespace mlx::core::cu

namespace mlx::core {

void apply_block_mask(
    cu::CommandEncoder& encoder,
    array& data,
    const array& mask,
    int block_size,
    int rows,
    int cols,
    int64_t data_batch_stride,
    int batch_count) {
  encoder.set_input_array(mask);
  encoder.set_output_array(data);

  int mask_ndim = mask.ndim();
  int64_t mask_row_stride = mask.strides()[mask_ndim - 2];
  int64_t mask_col_stride = mask.strides()[mask_ndim - 1];
  int mask_mat_size = mask.shape()[mask_ndim - 2] * mask.shape()[mask_ndim - 1];

  int64_t total = int64_t(batch_count) * rows * cols;
  constexpr int BLOCK = 256;
  int grid = (total + BLOCK - 1) / BLOCK;

  auto launch = [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    auto data_ptr = gpu_ptr<T>(data);

    auto do_mask = [&](auto mask_tag) {
      using MaskT = decltype(mask_tag);
      const MaskT* mask_ptr;
      if constexpr (std::is_same_v<MaskT, bool>) {
        mask_ptr = gpu_ptr<bool>(mask);
      } else {
        mask_ptr = gpu_ptr<T>(mask);
      }
      encoder.add_kernel_node(
          cu::block_mask_matrix<T, MaskT>,
          grid,
          BLOCK,
          data_ptr,
          mask_ptr,
          block_size,
          rows,
          cols,
          data_batch_stride,
          const_param(mask.shape()),
          const_param(mask.strides()),
          mask_ndim,
          mask_row_stride,
          mask_col_stride,
          mask_mat_size,
          batch_count);
    };

    if (mask.dtype() == bool_) {
      do_mask(bool{});
    } else {
      do_mask(T{});
    }
  };

  dispatch_float_types(data.dtype(), "block_mask_matrix", launch);
}

} // namespace mlx::core
