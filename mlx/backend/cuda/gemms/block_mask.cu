// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/utils.cuh"
#include "mlx/backend/cuda/gemms/block_mask.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"

namespace mlx::core {

namespace cu {

template <typename T, typename MaskT>
__global__ void block_mask_inplace(
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

  int64_t mask_batch_offset = elem_to_loc(
      int64_t(batch) * mask_mat_size,
      mask_shape.data(),
      mask_strides.data(),
      mask_ndim);
  MaskT mask_val = mask
      [mask_batch_offset + (within / cols) / block_size * mask_row_stride +
       (within % cols) / block_size * mask_col_stride];

  int64_t data_offset = int64_t(batch) * data_batch_stride + within;
  if constexpr (std::is_same_v<MaskT, bool>) {
    if (!mask_val) {
      data[data_offset] = T(0);
    }
  } else {
    data[data_offset] *= T(mask_val);
  }
}

template <typename T, typename MaskT, bool SrcContiguous>
__global__ void block_mask_copy(
    const T* src,
    T* dst,
    int block_size,
    int rows,
    int cols,
    const __grid_constant__ Shape src_shape,
    const __grid_constant__ Strides src_strides,
    int src_ndim,
    const MaskT* mask,
    const __grid_constant__ Shape mask_shape,
    const __grid_constant__ Strides mask_strides,
    int mask_ndim,
    int64_t mask_row_stride,
    int64_t mask_col_stride,
    int mask_mat_size,
    int batch_count) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int mat_size = rows * cols;
  int64_t total = int64_t(batch_count) * mat_size;
  if (idx >= total)
    return;

  int batch = idx / mat_size;
  int within = idx % mat_size;

  int64_t mask_batch_offset = elem_to_loc(
      int64_t(batch) * mask_mat_size,
      mask_shape.data(),
      mask_strides.data(),
      mask_ndim);
  MaskT mask_val = mask
      [mask_batch_offset + (within / cols) / block_size * mask_row_stride +
       (within % cols) / block_size * mask_col_stride];

  int64_t src_offset;
  if constexpr (SrcContiguous) {
    src_offset = idx;
  } else {
    src_offset = elem_to_loc(
        int64_t(batch) * mat_size + within,
        src_shape.data(),
        src_strides.data(),
        src_ndim);
  }

  if constexpr (std::is_same_v<MaskT, bool>) {
    dst[idx] = mask_val ? src[src_offset] : T(0);
  } else {
    dst[idx] = src[src_offset] * T(mask_val);
  }
}

} // namespace cu

namespace {

constexpr int BLOCK_DIM = 256;

} // namespace

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

  int64_t total = int64_t(batch_count) * rows * cols;
  int grid = (total + BLOCK_DIM - 1) / BLOCK_DIM;
  int mask_ndim = mask.ndim();
  int64_t mask_row_str = mask.strides()[mask_ndim - 2];
  int64_t mask_col_str = mask.strides()[mask_ndim - 1];
  int mask_mat_size = mask.shape()[mask_ndim - 2] * mask.shape()[mask_ndim - 1];
  auto mask_shape = const_param(mask.shape());
  auto mask_strides = const_param(mask.strides());
  auto& mask_nc = const_cast<array&>(mask);

  dispatch_float_types(data.dtype(), "apply_block_mask", [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;

    auto launch = [&](auto mask_tag) {
      using MaskT = decltype(mask_tag);
      MaskT* mask_ptr;
      if constexpr (std::is_same_v<MaskT, bool>) {
        mask_ptr = gpu_ptr<bool>(mask_nc);
      } else {
        mask_ptr = gpu_ptr<T>(mask_nc);
      }
      encoder.add_kernel_node(
          cu::block_mask_inplace<T, MaskT>,
          grid,
          BLOCK_DIM,
          gpu_ptr<T>(data),
          mask_ptr,
          block_size,
          rows,
          cols,
          data_batch_stride,
          mask_shape,
          mask_strides,
          mask_ndim,
          mask_row_str,
          mask_col_str,
          mask_mat_size,
          batch_count);
    };

    if (mask.dtype() == bool_) {
      launch(bool{});
    } else {
      launch(T{});
    }
  });
}

array copy_with_block_mask(
    cu::CommandEncoder& encoder,
    const array& src,
    const array& mask,
    int block_size,
    int rows,
    int cols,
    int batch_count) {
  array dst(src.shape(), src.dtype(), nullptr, {});
  dst.set_data(cu::malloc_async(dst.nbytes(), encoder));
  encoder.add_temporary(dst);

  encoder.set_input_array(src);
  encoder.set_input_array(mask);
  encoder.set_output_array(dst);

  int64_t total = int64_t(batch_count) * rows * cols;
  int grid = (total + BLOCK_DIM - 1) / BLOCK_DIM;
  int mask_ndim = mask.ndim();
  int64_t mask_row_str = mask.strides()[mask_ndim - 2];
  int64_t mask_col_str = mask.strides()[mask_ndim - 1];
  int mask_mat_size = mask.shape()[mask_ndim - 2] * mask.shape()[mask_ndim - 1];
  auto src_shape = const_param(src.shape());
  auto src_strides = const_param(src.strides());
  int src_ndim = src.ndim();
  auto mask_shape = const_param(mask.shape());
  auto mask_strides_p = const_param(mask.strides());
  bool src_contiguous = src.flags().row_contiguous;

  auto& src_nc = const_cast<array&>(src);
  auto& mask_nc = const_cast<array&>(mask);

  dispatch_float_types(src.dtype(), "copy_with_block_mask", [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    auto src_ptr = gpu_ptr<T>(src_nc);
    auto dst_ptr = gpu_ptr<T>(dst);

    auto launch = [&](auto mask_tag, auto contiguous_tag) {
      using MaskT = decltype(mask_tag);
      constexpr bool Contiguous = decltype(contiguous_tag)::value;
      MaskT* mask_ptr;
      if constexpr (std::is_same_v<MaskT, bool>) {
        mask_ptr = gpu_ptr<bool>(mask_nc);
      } else {
        mask_ptr = gpu_ptr<T>(mask_nc);
      }
      encoder.add_kernel_node(
          cu::block_mask_copy<T, MaskT, Contiguous>,
          grid,
          BLOCK_DIM,
          src_ptr,
          dst_ptr,
          block_size,
          rows,
          cols,
          src_shape,
          src_strides,
          src_ndim,
          mask_ptr,
          mask_shape,
          mask_strides_p,
          mask_ndim,
          mask_row_str,
          mask_col_str,
          mask_mat_size,
          batch_count);
    };

    auto dispatch_contiguous = [&](auto mask_tag) {
      if (src_contiguous) {
        launch(mask_tag, std::true_type{});
      } else {
        launch(mask_tag, std::false_type{});
      }
    };

    if (mask.dtype() == bool_) {
      dispatch_contiguous(bool{});
    } else {
      dispatch_contiguous(T{});
    }
  });

  return dst;
}

} // namespace mlx::core
