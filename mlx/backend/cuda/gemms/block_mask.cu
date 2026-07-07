// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/utils.cuh"
#include "mlx/backend/cuda/gemms/block_mask.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"

#include <cooperative_groups.h>

namespace mlx::core {

namespace cg = cooperative_groups;

namespace cu {

template <typename T, typename MaskT, bool SrcContiguous>
__global__ void block_mask_copy_kernel(
    const T* src,
    T* dst,
    int block_size,
    int64_t rows,
    int64_t cols,
    const __grid_constant__ Shape src_shape,
    const __grid_constant__ Strides src_strides,
    int src_ndim,
    MaskT* mask,
    const __grid_constant__ Shape mask_shape,
    const __grid_constant__ Strides mask_strides,
    int mask_ndim,
    int64_t mask_row_stride,
    int64_t mask_col_stride,
    int64_t mask_mat_size,
    int64_t batch_count) {
  int64_t mat_size = rows * cols;
  int64_t idx = cg::this_grid().thread_rank();
  if (idx >= batch_count * mat_size)
    return;

  int64_t batch = idx / mat_size;
  int64_t within = idx % mat_size;
  int64_t mask_batch_offset = elem_to_loc(
      batch * mask_mat_size, mask_shape.data(), mask_strides.data(), mask_ndim);
  MaskT mask_val = mask
      [mask_batch_offset + (within / cols) / block_size * mask_row_stride +
       (within % cols) / block_size * mask_col_stride];

  int64_t src_offset;
  if constexpr (SrcContiguous) {
    src_offset = idx;
  } else {
    src_offset = elem_to_loc(
        batch * mat_size + within,
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

template <typename T, typename F>
void dispatch_mask_type(Dtype mask_dtype, F&& f) {
  if (mask_dtype == bool_) {
    f.template operator()<bool>();
  } else {
    f.template operator()<T>();
  }
}

void block_mask_copy(
    cu::CommandEncoder& encoder,
    const array& src,
    array& dst,
    const array& mask,
    int block_size,
    int64_t rows,
    int64_t cols,
    bool src_contiguous,
    int64_t batch_count) {
  int mask_ndim = mask.ndim();
  int64_t mask_row_str = mask.strides()[mask_ndim - 2];
  int64_t mask_col_str = mask.strides()[mask_ndim - 1];
  int64_t mask_mat_size =
      int64_t(mask.shape()[mask_ndim - 2]) * mask.shape()[mask_ndim - 1];

  auto [num_blocks, block_dims] = get_launch_args(src, src.size() > INT32_MAX);

  dispatch_float_types(src.dtype(), "block_mask_copy", [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;

    dispatch_mask_type<T>(mask.dtype(), [&]<typename MaskT>() {
      dispatch_bool(src_contiguous, [&](auto contiguous_tag) {
        constexpr bool Contiguous = decltype(contiguous_tag)::value;
        encoder.add_kernel_node(
            cu::block_mask_copy_kernel<T, MaskT, Contiguous>,
            num_blocks,
            block_dims,
            gpu_ptr<T>(src),
            gpu_ptr<T>(dst),
            block_size,
            rows,
            cols,
            const_param(src.shape()),
            const_param(src.strides()),
            src.ndim(),
            gpu_ptr<MaskT>(mask),
            const_param(mask.shape()),
            const_param(mask.strides()),
            mask_ndim,
            mask_row_str,
            mask_col_str,
            mask_mat_size,
            batch_count);
      });
    });
  });
}

} // namespace

void apply_block_mask(
    cu::CommandEncoder& encoder,
    array& data,
    const array& mask,
    int block_size,
    int64_t rows,
    int64_t cols,
    int64_t batch_count) {
  encoder.set_input_array(mask);
  encoder.set_output_array(data);

  // Use block_mask_copy in-place (src == dst) with SrcContiguous=true.
  block_mask_copy(
      encoder, data, data, mask, block_size, rows, cols, true, batch_count);
}

array copy_with_block_mask(
    cu::CommandEncoder& encoder,
    const array& src,
    const array& mask,
    int block_size,
    int64_t rows,
    int64_t cols,
    int64_t batch_count) {
  array dst(src.shape(), src.dtype(), nullptr, {});
  dst.set_data(cu::malloc_async(dst.nbytes(), encoder));
  encoder.add_temporary(dst);

  encoder.set_input_array(src);
  encoder.set_input_array(mask);
  encoder.set_output_array(dst);

  block_mask_copy(
      encoder,
      src,
      dst,
      mask,
      block_size,
      rows,
      cols,
      src.flags().row_contiguous,
      batch_count);

  return dst;
}

} // namespace mlx::core
