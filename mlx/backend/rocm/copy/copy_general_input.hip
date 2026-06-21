// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/copy/copy.hpp"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/device/config.h"
#include "mlx/backend/rocm/device/utils.hpp"
#include "mlx/dtype_utils.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

static constexpr int TILE_SIZE = 16;

namespace rocm {

// General copy kernel - strided input to contiguous output (by-value args)
template <typename In, typename Out, typename IdxT>
__global__ void copy_g_byval(
    const In* in,
    Out* out,
    IdxT size,
    hip_array<int32_t, MAX_NDIM> shape,
    hip_array<int64_t, MAX_NDIM> strides,
    int ndim) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) return;

  IdxT loc = 0;
  IdxT elem = index;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * IdxT(strides[i]);
    elem /= shape[i];
  }
  out[index] = cast_to<Out>(in[loc]);
}

// Column to row transpose kernel
template <typename T>
__global__ void copy_col_row(
    const T* in,
    T* out,
    int64_t rows,
    int64_t cols) {
  __shared__ T tile[TILE_SIZE][TILE_SIZE + 1];

  int tile_row = blockIdx.x * TILE_SIZE;
  int tile_col = blockIdx.y * TILE_SIZE;

  int tidx = threadIdx.x;
  int tidy = threadIdx.y;

  int in_row = tile_row + tidx;
  int in_col = tile_col + tidy;
  if (in_row < rows && in_col < cols) {
    tile[tidx][tidy] = in[in_col * rows + in_row];
  }

  __syncthreads();

  int out_row = tile_row + tidy;
  int out_col = tile_col + tidx;
  if (out_row < rows && out_col < cols) {
    out[out_row * cols + out_col] = tile[tidy][tidx];
  }
}

} // namespace rocm

void copy_general_input(
    rocm::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t offset_in,
    int64_t offset_out,
    const Shape& shape,
    const Strides& strides_in) {

  int ndim = shape.size();
  size_t data_size = out.size();

  if (data_size == 0) {
    return;
  }

  // Column contiguous to row contiguous specialization (same type only)
  if (ndim == 2 && strides_in[0] == 1 && strides_in[1] == shape[0] && in.dtype() == out.dtype()) {
    dispatch_all_types(in.dtype(), [&](auto type_tag) {
      using T = hip_type_t<MLX_GET_TYPE(type_tag)>;
      dim3 block(TILE_SIZE, TILE_SIZE);
      dim3 grid((shape[0] + TILE_SIZE - 1) / TILE_SIZE,
                (shape[1] + TILE_SIZE - 1) / TILE_SIZE);
      const T* in_typed =
          reinterpret_cast<const T*>(gpu_ptr<void>(in)) + offset_in;
      T* out_typed = reinterpret_cast<T*>(gpu_ptr<void>(out)) + offset_out;
      int64_t rows_arg = static_cast<int64_t>(shape[0]);
      int64_t cols_arg = static_cast<int64_t>(shape[1]);
      encoder.add_kernel_node(
          &rocm::copy_col_row<T>,
          grid, block, 0,
          in_typed,
          out_typed,
          rows_arg,
          cols_arg);
    });
    return;
  }

  // Pack shape/strides into by-value structs (no device allocation or hipMemcpyAsync)
  rocm::hip_array<int32_t, MAX_NDIM> shape_arg = {};
  rocm::hip_array<int64_t, MAX_NDIM> strides_arg = {};
  for (int i = 0; i < ndim; i++) {
    shape_arg.data_[i] = static_cast<int32_t>(shape[i]);
    strides_arg.data_[i] = strides_in[i];
  }

  const void* in_ptr = gpu_ptr<void>(in);
  void* out_ptr = gpu_ptr<void>(out);

  int block_size = 256;
  int num_blocks = (data_size + block_size - 1) / block_size;
  int64_t size_arg = static_cast<int64_t>(data_size);

  dispatch_all_types(in.dtype(), [&](auto in_type_tag) {
    dispatch_all_types(out.dtype(), [&](auto out_type_tag) {
      using InType = hip_type_t<MLX_GET_TYPE(in_type_tag)>;
      using OutType = hip_type_t<MLX_GET_TYPE(out_type_tag)>;

      const InType* in_typed = static_cast<const InType*>(in_ptr) + offset_in;
      OutType* out_typed = static_cast<OutType*>(out_ptr) + offset_out;

      encoder.add_kernel_node(
          &rocm::copy_g_byval<InType, OutType, int64_t>,
          dim3(num_blocks), dim3(block_size), 0,
          in_typed,
          out_typed,
          size_arg,
          shape_arg,
          strides_arg,
          ndim);
    });
  });
}

} // namespace mlx::core
