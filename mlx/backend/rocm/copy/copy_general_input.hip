// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/copy/copy.hpp"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/dtype_utils.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

static constexpr int TILE_SIZE = 16;

namespace rocm {

// General copy kernel - strided input to contiguous output (dynamic ndim)
template <typename In, typename Out, typename IdxT>
__global__ void copy_g_dynamic(
    const In* in,
    Out* out,
    IdxT size_rest,
    const int* shape,
    const int64_t* strides,
    int ndim) {
  IdxT index_rest = blockIdx.y * blockDim.y + threadIdx.y;
  if (index_rest >= size_rest) {
    return;
  }

  int shape_x = shape[ndim - 1];
  int64_t stride_x = strides[ndim - 1];
  IdxT index_x = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (index_x >= shape_x) {
    return;
  }

  // Compute input offset
  IdxT idx = 0;
  IdxT tmp = index_rest;
  for (int i = ndim - 2; i >= 0; --i) {
    IdxT coord = tmp % shape[i];
    idx += coord * strides[i];
    tmp /= shape[i];
  }
  idx += index_x * stride_x;

  // Output is contiguous
  IdxT out_idx = index_rest * shape_x + index_x;
  out[out_idx] = cast_to<Out>(in[idx]);
}

// Column to row transpose kernel
template <typename T>
__global__ void copy_col_row(
    const T* in,
    T* out,
    int64_t rows,
    int64_t cols) {
  __shared__ T tile[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid bank conflicts

  int tile_row = blockIdx.x * TILE_SIZE;
  int tile_col = blockIdx.y * TILE_SIZE;

  int tidx = threadIdx.x;
  int tidy = threadIdx.y;

  // Load from column-major input
  int in_row = tile_row + tidx;
  int in_col = tile_col + tidy;
  if (in_row < rows && in_col < cols) {
    tile[tidx][tidy] = in[in_col * rows + in_row];
  }

  __syncthreads();

  // Store to row-major output
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
      encoder.launch_kernel([&](hipStream_t stream) {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((shape[0] + TILE_SIZE - 1) / TILE_SIZE,
                  (shape[1] + TILE_SIZE - 1) / TILE_SIZE);
        hipLaunchKernelGGL(
            (rocm::copy_col_row<T>),
            grid, block, 0, stream,
            reinterpret_cast<const T*>(in.data<void>()) + offset_in,
            reinterpret_cast<T*>(out.data<void>()) + offset_out,
            static_cast<int64_t>(shape[0]),
            static_cast<int64_t>(shape[1]));
      });
    });
    return;
  }

  auto dim0 = ndim > 0 ? shape.back() : 1;
  auto rest = data_size / dim0;

  // Allocate device memory for shape and strides
  array shape_arr({ndim}, int32, nullptr, {});
  array strides_arr({ndim}, int64, nullptr, {});
  shape_arr.set_data(allocator::malloc(shape_arr.nbytes()));
  strides_arr.set_data(allocator::malloc(strides_arr.nbytes()));
  encoder.add_temporary(shape_arr);
  encoder.add_temporary(strides_arr);

  dispatch_all_types(in.dtype(), [&](auto in_type_tag) {
    dispatch_all_types(out.dtype(), [&](auto out_type_tag) {
      using InType = hip_type_t<MLX_GET_TYPE(in_type_tag)>;
      using OutType = hip_type_t<MLX_GET_TYPE(out_type_tag)>;
      
      encoder.launch_kernel([&](hipStream_t stream) {
        // Copy shape and strides to device
        (void)hipMemcpyAsync(
            shape_arr.data<int32_t>(),
            shape.data(),
            ndim * sizeof(int32_t),
            hipMemcpyHostToDevice,
            stream);
        (void)hipMemcpyAsync(
            strides_arr.data<int64_t>(),
            strides_in.data(),
            ndim * sizeof(int64_t),
            hipMemcpyHostToDevice,
            stream);

        dim3 block(16, 16);
        dim3 grid((dim0 + block.x - 1) / block.x, (rest + block.y - 1) / block.y);

        hipLaunchKernelGGL(
            (rocm::copy_g_dynamic<InType, OutType, int64_t>),
            grid, block, 0, stream,
            reinterpret_cast<const InType*>(in.data<void>()) + offset_in,
            reinterpret_cast<OutType*>(out.data<void>()) + offset_out,
            static_cast<int64_t>(rest),
            shape_arr.data<int32_t>(),
            strides_arr.data<int64_t>(),
            ndim);
      });
    });
  });
}

} // namespace mlx::core
