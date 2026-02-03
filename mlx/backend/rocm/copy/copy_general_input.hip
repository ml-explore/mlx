// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/copy/copy.hpp"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/dtype_utils.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

static constexpr int TILE_SIZE = 16;

namespace rocm {

// General copy kernel - strided input to contiguous output (N-dimensional)
template <typename In, typename Out, typename IdxT, int NDIM>
__global__ void copy_g_nd(
    const In* in,
    Out* out,
    IdxT size_rest,
    const int* shape,
    const int64_t* strides) {
  IdxT index_rest = blockIdx.y * blockDim.y + threadIdx.y;
  if (index_rest >= size_rest) {
    return;
  }

  int shape_x = shape[NDIM - 1];
  int64_t stride_x = strides[NDIM - 1];
  IdxT index_x = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (index_x >= shape_x) {
    return;
  }

  // Compute input offset
  IdxT idx = 0;
  IdxT tmp = index_rest;
  #pragma unroll
  for (int i = NDIM - 2; i >= 0; --i) {
    IdxT coord = tmp % shape[i];
    idx += coord * strides[i];
    tmp /= shape[i];
  }
  idx += index_x * stride_x;

  // Output is contiguous
  IdxT out_idx = index_rest * shape_x + index_x;
  out[out_idx] = cast_to<Out>(in[idx]);
}

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
template <typename In, typename Out>
__global__ void copy_col_row(
    const In* in,
    Out* out,
    int64_t rows,
    int64_t cols) {
  __shared__ Out tile[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid bank conflicts

  int tile_row = blockIdx.x * TILE_SIZE;
  int tile_col = blockIdx.y * TILE_SIZE;

  int tidx = threadIdx.x;
  int tidy = threadIdx.y;

  // Load from column-major input
  int in_row = tile_row + tidx;
  int in_col = tile_col + tidy;
  if (in_row < rows && in_col < cols) {
    tile[tidx][tidy] = cast_to<Out>(in[in_col * rows + in_row]);
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

  // Column contiguous to row contiguous specialization
  if (ndim == 2 && strides_in[0] == 1 && strides_in[1] == shape[0]) {
    encoder.launch_kernel([&](hipStream_t stream) {
      dim3 block(TILE_SIZE, TILE_SIZE);
      dim3 grid((shape[0] + TILE_SIZE - 1) / TILE_SIZE,
                (shape[1] + TILE_SIZE - 1) / TILE_SIZE);

      #define LAUNCH_COL_ROW(InT, OutT) \
        hipLaunchKernelGGL( \
            (rocm::copy_col_row<InT, OutT>), \
            grid, block, 0, stream, \
            in.data<InT>() + offset_in, \
            out.data<OutT>() + offset_out, \
            static_cast<int64_t>(shape[0]), \
            static_cast<int64_t>(shape[1]))

      switch (in.dtype()) {
        case float32:
          switch (out.dtype()) {
            case float32: LAUNCH_COL_ROW(float, float); break;
            default: break;
          }
          break;
        case float16:
          switch (out.dtype()) {
            case float16: LAUNCH_COL_ROW(__half, __half); break;
            default: break;
          }
          break;
        default:
          break;
      }
      #undef LAUNCH_COL_ROW
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

    #define LAUNCH_COPY_G(InT, OutT) \
      hipLaunchKernelGGL( \
          (rocm::copy_g_dynamic<InT, OutT, int64_t>), \
          grid, block, 0, stream, \
          in.data<InT>() + offset_in, \
          out.data<OutT>() + offset_out, \
          static_cast<int64_t>(rest), \
          shape_arr.data<int32_t>(), \
          strides_arr.data<int64_t>(), \
          ndim)

    switch (in.dtype()) {
      case float32:
        switch (out.dtype()) {
          case float32: LAUNCH_COPY_G(float, float); break;
          case float16: LAUNCH_COPY_G(float, __half); break;
          case int32: LAUNCH_COPY_G(float, int32_t); break;
          default: throw std::runtime_error("Unsupported output type for copy_general_input");
        }
        break;
      case float16:
        switch (out.dtype()) {
          case float32: LAUNCH_COPY_G(__half, float); break;
          case float16: LAUNCH_COPY_G(__half, __half); break;
          default: throw std::runtime_error("Unsupported output type for copy_general_input");
        }
        break;
      case int32:
        switch (out.dtype()) {
          case float32: LAUNCH_COPY_G(int32_t, float); break;
          case int32: LAUNCH_COPY_G(int32_t, int32_t); break;
          case int64: LAUNCH_COPY_G(int32_t, int64_t); break;
          default: throw std::runtime_error("Unsupported output type for copy_general_input");
        }
        break;
      case int64:
        switch (out.dtype()) {
          case int64: LAUNCH_COPY_G(int64_t, int64_t); break;
          case int32: LAUNCH_COPY_G(int64_t, int32_t); break;
          default: throw std::runtime_error("Unsupported output type for copy_general_input");
        }
        break;
      case bool_:
        switch (out.dtype()) {
          case bool_: LAUNCH_COPY_G(bool, bool); break;
          default: throw std::runtime_error("Unsupported output type for copy_general_input");
        }
        break;
      default:
        throw std::runtime_error("Unsupported input type for copy_general_input");
    }
    #undef LAUNCH_COPY_G
  });
}

} // namespace mlx::core
