// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/copy/copy.hpp"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/dtype_utils.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

// General copy kernel - strided input to strided output (N-dimensional)
template <typename In, typename Out, typename IdxT, int NDIM>
__global__ void copy_gg_nd(
    const In* in,
    Out* out,
    IdxT size_rest,
    const int* shape,
    const int64_t* strides_in,
    const int64_t* strides_out) {
  IdxT index_rest = blockIdx.y * blockDim.y + threadIdx.y;
  if (index_rest >= size_rest) {
    return;
  }

  int shape_x = shape[NDIM - 1];
  int64_t in_stride_x = strides_in[NDIM - 1];
  int64_t out_stride_x = strides_out[NDIM - 1];
  IdxT index_x = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (index_x >= shape_x) {
    return;
  }

  // Compute base offsets for input and output
  IdxT idx_in = 0;
  IdxT idx_out = 0;
  IdxT tmp = index_rest;
  #pragma unroll
  for (int i = NDIM - 2; i >= 0; --i) {
    IdxT coord = tmp % shape[i];
    idx_in += coord * strides_in[i];
    idx_out += coord * strides_out[i];
    tmp /= shape[i];
  }

  // Add x-dimension offset
  idx_in += index_x * in_stride_x;
  idx_out += index_x * out_stride_x;

  out[idx_out] = cast_to<Out>(in[idx_in]);
}

// General copy kernel - strided input to strided output (dynamic ndim)
template <typename In, typename Out, typename IdxT>
__global__ void copy_gg_dynamic(
    const In* in,
    Out* out,
    IdxT size_rest,
    const int* shape,
    const int64_t* strides_in,
    const int64_t* strides_out,
    int ndim) {
  IdxT index_rest = blockIdx.y * blockDim.y + threadIdx.y;
  if (index_rest >= size_rest) {
    return;
  }

  int shape_x = shape[ndim - 1];
  int64_t in_stride_x = strides_in[ndim - 1];
  int64_t out_stride_x = strides_out[ndim - 1];
  IdxT index_x = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (index_x >= shape_x) {
    return;
  }

  // Compute base offsets for input and output
  IdxT idx_in = 0;
  IdxT idx_out = 0;
  IdxT tmp = index_rest;
  for (int i = ndim - 2; i >= 0; --i) {
    IdxT coord = tmp % shape[i];
    idx_in += coord * strides_in[i];
    idx_out += coord * strides_out[i];
    tmp /= shape[i];
  }

  // Add x-dimension offset
  idx_in += index_x * in_stride_x;
  idx_out += index_x * out_stride_x;

  out[idx_out] = cast_to<Out>(in[idx_in]);
}

} // namespace rocm

void copy_general(
    rocm::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t offset_in,
    int64_t offset_out,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out) {
  
  int ndim = shape.size();
  size_t data_size = 1;
  for (auto& s : shape) {
    data_size *= s;
  }
  
  if (data_size == 0) {
    return;
  }

  auto dim0 = ndim > 0 ? shape.back() : 1;
  auto rest = data_size / dim0;

  // Allocate device memory for shape and strides
  array shape_arr({ndim}, int32, nullptr, {});
  array strides_in_arr({ndim}, int64, nullptr, {});
  array strides_out_arr({ndim}, int64, nullptr, {});
  shape_arr.set_data(allocator::malloc(shape_arr.nbytes()));
  strides_in_arr.set_data(allocator::malloc(strides_in_arr.nbytes()));
  strides_out_arr.set_data(allocator::malloc(strides_out_arr.nbytes()));
  encoder.add_temporary(shape_arr);
  encoder.add_temporary(strides_in_arr);
  encoder.add_temporary(strides_out_arr);

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
            strides_in_arr.data<int64_t>(),
            strides_in.data(),
            ndim * sizeof(int64_t),
            hipMemcpyHostToDevice,
            stream);
        (void)hipMemcpyAsync(
            strides_out_arr.data<int64_t>(),
            strides_out.data(),
            ndim * sizeof(int64_t),
            hipMemcpyHostToDevice,
            stream);

        dim3 block(16, 16);
        dim3 grid((dim0 + block.x - 1) / block.x, (rest + block.y - 1) / block.y);

        hipLaunchKernelGGL(
            (rocm::copy_gg_dynamic<InType, OutType, int64_t>),
            grid, block, 0, stream,
            reinterpret_cast<const InType*>(in.data<void>()) + offset_in,
            reinterpret_cast<OutType*>(out.data<void>()) + offset_out,
            static_cast<int64_t>(rest),
            shape_arr.data<int32_t>(),
            strides_in_arr.data<int64_t>(),
            strides_out_arr.data<int64_t>(),
            ndim);
      });
    });
  });
}

} // namespace mlx::core
