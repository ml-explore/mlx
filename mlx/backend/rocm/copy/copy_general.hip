// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/copy/copy.hpp"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/dtype_utils.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

// Helper function to convert linear index to strided offset
template <typename IdxT>
__device__ IdxT linear_to_strided(
    IdxT elem,
    const int* shape,
    const int64_t* strides,
    int ndim) {
  IdxT loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * IdxT(strides[i]);
    elem /= shape[i];
  }
  return loc;
}

// Helper function to convert linear index to two strided offsets
template <typename IdxT>
__device__ void linear_to_strided_2(
    IdxT elem,
    const int* shape,
    const int64_t* strides_in,
    const int64_t* strides_out,
    int ndim,
    IdxT& loc_in,
    IdxT& loc_out) {
  loc_in = 0;
  loc_out = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    IdxT dim_idx = elem % shape[i];
    loc_in += dim_idx * IdxT(strides_in[i]);
    loc_out += dim_idx * IdxT(strides_out[i]);
    elem /= shape[i];
  }
}

// General copy kernel - strided input to strided output (dynamic ndim)
template <typename In, typename Out, typename IdxT>
__global__ void copy_gg_dynamic(
    const In* in,
    Out* out,
    IdxT size,
    const int* shape,
    const int64_t* strides_in,
    const int64_t* strides_out,
    int ndim) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) {
    return;
  }

  IdxT idx_in, idx_out;
  linear_to_strided_2(index, shape, strides_in, strides_out, ndim, idx_in, idx_out);
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

        int block_size = 256;
        int num_blocks = (data_size + block_size - 1) / block_size;

        hipLaunchKernelGGL(
            (rocm::copy_gg_dynamic<InType, OutType, int64_t>),
            dim3(num_blocks), dim3(block_size), 0, stream,
            reinterpret_cast<const InType*>(in.data<void>()) + offset_in,
            reinterpret_cast<OutType*>(out.data<void>()) + offset_out,
            static_cast<int64_t>(data_size),
            shape_arr.data<int32_t>(),
            strides_in_arr.data<int64_t>(),
            strides_out_arr.data<int64_t>(),
            ndim);
      });
    });
  });
}

} // namespace mlx::core
