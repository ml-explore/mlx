// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/copy/copy.hpp"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

namespace mlx::core {

namespace rocm {

template <typename In, typename Out, typename IdxT, int NDIM>
__global__ void copy_gg_dynamic_nd(
    const In* in,
    Out* out,
    IdxT size,
    const int32_t* shape,
    const int64_t* strides_in,
    const int64_t* strides_out,
    const int64_t* offset_in,
    const int64_t* offset_out) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) return;
  
  // Compute input and output locations
  IdxT idx_in = 0;
  IdxT idx_out = 0;
  IdxT elem = index;
  
  #pragma unroll
  for (int i = NDIM - 1; i >= 0; --i) {
    IdxT dim_idx = elem % shape[i];
    elem /= shape[i];
    idx_in += dim_idx * strides_in[i];
    idx_out += dim_idx * strides_out[i];
  }
  
  out[idx_out + *offset_out] = static_cast<Out>(in[idx_in + *offset_in]);
}

template <typename In, typename Out, typename IdxT>
__global__ void copy_gg_dynamic(
    const In* in,
    Out* out,
    IdxT size,
    const int32_t* shape,
    const int64_t* strides_in,
    const int64_t* strides_out,
    int ndim,
    const int64_t* offset_in,
    const int64_t* offset_out) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) return;
  
  // Compute input and output locations
  IdxT idx_in = 0;
  IdxT idx_out = 0;
  IdxT elem = index;
  
  for (int i = ndim - 1; i >= 0; --i) {
    IdxT dim_idx = elem % shape[i];
    elem /= shape[i];
    idx_in += dim_idx * strides_in[i];
    idx_out += dim_idx * strides_out[i];
  }
  
  out[idx_out + *offset_out] = static_cast<Out>(in[idx_in + *offset_in]);
}

} // namespace rocm

void copy_general_dynamic(
    rocm::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t offset_in,
    int64_t offset_out,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out,
    const array& dynamic_offset_in,
    const array& dynamic_offset_out) {
  
  encoder.set_input_array(in);
  encoder.set_input_array(dynamic_offset_in);
  encoder.set_input_array(dynamic_offset_out);
  encoder.set_output_array(out);
  
  int ndim = shape.size();
  size_t size = out.size();
  
  // Allocate device memory for shape and strides using allocator
  array shape_arr({ndim}, int32, nullptr, {});
  array strides_in_arr({ndim}, int64, nullptr, {});
  array strides_out_arr({ndim}, int64, nullptr, {});
  shape_arr.set_data(allocator::malloc(ndim * sizeof(int32_t)));
  strides_in_arr.set_data(allocator::malloc(ndim * sizeof(int64_t)));
  strides_out_arr.set_data(allocator::malloc(ndim * sizeof(int64_t)));
  
  encoder.add_temporary(shape_arr);
  encoder.add_temporary(strides_in_arr);
  encoder.add_temporary(strides_out_arr);
  
  // Prepare host data
  std::vector<int32_t> h_shape(shape.begin(), shape.end());
  std::vector<int64_t> h_strides_in(strides_in.begin(), strides_in.end());
  std::vector<int64_t> h_strides_out(strides_out.begin(), strides_out.end());
  
  int block_size = 256;
  int num_blocks = (size + block_size - 1) / block_size;
  
  bool large = in.data_size() > INT32_MAX || out.data_size() > INT32_MAX;
  
  // Get GPU pointers before lambda to avoid synchronization issues
  const void* in_ptr_base = gpu_ptr<void>(in);
  void* out_ptr_base = gpu_ptr<void>(out);
  int32_t* shape_ptr = gpu_ptr<int32_t>(shape_arr);
  int64_t* strides_in_ptr = gpu_ptr<int64_t>(strides_in_arr);
  int64_t* strides_out_ptr = gpu_ptr<int64_t>(strides_out_arr);
  const int64_t* dyn_offset_in_ptr = gpu_ptr<int64_t>(dynamic_offset_in);
  const int64_t* dyn_offset_out_ptr = gpu_ptr<int64_t>(dynamic_offset_out);
  
  fprintf(stderr, "DEBUG copy_general_dynamic: Starting launch_kernel\n");
  encoder.launch_kernel([&, h_shape, h_strides_in, h_strides_out, 
                         in_ptr_base, out_ptr_base, shape_ptr, strides_in_ptr, strides_out_ptr,
                         dyn_offset_in_ptr, dyn_offset_out_ptr](hipStream_t stream) {
    fprintf(stderr, "DEBUG copy_general_dynamic: Inside lambda, copying shape\n");
    // Copy data to device asynchronously
    (void)hipMemcpyAsync(shape_ptr, h_shape.data(), 
                         ndim * sizeof(int32_t), hipMemcpyHostToDevice, stream);
    fprintf(stderr, "DEBUG copy_general_dynamic: Copying strides_in\n");
    (void)hipMemcpyAsync(strides_in_ptr, h_strides_in.data(), 
                         ndim * sizeof(int64_t), hipMemcpyHostToDevice, stream);
    fprintf(stderr, "DEBUG copy_general_dynamic: Copying strides_out\n");
    (void)hipMemcpyAsync(strides_out_ptr, h_strides_out.data(), 
                         ndim * sizeof(int64_t), hipMemcpyHostToDevice, stream);
    fprintf(stderr, "DEBUG copy_general_dynamic: Launching kernel, ndim=%d, size=%zu\n", ndim, size);
    
    #define LAUNCH_COPY_DYNAMIC(InT, OutT, IdxT, NDIM) \
      hipLaunchKernelGGL( \
          (rocm::copy_gg_dynamic_nd<InT, OutT, IdxT, NDIM>), \
          dim3(num_blocks), dim3(block_size), 0, stream, \
          static_cast<const InT*>(in_ptr_base) + offset_in, static_cast<OutT*>(out_ptr_base) + offset_out, \
          static_cast<IdxT>(size), shape_ptr, \
          strides_in_ptr, strides_out_ptr, \
          dyn_offset_in_ptr, dyn_offset_out_ptr)
    
    #define LAUNCH_COPY_DYNAMIC_GENERAL(InT, OutT, IdxT) \
      hipLaunchKernelGGL( \
          (rocm::copy_gg_dynamic<InT, OutT, IdxT>), \
          dim3(num_blocks), dim3(block_size), 0, stream, \
          static_cast<const InT*>(in_ptr_base) + offset_in, static_cast<OutT*>(out_ptr_base) + offset_out, \
          static_cast<IdxT>(size), shape_ptr, \
          strides_in_ptr, strides_out_ptr, \
          ndim, dyn_offset_in_ptr, dyn_offset_out_ptr)
    
    #define DISPATCH_NDIM(InT, OutT, IdxT) \
      switch (ndim) { \
        case 1: LAUNCH_COPY_DYNAMIC(InT, OutT, IdxT, 1); break; \
        case 2: LAUNCH_COPY_DYNAMIC(InT, OutT, IdxT, 2); break; \
        case 3: LAUNCH_COPY_DYNAMIC(InT, OutT, IdxT, 3); break; \
        default: LAUNCH_COPY_DYNAMIC_GENERAL(InT, OutT, IdxT); break; \
      }
    
    #define DISPATCH_OUT_TYPE(InT, IdxT) \
      switch (out.dtype()) { \
        case float32: DISPATCH_NDIM(InT, float, IdxT); break; \
        case float16: DISPATCH_NDIM(InT, __half, IdxT); break; \
        case bfloat16: DISPATCH_NDIM(InT, hip_bfloat16, IdxT); break; \
        case int32: DISPATCH_NDIM(InT, int32_t, IdxT); break; \
        case int64: DISPATCH_NDIM(InT, int64_t, IdxT); break; \
        case uint32: DISPATCH_NDIM(InT, uint32_t, IdxT); break; \
        case uint8: DISPATCH_NDIM(InT, uint8_t, IdxT); break; \
        case bool_: DISPATCH_NDIM(InT, bool, IdxT); break; \
        default: throw std::runtime_error("Unsupported output dtype for copy_general_dynamic"); \
      }
    
    #define DISPATCH_IN_TYPE(IdxT) \
      switch (in.dtype()) { \
        case float32: DISPATCH_OUT_TYPE(float, IdxT); break; \
        case float16: DISPATCH_OUT_TYPE(__half, IdxT); break; \
        case bfloat16: DISPATCH_OUT_TYPE(hip_bfloat16, IdxT); break; \
        case int32: DISPATCH_OUT_TYPE(int32_t, IdxT); break; \
        case int64: DISPATCH_OUT_TYPE(int64_t, IdxT); break; \
        case uint32: DISPATCH_OUT_TYPE(uint32_t, IdxT); break; \
        case uint8: DISPATCH_OUT_TYPE(uint8_t, IdxT); break; \
        case bool_: DISPATCH_OUT_TYPE(bool, IdxT); break; \
        default: throw std::runtime_error("Unsupported input dtype for copy_general_dynamic"); \
      }
    
    if (large) {
      DISPATCH_IN_TYPE(int64_t);
    } else {
      DISPATCH_IN_TYPE(int32_t);
    }
    fprintf(stderr, "DEBUG copy_general_dynamic: Kernel launched\n");
    
    #undef DISPATCH_IN_TYPE
    #undef DISPATCH_OUT_TYPE
    #undef DISPATCH_NDIM
    #undef LAUNCH_COPY_DYNAMIC_GENERAL
    #undef LAUNCH_COPY_DYNAMIC
  });
  fprintf(stderr, "DEBUG copy_general_dynamic: Returning\n");
}

} // namespace mlx::core
