// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/copy/copy.hpp"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <array>

namespace mlx::core {

namespace rocm {

// Kernel with fixed-size arrays passed by value (no device memory needed)
template <typename In, typename Out, typename IdxT, int NDIM>
__global__ void copy_gg_dynamic_nd(
    const In* in,
    Out* out,
    IdxT size,
    const int32_t shape0, const int32_t shape1, const int32_t shape2,
    const int64_t strides_in0, const int64_t strides_in1, const int64_t strides_in2,
    const int64_t strides_out0, const int64_t strides_out1, const int64_t strides_out2,
    const int64_t* offset_in,
    const int64_t* offset_out) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) return;
  
  // Compute input and output locations
  IdxT idx_in = 0;
  IdxT idx_out = 0;
  IdxT elem = index;
  
  // Unroll based on NDIM
  if constexpr (NDIM >= 3) {
    IdxT dim_idx = elem % shape2;
    elem /= shape2;
    idx_in += dim_idx * strides_in2;
    idx_out += dim_idx * strides_out2;
  }
  if constexpr (NDIM >= 2) {
    IdxT dim_idx = elem % shape1;
    elem /= shape1;
    idx_in += dim_idx * strides_in1;
    idx_out += dim_idx * strides_out1;
  }
  if constexpr (NDIM >= 1) {
    IdxT dim_idx = elem % shape0;
    idx_in += dim_idx * strides_in0;
    idx_out += dim_idx * strides_out0;
  }
  
  out[idx_out + *offset_out] = static_cast<Out>(in[idx_in + *offset_in]);
}

// General kernel for ndim > 3 (still needs device memory for shape/strides)
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
  
  int block_size = 256;
  int num_blocks = (size + block_size - 1) / block_size;
  
  bool large = in.data_size() > INT32_MAX || out.data_size() > INT32_MAX;
  
  // Get GPU pointers before lambda to avoid synchronization issues
  const void* in_ptr_base = gpu_ptr<void>(in);
  void* out_ptr_base = gpu_ptr<void>(out);
  const int64_t* dyn_offset_in_ptr = gpu_ptr<int64_t>(dynamic_offset_in);
  const int64_t* dyn_offset_out_ptr = gpu_ptr<int64_t>(dynamic_offset_out);
  
  // For ndim <= 3, pass shape and strides as kernel arguments (no device memory needed)
  if (ndim <= 3) {
    // Pad arrays to size 3
    int32_t s0 = ndim > 0 ? static_cast<int32_t>(shape[0]) : 1;
    int32_t s1 = ndim > 1 ? static_cast<int32_t>(shape[1]) : 1;
    int32_t s2 = ndim > 2 ? static_cast<int32_t>(shape[2]) : 1;
    int64_t si0 = ndim > 0 ? strides_in[0] : 0;
    int64_t si1 = ndim > 1 ? strides_in[1] : 0;
    int64_t si2 = ndim > 2 ? strides_in[2] : 0;
    int64_t so0 = ndim > 0 ? strides_out[0] : 0;
    int64_t so1 = ndim > 1 ? strides_out[1] : 0;
    int64_t so2 = ndim > 2 ? strides_out[2] : 0;
    
    encoder.launch_kernel([&, in_ptr_base, out_ptr_base,
                           s0, s1, s2, si0, si1, si2, so0, so1, so2,
                           dyn_offset_in_ptr, dyn_offset_out_ptr](hipStream_t stream) {
      
      #define LAUNCH_COPY_DYNAMIC_ND(InT, OutT, IdxT, NDIM) \
        hipLaunchKernelGGL( \
            (rocm::copy_gg_dynamic_nd<InT, OutT, IdxT, NDIM>), \
            dim3(num_blocks), dim3(block_size), 0, stream, \
            static_cast<const InT*>(in_ptr_base) + offset_in, \
            static_cast<OutT*>(out_ptr_base) + offset_out, \
            static_cast<IdxT>(size), \
            s0, s1, s2, si0, si1, si2, so0, so1, so2, \
            dyn_offset_in_ptr, dyn_offset_out_ptr)
      
      #define DISPATCH_NDIM_ND(InT, OutT, IdxT) \
        switch (ndim) { \
          case 1: LAUNCH_COPY_DYNAMIC_ND(InT, OutT, IdxT, 1); break; \
          case 2: LAUNCH_COPY_DYNAMIC_ND(InT, OutT, IdxT, 2); break; \
          case 3: LAUNCH_COPY_DYNAMIC_ND(InT, OutT, IdxT, 3); break; \
          default: break; \
        }
      
      #define DISPATCH_OUT_TYPE_ND(InT, IdxT) \
        switch (out.dtype()) { \
          case float32: DISPATCH_NDIM_ND(InT, float, IdxT); break; \
          case float16: DISPATCH_NDIM_ND(InT, __half, IdxT); break; \
          case bfloat16: DISPATCH_NDIM_ND(InT, hip_bfloat16, IdxT); break; \
          case int32: DISPATCH_NDIM_ND(InT, int32_t, IdxT); break; \
          case int64: DISPATCH_NDIM_ND(InT, int64_t, IdxT); break; \
          case uint32: DISPATCH_NDIM_ND(InT, uint32_t, IdxT); break; \
          case uint8: DISPATCH_NDIM_ND(InT, uint8_t, IdxT); break; \
          case bool_: DISPATCH_NDIM_ND(InT, bool, IdxT); break; \
          default: break; \
        }
      
      #define DISPATCH_IN_TYPE_ND(IdxT) \
        switch (in.dtype()) { \
          case float32: DISPATCH_OUT_TYPE_ND(float, IdxT); break; \
          case float16: DISPATCH_OUT_TYPE_ND(__half, IdxT); break; \
          case bfloat16: DISPATCH_OUT_TYPE_ND(hip_bfloat16, IdxT); break; \
          case int32: DISPATCH_OUT_TYPE_ND(int32_t, IdxT); break; \
          case int64: DISPATCH_OUT_TYPE_ND(int64_t, IdxT); break; \
          case uint32: DISPATCH_OUT_TYPE_ND(uint32_t, IdxT); break; \
          case uint8: DISPATCH_OUT_TYPE_ND(uint8_t, IdxT); break; \
          case bool_: DISPATCH_OUT_TYPE_ND(bool, IdxT); break; \
          default: break; \
        }
      
      if (large) {
        DISPATCH_IN_TYPE_ND(int64_t);
      } else {
        DISPATCH_IN_TYPE_ND(int32_t);
      }
      
      #undef DISPATCH_IN_TYPE_ND
      #undef DISPATCH_OUT_TYPE_ND
      #undef DISPATCH_NDIM_ND
      #undef LAUNCH_COPY_DYNAMIC_ND
    });
    return;
  }
  
  // For ndim > 3, we need device memory for shape and strides
  // Allocate device memory synchronously before the lambda
  int32_t* d_shape = nullptr;
  int64_t* d_strides_in = nullptr;
  int64_t* d_strides_out = nullptr;
  
  (void)hipMalloc(&d_shape, ndim * sizeof(int32_t));
  (void)hipMalloc(&d_strides_in, ndim * sizeof(int64_t));
  (void)hipMalloc(&d_strides_out, ndim * sizeof(int64_t));
  
  // Prepare host data
  std::vector<int32_t> h_shape(shape.begin(), shape.end());
  std::vector<int64_t> h_strides_in(strides_in.begin(), strides_in.end());
  std::vector<int64_t> h_strides_out(strides_out.begin(), strides_out.end());
  
  encoder.launch_kernel([&, h_shape, h_strides_in, h_strides_out, 
                         in_ptr_base, out_ptr_base,
                         d_shape, d_strides_in, d_strides_out,
                         dyn_offset_in_ptr, dyn_offset_out_ptr](hipStream_t stream) {
    // Copy data to device asynchronously
    (void)hipMemcpyAsync(d_shape, h_shape.data(), 
                         ndim * sizeof(int32_t), hipMemcpyHostToDevice, stream);
    (void)hipMemcpyAsync(d_strides_in, h_strides_in.data(), 
                         ndim * sizeof(int64_t), hipMemcpyHostToDevice, stream);
    (void)hipMemcpyAsync(d_strides_out, h_strides_out.data(), 
                         ndim * sizeof(int64_t), hipMemcpyHostToDevice, stream);
    
    #define LAUNCH_COPY_DYNAMIC_GENERAL(InT, OutT, IdxT) \
      hipLaunchKernelGGL( \
          (rocm::copy_gg_dynamic<InT, OutT, IdxT>), \
          dim3(num_blocks), dim3(block_size), 0, stream, \
          static_cast<const InT*>(in_ptr_base) + offset_in, \
          static_cast<OutT*>(out_ptr_base) + offset_out, \
          static_cast<IdxT>(size), d_shape, \
          d_strides_in, d_strides_out, \
          ndim, dyn_offset_in_ptr, dyn_offset_out_ptr)
    
    #define DISPATCH_OUT_TYPE_GEN(InT, IdxT) \
      switch (out.dtype()) { \
        case float32: LAUNCH_COPY_DYNAMIC_GENERAL(InT, float, IdxT); break; \
        case float16: LAUNCH_COPY_DYNAMIC_GENERAL(InT, __half, IdxT); break; \
        case bfloat16: LAUNCH_COPY_DYNAMIC_GENERAL(InT, hip_bfloat16, IdxT); break; \
        case int32: LAUNCH_COPY_DYNAMIC_GENERAL(InT, int32_t, IdxT); break; \
        case int64: LAUNCH_COPY_DYNAMIC_GENERAL(InT, int64_t, IdxT); break; \
        case uint32: LAUNCH_COPY_DYNAMIC_GENERAL(InT, uint32_t, IdxT); break; \
        case uint8: LAUNCH_COPY_DYNAMIC_GENERAL(InT, uint8_t, IdxT); break; \
        case bool_: LAUNCH_COPY_DYNAMIC_GENERAL(InT, bool, IdxT); break; \
        default: break; \
      }
    
    #define DISPATCH_IN_TYPE_GEN(IdxT) \
      switch (in.dtype()) { \
        case float32: DISPATCH_OUT_TYPE_GEN(float, IdxT); break; \
        case float16: DISPATCH_OUT_TYPE_GEN(__half, IdxT); break; \
        case bfloat16: DISPATCH_OUT_TYPE_GEN(hip_bfloat16, IdxT); break; \
        case int32: DISPATCH_OUT_TYPE_GEN(int32_t, IdxT); break; \
        case int64: DISPATCH_OUT_TYPE_GEN(int64_t, IdxT); break; \
        case uint32: DISPATCH_OUT_TYPE_GEN(uint32_t, IdxT); break; \
        case uint8: DISPATCH_OUT_TYPE_GEN(uint8_t, IdxT); break; \
        case bool_: DISPATCH_OUT_TYPE_GEN(bool, IdxT); break; \
        default: break; \
      }
    
    if (large) {
      DISPATCH_IN_TYPE_GEN(int64_t);
    } else {
      DISPATCH_IN_TYPE_GEN(int32_t);
    }
    
    // Free device memory asynchronously on the stream after kernel completes
    (void)hipFreeAsync(d_shape, stream);
    (void)hipFreeAsync(d_strides_in, stream);
    (void)hipFreeAsync(d_strides_out, stream);
    
    #undef DISPATCH_IN_TYPE_GEN
    #undef DISPATCH_OUT_TYPE_GEN
    #undef LAUNCH_COPY_DYNAMIC_GENERAL
  });
}

} // namespace mlx::core
