// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/copy/copy.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/dtype_utils.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

template <typename In, typename Out, typename IdxT, int N_READS>
__global__ void copy_s(const In* in, Out* out, IdxT size) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  IdxT stride = blockDim.x * gridDim.x;

  for (IdxT i = index * N_READS; i < size; i += stride * N_READS) {
    if (i + N_READS <= size) {
      #pragma unroll
      for (int j = 0; j < N_READS; ++j) {
        out[i + j] = cast_to<Out>(in[0]);
      }
    } else {
      for (IdxT j = i; j < size; ++j) {
        out[j] = cast_to<Out>(in[0]);
      }
    }
  }
}

template <typename In, typename Out, typename IdxT, int N_READS>
__global__ void copy_v(const In* in, Out* out, IdxT size) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  IdxT stride = blockDim.x * gridDim.x;

  for (IdxT i = index * N_READS; i < size; i += stride * N_READS) {
    if (i + N_READS <= size) {
      #pragma unroll
      for (int j = 0; j < N_READS; ++j) {
        out[i + j] = cast_to<Out>(in[i + j]);
      }
    } else {
      for (IdxT j = i; j < size; ++j) {
        out[j] = cast_to<Out>(in[j]);
      }
    }
  }
}

} // namespace rocm

// Macro to launch copy kernel for a specific type combination
#define LAUNCH_COPY_KERNEL(InT, OutT) \
  do { \
    constexpr int N_READS = 4; \
    int block_size = 256; \
    int num_blocks = (size + block_size * N_READS - 1) / (block_size * N_READS); \
    num_blocks = std::min(num_blocks, 65535); \
    const InT* in_ptr = reinterpret_cast<const InT*>(in.data<void>()) + in_offset; \
    OutT* out_ptr = reinterpret_cast<OutT*>(out.data<void>()) + out_offset; \
    encoder.launch_kernel([&](hipStream_t stream) { \
      if (ctype == CopyType::Scalar) { \
        if (large) { \
          hipLaunchKernelGGL( \
              (rocm::copy_s<InT, OutT, int64_t, N_READS>), \
              dim3(num_blocks), dim3(block_size), 0, stream, \
              in_ptr, out_ptr, static_cast<int64_t>(size)); \
        } else { \
          hipLaunchKernelGGL( \
              (rocm::copy_s<InT, OutT, uint32_t, N_READS>), \
              dim3(num_blocks), dim3(block_size), 0, stream, \
              in_ptr, out_ptr, static_cast<uint32_t>(size)); \
        } \
      } else { \
        if (large) { \
          hipLaunchKernelGGL( \
              (rocm::copy_v<InT, OutT, int64_t, N_READS>), \
              dim3(num_blocks), dim3(block_size), 0, stream, \
              in_ptr, out_ptr, static_cast<int64_t>(size)); \
        } else { \
          hipLaunchKernelGGL( \
              (rocm::copy_v<InT, OutT, uint32_t, N_READS>), \
              dim3(num_blocks), dim3(block_size), 0, stream, \
              in_ptr, out_ptr, static_cast<uint32_t>(size)); \
        } \
      } \
    }); \
  } while(0)

void copy_contiguous(
    rocm::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t in_offset,
    int64_t out_offset) {
  
  bool large = out.data_size() > UINT32_MAX;
  size_t size = out.data_size();
  
  // Handle same-type copies (most common case)
  if (in.dtype() == out.dtype()) {
    switch (in.dtype()) {
      case float32: LAUNCH_COPY_KERNEL(float, float); return;
      case float16: LAUNCH_COPY_KERNEL(__half, __half); return;
      case bfloat16: LAUNCH_COPY_KERNEL(hip_bfloat16, hip_bfloat16); return;
      case int32: LAUNCH_COPY_KERNEL(int32_t, int32_t); return;
      case int64: LAUNCH_COPY_KERNEL(int64_t, int64_t); return;
      case uint32: LAUNCH_COPY_KERNEL(uint32_t, uint32_t); return;
      case uint64: LAUNCH_COPY_KERNEL(uint64_t, uint64_t); return;
      case int8: LAUNCH_COPY_KERNEL(int8_t, int8_t); return;
      case int16: LAUNCH_COPY_KERNEL(int16_t, int16_t); return;
      case uint8: LAUNCH_COPY_KERNEL(uint8_t, uint8_t); return;
      case uint16: LAUNCH_COPY_KERNEL(uint16_t, uint16_t); return;
      case bool_: LAUNCH_COPY_KERNEL(bool, bool); return;
      case float64: LAUNCH_COPY_KERNEL(double, double); return;
      default: break;
    }
  }
  
  // Handle cross-type copies - common conversions
  switch (in.dtype()) {
    case float32:
      switch (out.dtype()) {
        case float16: LAUNCH_COPY_KERNEL(float, __half); return;
        case bfloat16: LAUNCH_COPY_KERNEL(float, hip_bfloat16); return;
        case int32: LAUNCH_COPY_KERNEL(float, int32_t); return;
        case int64: LAUNCH_COPY_KERNEL(float, int64_t); return;
        case bool_: LAUNCH_COPY_KERNEL(float, bool); return;
        case float64: LAUNCH_COPY_KERNEL(float, double); return;
        default: break;
      }
      break;
    case float16:
      switch (out.dtype()) {
        case float32: LAUNCH_COPY_KERNEL(__half, float); return;
        case bfloat16: LAUNCH_COPY_KERNEL(__half, hip_bfloat16); return;
        case int32: LAUNCH_COPY_KERNEL(__half, int32_t); return;
        case bool_: LAUNCH_COPY_KERNEL(__half, bool); return;
        default: break;
      }
      break;
    case bfloat16:
      switch (out.dtype()) {
        case float32: LAUNCH_COPY_KERNEL(hip_bfloat16, float); return;
        case float16: LAUNCH_COPY_KERNEL(hip_bfloat16, __half); return;
        case int32: LAUNCH_COPY_KERNEL(hip_bfloat16, int32_t); return;
        case bool_: LAUNCH_COPY_KERNEL(hip_bfloat16, bool); return;
        default: break;
      }
      break;
    case int32:
      switch (out.dtype()) {
        case float32: LAUNCH_COPY_KERNEL(int32_t, float); return;
        case float16: LAUNCH_COPY_KERNEL(int32_t, __half); return;
        case int64: LAUNCH_COPY_KERNEL(int32_t, int64_t); return;
        case uint32: LAUNCH_COPY_KERNEL(int32_t, uint32_t); return;
        case bool_: LAUNCH_COPY_KERNEL(int32_t, bool); return;
        default: break;
      }
      break;
    case int64:
      switch (out.dtype()) {
        case float32: LAUNCH_COPY_KERNEL(int64_t, float); return;
        case int32: LAUNCH_COPY_KERNEL(int64_t, int32_t); return;
        case uint64: LAUNCH_COPY_KERNEL(int64_t, uint64_t); return;
        case bool_: LAUNCH_COPY_KERNEL(int64_t, bool); return;
        default: break;
      }
      break;
    case uint32:
      switch (out.dtype()) {
        case float32: LAUNCH_COPY_KERNEL(uint32_t, float); return;
        case int32: LAUNCH_COPY_KERNEL(uint32_t, int32_t); return;
        case int64: LAUNCH_COPY_KERNEL(uint32_t, int64_t); return;
        case uint64: LAUNCH_COPY_KERNEL(uint32_t, uint64_t); return;
        case bool_: LAUNCH_COPY_KERNEL(uint32_t, bool); return;
        default: break;
      }
      break;
    case uint64:
      switch (out.dtype()) {
        case float32: LAUNCH_COPY_KERNEL(uint64_t, float); return;
        case int64: LAUNCH_COPY_KERNEL(uint64_t, int64_t); return;
        case uint32: LAUNCH_COPY_KERNEL(uint64_t, uint32_t); return;
        case bool_: LAUNCH_COPY_KERNEL(uint64_t, bool); return;
        default: break;
      }
      break;
    case int8:
      switch (out.dtype()) {
        case float32: LAUNCH_COPY_KERNEL(int8_t, float); return;
        case int32: LAUNCH_COPY_KERNEL(int8_t, int32_t); return;
        case int16: LAUNCH_COPY_KERNEL(int8_t, int16_t); return;
        case bool_: LAUNCH_COPY_KERNEL(int8_t, bool); return;
        default: break;
      }
      break;
    case uint8:
      switch (out.dtype()) {
        case float32: LAUNCH_COPY_KERNEL(uint8_t, float); return;
        case int32: LAUNCH_COPY_KERNEL(uint8_t, int32_t); return;
        case uint16: LAUNCH_COPY_KERNEL(uint8_t, uint16_t); return;
        case bool_: LAUNCH_COPY_KERNEL(uint8_t, bool); return;
        default: break;
      }
      break;
    case bool_:
      switch (out.dtype()) {
        case float32: LAUNCH_COPY_KERNEL(bool, float); return;
        case int32: LAUNCH_COPY_KERNEL(bool, int32_t); return;
        case int8: LAUNCH_COPY_KERNEL(bool, int8_t); return;
        case uint8: LAUNCH_COPY_KERNEL(bool, uint8_t); return;
        default: break;
      }
      break;
    case float64:
      switch (out.dtype()) {
        case float32: LAUNCH_COPY_KERNEL(double, float); return;
        case int64: LAUNCH_COPY_KERNEL(double, int64_t); return;
        case bool_: LAUNCH_COPY_KERNEL(double, bool); return;
        default: break;
      }
      break;
    default:
      break;
  }
  
  throw std::runtime_error(
      std::string("Unsupported type conversion in copy: ") + 
      dtype_to_string(in.dtype()) + " -> " + dtype_to_string(out.dtype()));
}

#undef LAUNCH_COPY_KERNEL

} // namespace mlx::core
