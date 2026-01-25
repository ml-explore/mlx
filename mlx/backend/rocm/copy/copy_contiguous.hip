// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/copy/copy.hpp"
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

void copy_contiguous(
    rocm::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t in_offset,
    int64_t out_offset) {
  
  bool large = out.data_size() > UINT32_MAX;
  
  auto launch_kernel = [&](auto in_ptr, auto out_ptr, auto size) {
    using InType = std::remove_pointer_t<decltype(in_ptr)>;
    using OutType = std::remove_pointer_t<decltype(out_ptr)>;
    
    constexpr int N_READS = 4;
    int block_size = 256;
    int num_blocks = (size + block_size * N_READS - 1) / (block_size * N_READS);
    num_blocks = std::min(num_blocks, 65535);
    
    encoder.launch_kernel([&](hipStream_t stream) {
      if (ctype == CopyType::Scalar) {
        if (large) {
          hipLaunchKernelGGL(
              (rocm::copy_s<InType, OutType, int64_t, N_READS>),
              dim3(num_blocks), dim3(block_size), 0, stream,
              in_ptr + in_offset, out_ptr + out_offset, static_cast<int64_t>(size));
        } else {
          hipLaunchKernelGGL(
              (rocm::copy_s<InType, OutType, uint32_t, N_READS>),
              dim3(num_blocks), dim3(block_size), 0, stream,
              in_ptr + in_offset, out_ptr + out_offset, static_cast<uint32_t>(size));
        }
      } else {
        if (large) {
          hipLaunchKernelGGL(
              (rocm::copy_v<InType, OutType, int64_t, N_READS>),
              dim3(num_blocks), dim3(block_size), 0, stream,
              in_ptr + in_offset, out_ptr + out_offset, static_cast<int64_t>(size));
        } else {
          hipLaunchKernelGGL(
              (rocm::copy_v<InType, OutType, uint32_t, N_READS>),
              dim3(num_blocks), dim3(block_size), 0, stream,
              in_ptr + in_offset, out_ptr + out_offset, static_cast<uint32_t>(size));
        }
      }
    });
  };
  
  // Type dispatch - same type copy is most common
  if (in.dtype() == out.dtype()) {
    switch (in.dtype()) {
      case float32:
        launch_kernel(in.data<float>(), out.data<float>(), out.data_size());
        break;
      case float16:
        launch_kernel(in.data<__half>(), out.data<__half>(), out.data_size());
        break;
      case bfloat16:
        launch_kernel(in.data<hip_bfloat16>(), out.data<hip_bfloat16>(), out.data_size());
        break;
      case int32:
        launch_kernel(in.data<int32_t>(), out.data<int32_t>(), out.data_size());
        break;
      case int64:
        launch_kernel(in.data<int64_t>(), out.data<int64_t>(), out.data_size());
        break;
      case uint32:
        launch_kernel(in.data<uint32_t>(), out.data<uint32_t>(), out.data_size());
        break;
      case uint64:
        launch_kernel(in.data<uint64_t>(), out.data<uint64_t>(), out.data_size());
        break;
      case int8:
        launch_kernel(in.data<int8_t>(), out.data<int8_t>(), out.data_size());
        break;
      case uint8:
        launch_kernel(in.data<uint8_t>(), out.data<uint8_t>(), out.data_size());
        break;
      case bool_:
        launch_kernel(in.data<bool>(), out.data<bool>(), out.data_size());
        break;
      default:
        throw std::runtime_error(
            std::string("Unsupported type for copy: ") + dtype_to_string(in.dtype()));
    }
  } else {
    // Cross-type copy - handle common conversions
    throw std::runtime_error("Cross-type copy not yet fully implemented for ROCm.");
  }
}

} // namespace mlx::core
