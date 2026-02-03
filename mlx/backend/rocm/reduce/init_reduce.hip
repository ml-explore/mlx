// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/reduce/reduce.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

template <typename T, typename U, typename Op>
__global__ void init_reduce_kernel(U* out, size_t size) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    out[index] = ReduceInit<Op, T>::value();
  }
}

} // namespace rocm

void init_reduce(
    rocm::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type) {
  // Allocate if needed
  if (out.data_shared_ptr() == nullptr) {
    out.set_data(allocator::malloc(out.nbytes()));
  }
  
  encoder.set_output_array(out);
  
  int block_size = 256;
  int num_blocks = (out.size() + block_size - 1) / block_size;
  
  encoder.launch_kernel([&](hipStream_t stream) {
    #define LAUNCH_INIT_REDUCE(T, U, OP) \
      hipLaunchKernelGGL( \
          (rocm::init_reduce_kernel<T, U, rocm::OP>), \
          dim3(num_blocks), dim3(block_size), 0, stream, \
          out.data<U>(), out.size())
    
    switch (in.dtype()) {
      case float32:
        switch (reduce_type) {
          case Reduce::Sum: LAUNCH_INIT_REDUCE(float, float, Sum); break;
          case Reduce::Prod: LAUNCH_INIT_REDUCE(float, float, Prod); break;
          case Reduce::Max: LAUNCH_INIT_REDUCE(float, float, Max); break;
          case Reduce::Min: LAUNCH_INIT_REDUCE(float, float, Min); break;
          default: break;
        }
        break;
      case float16:
        switch (reduce_type) {
          case Reduce::Sum: LAUNCH_INIT_REDUCE(__half, __half, Sum); break;
          case Reduce::Prod: LAUNCH_INIT_REDUCE(__half, __half, Prod); break;
          case Reduce::Max: LAUNCH_INIT_REDUCE(__half, __half, Max); break;
          case Reduce::Min: LAUNCH_INIT_REDUCE(__half, __half, Min); break;
          default: break;
        }
        break;
      case bfloat16:
        switch (reduce_type) {
          case Reduce::Sum: LAUNCH_INIT_REDUCE(hip_bfloat16, hip_bfloat16, Sum); break;
          case Reduce::Prod: LAUNCH_INIT_REDUCE(hip_bfloat16, hip_bfloat16, Prod); break;
          case Reduce::Max: LAUNCH_INIT_REDUCE(hip_bfloat16, hip_bfloat16, Max); break;
          case Reduce::Min: LAUNCH_INIT_REDUCE(hip_bfloat16, hip_bfloat16, Min); break;
          default: break;
        }
        break;
      case int32:
        switch (reduce_type) {
          case Reduce::Sum: LAUNCH_INIT_REDUCE(int32_t, int32_t, Sum); break;
          case Reduce::Prod: LAUNCH_INIT_REDUCE(int32_t, int32_t, Prod); break;
          case Reduce::Max: LAUNCH_INIT_REDUCE(int32_t, int32_t, Max); break;
          case Reduce::Min: LAUNCH_INIT_REDUCE(int32_t, int32_t, Min); break;
          default: break;
        }
        break;
      case int64:
        switch (reduce_type) {
          case Reduce::Sum: LAUNCH_INIT_REDUCE(int64_t, int64_t, Sum); break;
          case Reduce::Prod: LAUNCH_INIT_REDUCE(int64_t, int64_t, Prod); break;
          case Reduce::Max: LAUNCH_INIT_REDUCE(int64_t, int64_t, Max); break;
          case Reduce::Min: LAUNCH_INIT_REDUCE(int64_t, int64_t, Min); break;
          default: break;
        }
        break;
      case bool_:
        switch (reduce_type) {
          case Reduce::And: LAUNCH_INIT_REDUCE(bool, bool, And); break;
          case Reduce::Or: LAUNCH_INIT_REDUCE(bool, bool, Or); break;
          default: break;
        }
        break;
      default:
        // For unsupported types, just zero-fill
        (void)hipMemsetAsync(out.data<void>(), 0, out.nbytes(), stream);
        break;
    }
    #undef LAUNCH_INIT_REDUCE
  });
}

} // namespace mlx::core
