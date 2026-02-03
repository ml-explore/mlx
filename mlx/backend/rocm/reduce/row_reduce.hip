// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/config.h"
#include "mlx/backend/rocm/reduce/reduce.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/device/fp16_math.hpp"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

// Use WARP_SIZE from config.h (architecture-dependent)
constexpr int WARP_SIZE_ROW = WARP_SIZE;

// Helper to handle warp shuffle for different types
template <typename T>
__device__ T warp_shfl_down(T val, int offset) {
  return __shfl_down(val, offset);
}

// Specialization for hip_bfloat16 - convert to float for shuffle
template <>
__device__ hip_bfloat16 warp_shfl_down(hip_bfloat16 val, int offset) {
  float f = bf16_to_float(val);
  f = __shfl_down(f, offset);
  return float_to_bf16(f);
}

// Specialization for __half - convert to float for shuffle
template <>
__device__ __half warp_shfl_down(__half val, int offset) {
  float f = __half2float(val);
  f = __shfl_down(f, offset);
  return __float2half(f);
}

template <typename T, typename U, typename Op, int N = 4>
__global__ void row_reduce_simple_kernel(
    const T* __restrict__ in,
    U* __restrict__ out,
    size_t n_rows,
    int row_size) {
  __shared__ U shared_data[32];
  
  const U init = ReduceInit<Op, T>::value();
  Op op;
  
  size_t row = blockIdx.x;
  if (row >= n_rows) return;
  
  const T* row_in = in + row * row_size;
  U acc = init;
  
  // Each thread processes multiple elements
  for (int i = threadIdx.x * N; i < row_size; i += blockDim.x * N) {
    #pragma unroll
    for (int j = 0; j < N && (i + j) < row_size; ++j) {
      acc = op(acc, static_cast<U>(row_in[i + j]));
    }
  }
  
  // Warp-level reduction using helper
  int lane = threadIdx.x % WARP_SIZE_ROW;
  int warp_id = threadIdx.x / WARP_SIZE_ROW;
  
  for (int offset = WARP_SIZE_ROW / 2; offset > 0; offset /= 2) {
    acc = op(acc, warp_shfl_down(acc, offset));
  }
  
  if (lane == 0) {
    shared_data[warp_id] = acc;
  }
  __syncthreads();
  
  // Final reduction by first warp
  int num_warps = (blockDim.x + WARP_SIZE_ROW - 1) / WARP_SIZE_ROW;
  if (warp_id == 0) {
    acc = (lane < num_warps) ? shared_data[lane] : init;
    for (int offset = WARP_SIZE_ROW / 2; offset > 0; offset /= 2) {
      acc = op(acc, warp_shfl_down(acc, offset));
    }
    
    if (lane == 0) {
      out[row] = acc;
    }
  }
}

template <typename T, typename U, typename Op>
__global__ void row_reduce_looped_kernel(
    const T* __restrict__ in,
    U* __restrict__ out,
    size_t out_size,
    int row_size,
    const int64_t* __restrict__ in_strides,
    const int* __restrict__ shape,
    int ndim,
    size_t non_row_reductions,
    const int64_t* __restrict__ reduce_strides,
    const int* __restrict__ reduce_shape,
    int reduce_ndim) {
  __shared__ U shared_data[32];
  
  const U init = ReduceInit<Op, T>::value();
  Op op;
  
  size_t out_idx = blockIdx.x;
  if (out_idx >= out_size) return;
  
  // Compute base input offset from output index
  int64_t base_offset = 0;
  size_t tmp = out_idx;
  for (int i = ndim - 1; i >= 0; --i) {
    int coord = tmp % shape[i];
    base_offset += coord * in_strides[i];
    tmp /= shape[i];
  }
  
  U acc = init;
  
  // Loop over non-row reductions
  for (size_t n = 0; n < non_row_reductions; ++n) {
    // Compute reduction offset
    int64_t reduce_offset = 0;
    size_t rtmp = n;
    for (int i = reduce_ndim - 1; i >= 0; --i) {
      int coord = rtmp % reduce_shape[i];
      reduce_offset += coord * reduce_strides[i];
      rtmp /= reduce_shape[i];
    }
    
    const T* row_in = in + base_offset + reduce_offset;
    
    // Reduce the row
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
      acc = op(acc, static_cast<U>(row_in[i]));
    }
  }
  
  // Warp-level reduction
  int lane = threadIdx.x % WARP_SIZE_ROW;
  int warp_id = threadIdx.x / WARP_SIZE_ROW;
  
  for (int offset = WARP_SIZE_ROW / 2; offset > 0; offset /= 2) {
    acc = op(acc, warp_shfl_down(acc, offset));
  }
  
  if (lane == 0) {
    shared_data[warp_id] = acc;
  }
  __syncthreads();
  
  int num_warps = (blockDim.x + WARP_SIZE_ROW - 1) / WARP_SIZE_ROW;
  if (warp_id == 0) {
    acc = (lane < num_warps) ? shared_data[lane] : init;
    for (int offset = WARP_SIZE_ROW / 2; offset > 0; offset /= 2) {
      acc = op(acc, warp_shfl_down(acc, offset));
    }
    
    if (lane == 0) {
      out[out_idx] = acc;
    }
  }
}

} // namespace rocm

void row_reduce(
    rocm::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan) {
  out.set_data(allocator::malloc(out.nbytes()));
  
  int row_size = plan.shape.back();
  size_t out_size = out.size();
  
  // Calculate threads based on row size
  int threads = std::min(256, ((row_size + 3) / 4 + rocm::WARP_SIZE_ROW - 1) / rocm::WARP_SIZE_ROW * rocm::WARP_SIZE_ROW);
  threads = std::max(threads, rocm::WARP_SIZE_ROW);
  
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  
  // Simple row reduce for single reduction axis
  if (plan.shape.size() == 1) {
    encoder.launch_kernel([&](hipStream_t stream) {
      #define LAUNCH_ROW_REDUCE(T, U, OP) \
        hipLaunchKernelGGL( \
            (rocm::row_reduce_simple_kernel<T, U, rocm::OP, 4>), \
            dim3(out_size), dim3(threads), 0, stream, \
            in.data<T>(), out.data<U>(), out_size, row_size)
      
      switch (in.dtype()) {
        case float32:
          switch (reduce_type) {
            case Reduce::Sum: LAUNCH_ROW_REDUCE(float, float, Sum); break;
            case Reduce::Prod: LAUNCH_ROW_REDUCE(float, float, Prod); break;
            case Reduce::Max: LAUNCH_ROW_REDUCE(float, float, Max); break;
            case Reduce::Min: LAUNCH_ROW_REDUCE(float, float, Min); break;
            default: break;
          }
          break;
        case float16:
          switch (reduce_type) {
            case Reduce::Sum: LAUNCH_ROW_REDUCE(__half, __half, Sum); break;
            case Reduce::Prod: LAUNCH_ROW_REDUCE(__half, __half, Prod); break;
            case Reduce::Max: LAUNCH_ROW_REDUCE(__half, __half, Max); break;
            case Reduce::Min: LAUNCH_ROW_REDUCE(__half, __half, Min); break;
            default: break;
          }
          break;
        case bfloat16:
          switch (reduce_type) {
            case Reduce::Sum: LAUNCH_ROW_REDUCE(hip_bfloat16, hip_bfloat16, Sum); break;
            case Reduce::Prod: LAUNCH_ROW_REDUCE(hip_bfloat16, hip_bfloat16, Prod); break;
            case Reduce::Max: LAUNCH_ROW_REDUCE(hip_bfloat16, hip_bfloat16, Max); break;
            case Reduce::Min: LAUNCH_ROW_REDUCE(hip_bfloat16, hip_bfloat16, Min); break;
            default: break;
          }
          break;
        case int32:
          switch (reduce_type) {
            case Reduce::Sum: LAUNCH_ROW_REDUCE(int32_t, int32_t, Sum); break;
            case Reduce::Prod: LAUNCH_ROW_REDUCE(int32_t, int32_t, Prod); break;
            case Reduce::Max: LAUNCH_ROW_REDUCE(int32_t, int32_t, Max); break;
            case Reduce::Min: LAUNCH_ROW_REDUCE(int32_t, int32_t, Min); break;
            default: break;
          }
          break;
        case int64:
          switch (reduce_type) {
            case Reduce::Sum: LAUNCH_ROW_REDUCE(int64_t, int64_t, Sum); break;
            case Reduce::Prod: LAUNCH_ROW_REDUCE(int64_t, int64_t, Prod); break;
            case Reduce::Max: LAUNCH_ROW_REDUCE(int64_t, int64_t, Max); break;
            case Reduce::Min: LAUNCH_ROW_REDUCE(int64_t, int64_t, Min); break;
            default: break;
          }
          break;
        case bool_:
          switch (reduce_type) {
            case Reduce::And: LAUNCH_ROW_REDUCE(bool, bool, And); break;
            case Reduce::Or: LAUNCH_ROW_REDUCE(bool, bool, Or); break;
            default: break;
          }
          break;
        default:
          throw std::runtime_error("Unsupported type for row_reduce");
      }
      #undef LAUNCH_ROW_REDUCE
    });
  } else {
    // Looped row reduce for multiple reduction axes
    // For now, fall back to simple implementation
    encoder.launch_kernel([&](hipStream_t stream) {
      #define LAUNCH_ROW_REDUCE_SIMPLE(T, U, OP) \
        hipLaunchKernelGGL( \
            (rocm::row_reduce_simple_kernel<T, U, rocm::OP, 4>), \
            dim3(out_size), dim3(threads), 0, stream, \
            in.data<T>(), out.data<U>(), out_size, row_size)
      
      switch (in.dtype()) {
        case float32:
          switch (reduce_type) {
            case Reduce::Sum: LAUNCH_ROW_REDUCE_SIMPLE(float, float, Sum); break;
            case Reduce::Prod: LAUNCH_ROW_REDUCE_SIMPLE(float, float, Prod); break;
            case Reduce::Max: LAUNCH_ROW_REDUCE_SIMPLE(float, float, Max); break;
            case Reduce::Min: LAUNCH_ROW_REDUCE_SIMPLE(float, float, Min); break;
            default: break;
          }
          break;
        default:
          throw std::runtime_error("Unsupported type for looped row_reduce");
      }
      #undef LAUNCH_ROW_REDUCE_SIMPLE
    });
  }
}

} // namespace mlx::core
