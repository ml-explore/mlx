// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/config.h"
#include "mlx/backend/rocm/reduce/reduce.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/device/fp16_math.hpp"

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>

namespace mlx::core {

namespace rocm {

// Helper to handle warp shuffle for different types
template <typename T>
__device__ T warp_shfl_down_all(T val, int offset) {
  return __shfl_down(val, offset);
}

// Specialization for hip_bfloat16 - convert to float for shuffle
template <>
__device__ hip_bfloat16 warp_shfl_down_all(hip_bfloat16 val, int offset) {
  float f = bf16_to_float(val);
  f = __shfl_down(f, offset);
  return float_to_bf16(f);
}

// Specialization for __half - convert to float for shuffle
template <>
__device__ __half warp_shfl_down_all(__half val, int offset) {
  float f = __half2float(val);
  f = __shfl_down(f, offset);
  return __float2half(f);
}

template <typename T, typename U, typename Op>
__device__ U warp_reduce(U val, Op op) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val = op(val, warp_shfl_down_all(val, offset));
  }
  return val;
}

template <typename T, typename U, typename Op, int N = 4>
__global__ void all_reduce_kernel(
    const T* __restrict__ in,
    U* __restrict__ out,
    size_t block_step,
    size_t size) {
  __shared__ U shared_data[32];
  
  const U init = ReduceInit<Op, T>::value();
  Op op;
  
  U acc = init;
  
  size_t start = blockIdx.x * block_step;
  size_t end = min(start + block_step, size);
  
  // Each thread processes multiple elements
  for (size_t i = start + threadIdx.x * N; i < end; i += blockDim.x * N) {
    #pragma unroll
    for (int j = 0; j < N && (i + j) < end; ++j) {
      acc = op(acc, static_cast<U>(in[i + j]));
    }
  }
  
  // Warp-level reduction
  int lane = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;
  
  acc = warp_reduce<T, U, Op>(acc, op);
  
  if (lane == 0) {
    shared_data[warp_id] = acc;
  }
  __syncthreads();
  
  // Final reduction by first warp
  int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
  if (warp_id == 0) {
    acc = (lane < num_warps) ? shared_data[lane] : init;
    acc = warp_reduce<T, U, Op>(acc, op);
    
    if (lane == 0) {
      out[blockIdx.x] = acc;
    }
  }
}

} // namespace rocm

void all_reduce(
    rocm::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type) {
  constexpr int N_READS = 4;
  
  out.set_data(allocator::malloc(out.nbytes()));
  
  auto get_args = [](size_t size, int N) {
    int threads = std::min(512, static_cast<int>((size + N - 1) / N));
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    int reductions_per_step = threads * N;
    size_t steps_needed = (size + reductions_per_step - 1) / reductions_per_step;
    
    int blocks;
    if (steps_needed < 32) {
      blocks = 1;
    } else if (steps_needed < 128) {
      blocks = 32;
    } else if (steps_needed < 512) {
      blocks = 128;
    } else if (steps_needed < 1024) {
      blocks = 512;
    } else {
      blocks = 1024;
    }
    
    size_t steps_per_block = (steps_needed + blocks - 1) / blocks;
    size_t block_step = steps_per_block * reductions_per_step;
    
    return std::make_tuple(blocks, threads, block_step);
  };
  
  int blocks, threads;
  size_t block_step;
  size_t insize = in.size();
  
  std::tie(blocks, threads, block_step) = get_args(insize, N_READS);
  
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  
  // For multi-block reduction, we need an intermediate buffer
  if (blocks > 1) {
    array intermediate({blocks}, out.dtype(), nullptr, {});
    intermediate.set_data(allocator::malloc(intermediate.nbytes()));
    encoder.add_temporary(intermediate);
    
    // First pass: reduce to intermediate
    encoder.launch_kernel([&](hipStream_t stream) {
      #define LAUNCH_ALL_REDUCE(T, U, OP) \
        hipLaunchKernelGGL( \
            (rocm::all_reduce_kernel<T, U, rocm::OP, N_READS>), \
            dim3(blocks), dim3(threads), 0, stream, \
            in.data<T>(), intermediate.data<U>(), block_step, insize)
      
      switch (in.dtype()) {
        case float32:
          switch (reduce_type) {
            case Reduce::Sum: LAUNCH_ALL_REDUCE(float, float, Sum); break;
            case Reduce::Prod: LAUNCH_ALL_REDUCE(float, float, Prod); break;
            case Reduce::Max: LAUNCH_ALL_REDUCE(float, float, Max); break;
            case Reduce::Min: LAUNCH_ALL_REDUCE(float, float, Min); break;
            default: break;
          }
          break;
        case float16:
          switch (reduce_type) {
            case Reduce::Sum: LAUNCH_ALL_REDUCE(__half, float, Sum); break;
            case Reduce::Prod: LAUNCH_ALL_REDUCE(__half, float, Prod); break;
            case Reduce::Max: LAUNCH_ALL_REDUCE(__half, __half, Max); break;
            case Reduce::Min: LAUNCH_ALL_REDUCE(__half, __half, Min); break;
            default: break;
          }
          break;
        case int32:
          switch (reduce_type) {
            case Reduce::Sum: LAUNCH_ALL_REDUCE(int32_t, int32_t, Sum); break;
            case Reduce::Prod: LAUNCH_ALL_REDUCE(int32_t, int32_t, Prod); break;
            case Reduce::Max: LAUNCH_ALL_REDUCE(int32_t, int32_t, Max); break;
            case Reduce::Min: LAUNCH_ALL_REDUCE(int32_t, int32_t, Min); break;
            default: break;
          }
          break;
        case int64:
          switch (reduce_type) {
            case Reduce::Sum: LAUNCH_ALL_REDUCE(int64_t, int64_t, Sum); break;
            case Reduce::Prod: LAUNCH_ALL_REDUCE(int64_t, int64_t, Prod); break;
            case Reduce::Max: LAUNCH_ALL_REDUCE(int64_t, int64_t, Max); break;
            case Reduce::Min: LAUNCH_ALL_REDUCE(int64_t, int64_t, Min); break;
            default: break;
          }
          break;
        case bool_:
          switch (reduce_type) {
            case Reduce::And: LAUNCH_ALL_REDUCE(bool, bool, And); break;
            case Reduce::Or: LAUNCH_ALL_REDUCE(bool, bool, Or); break;
            default: break;
          }
          break;
        default:
          throw std::runtime_error("Unsupported type for all_reduce");
      }
      #undef LAUNCH_ALL_REDUCE
    });
    
    // Second pass: reduce intermediate to output
    std::tie(blocks, threads, block_step) = get_args(intermediate.size(), N_READS);
    
    encoder.launch_kernel([&](hipStream_t stream) {
      #define LAUNCH_ALL_REDUCE_FINAL(T, U, OP) \
        hipLaunchKernelGGL( \
            (rocm::all_reduce_kernel<T, U, rocm::OP, N_READS>), \
            dim3(1), dim3(threads), 0, stream, \
            intermediate.data<T>(), out.data<U>(), block_step, intermediate.size())
      
      switch (out.dtype()) {
        case float32:
          switch (reduce_type) {
            case Reduce::Sum: LAUNCH_ALL_REDUCE_FINAL(float, float, Sum); break;
            case Reduce::Prod: LAUNCH_ALL_REDUCE_FINAL(float, float, Prod); break;
            case Reduce::Max: LAUNCH_ALL_REDUCE_FINAL(float, float, Max); break;
            case Reduce::Min: LAUNCH_ALL_REDUCE_FINAL(float, float, Min); break;
            default: break;
          }
          break;
        case float16:
          switch (reduce_type) {
            case Reduce::Sum: LAUNCH_ALL_REDUCE_FINAL(float, __half, Sum); break;
            case Reduce::Prod: LAUNCH_ALL_REDUCE_FINAL(float, __half, Prod); break;
            case Reduce::Max: LAUNCH_ALL_REDUCE_FINAL(__half, __half, Max); break;
            case Reduce::Min: LAUNCH_ALL_REDUCE_FINAL(__half, __half, Min); break;
            default: break;
          }
          break;
        case int32:
          switch (reduce_type) {
            case Reduce::Sum: LAUNCH_ALL_REDUCE_FINAL(int32_t, int32_t, Sum); break;
            case Reduce::Prod: LAUNCH_ALL_REDUCE_FINAL(int32_t, int32_t, Prod); break;
            case Reduce::Max: LAUNCH_ALL_REDUCE_FINAL(int32_t, int32_t, Max); break;
            case Reduce::Min: LAUNCH_ALL_REDUCE_FINAL(int32_t, int32_t, Min); break;
            default: break;
          }
          break;
        case int64:
          switch (reduce_type) {
            case Reduce::Sum: LAUNCH_ALL_REDUCE_FINAL(int64_t, int64_t, Sum); break;
            case Reduce::Prod: LAUNCH_ALL_REDUCE_FINAL(int64_t, int64_t, Prod); break;
            case Reduce::Max: LAUNCH_ALL_REDUCE_FINAL(int64_t, int64_t, Max); break;
            case Reduce::Min: LAUNCH_ALL_REDUCE_FINAL(int64_t, int64_t, Min); break;
            default: break;
          }
          break;
        case bool_:
          switch (reduce_type) {
            case Reduce::And: LAUNCH_ALL_REDUCE_FINAL(bool, bool, And); break;
            case Reduce::Or: LAUNCH_ALL_REDUCE_FINAL(bool, bool, Or); break;
            default: break;
          }
          break;
        default:
          throw std::runtime_error("Unsupported type for all_reduce");
      }
      #undef LAUNCH_ALL_REDUCE_FINAL
    });
  } else {
    // Single block reduction
    encoder.launch_kernel([&](hipStream_t stream) {
      #define LAUNCH_ALL_REDUCE_SINGLE(T, U, OP) \
        hipLaunchKernelGGL( \
            (rocm::all_reduce_kernel<T, U, rocm::OP, N_READS>), \
            dim3(1), dim3(threads), 0, stream, \
            in.data<T>(), out.data<U>(), block_step, insize)
      
      switch (in.dtype()) {
        case float32:
          switch (reduce_type) {
            case Reduce::Sum: LAUNCH_ALL_REDUCE_SINGLE(float, float, Sum); break;
            case Reduce::Prod: LAUNCH_ALL_REDUCE_SINGLE(float, float, Prod); break;
            case Reduce::Max: LAUNCH_ALL_REDUCE_SINGLE(float, float, Max); break;
            case Reduce::Min: LAUNCH_ALL_REDUCE_SINGLE(float, float, Min); break;
            default: break;
          }
          break;
        case float16:
          switch (reduce_type) {
            case Reduce::Sum: LAUNCH_ALL_REDUCE_SINGLE(__half, __half, Sum); break;
            case Reduce::Prod: LAUNCH_ALL_REDUCE_SINGLE(__half, __half, Prod); break;
            case Reduce::Max: LAUNCH_ALL_REDUCE_SINGLE(__half, __half, Max); break;
            case Reduce::Min: LAUNCH_ALL_REDUCE_SINGLE(__half, __half, Min); break;
            default: break;
          }
          break;
        case int32:
          switch (reduce_type) {
            case Reduce::Sum: LAUNCH_ALL_REDUCE_SINGLE(int32_t, int32_t, Sum); break;
            case Reduce::Prod: LAUNCH_ALL_REDUCE_SINGLE(int32_t, int32_t, Prod); break;
            case Reduce::Max: LAUNCH_ALL_REDUCE_SINGLE(int32_t, int32_t, Max); break;
            case Reduce::Min: LAUNCH_ALL_REDUCE_SINGLE(int32_t, int32_t, Min); break;
            default: break;
          }
          break;
        case int64:
          switch (reduce_type) {
            case Reduce::Sum: LAUNCH_ALL_REDUCE_SINGLE(int64_t, int64_t, Sum); break;
            case Reduce::Prod: LAUNCH_ALL_REDUCE_SINGLE(int64_t, int64_t, Prod); break;
            case Reduce::Max: LAUNCH_ALL_REDUCE_SINGLE(int64_t, int64_t, Max); break;
            case Reduce::Min: LAUNCH_ALL_REDUCE_SINGLE(int64_t, int64_t, Min); break;
            default: break;
          }
          break;
        case bool_:
          switch (reduce_type) {
            case Reduce::And: LAUNCH_ALL_REDUCE_SINGLE(bool, bool, And); break;
            case Reduce::Or: LAUNCH_ALL_REDUCE_SINGLE(bool, bool, Or); break;
            default: break;
          }
          break;
        default:
          throw std::runtime_error("Unsupported type for all_reduce");
      }
      #undef LAUNCH_ALL_REDUCE_SINGLE
    });
  }
}

} // namespace mlx::core
