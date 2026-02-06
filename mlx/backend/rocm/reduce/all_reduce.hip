// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/config.h"
#include "mlx/backend/rocm/reduce/reduce.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/device/fp16_math.hpp"
#include "mlx/dtype_utils.h"

#include <hip/hip_runtime.h>

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

// Specialization for hipFloatComplex
template <>
__device__ hipFloatComplex warp_shfl_down_all(hipFloatComplex val, int offset) {
  return make_hipFloatComplex(
      __shfl_down(val.x, offset),
      __shfl_down(val.y, offset));
}

template <typename T, typename U, typename Op>
__device__ U warp_reduce(U val, Op op) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val = op(val, warp_shfl_down_all(val, offset));
  }
  return val;
}

// Helper to cast input to accumulator type
template <typename U, typename T>
__device__ U cast_to_acc(T val) {
  if constexpr (std::is_same_v<U, bool>) {
    // For And/Or operations, convert to bool
    if constexpr (is_complex_v<T>) {
      return val.x != 0 || val.y != 0;
    } else {
      return static_cast<bool>(val);
    }
  } else {
    return static_cast<U>(val);
  }
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
      acc = op(acc, cast_to_acc<U>(in[i + j]));
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

// Dispatch reduce operations
template <typename F>
void dispatch_reduce_ops(Reduce::ReduceType reduce_type, F&& f) {
  switch (reduce_type) {
    case Reduce::Sum:
      f(type_identity<rocm::Sum>{});
      break;
    case Reduce::Prod:
      f(type_identity<rocm::Prod>{});
      break;
    case Reduce::Max:
      f(type_identity<rocm::Max>{});
      break;
    case Reduce::Min:
      f(type_identity<rocm::Min>{});
      break;
    case Reduce::And:
      f(type_identity<rocm::And>{});
      break;
    case Reduce::Or:
      f(type_identity<rocm::Or>{});
      break;
    default:
      throw std::runtime_error("Unsupported reduce type");
  }
}

// ReduceResult type trait - determines output type for reduction
template <typename Op, typename T>
struct ReduceResult {
  using type = T;
};

// And always produces bool
template <typename T>
struct ReduceResult<rocm::And, T> {
  using type = bool;
};

// Or always produces bool
template <typename T>
struct ReduceResult<rocm::Or, T> {
  using type = bool;
};

// Sum on small integers produces int32
template <typename T>
struct ReduceResult<rocm::Sum, T> {
  using type = std::conditional_t<
      (std::is_integral_v<T> && sizeof(T) <= 4),
      int32_t,
      T>;
};

// Prod on small integers produces int32
template <typename T>
struct ReduceResult<rocm::Prod, T> {
  using type = std::conditional_t<
      (std::is_integral_v<T> && sizeof(T) <= 4),
      int32_t,
      T>;
};

// Check if a reduce operation is valid for a type
template <typename Op, typename T>
constexpr bool is_valid_reduce_op() {
  // All reduce operations work on all types
  // And/Or will cast to bool internally
  return true;
}

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
  Dtype dt = in.dtype();
  
  std::tie(blocks, threads, block_step) = get_args(insize, N_READS);
  
  encoder.set_input_array(in);
  
  // For multi-block reduction, we need an intermediate buffer
  if (blocks > 1) {
    array intermediate({blocks}, out.dtype(), nullptr, {});
    intermediate.set_data(allocator::malloc(intermediate.nbytes()));
    encoder.add_temporary(intermediate);
    encoder.set_output_array(intermediate);
    
    // First pass: reduce to intermediate
    dispatch_all_types(dt, [&](auto type_tag) {
      dispatch_reduce_ops(reduce_type, [&](auto reduce_type_tag) {
        using OP = MLX_GET_TYPE(reduce_type_tag);
        using T = hip_type_t<MLX_GET_TYPE(type_tag)>;
        using U = typename ReduceResult<OP, T>::type;
        
        if constexpr (is_valid_reduce_op<OP, T>()) {
          encoder.launch_kernel([&](hipStream_t stream) {
            hipLaunchKernelGGL(
                (rocm::all_reduce_kernel<T, U, OP, N_READS>),
                dim3(blocks), dim3(threads), 0, stream,
                gpu_ptr<const T>(in), gpu_ptr<U>(intermediate), block_step, insize);
          });
        }
      });
    });
    
    // Set the input for the next step and recalculate the blocks
    dt = intermediate.dtype();
    insize = intermediate.size();
    std::tie(blocks, threads, block_step) = get_args(insize, N_READS);
    encoder.set_input_array(intermediate);
    
    // Second pass: reduce intermediate to output
    encoder.set_output_array(out);
    dispatch_all_types(dt, [&](auto type_tag) {
      dispatch_reduce_ops(reduce_type, [&](auto reduce_type_tag) {
        using OP = MLX_GET_TYPE(reduce_type_tag);
        using T = hip_type_t<MLX_GET_TYPE(type_tag)>;
        using U = typename ReduceResult<OP, T>::type;
        
        if constexpr (is_valid_reduce_op<OP, T>()) {
          encoder.launch_kernel([&](hipStream_t stream) {
            hipLaunchKernelGGL(
                (rocm::all_reduce_kernel<T, U, OP, N_READS>),
                dim3(1), dim3(threads), 0, stream,
                gpu_ptr<const T>(intermediate), gpu_ptr<U>(out), block_step, insize);
          });
        }
      });
    });
  } else {
    // Single block reduction
    encoder.set_output_array(out);
    dispatch_all_types(dt, [&](auto type_tag) {
      dispatch_reduce_ops(reduce_type, [&](auto reduce_type_tag) {
        using OP = MLX_GET_TYPE(reduce_type_tag);
        using T = hip_type_t<MLX_GET_TYPE(type_tag)>;
        using U = typename ReduceResult<OP, T>::type;
        
        if constexpr (is_valid_reduce_op<OP, T>()) {
          encoder.launch_kernel([&](hipStream_t stream) {
            hipLaunchKernelGGL(
                (rocm::all_reduce_kernel<T, U, OP, N_READS>),
                dim3(1), dim3(threads), 0, stream,
                gpu_ptr<const T>(in), gpu_ptr<U>(out), block_step, insize);
          });
        }
      });
    });
  }
}

} // namespace mlx::core
