// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/config.h"
#include "mlx/backend/rocm/reduce/reduce.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/device/fp16_math.hpp"
#include "mlx/backend/rocm/device/utils.hpp"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

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

// Helper to cast input to accumulator type
template <typename U, typename T>
__device__ U cast_to_row(T val) {
  if constexpr (std::is_same_v<U, bool>) {
    // For And/Or operations, convert to bool
    return static_cast<bool>(val);
  } else {
    return static_cast<U>(val);
  }
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
      acc = op(acc, cast_to_row<U>(row_in[i + j]));
    }
  }
  
  // Warp-level reduction using runtime warpSize
  int lane = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    acc = op(acc, warp_shfl_down(acc, offset));
  }
  
  if (lane == 0) {
    shared_data[warp_id] = acc;
  }
  __syncthreads();
  
  // Final reduction by first warp
  int num_warps = (blockDim.x + warpSize - 1) / warpSize;
  if (warp_id == 0) {
    acc = (lane < num_warps) ? shared_data[lane] : init;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      acc = op(acc, warp_shfl_down(acc, offset));
    }
    
    if (lane == 0) {
      out[row] = acc;
    }
  }
}

template <typename T, typename U, typename Op, int NDIM>
__global__ void row_reduce_looped_kernel(
    const T* __restrict__ in,
    U* __restrict__ out,
    size_t out_size,
    int row_size,
    Shape shape,
    Strides in_strides,
    int ndim,
    size_t non_row_reductions,
    Shape reduce_shape,
    Strides reduce_strides,
    int reduce_ndim) {
  __shared__ U shared_data[32];
  
  const U init = ReduceInit<Op, T>::value();
  Op op;
  
  size_t out_idx = blockIdx.x;
  if (out_idx >= out_size) return;
  
  // Compute base input offset from output index
  int64_t base_offset = elem_to_loc(out_idx, shape.data(), in_strides.data(), ndim);
  
  U acc = init;
  
  // Loop over non-row reductions
  LoopedElemToLoc<NDIM, (NDIM > 2)> loop(reduce_ndim);
  for (size_t n = 0; n < non_row_reductions; ++n) {
    const T* row_in = in + base_offset + loop.location();
    
    // Reduce the row
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
      acc = op(acc, cast_to_row<U>(row_in[i]));
    }
    
    loop.next(reduce_shape.data(), reduce_strides.data());
  }
  
  // Warp-level reduction
  int lane = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    acc = op(acc, warp_shfl_down(acc, offset));
  }
  
  if (lane == 0) {
    shared_data[warp_id] = acc;
  }
  __syncthreads();
  
  int num_warps = (blockDim.x + warpSize - 1) / warpSize;
  if (warp_id == 0) {
    acc = (lane < num_warps) ? shared_data[lane] : init;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      acc = op(acc, warp_shfl_down(acc, offset));
    }
    
    if (lane == 0) {
      out[out_idx] = acc;
    }
  }
}

} // namespace rocm

// Dispatch for reduce types - excludes complex64 which doesn't support most reduce ops
template <typename Func>
void dispatch_reduce_types_row(Dtype dt, Func&& func) {
  switch (dt) {
    case bool_:
      func(type_identity<bool>{});
      break;
    case uint8:
      func(type_identity<uint8_t>{});
      break;
    case uint16:
      func(type_identity<uint16_t>{});
      break;
    case uint32:
      func(type_identity<uint32_t>{});
      break;
    case uint64:
      func(type_identity<uint64_t>{});
      break;
    case int8:
      func(type_identity<int8_t>{});
      break;
    case int16:
      func(type_identity<int16_t>{});
      break;
    case int32:
      func(type_identity<int32_t>{});
      break;
    case int64:
      func(type_identity<int64_t>{});
      break;
    case float16:
      func(type_identity<float16_t>{});
      break;
    case bfloat16:
      func(type_identity<bfloat16_t>{});
      break;
    case float32:
      func(type_identity<float>{});
      break;
    case float64:
      func(type_identity<double>{});
      break;
    case complex64:
      throw std::runtime_error("Complex types not yet supported for reduce operations on ROCm");
    default:
      throw std::runtime_error("Unsupported dtype for reduce");
  }
}

// Dispatch helper for reduce operations - no type restrictions
// The cast_to function handles conversion to bool for And/Or
template <typename Func>
void dispatch_reduce_ops_row(Reduce::ReduceType reduce_type, Func&& func) {
  switch (reduce_type) {
    case Reduce::Sum:
      func(type_identity<rocm::Sum>{});
      break;
    case Reduce::Prod:
      func(type_identity<rocm::Prod>{});
      break;
    case Reduce::Max:
      func(type_identity<rocm::Max>{});
      break;
    case Reduce::Min:
      func(type_identity<rocm::Min>{});
      break;
    case Reduce::And:
      func(type_identity<rocm::And>{});
      break;
    case Reduce::Or:
      func(type_identity<rocm::Or>{});
      break;
    default:
      throw std::runtime_error("Unsupported reduce type");
  }
}

// Dispatch helper for reduce ndim
template <typename Func>
void dispatch_reduce_ndim_row(int ndim, Func&& func) {
  switch (ndim) {
    case 1:
      func(std::integral_constant<int, 1>{});
      break;
    case 2:
      func(std::integral_constant<int, 2>{});
      break;
    case 3:
      func(std::integral_constant<int, 3>{});
      break;
    case 4:
      func(std::integral_constant<int, 4>{});
      break;
    default:
      func(std::integral_constant<int, 5>{});
      break;
  }
}

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
  int threads = std::min(256, ((row_size + 3) / 4 + 32 - 1) / 32 * 32);
  threads = std::max(threads, 32);
  
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  
  // Simple row reduce for single reduction axis with contiguous data
  // Only use simple kernel for ContiguousReduce (row-contiguous input)
  if (plan.shape.size() == 1 && plan.type == ContiguousReduce) {
    dispatch_reduce_types_row(in.dtype(), [&](auto type_tag) {
      using T = hip_type_t<typename decltype(type_tag)::type>;
      dispatch_reduce_ops_row(reduce_type, [&](auto reduce_type_tag) {
        using OP = typename decltype(reduce_type_tag)::type;
        using U = typename rocm::ReduceResult<OP, T>::type;
        
        encoder.launch_kernel([&](hipStream_t stream) {
          hipLaunchKernelGGL(
              (rocm::row_reduce_simple_kernel<T, U, OP, 4>),
              dim3(out_size), dim3(threads), 0, stream,
              in.data<T>(), out.data<U>(), out_size, row_size);
        });
      });
    });
  } else {
    // Looped row reduce for multiple reduction axes
    // Build shape/strides for non-reduction axes
    auto [shape_vec, strides_vec] = shapes_without_reduction_axes(in, axes);
    
    rocm::Shape shape;
    rocm::Strides strides;
    int ndim = shape_vec.size();
    for (int i = 0; i < ndim && i < MAX_NDIM; i++) {
      shape[i] = shape_vec[i];
      strides[i] = strides_vec[i];
    }
    
    // Build reduce shape/strides (excluding last axis which is the row)
    rocm::Shape reduce_shape;
    rocm::Strides reduce_strides;
    int reduce_ndim = plan.shape.size() - 1;
    size_t non_row_reductions = 1;
    for (int i = 0; i < reduce_ndim && i < MAX_NDIM; i++) {
      reduce_shape[i] = plan.shape[i];
      reduce_strides[i] = plan.strides[i];
      non_row_reductions *= plan.shape[i];
    }
    
    dispatch_reduce_types_row(in.dtype(), [&](auto type_tag) {
      using T = hip_type_t<typename decltype(type_tag)::type>;
      dispatch_reduce_ops_row(reduce_type, [&](auto reduce_type_tag) {
        dispatch_reduce_ndim_row(reduce_ndim, [&](auto reduce_ndim_val) {
          using OP = typename decltype(reduce_type_tag)::type;
          using U = typename rocm::ReduceResult<OP, T>::type;
          
          encoder.launch_kernel([&](hipStream_t stream) {
            hipLaunchKernelGGL(
                (rocm::row_reduce_looped_kernel<T, U, OP, reduce_ndim_val()>),
                dim3(out_size), dim3(threads), 0, stream,
                in.data<T>(), out.data<U>(), out_size, row_size,
                shape, strides, ndim,
                non_row_reductions, reduce_shape, reduce_strides, reduce_ndim);
          });
        });
      });
    });
  }
}

} // namespace mlx::core
