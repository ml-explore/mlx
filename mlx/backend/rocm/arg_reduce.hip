// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/config.h"
#include "mlx/backend/rocm/device/fp16_math.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>

#include <cassert>

namespace mlx::core {

namespace rocm {

template <typename T>
struct IndexValPair {
  uint32_t index;
  T val;
};

// Type-safe shuffle wrappers for __shfl_xor
template <typename T>
__device__ __forceinline__ T shfl_xor_arg(T val, int lane_mask) {
  return __shfl_xor(val, lane_mask);
}

// Specialization for __half - __shfl_xor returns float
template <>
__device__ __forceinline__ __half shfl_xor_arg(__half val, int lane_mask) {
  return __half(__shfl_xor(__half2float(val), lane_mask));
}

// Specialization for hip_bfloat16
template <>
__device__ __forceinline__ hip_bfloat16 shfl_xor_arg(hip_bfloat16 val, int lane_mask) {
  return hip_bfloat16(__shfl_xor(static_cast<float>(val), lane_mask));
}

template <typename T>
struct ArgMin {
  __device__ T init() const {
    return numeric_limits<T>::max();
  }

  __device__ IndexValPair<T> operator()(
      const IndexValPair<T>& best,
      const IndexValPair<T>& current) const {
    if (best.val > current.val ||
        (best.val == current.val && best.index > current.index)) {
      return current;
    } else {
      return best;
    }
  }
};

template <typename T>
struct ArgMax {
  __device__ T init() const {
    return numeric_limits<T>::lowest();
  }

  __device__ IndexValPair<T> operator()(
      const IndexValPair<T>& best,
      const IndexValPair<T>& current) const {
    if (best.val < current.val ||
        (best.val == current.val && best.index > current.index)) {
      return current;
    } else {
      return best;
    }
  }
};

// Warp reduce for IndexValPair - uses runtime warp size
template <typename T, typename Op>
__device__ IndexValPair<T> warp_reduce_arg(IndexValPair<T> val, Op op) {
  // Use warpSize which is a built-in variable in HIP
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    IndexValPair<T> other;
    other.index = __shfl_xor(val.index, offset);
    other.val = shfl_xor_arg(val.val, offset);
    val = op(val, other);
  }
  return val;
}

// Block reduce for IndexValPair
template <typename T, typename Op, int BLOCK_DIM>
__device__ IndexValPair<T> block_reduce_arg(IndexValPair<T> val, Op op) {
  // Use warpSize built-in for correct behavior on both RDNA (32) and CDNA (64)
  constexpr int MAX_WARPS = BLOCK_DIM / 32 + 1;  // Conservative estimate
  __shared__ IndexValPair<T> shared[MAX_WARPS];
  
  int lane = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  int num_warps = (BLOCK_DIM + warpSize - 1) / warpSize;
  
  // Warp-level reduction
  val = warp_reduce_arg(val, op);
  
  // Write reduced value to shared memory
  if (lane == 0) {
    shared[warp_id] = val;
  }
  __syncthreads();
  
  // Final reduction in first warp
  if (warp_id == 0) {
    val = (lane < num_warps) ? shared[lane] : IndexValPair<T>{0, op.init()};
    val = warp_reduce_arg(val, op);
  }
  
  return val;
}

template <typename T, typename Op, int BLOCK_DIM, int N_READS = 4>
__global__ void arg_reduce_general(
    const T* in,
    uint32_t* out,
    size_t size,
    const Shape shape,
    const Strides in_strides,
    const Strides out_strides,
    int32_t ndim,
    int64_t axis_stride,
    int32_t axis_size) {
  int64_t index = blockIdx.x + blockIdx.y * gridDim.x;
  if (index >= size) {
    return;
  }

  // Compute input and output indices using elem_to_loc
  int64_t in_idx = elem_to_loc(index, shape.data_, in_strides.data_, ndim);
  int64_t out_idx = elem_to_loc(index, shape.data_, out_strides.data_, ndim);
  in += in_idx;

  Op op;
  T init_val = op.init();
  IndexValPair<T> best{0, init_val};

  // Each thread processes multiple elements
  for (int i = threadIdx.x; i < axis_size; i += BLOCK_DIM) {
    T val = in[i * axis_stride];
    IndexValPair<T> current{static_cast<uint32_t>(i), val};
    best = op(best, current);
  }

  // Block reduction
  best = block_reduce_arg<T, Op, BLOCK_DIM>(best, op);

  if (threadIdx.x == 0) {
    out[out_idx] = best.index;
  }
}

} // namespace rocm

void ArgReduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  out.set_data(allocator::malloc(out.nbytes()));
  auto& s = stream();

  // Handle scalar case - just output 0
  if (in.ndim() == 0 || in.size() == 1) {
    auto& encoder = rocm::get_command_encoder(s);
    encoder.set_output_array(out);
    encoder.launch_kernel([&](hipStream_t stream) {
      uint32_t zero = 0;
      (void)hipMemcpyAsync(out.data<uint32_t>(), &zero, sizeof(uint32_t), hipMemcpyHostToDevice, stream);
    });
    return;
  }

  // Prepare the shapes, strides and axis arguments.
  Shape shape = remove_index(in.shape(), axis_);
  Strides in_strides = remove_index(in.strides(), axis_);
  Strides out_strides = out.ndim() == in.ndim()
      ? remove_index(out.strides(), axis_)
      : out.strides();
  int64_t axis_stride = in.strides()[axis_];
  int32_t axis_size = in.shape()[axis_];
  int32_t ndim = shape.size();

  auto& encoder = rocm::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  
  constexpr int BLOCK_DIM = 256;
  dim3 num_blocks = get_2d_grid_dims(out.shape(), out.strides());
  
  // Use const_param to pass shape and strides by value (like CUDA)
  auto shape_param = const_param(shape);
  auto in_strides_param = const_param(in_strides);
  auto out_strides_param = const_param(out_strides);
  
  encoder.launch_kernel([&](hipStream_t stream) {
    switch (in.dtype()) {
      case float32:
        if (reduce_type_ == ArgReduce::ArgMax) {
          hipLaunchKernelGGL(
              (rocm::arg_reduce_general<float, rocm::ArgMax<float>, BLOCK_DIM, 4>),
              num_blocks, dim3(BLOCK_DIM), 0, stream,
              in.data<float>(), out.data<uint32_t>(), out.size(),
              shape_param, in_strides_param, out_strides_param,
              ndim, axis_stride, axis_size);
        } else {
          hipLaunchKernelGGL(
              (rocm::arg_reduce_general<float, rocm::ArgMin<float>, BLOCK_DIM, 4>),
              num_blocks, dim3(BLOCK_DIM), 0, stream,
              in.data<float>(), out.data<uint32_t>(), out.size(),
              shape_param, in_strides_param, out_strides_param,
              ndim, axis_stride, axis_size);
        }
        break;
      case int32:
        if (reduce_type_ == ArgReduce::ArgMax) {
          hipLaunchKernelGGL(
              (rocm::arg_reduce_general<int32_t, rocm::ArgMax<int32_t>, BLOCK_DIM, 4>),
              num_blocks, dim3(BLOCK_DIM), 0, stream,
              in.data<int32_t>(), out.data<uint32_t>(), out.size(),
              shape_param, in_strides_param, out_strides_param,
              ndim, axis_stride, axis_size);
        } else {
          hipLaunchKernelGGL(
              (rocm::arg_reduce_general<int32_t, rocm::ArgMin<int32_t>, BLOCK_DIM, 4>),
              num_blocks, dim3(BLOCK_DIM), 0, stream,
              in.data<int32_t>(), out.data<uint32_t>(), out.size(),
              shape_param, in_strides_param, out_strides_param,
              ndim, axis_stride, axis_size);
        }
        break;
      case float16:
        if (reduce_type_ == ArgReduce::ArgMax) {
          hipLaunchKernelGGL(
              (rocm::arg_reduce_general<__half, rocm::ArgMax<__half>, BLOCK_DIM, 4>),
              num_blocks, dim3(BLOCK_DIM), 0, stream,
              in.data<__half>(), out.data<uint32_t>(), out.size(),
              shape_param, in_strides_param, out_strides_param,
              ndim, axis_stride, axis_size);
        } else {
          hipLaunchKernelGGL(
              (rocm::arg_reduce_general<__half, rocm::ArgMin<__half>, BLOCK_DIM, 4>),
              num_blocks, dim3(BLOCK_DIM), 0, stream,
              in.data<__half>(), out.data<uint32_t>(), out.size(),
              shape_param, in_strides_param, out_strides_param,
              ndim, axis_stride, axis_size);
        }
        break;
      default:
        throw std::runtime_error("Unsupported type for ArgReduce");
    }
  });
}

} // namespace mlx::core
