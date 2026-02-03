// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/rocm/device.h"
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

// Warp reduce for IndexValPair
template <typename T, typename Op>
__device__ IndexValPair<T> warp_reduce_arg(IndexValPair<T> val, Op op) {
  for (int offset = 32; offset > 0; offset /= 2) {
    IndexValPair<T> other;
    other.index = __shfl_xor(val.index, offset);
    other.val = __shfl_xor(val.val, offset);
    val = op(val, other);
  }
  return val;
}

// Block reduce for IndexValPair
template <typename T, typename Op, int BLOCK_DIM>
__device__ IndexValPair<T> block_reduce_arg(IndexValPair<T> val, Op op) {
  __shared__ IndexValPair<T> shared[BLOCK_DIM / 64 + 1];
  
  int lane = threadIdx.x % 64;
  int warp_id = threadIdx.x / 64;
  
  // Warp-level reduction
  val = warp_reduce_arg(val, op);
  
  // Write reduced value to shared memory
  if (lane == 0) {
    shared[warp_id] = val;
  }
  __syncthreads();
  
  // Final reduction in first warp
  if (warp_id == 0) {
    val = (lane < (BLOCK_DIM + 63) / 64) ? shared[lane] : IndexValPair<T>{0, op.init()};
    val = warp_reduce_arg(val, op);
  }
  
  return val;
}

template <typename T, typename Op, int BLOCK_DIM, int N_READS = 4>
__global__ void arg_reduce_general(
    const T* in,
    uint32_t* out,
    size_t size,
    const int* shape,
    const int64_t* in_strides,
    const int64_t* out_strides,
    int32_t ndim,
    int64_t axis_stride,
    int32_t axis_size) {
  int64_t index = blockIdx.x + blockIdx.y * gridDim.x;
  if (index >= size) {
    return;
  }

  // Compute input and output indices
  int64_t in_idx = 0;
  int64_t out_idx = 0;
  int64_t tmp = index;
  for (int i = ndim - 1; i >= 0; --i) {
    int64_t coord = tmp % shape[i];
    in_idx += coord * in_strides[i];
    out_idx += coord * out_strides[i];
    tmp /= shape[i];
  }
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
  
  // Allocate device memory for shapes and strides
  constexpr int BLOCK_DIM = 256;
  dim3 num_blocks = get_2d_grid_dims(out.shape(), out.strides());
  
  // Copy shapes and strides to device
  array shape_arr({ndim}, int32);
  array in_strides_arr({ndim}, int64);
  array out_strides_arr({ndim}, int64);
  shape_arr.set_data(allocator::malloc(shape_arr.nbytes()));
  in_strides_arr.set_data(allocator::malloc(in_strides_arr.nbytes()));
  out_strides_arr.set_data(allocator::malloc(out_strides_arr.nbytes()));
  
  encoder.add_temporary(shape_arr);
  encoder.add_temporary(in_strides_arr);
  encoder.add_temporary(out_strides_arr);
  
  encoder.launch_kernel([&](hipStream_t stream) {
    // Copy shape and stride data
    (void)hipMemcpyAsync(shape_arr.data<int32_t>(), shape.data(), ndim * sizeof(int32_t), hipMemcpyHostToDevice, stream);
    (void)hipMemcpyAsync(in_strides_arr.data<int64_t>(), in_strides.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice, stream);
    (void)hipMemcpyAsync(out_strides_arr.data<int64_t>(), out_strides.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice, stream);
    
    switch (in.dtype()) {
      case float32:
        if (reduce_type_ == ArgReduce::ArgMax) {
          hipLaunchKernelGGL(
              (rocm::arg_reduce_general<float, rocm::ArgMax<float>, BLOCK_DIM, 4>),
              num_blocks, dim3(BLOCK_DIM), 0, stream,
              in.data<float>(), out.data<uint32_t>(), out.size(),
              shape_arr.data<int32_t>(), in_strides_arr.data<int64_t>(), out_strides_arr.data<int64_t>(),
              ndim, axis_stride, axis_size);
        } else {
          hipLaunchKernelGGL(
              (rocm::arg_reduce_general<float, rocm::ArgMin<float>, BLOCK_DIM, 4>),
              num_blocks, dim3(BLOCK_DIM), 0, stream,
              in.data<float>(), out.data<uint32_t>(), out.size(),
              shape_arr.data<int32_t>(), in_strides_arr.data<int64_t>(), out_strides_arr.data<int64_t>(),
              ndim, axis_stride, axis_size);
        }
        break;
      case int32:
        if (reduce_type_ == ArgReduce::ArgMax) {
          hipLaunchKernelGGL(
              (rocm::arg_reduce_general<int32_t, rocm::ArgMax<int32_t>, BLOCK_DIM, 4>),
              num_blocks, dim3(BLOCK_DIM), 0, stream,
              in.data<int32_t>(), out.data<uint32_t>(), out.size(),
              shape_arr.data<int32_t>(), in_strides_arr.data<int64_t>(), out_strides_arr.data<int64_t>(),
              ndim, axis_stride, axis_size);
        } else {
          hipLaunchKernelGGL(
              (rocm::arg_reduce_general<int32_t, rocm::ArgMin<int32_t>, BLOCK_DIM, 4>),
              num_blocks, dim3(BLOCK_DIM), 0, stream,
              in.data<int32_t>(), out.data<uint32_t>(), out.size(),
              shape_arr.data<int32_t>(), in_strides_arr.data<int64_t>(), out_strides_arr.data<int64_t>(),
              ndim, axis_stride, axis_size);
        }
        break;
      case float16:
        if (reduce_type_ == ArgReduce::ArgMax) {
          hipLaunchKernelGGL(
              (rocm::arg_reduce_general<__half, rocm::ArgMax<__half>, BLOCK_DIM, 4>),
              num_blocks, dim3(BLOCK_DIM), 0, stream,
              in.data<__half>(), out.data<uint32_t>(), out.size(),
              shape_arr.data<int32_t>(), in_strides_arr.data<int64_t>(), out_strides_arr.data<int64_t>(),
              ndim, axis_stride, axis_size);
        } else {
          hipLaunchKernelGGL(
              (rocm::arg_reduce_general<__half, rocm::ArgMin<__half>, BLOCK_DIM, 4>),
              num_blocks, dim3(BLOCK_DIM), 0, stream,
              in.data<__half>(), out.data<uint32_t>(), out.size(),
              shape_arr.data<int32_t>(), in_strides_arr.data<int64_t>(), out_strides_arr.data<int64_t>(),
              ndim, axis_stride, axis_size);
        }
        break;
      default:
        throw std::runtime_error("Unsupported type for ArgReduce");
    }
  });
}

} // namespace mlx::core
