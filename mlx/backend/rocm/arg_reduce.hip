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

template <typename T, typename Op, int BLOCK_DIM>
__global__ void arg_reduce_partial(
    const T* in,
    T* part_val,
    uint32_t* part_idx,
    size_t n_rows,
    const Shape shape,
    const Strides in_strides,
    int32_t ndim,
    int64_t axis_stride,
    int32_t axis_size,
    int blocks_per_row) {
  int row = blockIdx.y;
  int blk = blockIdx.x;
  if (row >= n_rows) {
    return;
  }
  int64_t in_idx = elem_to_loc(row, shape.data_, in_strides.data_, ndim);
  const T* rin = in + in_idx;

  Op op;
  IndexValPair<T> best{0, op.init()};
  for (int i = blk * BLOCK_DIM + threadIdx.x; i < axis_size;
       i += blocks_per_row * BLOCK_DIM) {
    IndexValPair<T> current{static_cast<uint32_t>(i), rin[i * axis_stride]};
    best = op(best, current);
  }
  best = block_reduce_arg<T, Op, BLOCK_DIM>(best, op);
  if (threadIdx.x == 0) {
    part_val[row * blocks_per_row + blk] = best.val;
    part_idx[row * blocks_per_row + blk] = best.index;
  }
}

template <typename T, typename Op, int BLOCK_DIM>
__global__ void arg_reduce_final(
    const T* part_val,
    const uint32_t* part_idx,
    uint32_t* out,
    const Shape shape,
    const Strides out_strides,
    int32_t ndim,
    size_t n_rows,
    int blocks_per_row) {
  int row = blockIdx.x;
  if (row >= n_rows) {
    return;
  }
  int64_t out_idx = elem_to_loc(row, shape.data_, out_strides.data_, ndim);

  Op op;
  IndexValPair<T> best{0, op.init()};
  for (int i = threadIdx.x; i < blocks_per_row; i += BLOCK_DIM) {
    IndexValPair<T> current{part_idx[row * blocks_per_row + i],
                            part_val[row * blocks_per_row + i]};
    best = op(best, current);
  }
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
      (void)hipMemsetAsync(gpu_ptr<uint32_t>(out), 0, sizeof(uint32_t), stream);
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
  
  auto shape_param = const_param(shape);
  auto in_strides_param = const_param(in_strides);
  auto out_strides_param = const_param(out_strides);
  size_t out_size = out.size();

  const bool is_max = (reduce_type_ == ArgReduce::ArgMax);
  auto dispatch_op = [&](auto&& fn) {
    switch (in.dtype()) {
      case float32:
        is_max ? fn(float{}, rocm::ArgMax<float>{})
               : fn(float{}, rocm::ArgMin<float>{});
        break;
      case int32:
        is_max ? fn(int32_t{}, rocm::ArgMax<int32_t>{})
               : fn(int32_t{}, rocm::ArgMin<int32_t>{});
        break;
      case float16:
        is_max ? fn(__half{}, rocm::ArgMax<__half>{})
               : fn(__half{}, rocm::ArgMin<__half>{});
        break;
      case bfloat16:
        is_max ? fn(hip_bfloat16{}, rocm::ArgMax<hip_bfloat16>{})
               : fn(hip_bfloat16{}, rocm::ArgMin<hip_bfloat16>{});
        break;
      default:
        throw std::runtime_error("Unsupported type for ArgReduce");
    }
  };

  constexpr int PART_BLOCK = 256;
  int blocks_per_row = (axis_size + PART_BLOCK - 1) / PART_BLOCK;
  if (blocks_per_row > 64) blocks_per_row = 64;

  if (blocks_per_row <= 1) {
    constexpr int BLOCK_DIM = 1024;
    dim3 num_blocks = get_2d_grid_dims(out.shape(), out.strides());
    dispatch_op([&](auto tt, auto opp) {
      using T = decltype(tt);
      using Op = decltype(opp);
      encoder.add_kernel_node(
          &rocm::arg_reduce_general<T, Op, BLOCK_DIM, 4>,
          num_blocks, dim3(BLOCK_DIM), 0,
          gpu_ptr<T>(in), gpu_ptr<uint32_t>(out), out_size,
          shape_param, in_strides_param, out_strides_param,
          ndim, axis_stride, axis_size);
    });
    return;
  }

  dispatch_op([&](auto tt, auto opp) {
    using T = decltype(tt);
    using Op = decltype(opp);
    array part_val({static_cast<int>(out_size * blocks_per_row)}, in.dtype(), nullptr, {});
    part_val.set_data(rocm::malloc_async(part_val.nbytes(), encoder));
    encoder.add_temporary(part_val);
    array part_idx({static_cast<int>(out_size * blocks_per_row)}, uint32, nullptr, {});
    part_idx.set_data(rocm::malloc_async(part_idx.nbytes(), encoder));
    encoder.add_temporary(part_idx);

    encoder.set_output_array(part_val);
    encoder.set_output_array(part_idx);
    encoder.add_kernel_node(
        &rocm::arg_reduce_partial<T, Op, PART_BLOCK>,
        dim3(blocks_per_row, out_size), dim3(PART_BLOCK), 0,
        gpu_ptr<T>(in), gpu_ptr<T>(part_val), gpu_ptr<uint32_t>(part_idx),
        out_size, shape_param, in_strides_param, ndim, axis_stride, axis_size,
        blocks_per_row);

    encoder.set_input_array(part_val);
    encoder.set_input_array(part_idx);
    encoder.set_output_array(out);
    encoder.add_kernel_node(
        &rocm::arg_reduce_final<T, Op, PART_BLOCK>,
        dim3(out_size), dim3(PART_BLOCK), 0,
        gpu_ptr<T>(part_val), gpu_ptr<uint32_t>(part_idx), gpu_ptr<uint32_t>(out),
        shape_param, out_strides_param, ndim, out_size, blocks_per_row);
  });
}

} // namespace mlx::core
