// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/fp16_math.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <cooperative_groups.h>
#include <nvtx3/nvtx3.hpp>
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>

#include <cassert>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename T>
struct IndexValPair {
  uint32_t index;
  T val;
};

template <typename T>
struct ArgMin {
  constexpr __device__ T init() {
    return Limits<T>::max();
  }

  __device__ IndexValPair<T> operator()(
      const IndexValPair<T>& best,
      const IndexValPair<T>& current) {
    if (best.val > current.val ||
        (best.val == current.val && best.index > current.index)) {
      return current;
    } else {
      return best;
    }
  }

  template <int N>
  __device__ IndexValPair<T>
  reduce_many(IndexValPair<T> best, T (&vals)[N], uint32_t offset) {
    for (int i = 0; i < N; i++) {
      if (vals[i] < best.val) {
        best.val = vals[i];
        best.index = offset + i;
      }
    }
    return best;
  }
};

template <typename T>
struct ArgMax {
  constexpr __device__ T init() {
    return Limits<T>::min();
  }

  __device__ IndexValPair<T> operator()(
      const IndexValPair<T>& best,
      const IndexValPair<T>& current) {
    if (best.val < current.val ||
        (best.val == current.val && best.index > current.index)) {
      return current;
    } else {
      return best;
    }
  }

  template <int N>
  __device__ IndexValPair<T>
  reduce_many(IndexValPair<T> best, T (&vals)[N], uint32_t offset) {
    for (int i = 0; i < N; i++) {
      if (vals[i] > best.val) {
        best.val = vals[i];
        best.index = offset + i;
      }
    }
    return best;
  }
};

template <typename T, typename Op, int BLOCK_DIM, int N_READS = 4>
__global__ void arg_reduce_general(
    const T* in,
    uint32_t* out,
    size_t size,
    const __grid_constant__ Shape shape,
    const __grid_constant__ Strides in_strides,
    const __grid_constant__ Strides out_strides,
    int32_t ndim,
    int64_t axis_stride,
    int32_t axis_size) {
  auto block = cg::this_thread_block();

  int64_t index = cg::this_grid().block_rank();
  if (index >= size) {
    return;
  }

  int64_t in_idx = elem_to_loc(index, shape.data(), in_strides.data(), ndim);
  int64_t out_idx = elem_to_loc(index, shape.data(), out_strides.data(), ndim);

  Op op;
  T init = op.init();
  IndexValPair<T> best{0, init};

  for (int r = 0; r < cuda::ceil_div(axis_size, BLOCK_DIM * N_READS); ++r) {
    T vals[N_READS];
    auto tid = r * BLOCK_DIM + block.thread_index().x;
    cub::LoadDirectBlocked(
        tid, StridedIterator(in + in_idx, axis_stride), vals, axis_size, init);
    best = op.reduce_many(best, vals, tid * N_READS);
  }

  typedef cub::BlockReduce<IndexValPair<T>, BLOCK_DIM> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp;

  best = BlockReduceT(temp).Reduce(best, op);

  if (block.thread_rank() == 0) {
    out[out_idx] = best.index;
  }
}

} // namespace cu

void ArgReduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("ArgReduce::eval_gpu");
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

  // ArgReduce.
  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  dispatch_real_types(in.dtype(), "ArgReduce", [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    constexpr uint32_t N_READS = 4;
    dispatch_block_dim(cuda::ceil_div(axis_size, N_READS), [&](auto block_dim) {
      dim3 num_blocks = get_2d_grid_dims(out.shape(), out.strides());
      auto kernel =
          cu::arg_reduce_general<T, cu::ArgMax<T>, block_dim(), N_READS>;
      if (reduce_type_ == ArgReduce::ArgMin) {
        kernel = cu::arg_reduce_general<T, cu::ArgMin<T>, block_dim(), N_READS>;
      }
      encoder.add_kernel_node(
          kernel,
          num_blocks,
          block_dim(),
          in.data<T>(),
          out.data<uint32_t>(),
          out.size(),
          const_param(shape),
          const_param(in_strides),
          const_param(out_strides),
          ndim,
          axis_stride,
          axis_size);
    });
  });
}

} // namespace mlx::core
