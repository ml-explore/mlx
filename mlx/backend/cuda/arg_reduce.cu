// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/iterators/strided_iterator.cuh"
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

template <typename U>
struct IndexValPair {
  uint32_t index;
  U val;
};

template <typename U>
struct ArgMin {
  static constexpr U init = Limits<U>::max;

  __device__ IndexValPair<U> operator()(
      const IndexValPair<U>& best,
      const IndexValPair<U>& current) {
    if (best.val > current.val ||
        (best.val == current.val && best.index > current.index)) {
      return current;
    } else {
      return best;
    }
  }

  template <int N>
  __device__ IndexValPair<U>
  reduce_many(IndexValPair<U> best, U (&vals)[N], uint32_t offset) {
    for (int i = 0; i < N; i++) {
      if (vals[i] < best.val) {
        best.val = vals[i];
        best.index = offset + i;
      }
    }
    return best;
  }
};

template <typename U>
struct ArgMax {
  static constexpr U init = Limits<U>::min;

  __device__ IndexValPair<U> operator()(
      const IndexValPair<U>& best,
      const IndexValPair<U>& current) {
    if (best.val < current.val ||
        (best.val == current.val && best.index > current.index)) {
      return current;
    } else {
      return best;
    }
  }

  template <int N>
  __device__ IndexValPair<U>
  reduce_many(IndexValPair<U> best, U (&vals)[N], uint32_t offset) {
    for (int i = 0; i < N; i++) {
      if (vals[i] > best.val) {
        best.val = vals[i];
        best.index = offset + i;
      }
    }
    return best;
  }
};

template <typename U>
inline __device__ IndexValPair<U> warp_shuffle_down(
    const cg::thread_block_tile<WARP_SIZE>& g,
    const IndexValPair<U>& data,
    int delta) {
  return {g.shfl_down(data.index, delta), g.shfl_down(data.val, delta)};
}

template <typename T, typename Op, int BLOCK_DIM, int N_READS = 4>
__global__ void arg_reduce_general(
    const T* in,
    uint32_t* out,
    const __grid_constant__ Shape shape,
    const __grid_constant__ Strides in_strides,
    const __grid_constant__ Strides out_strides,
    size_t ndim,
    int64_t axis_stride,
    size_t axis_size) {
  // Shapes and strides *do not* contain the reduction axis. The reduction size
  // and stride are provided in axis_stride and axis_size.
  //
  // Note: in shape == out shape with this convention.
  Op op;

  // Compute the input/output index. There is one beginning and one output for
  // the whole block.
  auto elem = cg::this_grid().block_rank();
  auto in_idx = elem_to_loc(elem, shape.data(), in_strides.data(), ndim);
  auto out_idx = elem_to_loc(elem, shape.data(), out_strides.data(), ndim);

  IndexValPair<T> best{0, Op::init};

  auto block = cg::this_thread_block();
  for (size_t r = 0; r < cuda::ceil_div(axis_size, BLOCK_DIM * N_READS); r++) {
    T vals[N_READS];
    auto index = r * BLOCK_DIM + block.thread_index().z;
    cub::LoadDirectBlocked(
        index,
        strided_iterator(in + in_idx, axis_stride),
        vals,
        axis_size,
        Op::init);
    best = op.reduce_many(best, vals, index * N_READS);
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
  auto in_strides = in.strides();
  auto shape = in.shape();
  auto out_strides = out.strides();
  auto axis_stride = in_strides[axis_];
  size_t axis_size = shape[axis_];
  if (out_strides.size() == in_strides.size()) {
    out_strides.erase(out_strides.begin() + axis_);
  }
  in_strides.erase(in_strides.begin() + axis_);
  shape.erase(shape.begin() + axis_);
  size_t ndim = shape.size();

  // ArgReduce.
  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_REAL_TYPES_CHECKED(in.dtype(), "ArgReduce", CTYPE, {
      using InType = cuda_type_t<CTYPE>;
      constexpr uint32_t N_READS = 4;
      MLX_SWITCH_BLOCK_DIM(cuda::ceil_div(axis_size, N_READS), BLOCK_DIM, {
        dim3 num_blocks = get_2d_grid_dims(out.shape(), out.strides());
        dim3 block_dims{1, 1, BLOCK_DIM};
        auto kernel = &cu::arg_reduce_general<
            InType,
            cu::ArgMax<InType>,
            BLOCK_DIM,
            N_READS>;
        if (reduce_type_ == ArgReduce::ArgMin) {
          kernel = &cu::arg_reduce_general<
              InType,
              cu::ArgMin<InType>,
              BLOCK_DIM,
              N_READS>;
        }
        kernel<<<num_blocks, block_dims, 0, stream>>>(
            in.data<InType>(),
            out.data<uint32_t>(),
            const_param(shape),
            const_param(in_strides),
            const_param(out_strides),
            ndim,
            axis_stride,
            axis_size);
      });
    });
  });
}

} // namespace mlx::core
