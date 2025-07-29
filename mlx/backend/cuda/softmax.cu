// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/cast_op.cuh"
#include "mlx/backend/cuda/device/fp16_math.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <nvtx3/nvtx3.hpp>

#include <cassert>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename T>
inline __device__ T softmax_exp(T x) {
  // Softmax doesn't need high precision exponential cause x is gonna be in
  // (-oo, 0] anyway and subsequently it will be divided by sum(exp(x_i)).
  return __expf(x);
}

template <typename T, typename AccT, int BLOCK_DIM, int N_READS = 4>
__global__ void softmax(const T* in, T* out, int axis_size) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  in += grid.block_rank() * axis_size;
  out += grid.block_rank() * axis_size;

  cg::greater<AccT> max_op;
  cg::plus<AccT> plus_op;

  // Thread reduce.
  AccT prevmax;
  AccT maxval = Limits<AccT>::finite_min();
  AccT normalizer = cast_to<AccT>(0);
  for (int r = 0; r < cuda::ceil_div(axis_size, BLOCK_DIM * N_READS); r++) {
    auto index = r * BLOCK_DIM + block.thread_rank();
    auto vals = load_vector<N_READS>(in, index, axis_size, Limits<T>::min());
    prevmax = maxval;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      maxval = max_op(maxval, static_cast<AccT>(vals[i]));
    }

    // Online normalizer calculation for softmax:
    // https://github.com/NVIDIA/online-softmax
    normalizer = normalizer * softmax_exp(prevmax - maxval);
#pragma unroll
    for (int i = 0; i < N_READS; i++) {
      normalizer =
          normalizer + softmax_exp(static_cast<AccT>(vals[i]) - maxval);
    }
  }

  // First warp reduce.
  prevmax = maxval;
  maxval = cg::reduce(warp, maxval, max_op);
  normalizer = normalizer * softmax_exp(prevmax - maxval);
  normalizer = cg::reduce(warp, normalizer, plus_op);

  __shared__ AccT local_max[WARP_SIZE];
  __shared__ AccT local_normalizer[WARP_SIZE];

  // Write to shared memory and do second warp reduce.
  prevmax = maxval;
  if (warp.thread_rank() == 0) {
    local_max[warp.meta_group_rank()] = maxval;
  }
  block.sync();
  maxval = warp.thread_rank() < warp.meta_group_size()
      ? local_max[warp.thread_rank()]
      : Limits<AccT>::min();
  maxval = cg::reduce(warp, maxval, max_op);
  normalizer = normalizer * softmax_exp(prevmax - maxval);
  if (warp.thread_rank() == 0) {
    local_normalizer[warp.meta_group_rank()] = normalizer;
  }
  block.sync();
  normalizer = warp.thread_rank() < warp.meta_group_size()
      ? local_normalizer[warp.thread_rank()]
      : AccT{};
  normalizer = cg::reduce(warp, normalizer, plus_op);
  normalizer = 1 / normalizer;

  // Write output.
  for (int r = 0; r < cuda::ceil_div(axis_size, BLOCK_DIM * N_READS); r++) {
    auto index = r * BLOCK_DIM + block.thread_rank();
    auto vals = load_vector<N_READS>(in, index, axis_size, T(0));
    for (int i = 0; i < N_READS; i++) {
      vals[i] = softmax_exp(static_cast<AccT>(vals[i]) - maxval) * normalizer;
    }
    store_vector<N_READS>(out, index, vals, axis_size);
  }
}

} // namespace cu

void Softmax::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Softmax::eval_gpu");
  assert(inputs.size() == 1);
  auto& s = stream();

  // Make sure that the last dimension is contiguous.
  auto set_output = [&s, &out](const array& x) {
    if (x.flags().contiguous && x.strides()[x.ndim() - 1] == 1) {
      if (x.is_donatable()) {
        out.copy_shared_buffer(x);
      } else {
        out.set_data(
            allocator::malloc(x.data_size() * x.itemsize()),
            x.data_size(),
            x.strides(),
            x.flags());
      }
      return x;
    } else {
      array x_copy = contiguous_copy_gpu(x, s);
      out.copy_shared_buffer(x_copy);
      return x_copy;
    }
  };

  array in = set_output(inputs[0]);
  bool precise = in.dtype() != float32 && precise_;

  int axis_size = in.shape().back();
  int n_rows = in.data_size() / axis_size;

  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  dispatch_float_types(out.dtype(), "softmax", [&](auto type_tag) {
    using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    constexpr int N_READS = 16 / sizeof(DataType);
    dispatch_block_dim(cuda::ceil_div(axis_size, N_READS), [&](auto block_dim) {
      auto kernel = cu::softmax<DataType, DataType, block_dim(), N_READS>;
      if (precise) {
        kernel = cu::softmax<DataType, float, block_dim(), N_READS>;
      }
      encoder.add_kernel_node(
          kernel,
          n_rows,
          block_dim(),
          in.data<DataType>(),
          out.data<DataType>(),
          axis_size);
    });
  });
}

} // namespace mlx::core
