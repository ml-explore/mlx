// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/cast_op.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <nvtx3/nvtx3.hpp>
#include <cub/block/block_load.cuh>

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
__global__ void logsumexp(const T* in, T* out, int axis_size) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  in += grid.block_rank() * axis_size;

  cg::greater<AccT> max_op;
  cg::plus<AccT> plus_op;

  // Thread reduce.
  AccT prevmax;
  AccT maxval = Limits<AccT>::finite_min();
  AccT normalizer = 0;
  for (int r = 0; r < cuda::ceil_div(axis_size, BLOCK_DIM * N_READS); r++) {
    AccT vals[N_READS];
    cub::LoadDirectBlocked(
        r * BLOCK_DIM + block.thread_rank(),
        make_cast_iterator<AccT>(in),
        vals,
        axis_size,
        Limits<AccT>::min());
    prevmax = maxval;
    maxval = max_op(maxval, cub::ThreadReduce(vals, max_op));
    // Online normalizer calculation for softmax:
    // https://github.com/NVIDIA/online-softmax
    normalizer = normalizer * softmax_exp(prevmax - maxval);
    for (int i = 0; i < N_READS; i++) {
      normalizer = normalizer + softmax_exp(vals[i] - maxval);
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
      : Limits<AccT>::finite_min();
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

  // Write output.
  if (block.thread_rank() == 0) {
    out[grid.block_rank()] = isinf(maxval) ? maxval : log(normalizer) + maxval;
  }
}

} // namespace cu

void LogSumExp::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("LogSumExp::eval_gpu");
  assert(inputs.size() == 1);
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  // Make sure that the last dimension is contiguous.
  auto ensure_contiguous = [&s, &encoder](const array& x) {
    if (x.flags().contiguous && x.strides()[x.ndim() - 1] == 1) {
      return x;
    } else {
      auto x_copy = array(x.shape(), x.dtype(), nullptr, {});
      copy_gpu(x, x_copy, CopyType::General, s);
      encoder.add_temporary(x_copy);
      return x_copy;
    }
  };

  auto in = ensure_contiguous(inputs[0]);
  if (in.flags().row_contiguous) {
    out.set_data(allocator::malloc(out.nbytes()));
  } else {
    auto n = in.shape(-1);
    auto flags = in.flags();
    auto strides = in.strides();
    for (auto& s : strides) {
      s /= n;
    }
    bool col_contig = strides[0] == 1;
    for (int i = 1; col_contig && i < strides.size(); ++i) {
      col_contig &=
          (out.shape(i) == 1 || strides[i - 1] == out.shape(i) * strides[i]);
    }
    flags.col_contiguous = col_contig;
    out.set_data(
        allocator::malloc(in.nbytes() / n),
        in.data_size() / n,
        std::move(strides),
        flags);
  }

  int axis_size = in.shape().back();
  int n_rows = in.data_size() / axis_size;

  encoder.set_input_array(in);
  encoder.set_output_array(out);
  dispatch_float_types(out.dtype(), "logsumexp", [&](auto type_tag) {
    constexpr int N_READS = 4;
    dispatch_block_dim(cuda::ceil_div(axis_size, N_READS), [&](auto block_dim) {
      using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;
      auto kernel = cu::logsumexp<DataType, float, block_dim(), N_READS>;
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
