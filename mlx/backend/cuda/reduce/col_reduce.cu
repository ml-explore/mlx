// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/cast_op.cuh"
#include "mlx/backend/cuda/reduce/reduce.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/block/block_load.cuh>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

struct ColReduceArgs {
  // The size of the contiguous column reduction.
  size_t reduction_size;
  int64_t reduction_stride;

  // Input shape and strides excluding the reduction axes.
  Shape shape;
  Strides strides;
  int ndim;

  // Input shape and strides of the reduction axes (including last dimension).
  Shape reduce_shape;
  Strides reduce_strides;
  int reduce_ndim;

  // The number of column we are reducing. Namely prod(reduce_shape).
  size_t non_col_reductions;

  ColReduceArgs(
      const array& in,
      const ReductionPlan& plan,
      const std::vector<int>& axes) {
    assert(!plan.shape.empty());
    reduction_size = plan.shape.back();
    reduction_stride = plan.strides.back();

    int64_t stride_back = 1;
    auto [shape_vec, strides_vec] = shapes_without_reduction_axes(in, axes);
    while (!shape_vec.empty() && stride_back < reduction_stride) {
      stride_back *= shape_vec.back();
      shape_vec.pop_back();
      strides_vec.pop_back();
    }
    std::tie(shape_vec, strides_vec) =
        collapse_contiguous_dims(shape_vec, strides_vec);
    shape = const_param(shape_vec);
    strides = const_param(strides_vec);
    ndim = shape_vec.size();

    reduce_shape = const_param(plan.shape);
    reduce_strides = const_param(plan.strides);
    reduce_ndim = plan.shape.size();

    non_col_reductions = 1;
    for (int i = 0; i < reduce_ndim - 1; i++) {
      non_col_reductions *= reduce_shape[i];
    }
  }
};

template <typename T, typename U, typename Op, int NDIM, int N_READS = 4>
__global__ void col_reduce_small(
    const T* in,
    U* out,
    const __grid_constant__ ColReduceArgs args) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();

  int column =
      grid.block_index().x * block.dim_threads().x + block.thread_index().x;
  if (column * N_READS >= args.reduction_stride) {
    return;
  }

  int out_idx = grid.block_rank() / grid.dim_blocks().x;
  in += elem_to_loc(out_idx, args.shape.data(), args.strides.data(), args.ndim);

  Op op;
  U totals[N_READS];
  for (int i = 0; i < N_READS; i++) {
    totals[i] = ReduceInit<Op, T>::value();
  }

  // Read input to local.
  LoopedElemToLoc<NDIM, (NDIM > 2)> loop(args.reduce_ndim);
  loop.next(
      block.thread_index().y,
      args.reduce_shape.data(),
      args.reduce_strides.data());
  for (size_t r = block.thread_index().y;
       r < args.non_col_reductions * args.reduction_size;
       r += block.dim_threads().y) {
    U vals[N_READS];
    cub::LoadDirectBlocked(
        column,
        make_cast_iterator<U>(in + loop.location()),
        vals,
        args.reduction_stride,
        ReduceInit<Op, T>::value());
    for (int i = 0; i < N_READS; i++) {
      totals[i] = op(vals[i], totals[i]);
    }
    loop.next(
        block.dim_threads().y,
        args.reduce_shape.data(),
        args.reduce_strides.data());
  }

  // Do block reduce when each column has more than 1 element to reduce.
  if (block.dim_threads().y > 1) {
    __shared__ U shared_vals[32 * 8 * N_READS];
    size_t col =
        block.thread_index().y * block.dim_threads().x + block.thread_index().x;
    for (int i = 0; i < N_READS; i++) {
      shared_vals[col * N_READS + i] = totals[i];
    }
    block.sync();
    if (block.thread_index().y == 0) {
      for (int i = 0; i < N_READS; i++) {
        totals[i] = shared_vals[block.thread_index().x * N_READS + i];
      }
      for (int j = 1; j < block.dim_threads().y; j++) {
        col = j * block.dim_threads().x + block.thread_index().x;
        for (int i = 0; i < N_READS; i++) {
          totals[i] = op(shared_vals[col * N_READS + i], totals[i]);
        }
      }
    }
  }

  // Write result.
  if (block.thread_index().y == 0) {
    cub::StoreDirectBlocked(
        column,
        out + out_idx * args.reduction_stride,
        totals,
        args.reduction_stride);
  }
}

template <
    typename T,
    typename U,
    typename Op,
    int NDIM,
    int BM,
    int BN,
    int N_READS = 4>
__global__ void col_reduce_looped(
    const T* in,
    U* out,
    const __grid_constant__ ColReduceArgs args) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  constexpr int n_warps = BN / N_READS;

  int out_idx = grid.block_rank() / grid.dim_blocks().x;
  in += elem_to_loc(out_idx, args.shape.data(), args.strides.data(), args.ndim);

  Op op;
  U totals[N_READS];
  for (int i = 0; i < N_READS; i++) {
    totals[i] = ReduceInit<Op, T>::value();
  }

  // Read input to local.
  int r = block.thread_rank() / n_warps;
  int column = block.thread_rank() % n_warps;
  int in_offset = grid.block_index().x * BN;
  LoopedElemToLoc<NDIM, (NDIM > 2)> loop(args.reduce_ndim);
  loop.next(r, args.reduce_shape.data(), args.reduce_strides.data());
  for (; r < args.non_col_reductions * args.reduction_size; r += BM) {
    U vals[N_READS];
    cub::LoadDirectBlocked(
        column,
        make_cast_iterator<U>(in + loop.location() + in_offset),
        vals,
        args.reduction_stride - in_offset,
        ReduceInit<Op, T>::value());
    for (int i = 0; i < N_READS; i++) {
      totals[i] = op(vals[i], totals[i]);
    }
    loop.next(BM, args.reduce_shape.data(), args.reduce_strides.data());
  }

  // Do warp reduce for each output.
  constexpr int n_outputs = BN / n_warps;
  static_assert(BM == 32 && n_outputs == N_READS);
  __shared__ U shared_vals[BM * BN];
  size_t col = block.thread_index().y * BN + block.thread_index().x * N_READS;
  for (int i = 0; i < N_READS; i++) {
    shared_vals[col + i] = totals[i];
  }
  block.sync();
  col = warp.thread_rank() * BN + warp.meta_group_rank() * n_outputs;
  for (int i = 0; i < n_outputs; i++) {
    totals[i] = cg::reduce(warp, shared_vals[col + i], op);
  }

  // Write result.
  if (warp.thread_rank() == 0) {
    size_t out_offset = grid.block_index().x * BN;
    cub::StoreDirectBlocked(
        warp.meta_group_rank(),
        out + out_idx * args.reduction_stride + out_offset,
        totals,
        args.reduction_stride - out_offset);
  }
}

} // namespace cu

inline auto output_grid_for_col_reduce(
    const array& out,
    const cu::ColReduceArgs& args) {
  auto out_shape = out.shape();
  auto out_strides = out.strides();
  while (!out_shape.empty() && out_strides.back() < args.reduction_stride) {
    out_shape.pop_back();
    out_strides.pop_back();
  }
  return get_2d_grid_dims(out_shape, out_strides);
}

void col_reduce(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan) {
  cu::ColReduceArgs args(in, plan, axes);

  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_ALL_TYPES(in.dtype(), CTYPE, {
      using InType = cuda_type_t<CTYPE>;
      MLX_SWITCH_REDUCE_OPS(reduce_type, OP, {
        using OutType = cu::ReduceResult<OP, InType>::type;
        MLX_SWITCH_REDUCE_NDIM(args.reduce_ndim, NDIM, {
          constexpr int N_READS = 4;
          dim3 block_dims;
          dim3 num_blocks = output_grid_for_col_reduce(out, args);
          num_blocks.z = num_blocks.y;
          num_blocks.y = num_blocks.x;
          auto kernel =
              cu::col_reduce_small<InType, OutType, OP, NDIM, N_READS>;
          size_t total = args.non_col_reductions * args.reduction_size;
          if (total < 32) {
            size_t stride_blocks =
                cuda::ceil_div(args.reduction_stride, N_READS);
            block_dims.x = std::min(stride_blocks, 32ul);
            block_dims.y = std::min(total, 8ul);
            num_blocks.x = cuda::ceil_div(stride_blocks, block_dims.x);
          } else {
            constexpr int BM = 32;
            constexpr int BN = 32;
            block_dims.x = BM * BN / N_READS;
            num_blocks.x = cuda::ceil_div(args.reduction_stride, BN);
            kernel = cu::
                col_reduce_looped<InType, OutType, OP, NDIM, BM, BN, N_READS>;
          }
          kernel<<<num_blocks, block_dims, 0, stream>>>(
              in.data<InType>(), out.data<OutType>(), args);
        });
      });
    });
  });
}

} // namespace mlx::core
