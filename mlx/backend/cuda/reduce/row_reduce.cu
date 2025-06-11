// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernels/cast_op.cuh"
#include "mlx/backend/cuda/reduce/reduce.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

struct RowReduceArgs {
  // The size of the row being reduced, i.e. the size of last dimension.
  int row_size;

  // Input shape and strides excluding the reduction axes.
  Shape shape;
  Strides strides;
  int ndim;

  // Input shape and strides of the reduction axes excluding last dimension.
  Shape reduce_shape;
  Strides reduce_strides;
  int reduce_ndim;

  // The number of rows we are reducing. Namely prod(reduce_shape).
  size_t non_row_reductions;

  RowReduceArgs(
      const array& in,
      const ReductionPlan& plan,
      const std::vector<int>& axes) {
    assert(!plan.shape.empty());
    row_size = plan.shape.back();

    auto [shape_vec, strides_vec] = shapes_without_reduction_axes(in, axes);
    std::tie(shape_vec, strides_vec) =
        collapse_contiguous_dims(shape_vec, strides_vec);
    shape = const_param(shape_vec);
    strides = const_param(strides_vec);
    ndim = shape_vec.size();

    reduce_shape = const_param(plan.shape);
    reduce_strides = const_param(plan.strides);
    reduce_ndim = plan.shape.size() - 1;

    non_row_reductions = 1;
    for (int i = 0; i < reduce_ndim; i++) {
      non_row_reductions *= reduce_shape[i];
    }
  }
};

template <typename T, typename U, typename Op, int NDIM, int N_READS = 4>
__global__ void row_reduce_small(
    const T* in,
    U* out,
    size_t out_size,
    const __grid_constant__ RowReduceArgs args) {
  size_t out_idx = cg::this_grid().thread_rank();
  if (out_idx >= out_size) {
    return;
  }

  Op op;

  U total_val = ReduceInit<Op, T>::value();
  LoopedElemToLoc<NDIM, (NDIM > 2)> loop(args.reduce_ndim);

  in += elem_to_loc(out_idx, args.shape.data(), args.strides.data(), args.ndim);

  for (size_t n = 0; n < args.non_row_reductions; n++) {
    for (int r = 0; r < cuda::ceil_div(args.row_size, N_READS); r++) {
      U vals[N_READS];
      cub::LoadDirectBlocked(
          r,
          make_cast_iterator<U>(in + loop.location()),
          vals,
          args.row_size,
          ReduceInit<Op, T>::value());
      total_val = op(total_val, cub::ThreadReduce(vals, op));
    }
    loop.next(args.reduce_shape.data(), args.reduce_strides.data());
  }

  out[out_idx] = total_val;
}

template <typename T, typename U, typename Op, int NDIM, int N_READS = 4>
__global__ void row_reduce_small_warp(
    const T* in,
    U* out,
    size_t out_size,
    const __grid_constant__ RowReduceArgs args) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  size_t out_idx = grid.thread_rank() / WARP_SIZE;
  if (out_idx >= out_size) {
    return;
  }

  Op op;

  U total_val = ReduceInit<Op, T>::value();
  LoopedElemToLoc<NDIM, (NDIM > 2)> loop(args.reduce_ndim);

  in += elem_to_loc(out_idx, args.shape.data(), args.strides.data(), args.ndim);

  for (size_t n = warp.thread_rank(); n < args.non_row_reductions;
       n += WARP_SIZE) {
    for (int r = 0; r < cuda::ceil_div(args.row_size, N_READS); r++) {
      U vals[N_READS];
      cub::LoadDirectBlocked(
          r,
          make_cast_iterator<U>(in + loop.location()),
          vals,
          args.row_size,
          ReduceInit<Op, T>::value());
      total_val = op(total_val, cub::ThreadReduce(vals, op));
    }
    loop.next(WARP_SIZE, args.reduce_shape.data(), args.reduce_strides.data());
  }

  total_val = cg::reduce(warp, total_val, op);

  if (warp.thread_rank() == 0) {
    out[out_idx] = total_val;
  }
}

template <
    typename T,
    typename U,
    typename Op,
    int NDIM,
    int BLOCK_DIM_X,
    int N_READS = 4>
__global__ void row_reduce_looped(
    const T* in,
    U* out,
    size_t out_size,
    const __grid_constant__ RowReduceArgs args) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();

  size_t out_idx = grid.thread_rank() / BLOCK_DIM_X;
  if (out_idx >= out_size) {
    return;
  }

  Op op;

  U total_val = ReduceInit<Op, T>::value();
  LoopedElemToLoc<NDIM, (NDIM > 2)> loop(args.reduce_ndim);

  in += elem_to_loc(out_idx, args.shape.data(), args.strides.data(), args.ndim);

  for (size_t n = 0; n < args.non_row_reductions; n++) {
    for (size_t r = 0; r < cuda::ceil_div(args.row_size, BLOCK_DIM_X * N_READS);
         r++) {
      U vals[N_READS];
      cub::LoadDirectBlocked(
          r * BLOCK_DIM_X + block.thread_index().x,
          make_cast_iterator<U>(in + loop.location()),
          vals,
          args.row_size,
          ReduceInit<Op, T>::value());
      total_val = op(total_val, cub::ThreadReduce(vals, op));
    }
    loop.next(args.reduce_shape.data(), args.reduce_strides.data());
  }

  typedef cub::BlockReduce<U, BLOCK_DIM_X> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp;

  total_val = BlockReduceT(temp).Reduce(total_val, op);

  if (block.thread_rank() == 0) {
    out[out_idx] = total_val;
  }
}

} // namespace cu

void row_reduce(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan) {
  cu::RowReduceArgs args(in, plan, axes);

  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_ALL_TYPES(in.dtype(), CTYPE, {
      using InType = cuda_type_t<CTYPE>;
      MLX_SWITCH_REDUCE_OPS(reduce_type, OP, {
        using OutType = cu::ReduceResult<OP, InType>::type;
        MLX_SWITCH_REDUCE_NDIM(args.reduce_ndim, NDIM, {
          constexpr size_t N_READS = 4;
          dim3 out_dims = get_2d_grid_dims(out.shape(), out.strides());
          dim3 block_dims, num_blocks;
          auto kernel =
              cu::row_reduce_small<InType, OutType, OP, NDIM, N_READS>;
          if (args.row_size <= 64) {
            if ((args.non_row_reductions < 32 && args.row_size <= 8) ||
                (args.non_row_reductions <= 8)) {
              block_dims.x = std::min(out_dims.x, 1024u);
              num_blocks.x = cuda::ceil_div(out_dims.x, block_dims.x);
              num_blocks.y = out_dims.y;
            } else {
              block_dims.x = WARP_SIZE;
              num_blocks.y = out_dims.x;
              num_blocks.z = out_dims.y;
              kernel =
                  cu::row_reduce_small_warp<InType, OutType, OP, NDIM, N_READS>;
            }
          } else {
            size_t num_threads = cuda::ceil_div(args.row_size, N_READS);
            num_threads = cuda::ceil_div(num_threads, WARP_SIZE) * WARP_SIZE;
            MLX_SWITCH_BLOCK_DIM(num_threads, BLOCK_DIM_X, {
              num_blocks.y = out_dims.x;
              num_blocks.z = out_dims.y;
              block_dims.x = BLOCK_DIM_X;
              kernel = cu::row_reduce_looped<
                  InType,
                  OutType,
                  OP,
                  NDIM,
                  BLOCK_DIM_X,
                  N_READS>;
            });
          }
          kernel<<<num_blocks, block_dims, 0, stream>>>(
              in.data<InType>(), out.data<OutType>(), out.size(), args);
        });
      });
    });
  });
}

} // namespace mlx::core
