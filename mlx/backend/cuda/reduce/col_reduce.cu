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

template <
    typename T,
    typename U,
    typename Op,
    int NDIM,
    int BM,
    int BN,
    int N_READS = 4>
__global__ void
col_reduce_looped(T* in, U* out, const __grid_constant__ ColReduceArgs args) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  constexpr int threads_per_row = BN / N_READS;

  // Compute the indices for the tile
  size_t tile_idx = grid.block_rank();
  size_t tile_x = tile_idx % ((args.reduction_stride + BN - 1) / BN);
  size_t tile_y = tile_idx / ((args.reduction_stride + BN - 1) / BN);

  // Compute the indices for the thread within the tile
  short thread_x = block.thread_rank() % threads_per_row;
  short thread_y = block.thread_rank() / threads_per_row;

  // Move the input pointer
  in += elem_to_loc(tile_y, args.shape.data(), args.strides.data(), args.ndim) +
      tile_x * BN;

  // Initialize the running totals
  Op op;
  U totals[N_READS];
  for (int i = 0; i < N_READS; i++) {
    totals[i] = ReduceInit<Op, T>::value();
  }

  LoopedElemToLoc<NDIM, (NDIM > 2)> loop(args.reduce_ndim);
  loop.next(thread_y, args.reduce_shape.data(), args.reduce_strides.data());
  size_t total = args.non_col_reductions * args.reduction_size;
  if (tile_x * BN + BN <= args.reduction_stride) {
    for (size_t r = thread_y; r < total; r += BM) {
      T vals[N_READS];
      cub::LoadDirectBlockedVectorized(thread_x, in + loop.location(), vals);
      for (int i = 0; i < N_READS; i++) {
        totals[i] = op(totals[i], __cast<U, T>(vals[i]));
      }
      loop.next(BM, args.reduce_shape.data(), args.reduce_strides.data());
    }
  } else {
    for (size_t r = thread_y; r < total; r += BM) {
      T vals[N_READS];
      cub::LoadDirectBlocked(
          thread_x,
          in + loop.location(),
          vals,
          args.reduction_stride - tile_x * BN,
          __cast<T, U>(ReduceInit<Op, T>::value()));
      for (int i = 0; i < N_READS; i++) {
        totals[i] = op(totals[i], __cast<U, T>(vals[i]));
      }
      loop.next(BM, args.reduce_shape.data(), args.reduce_strides.data());
    }
  }

  // Do warp reduce for each output.
  constexpr int n_outputs = BN / threads_per_row;
  static_assert(BM == 32 && n_outputs == N_READS);
  __shared__ U shared_vals[BM * BN];
  short s_idx = thread_y * BN + thread_x * N_READS;
  for (int i = 0; i < N_READS; i++) {
    shared_vals[s_idx + i] = totals[i];
  }
  block.sync();
  s_idx = warp.thread_rank() * BN + warp.meta_group_rank() * n_outputs;
  for (int i = 0; i < n_outputs; i++) {
    totals[i] = cg::reduce(warp, shared_vals[s_idx + i], op);
  }

  // Write result.
  if (warp.thread_rank() == 0) {
    cub::StoreDirectBlocked(
        warp.meta_group_rank(),
        out + tile_y * args.reduction_stride + tile_x * BN,
        totals,
        args.reduction_stride - tile_x * BN);
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

void col_reduce_looped(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan,
    cu::ColReduceArgs args) {
  // Allocate data for the output using in's layout to access them as
  // contiguously as possible.
  allocate_same_layout(out, in, axes);

  // Just a way to get out of the constness because cub doesn't like it ...
  // (sigh)
  array x = in;

  encoder.set_input_array(x);
  encoder.set_output_array(out);
  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_ALL_TYPES(x.dtype(), CTYPE, {
      MLX_SWITCH_REDUCE_OPS(reduce_type, OP, {
        MLX_SWITCH_REDUCE_NDIM(args.reduce_ndim, NDIM, {
          using T = cuda_type_t<CTYPE>;
          using U = cu::ReduceResult<OP, T>::type;

          constexpr int N_READS = 4;
          constexpr int BM = 32;
          constexpr int BN = 32;
          dim3 grid = output_grid_for_col_reduce(out, args);
          size_t extra_blocks = cuda::ceil_div(args.reduction_stride, BN);
          if (grid.x * extra_blocks < INT32_MAX) {
            grid.x *= extra_blocks;
          } else if (grid.y * extra_blocks < 65536) {
            grid.y *= extra_blocks;
          } else {
            throw std::runtime_error(
                "[col_reduce_looped] Need to factorize reduction_stride");
          }
          int blocks = BM * BN / N_READS;
          auto kernel = cu::col_reduce_looped<T, U, OP, NDIM, BM, BN, N_READS>;
          kernel<<<grid, blocks, 0, stream>>>(x.data<T>(), out.data<U>(), args);
        });
      });
    });
  });
}

void col_reduce(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan) {
  // Current col reduce options
  //
  // - col_reduce_looped
  //
  //   It is a general strided reduce. Each threadblock computes the output for
  //   a subrow of the fast moving axis. For instance 32 elements.
  //
  // Notes: As in row reduce we opt to read as much in order as possible and
  // leave
  //        transpositions as they are (contrary to our Metal backend).
  //
  //        Moreover we need different kernels for short rows and tuning

  // Make the args struct to help route to the best kernel
  cu::ColReduceArgs args(in, plan, axes);

  // Fallback col reduce
  col_reduce_looped(encoder, in, out, reduce_type, axes, plan, args);
}

} // namespace mlx::core
