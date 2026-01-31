// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/reduce/reduce.cuh"
#include "mlx/backend/cuda/reduce/reduce_ops.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/block/block_load.cuh>
#include <cub/cub.cuh>

#include <numeric>

namespace mlx::core::cu {

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
    using ShapeVector = decltype(plan.shape);
    using StridesVector = decltype(plan.strides);

    ShapeVector shape_vec;
    StridesVector strides_vec;

    assert(!plan.shape.empty());
    reduction_size = plan.shape.back();
    reduction_stride = plan.strides.back();

    int64_t stride_back = 1;
    std::tie(shape_vec, strides_vec) = shapes_without_reduction_axes(in, axes);
    while (!shape_vec.empty() && stride_back < reduction_stride) {
      stride_back *= shape_vec.back();
      shape_vec.pop_back();
      strides_vec.pop_back();
    }
    std::vector<int> indices(shape_vec.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int left, int right) {
      return strides_vec[left] > strides_vec[right];
    });
    ShapeVector sorted_shape;
    StridesVector sorted_strides;
    for (auto idx : indices) {
      sorted_shape.push_back(shape_vec[idx]);
      sorted_strides.push_back(strides_vec[idx]);
    }
    std::tie(shape_vec, strides_vec) =
        collapse_contiguous_dims(sorted_shape, sorted_strides);
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
    int N_READS = 4,
    int BLOCKS = 1,
    typename PrefixOp = Identity>
__global__ void col_reduce_looped(
    T* in,
    U* out,
    const __grid_constant__ ColReduceArgs args,
    int64_t out_size) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  constexpr int threads_per_row = BN / N_READS;

  // Compute the indices for the tile
  size_t tile_idx = grid.block_rank();
  size_t tile_x = tile_idx % ((args.reduction_stride + BN - 1) / BN);
  size_t tile_y = tile_idx / ((args.reduction_stride + BN - 1) / BN);
  size_t tile_out = tile_y / out_size;
  tile_y = tile_y % out_size;

  // Compute the indices for the thread within the tile
  short thread_x = block.thread_rank() % threads_per_row;
  short thread_y = block.thread_rank() / threads_per_row;

  // Move the input pointer
  in += elem_to_loc(tile_y, args.shape.data(), args.strides.data(), args.ndim) +
      tile_x * BN;

  // Initialize the running totals
  Op op;
  PrefixOp prefix;
  U totals[N_READS];
  for (int i = 0; i < N_READS; i++) {
    totals[i] = ReduceInit<Op, T>::value();
  }

  size_t total = args.non_col_reductions * args.reduction_size;
  size_t per_block, start, end;
  if constexpr (BLOCKS > 1) {
    per_block = (total + BLOCKS - 1) / BLOCKS;
    start = tile_out * per_block + thread_y;
    end = min((tile_out + 1) * per_block, total);
  } else {
    per_block = total;
    start = thread_y;
    end = total;
  }

  LoopedElemToLoc<NDIM, (NDIM > 2)> loop(args.reduce_ndim);
  loop.next(start, args.reduce_shape.data(), args.reduce_strides.data());
  if (tile_x * BN + BN <= args.reduction_stride) {
    if (args.reduction_stride % N_READS == 0) {
      for (size_t r = start; r < end; r += BM) {
        T vals[N_READS];
        cub::LoadDirectBlockedVectorized(thread_x, in + loop.location(), vals);
        for (int i = 0; i < N_READS; i++) {
          totals[i] = op(totals[i], cast_to<U>(prefix(vals[i])));
        }
        loop.next(BM, args.reduce_shape.data(), args.reduce_strides.data());
      }
    } else {
      for (size_t r = start; r < end; r += BM) {
        T vals[N_READS];
        cub::LoadDirectBlocked(thread_x, in + loop.location(), vals);
        for (int i = 0; i < N_READS; i++) {
          totals[i] = op(totals[i], cast_to<U>(prefix(vals[i])));
        }
        loop.next(BM, args.reduce_shape.data(), args.reduce_strides.data());
      }
    }
  } else {
    for (size_t r = start; r < end; r += BM) {
      T vals[N_READS];
      cub::LoadDirectBlocked(
          thread_x,
          in + loop.location(),
          vals,
          args.reduction_stride - tile_x * BN,
          cast_to<T>(ReduceInit<Op, T>::value()));
      for (int i = 0; i < N_READS; i++) {
        totals[i] = op(totals[i], cast_to<U>(prefix(vals[i])));
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
    if (BLOCKS > 1) {
      out += tile_out * out_size * args.reduction_stride;
    }
    cub::StoreDirectBlocked(
        warp.meta_group_rank(),
        out + tile_y * args.reduction_stride + tile_x * BN,
        totals,
        args.reduction_stride - tile_x * BN);
  }
}

template <
    typename T,
    typename U,
    typename Op,
    int N_READS = 4,
    typename PrefixOp = Identity>
__global__ void col_reduce_small(
    const T* in,
    U* out,
    const __grid_constant__ ColReduceArgs args,
    size_t total) {
  Op op;
  PrefixOp prefix;
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();

  const auto idx = grid.thread_rank() * N_READS;
  const auto before_axis = idx / args.reduction_stride;
  const auto after_axis = idx % args.reduction_stride;
  const auto offset =
      before_axis * args.reduction_stride * args.reduction_size + after_axis;

  if (idx >= total) {
    return;
  }

  in += offset;
  out += idx;

  AlignedVector<U, N_READS> accumulator;
  for (int i = 0; i < N_READS; i++) {
    accumulator[i] = ReduceInit<Op, T>::value();
  }

  for (int i = 0; i < args.reduction_size; i++) {
    auto values = load_vector<N_READS>(in, 0);

    for (int j = 0; j < N_READS; j++) {
      accumulator[j] = op(accumulator[j], cast_to<U>(prefix(values[j])));
    }

    in += args.reduction_stride;
  }

  store_vector(out, 0, accumulator);
}

} // namespace mlx::core::cu
