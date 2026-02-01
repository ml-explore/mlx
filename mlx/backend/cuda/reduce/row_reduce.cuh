// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/reduce/reduce.cuh"
#include "mlx/backend/cuda/reduce/reduce_ops.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <numeric>

namespace mlx::core::cu {

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

  // Convert shape and strides as if in was contiguous
  void sort_access_pattern(const array& in, const std::vector<int>& axes) {
    auto shape_vec = in.shape();
    auto strides_vec = in.strides();
    std::tie(shape_vec, strides_vec) =
        shapes_without_reduction_axes(shape_vec, strides_vec, axes);
    std::vector<int> indices(shape_vec.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int left, int right) {
      return strides_vec[left] > strides_vec[right];
    });
    decltype(shape_vec) sorted_shape;
    decltype(strides_vec) sorted_strides;
    for (auto idx : indices) {
      sorted_shape.push_back(shape_vec[idx]);
      sorted_strides.push_back(strides_vec[idx]);
    }
    std::tie(shape_vec, strides_vec) =
        collapse_contiguous_dims(sorted_shape, sorted_strides);
    shape = const_param(shape_vec);
    strides = const_param(strides_vec);
    ndim = shape_vec.size();
  }
};

template <
    typename T,
    typename U,
    typename ReduceOp,
    int N,
    int M,
    typename PrefixOp>
__device__ void row_reduce_simple_impl(
    const T* in,
    U* out,
    size_t n_rows,
    int size,
    PrefixOp prefix) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  const U init = ReduceInit<ReduceOp, T>::value();
  ReduceOp op;

  AlignedVector<T, N> vals[M];
  AlignedVector<U, M> accs;
  for (int i = 0; i < M; i++) {
    accs[i] = init;
  }

  const size_t start_row =
      min(n_rows - M, static_cast<size_t>(grid.block_rank() * M));
  const size_t full_blocks = size / (block.size() * N);
  const size_t final_offset = full_blocks * (block.size() * N);
  in += start_row * size + block.thread_rank() * N;
  out += start_row;

  for (size_t r = 0; r < full_blocks; r++) {
    for (int k = 0; k < M; k++) {
      vals[k] = load_vector<N>(in + k * size, 0);
    }
    for (int k = 0; k < M; k++) {
      for (int j = 0; j < N; j++) {
        accs[k] = op(accs[k], cast_to<U>(prefix(vals[k][j])));
      }
    }

    in += block.size() * N;
  }

  if (final_offset < size) {
    for (int k = 0; k < M; k++) {
      for (int i = 0; i < N; i++) {
        vals[k][i] = ((final_offset + block.thread_rank() * N + i) < size)
            ? in[k * size + i]
            : cast_to<T>(init);
      }
    }
    for (int k = 0; k < M; k++) {
      for (int j = 0; j < N; j++) {
        accs[k] = op(accs[k], cast_to<U>(prefix(vals[k][j])));
      }
    }
  }

  __shared__ U shared_accumulators[32 * M];
  block_reduce(block, warp, accs.val, shared_accumulators, op, init);

  if (block.thread_rank() == 0) {
    if (grid.block_rank() * M + M <= n_rows) {
      store_vector(out, 0, accs);
    } else {
      short offset = grid.block_rank() * M + M - n_rows;
      for (int i = offset; i < M; i++) {
        out[i] = accs[i];
      }
    }
  }
}

// Kernel with prefix parameter
template <
    typename T,
    typename U,
    typename ReduceOp,
    int N,
    int M,
    typename PrefixOp>
__global__ void row_reduce_simple(
    const T* in,
    U* out,
    size_t n_rows,
    int size,
    PrefixOp prefix) {
  row_reduce_simple_impl<T, U, ReduceOp, N, M, PrefixOp>(
      in, out, n_rows, size, prefix);
}

// Kernel without prefix parameter (default Identity)
template <
    typename T,
    typename U,
    typename ReduceOp,
    int N = 4,
    int M = 1,
    typename PrefixOp = Identity>
__global__ void
row_reduce_simple(const T* in, U* out, size_t n_rows, int size) {
  row_reduce_simple_impl<T, U, ReduceOp, N, M, PrefixOp>(
      in, out, n_rows, size, PrefixOp{});
}

// Device function for row_reduce_looped
template <
    typename T,
    typename U,
    typename Op,
    int NDIM,
    int N_READS,
    typename PrefixOp>
__device__ void row_reduce_looped_impl(
    const T* in,
    U* out,
    const RowReduceArgs& args,
    PrefixOp prefix) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  size_t out_idx = grid.block_rank();

  Op op;

  U total[1];
  U init = ReduceInit<Op, T>::value();
  total[0] = init;
  LoopedElemToLoc<NDIM, (NDIM > 2)> loop(args.reduce_ndim);
  const size_t full_blocks = args.row_size / (block.size() * N_READS);
  const size_t final_offset = full_blocks * (block.size() * N_READS);

  in += elem_to_loc(out_idx, args.shape.data(), args.strides.data(), args.ndim);
  in += block.thread_rank() * N_READS;

  // Unaligned reduce
  if (final_offset < args.row_size) {
    bool mask[N_READS];
    for (int i = 0; i < N_READS; i++) {
      mask[i] =
          (final_offset + block.thread_rank() * N_READS + i) < args.row_size;
    }

    for (size_t n = 0; n < args.non_row_reductions; n++) {
      const T* inlocal = in + loop.location();

      for (size_t r = 0; r < full_blocks; r++) {
        auto vals = load_vector<N_READS>(inlocal, 0);
        for (int i = 0; i < N_READS; i++) {
          total[0] = op(total[0], cast_to<U>(prefix(vals[i])));
        }
        inlocal += block.size() * N_READS;
      }

      {
        T vals[N_READS];
        for (int i = 0; i < N_READS; i++) {
          vals[i] = mask[i] ? inlocal[i] : cast_to<T>(init);
        }
        for (int i = 0; i < N_READS; i++) {
          total[0] = op(total[0], cast_to<U>(prefix(vals[i])));
        }
      }

      loop.next(args.reduce_shape.data(), args.reduce_strides.data());
    }
  }

  // Aligned case
  else {
    for (size_t n = 0; n < args.non_row_reductions; n++) {
      const T* inlocal = in + loop.location();

      for (size_t r = 0; r < full_blocks; r++) {
        auto vals = load_vector<N_READS>(inlocal, 0);
        for (int i = 0; i < N_READS; i++) {
          total[0] = op(total[0], cast_to<U>(prefix(vals[i])));
        }
        inlocal += block.size() * N_READS;
      }

      loop.next(args.reduce_shape.data(), args.reduce_strides.data());
    }
  }

  __shared__ U shared_accumulators[32];
  block_reduce(block, warp, total, shared_accumulators, op, init);

  if (block.thread_rank() == 0) {
    out[out_idx] = total[0];
  }
}

// Kernel with prefix parameter
template <
    typename T,
    typename U,
    typename Op,
    int NDIM,
    int N_READS,
    typename PrefixOp>
__global__ void row_reduce_looped(
    const T* in,
    U* out,
    const __grid_constant__ RowReduceArgs args,
    PrefixOp prefix) {
  row_reduce_looped_impl<T, U, Op, NDIM, N_READS, PrefixOp>(
      in, out, args, prefix);
}

// Kernel without prefix parameter (default Identity)
template <
    typename T,
    typename U,
    typename Op,
    int NDIM,
    int N_READS = 4,
    typename PrefixOp = Identity>
__global__ void row_reduce_looped(
    const T* in,
    U* out,
    const __grid_constant__ RowReduceArgs args) {
  row_reduce_looped_impl<T, U, Op, NDIM, N_READS, PrefixOp>(
      in, out, args, PrefixOp{});
}

} // namespace mlx::core::cu
