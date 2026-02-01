// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/reduce/reduce.cuh"
#include "mlx/backend/cuda/reduce/reduce_ops.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/block/block_load.cuh>

namespace mlx::core::cu {

namespace cg = cooperative_groups;

template <typename T, typename U, typename ReduceOp, int N, typename PrefixOp>
__device__ void all_reduce_impl(
    T* in,
    U* out,
    size_t block_step,
    size_t size,
    PrefixOp prefix) {
  // TODO: Process multiple "rows" in each thread
  constexpr int M = 1;

  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  const U init = ReduceInit<ReduceOp, T>::value();
  ReduceOp op;

  T vals[N];
  U accs[M];
  accs[0] = init;

  size_t start = grid.block_rank() * block_step;
  size_t end = start + block_step;
  size_t check = min(end, size);

  size_t i = start;
  for (; i + block.size() * N <= check; i += block.size() * N) {
    cub::LoadDirectBlockedVectorized<T, N>(block.thread_rank(), in + i, vals);
    for (int j = 0; j < N; j++) {
      accs[0] = op(accs[0], cast_to<U>(prefix(vals[j])));
    }
  }

  if (i < check) {
    cub::LoadDirectBlocked(
        block.thread_rank(), in + i, vals, check - i, cast_to<T>(init));
    for (int j = 0; j < N; j++) {
      accs[0] = op(accs[0], cast_to<U>(prefix(vals[j])));
    }
  }

  __shared__ U shared_accumulators[32];
  block_reduce(block, warp, accs, shared_accumulators, op, init);

  if (block.thread_rank() == 0) {
    out[grid.block_rank()] = accs[0];
  }
}

template <typename T, typename U, typename ReduceOp, int N, typename PrefixOp>
__global__ void
all_reduce(T* in, U* out, size_t block_step, size_t size, PrefixOp prefix) {
  all_reduce_impl<T, U, ReduceOp, N, PrefixOp>(
      in, out, block_step, size, prefix);
}

template <
    typename T,
    typename U,
    typename ReduceOp,
    int N = 4,
    typename PrefixOp = Identity>
__global__ void all_reduce(T* in, U* out, size_t block_step, size_t size) {
  all_reduce_impl<T, U, ReduceOp, N, PrefixOp>(
      in, out, block_step, size, PrefixOp{});
}

} // namespace mlx::core::cu
