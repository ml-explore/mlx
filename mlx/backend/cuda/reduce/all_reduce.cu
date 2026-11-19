// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/reduce/reduce.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/block/block_load.cuh>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename T, typename U, typename A, typename ReduceOp, int N = 4>
__global__ void all_reduce(T* in, U* out, size_t block_step, size_t size) {
  // TODO: Process multiple "rows" in each thread
  constexpr int M = 1;

  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  const A init = cast_to<A>(cu::ReduceInit<ReduceOp, T>::value());
  ReduceOp op;

  T vals[N];
  A accs[M];
  accs[0] = init;

  size_t start = grid.block_rank() * block_step;
  size_t end = start + block_step;
  size_t check = min(end, size);

  size_t i = start;
  for (; i + block.size() * N <= check; i += block.size() * N) {
    cub::LoadDirectBlockedVectorized<T, N>(block.thread_rank(), in + i, vals);
    for (int j = 0; j < N; j++) {
      accs[0] = op(accs[0], cast_to<A>(vals[j]));
    }
  }

  if (i < check) {
    cub::LoadDirectBlocked(
        block.thread_rank(), in + i, vals, check - i, cast_to<T>(init));
    for (int i = 0; i < N; i++) {
      accs[0] = op(accs[0], cast_to<A>(vals[i]));
    }
  }

  __shared__ A shared_accumulators[32];
  block_reduce(block, warp, accs, shared_accumulators, op, init);

  if (block.thread_rank() == 0) {
    out[grid.block_rank()] = cast_to<U>(accs[0]);
  }
}

} // namespace cu

void all_reduce(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type) {
  constexpr int N_READS = 8;

  out.set_data(cu::malloc_async(out.nbytes(), encoder));

  auto get_args = [](size_t size, size_t N) {
    int threads =
        static_cast<int>(std::min(size_t{512}, cuda::ceil_div(size, N)));
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    size_t reductions_per_step = threads * N;
    size_t steps_needed = cuda::ceil_div(size, reductions_per_step);

    int blocks;
    if (steps_needed < 32) {
      blocks = 1;
    } else if (steps_needed < 128) {
      blocks = 32;
    } else if (steps_needed < 512) {
      blocks = 128;
    } else if (steps_needed < 1024) {
      blocks = 512;
    } else {
      blocks = 1024;
    }

    size_t steps_per_block = cuda::ceil_div(steps_needed, blocks);
    size_t block_step = steps_per_block * reductions_per_step;

    return std::make_tuple(blocks, threads, block_step);
  };

  int blocks, threads;
  size_t block_step;
  size_t insize = in.size();

  std::tie(blocks, threads, block_step) = get_args(insize, N_READS);
  encoder.set_input_array(in);
  dispatch_all_types(in.dtype(), [&](auto type_tag) {
    dispatch_reduce_ops(reduce_type, [&](auto reduce_type_tag) {
      using OP = MLX_GET_TYPE(reduce_type_tag);
      using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
      using U = typename cu::ReduceResult<OP, T>::type;
      using A = typename cu::ReduceAccum<OP, T>::type;

      if (blocks == 1) {
        encoder.set_output_array(out);
        encoder.add_kernel_node(
            cu::all_reduce<T, U, A, OP, N_READS>,
            blocks,
            threads,
            gpu_ptr<T>(in),
            gpu_ptr<U>(out),
            block_step,
            insize);
        return;
      }

      // Large array so allocate an intermediate in the accumulator type and
      // accumulate there
      Dtype acc_dtype =
          std::is_same_v<A, U> ? out.dtype() : static_cast<Dtype>(float32);
      array intermediate({blocks}, acc_dtype, nullptr, {});
      intermediate.set_data(cu::malloc_async(intermediate.nbytes(), encoder));
      encoder.add_temporary(intermediate);
      encoder.set_output_array(intermediate);
      encoder.add_kernel_node(
          cu::all_reduce<T, A, A, OP, N_READS>,
          blocks,
          threads,
          gpu_ptr<T>(in),
          gpu_ptr<A>(intermediate),
          block_step,
          insize);

      // Reduce the intermediate into the output
      insize = intermediate.size();
      std::tie(blocks, threads, block_step) = get_args(insize, N_READS);
      encoder.set_input_array(intermediate);
      encoder.set_output_array(out);
      encoder.add_kernel_node(
          cu::all_reduce<A, U, A, OP, N_READS>,
          blocks,
          threads,
          gpu_ptr<A>(intermediate),
          gpu_ptr<U>(out),
          block_step,
          insize);
    });
  });
}

} // namespace mlx::core
