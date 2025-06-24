// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/reduce/reduce.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/block/block_load.cuh>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename T, typename U, typename ReduceOp, int N = 4>
__global__ void all_reduce(T* in, U* out, size_t block_step, size_t size) {
  // TODO: Process multiple "rows" in each thread
  constexpr int M = 1;

  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  const U init = cu::ReduceInit<ReduceOp, T>::value();
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
      accs[0] = op(accs[0], __cast<U, T>(vals[j]));
    }
  }

  if (i < check) {
    cub::LoadDirectBlocked(
        block.thread_rank(), in + i, vals, check - i, __cast<T, U>(init));
    for (int i = 0; i < N; i++) {
      accs[0] = op(accs[0], __cast<U, T>(vals[i]));
    }
  }

  __shared__ U shared_accumulators[32];
  block_reduce(block, warp, accs, shared_accumulators, op, init);

  if (block.thread_rank() == 0) {
    out[grid.block_rank()] = accs[0];
  }
}

} // namespace cu

void all_reduce(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type) {
  constexpr int N_READS = 8;

  out.set_data(allocator::malloc(out.nbytes()));

  auto get_args = [](size_t size, int N) {
    int threads = std::min(512UL, (size + N - 1) / N);
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    int reductions_per_step = threads * N;
    size_t steps_needed =
        (size + reductions_per_step - 1) / reductions_per_step;

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

    size_t steps_per_block = (steps_needed + blocks - 1) / blocks;
    size_t block_step = steps_per_block * reductions_per_step;

    return std::make_tuple(blocks, threads, block_step);
  };

  int blocks, threads;
  size_t block_step;
  array x = in;

  // Large array so allocate an intermediate and accumulate there
  std::tie(blocks, threads, block_step) = get_args(x.size(), N_READS);
  if (blocks > 1) {
    array intermediate({blocks}, out.dtype(), nullptr, {});
    intermediate.set_data(allocator::malloc(intermediate.nbytes()));
    encoder.add_temporary(intermediate);
    encoder.set_input_array(x);
    encoder.set_output_array(intermediate);
    encoder.launch_kernel([&](cudaStream_t stream) {
      MLX_SWITCH_ALL_TYPES(x.dtype(), CTYPE, {
        MLX_SWITCH_REDUCE_OPS(reduce_type, OP, {
          using T = cuda_type_t<CTYPE>;
          using U = cu::ReduceResult<OP, T>::type;
          auto kernel = cu::all_reduce<T, U, OP, N_READS>;
          kernel<<<blocks, threads, 0, stream>>>(
              x.data<T>(), intermediate.data<U>(), block_step, x.size());
        });
      });
    });

    // Set the input for the next step and recalculate the blocks
    x = intermediate;
    std::tie(blocks, threads, block_step) = get_args(x.size(), N_READS);
  }

  encoder.set_input_array(x);
  encoder.set_output_array(out);
  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_ALL_TYPES(x.dtype(), CTYPE, {
      MLX_SWITCH_REDUCE_OPS(reduce_type, OP, {
        using T = cuda_type_t<CTYPE>;
        using U = cu::ReduceResult<OP, T>::type;
        auto kernel = cu::all_reduce<T, U, OP, N_READS>;
        kernel<<<blocks, threads, 0, stream>>>(
            x.data<T>(), out.data<U>(), block_step, x.size());
      });
    });
  });
}

} // namespace mlx::core
