// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/reduce/reduce.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/block/block_load.cuh>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

namespace {

// TODO: Should make a custom complex type
template <typename U, typename T>
inline __device__ U __cast(T x) {
  return static_cast<U>(x);
}

template <>
inline __device__ bool __cast<bool, cuComplex>(cuComplex x) {
  return x.x != 0 && x.y != 0;
}

template <>
inline __device__ cuComplex __cast<cuComplex, bool>(bool x) {
  return x ? make_cuFloatComplex(1, 1) : make_cuFloatComplex(0, 0);
}

} // namespace

template <typename T, typename U, typename ReduceOp, int N = 4>
__global__ void all_reduce(T* in, U* out, size_t block_step, size_t size) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  const U init = cu::ReduceInit<ReduceOp, T>::value();
  ReduceOp op;

  T vals[N];
  U accs[N];
  for (int i = 0; i < N; i++) {
    accs[i] = init;
  }

  size_t start = grid.block_rank() * block_step;
  size_t end = start + block_step;
  size_t check = min(end, size);

  for (size_t i = start; i + block.size() * N <= check; i += block.size() * N) {
    cub::LoadDirectBlockedVectorized<T, N>(block.thread_rank(), in + i, vals);
    for (int i = 0; i < N; i++) {
      accs[i] = op(accs[i], __cast<U, T>(vals[i]));
    }
  }

  if (end > size) {
    size_t offset = end - block.size() * N;
    int block_end = size - offset;
    cub::LoadDirectBlocked(
        block.thread_rank(), in + offset, vals, block_end, __cast<T, U>(init));
    for (int i = 0; i < N; i++) {
      accs[i] = op(accs[i], __cast<U, T>(vals[i]));
    }
  }

  for (int i = 1; i < N; i++) {
    accs[0] = op(accs[0], accs[i]);
  }
  accs[0] = cg::reduce(warp, accs[0], op);

  __shared__ U shared_accumulators[32];
  if (warp.thread_rank() == 0) {
    shared_accumulators[warp.meta_group_rank()] = accs[0];
  }
  block.sync();
  accs[0] = (warp.thread_rank() < warp.meta_group_size())
      ? shared_accumulators[warp.thread_rank()]
      : init;
  accs[0] = cg::reduce(warp, accs[0], op);

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
  constexpr int N_READS = 4;

  out.set_data(allocator::malloc(out.nbytes()));

  auto get_args = [](size_t size, int N) {
    size_t reductions = size / N;
    int threads = 1024;
    int blocks = std::min(1024UL, (reductions + threads - 1) / threads);
    size_t reductions_per_block = std::max(
        static_cast<size_t>(threads), (reductions + blocks - 1) / blocks);
    size_t block_step = reductions_per_block * N_READS;

    return std::make_tuple(blocks, threads, block_step);
  };

  int blocks, threads;
  size_t block_step;
  bool large = in.size() > N_READS * 1024;
  array x = in;

  // Large array so allocate an intermediate and accumulate there
  if (large) {
    std::tie(blocks, threads, block_step) = get_args(x.size(), N_READS);
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
    x = intermediate;
  }

  // Final reduction
  {
    std::tie(blocks, threads, block_step) = get_args(x.size(), N_READS);
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
}

} // namespace mlx::core
