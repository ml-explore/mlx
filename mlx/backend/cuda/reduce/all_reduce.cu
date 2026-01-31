// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/reduce/all_reduce.cuh"

namespace mlx::core {

void all_reduce(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type) {
  constexpr int N_READS = 8;

  out.set_data(cu::malloc_async(out.nbytes(), encoder));

  auto get_args = [](int size, int N) {
    int threads = std::min(512, (size + N - 1) / N);
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
  size_t insize = in.size();
  Dtype dt = in.dtype();

  // Cub doesn't like const pointers for load (sigh).
  void* indata = const_cast<void*>(gpu_ptr<void>(in));

  // Large array so allocate an intermediate and accumulate there
  std::tie(blocks, threads, block_step) = get_args(insize, N_READS);
  encoder.set_input_array(in);
  if (blocks > 1) {
    array intermediate({blocks}, out.dtype(), nullptr, {});
    intermediate.set_data(cu::malloc_async(intermediate.nbytes(), encoder));
    encoder.add_temporary(intermediate);
    encoder.set_output_array(intermediate);
    dispatch_all_types(dt, [&](auto type_tag) {
      dispatch_reduce_ops(reduce_type, [&](auto reduce_type_tag) {
        using OP = MLX_GET_TYPE(reduce_type_tag);
        using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
        using U = typename cu::ReduceResult<OP, T>::type;
        auto kernel = cu::all_reduce<T, U, OP, N_READS>;
        encoder.add_kernel_node(
            kernel,
            blocks,
            threads,
            0,
            static_cast<T*>(indata),
            gpu_ptr<U>(intermediate),
            block_step,
            insize);
      });
    });

    // Set the input for the next step and recalculate the blocks
    indata = gpu_ptr<void>(intermediate);
    dt = intermediate.dtype();
    insize = intermediate.size();
    std::tie(blocks, threads, block_step) = get_args(insize, N_READS);
    encoder.set_input_array(intermediate);
  }

  encoder.set_output_array(out);
  dispatch_all_types(dt, [&](auto type_tag) {
    dispatch_reduce_ops(reduce_type, [&](auto reduce_type_tag) {
      using OP = MLX_GET_TYPE(reduce_type_tag);
      using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
      using U = typename cu::ReduceResult<OP, T>::type;
      auto kernel = cu::all_reduce<T, U, OP, N_READS>;
      encoder.add_kernel_node(
          kernel,
          blocks,
          threads,
          0,
          static_cast<T*>(indata),
          gpu_ptr<U>(out),
          block_step,
          insize);
    });
  });
}

} // namespace mlx::core
