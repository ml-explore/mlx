// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/reduce/row_reduce.cuh"

namespace mlx::core {

void row_reduce_simple(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan) {
  // Allocate data for the output using in's layout to avoid elem_to_loc in the
  // kernel.
  allocate_same_layout(out, in, axes, encoder);

  // TODO: If out.size() < 1024 which will be a common case then write this in
  //       2 passes. Something like 32 * out.size() and then do a warp reduce.
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  dispatch_all_types(in.dtype(), [&](auto type_tag) {
    dispatch_reduce_ops(reduce_type, [&](auto reduce_type_tag) {
      using OP = MLX_GET_TYPE(reduce_type_tag);
      using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
      using U = typename cu::ReduceResult<OP, T>::type;

      constexpr int N_READS = 16 / sizeof(T);

      // Calculate the grid and block dims
      size_t reductions = (plan.shape.back() + N_READS - 1) / N_READS;
      dim3 grid = get_2d_grid_dims(out.shape(), out.strides());
      int warps = (reductions + WARP_SIZE - 1) / WARP_SIZE;
      warps /= 4;
      warps = std::max(std::min(warps, 32), 1);
      int threads = warps * WARP_SIZE;
      dim3 block(threads, 1, 1);

      // Pick the kernel
      auto kernel = cu::row_reduce_simple<T, U, OP, N_READS>;
      if (grid.x >= 1024) {
        grid.x = (grid.x + 1) / 2;
        kernel = cu::row_reduce_simple<T, U, OP, N_READS, 2>;
      }

      T* indata = const_cast<T*>(gpu_ptr<T>(in));
      int size = plan.shape.back();
      encoder.add_kernel_node(
          kernel, grid, block, 0, indata, gpu_ptr<U>(out), out.size(), size);
    });
  });
}

void row_reduce_looped(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan,
    cu::RowReduceArgs args) {
  // Allocate data for the output using in's layout to access them as
  // contiguously as possible.
  allocate_same_layout(out, in, axes, encoder);

  encoder.set_input_array(in);
  encoder.set_output_array(out);
  dispatch_all_types(in.dtype(), [&](auto type_tag) {
    dispatch_reduce_ops(reduce_type, [&](auto reduce_type_tag) {
      using OP = MLX_GET_TYPE(reduce_type_tag);
      using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
      using U = typename cu::ReduceResult<OP, T>::type;

      constexpr int N_READS = 16 / sizeof(T);

      // Calculate the grid and block dims
      args.sort_access_pattern(in, axes);
      dim3 grid = get_2d_grid_dims(out.shape(), out.strides());
      size_t reductions = (args.row_size + N_READS - 1) / N_READS;
      int warps = (reductions + WARP_SIZE - 1) / WARP_SIZE;
      warps /= 4;
      warps = std::max(std::min(warps, 32), 1);
      int threads = warps * WARP_SIZE;
      dim3 block(threads, 1, 1);

      // Pick the kernel
      auto kernel = cu::row_reduce_looped<T, U, OP, 1, N_READS>;
      dispatch_reduce_ndim(args.reduce_ndim, [&](auto reduce_ndim) {
        kernel = cu::row_reduce_looped<T, U, OP, reduce_ndim.value, N_READS>;
      });

      encoder.add_kernel_node(
          kernel, grid, block, 0, gpu_ptr<T>(in), gpu_ptr<U>(out), args);
    });
  });
}

void row_reduce(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan) {
  // Current row reduction options
  //
  // - row_reduce_simple
  //
  //   That means that we are simply reducing across the fastest moving axis.
  //   We are reducing 1 or 2 rows per threadblock depending on the size of
  //   output.
  //
  // - row_reduce_looped
  //
  //   It is a general row reduction. We are computing 1 output per
  //   threadblock. We read the fastest moving axis vectorized and loop over
  //   the rest of the axes.
  //
  // Notes: We opt to read as much in order as possible and leave
  //        transpositions as they are (contrary to our Metal backend).

  // Simple row reduce means that we have 1 axis that we are reducing over and
  // it has stride 1.
  if (plan.shape.size() == 1) {
    row_reduce_simple(encoder, in, out, reduce_type, axes, plan);
    return;
  }

  // Make the args struct to help route to the best kernel
  cu::RowReduceArgs args(in, plan, axes);

  // Fallback row reduce
  row_reduce_looped(encoder, in, out, reduce_type, axes, plan, std::move(args));
}

} // namespace mlx::core
