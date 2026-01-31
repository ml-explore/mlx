// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/reduce/col_reduce.cuh"

namespace mlx::core {

inline auto output_grid_for_col_reduce(
    const array& out,
    const cu::ColReduceArgs& args,
    int bn,
    int outer = 1) {
  int gx, gy = 1;
  size_t n_inner_blocks = cuda::ceil_div(args.reduction_stride, bn);
  size_t n_outer_blocks = out.size() / args.reduction_stride;
  size_t n_blocks = n_outer_blocks * n_inner_blocks * outer;
  while (n_blocks / gy > INT32_MAX) {
    gy *= 2;
  }
  gx = cuda::ceil_div(n_blocks, gy);

  return dim3(gx, gy, 1);
}

void col_reduce_looped(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan,
    const cu::ColReduceArgs& args) {
  // Allocate data for the output using in's layout to access them as
  // contiguously as possible.
  allocate_same_layout(out, in, axes, encoder);

  encoder.set_input_array(in);
  encoder.set_output_array(out);
  dispatch_all_types(in.dtype(), [&](auto type_tag) {
    dispatch_reduce_ops(reduce_type, [&](auto reduce_type_tag) {
      dispatch_reduce_ndim(args.reduce_ndim, [&](auto reduce_ndim) {
        using OP = MLX_GET_TYPE(reduce_type_tag);
        using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
        using U = typename cu::ReduceResult<OP, T>::type;
        // Cub doesn't like const pointers for vectorized loads. (sigh)
        T* indata = const_cast<T*>(gpu_ptr<T>(in));

        constexpr int N_READS = 4;
        constexpr int BM = 32;
        constexpr int BN = 32;
        dim3 grid = output_grid_for_col_reduce(out, args, BN);
        int blocks = BM * BN / N_READS;
        auto kernel =
            cu::col_reduce_looped<T, U, OP, reduce_ndim(), BM, BN, N_READS>;
        encoder.add_kernel_node(
            kernel,
            grid,
            blocks,
            0,
            indata,
            gpu_ptr<U>(out),
            static_cast<cu::ColReduceArgs>(args),
            out.size() / args.reduction_stride);
      });
    });
  });
}

void col_reduce_small(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan,
    const cu::ColReduceArgs& args) {
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
      auto tmp_grid = get_2d_grid_dims(out.shape(), out.strides());
      auto [grid, block] = get_grid_and_block(tmp_grid.x, tmp_grid.y, 1);
      auto kernel = cu::col_reduce_small<T, U, OP, N_READS>;
      encoder.add_kernel_node(
          kernel,
          grid,
          block,
          0,
          gpu_ptr<T>(in),
          gpu_ptr<U>(out),
          static_cast<cu::ColReduceArgs>(args),
          out.size());
    });
  });
}

void col_reduce_two_pass(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan,
    const cu::ColReduceArgs& args) {
  // Allocate data for the output using in's layout to access them as
  // contiguously as possible.
  allocate_same_layout(out, in, axes, encoder);

  // Allocate an intermediate array to hold the 1st pass result
  constexpr int outer = 32;

  Shape intermediate_shape;
  intermediate_shape.push_back(outer);
  intermediate_shape.insert(
      intermediate_shape.end(), out.shape().begin(), out.shape().end());

  Strides intermediate_strides;
  intermediate_strides.push_back(out.size());
  intermediate_strides.insert(
      intermediate_strides.end(), out.strides().begin(), out.strides().end());

  array intermediate(intermediate_shape, out.dtype(), nullptr, {});
  auto [data_size, rc, cc] =
      check_contiguity(intermediate_shape, intermediate_strides);
  auto fl = out.flags();
  fl.row_contiguous = rc;
  fl.col_contiguous = cc;
  fl.contiguous = true;
  intermediate.set_data(
      cu::malloc_async(intermediate.nbytes(), encoder),
      data_size,
      intermediate_strides,
      fl,
      allocator::free);

  encoder.add_temporary(intermediate);
  encoder.set_input_array(in);
  encoder.set_output_array(intermediate);
  dispatch_all_types(in.dtype(), [&](auto type_tag) {
    dispatch_reduce_ops(reduce_type, [&](auto reduce_type_tag) {
      dispatch_reduce_ndim(args.reduce_ndim, [&](auto reduce_ndim) {
        using OP = MLX_GET_TYPE(reduce_type_tag);
        using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
        using U = typename cu::ReduceResult<OP, T>::type;
        // Cub doesn't like const pointers for vectorized loads. (sigh)
        T* indata = const_cast<T*>(gpu_ptr<T>(in));

        constexpr int N_READS = 4;
        constexpr int BM = 32;
        constexpr int BN = 32;
        dim3 grid = output_grid_for_col_reduce(out, args, BN, outer);
        int blocks = BM * BN / N_READS;
        auto kernel = cu::
            col_reduce_looped<T, U, OP, reduce_ndim(), BM, BN, N_READS, outer>;
        encoder.add_kernel_node(
            kernel,
            grid,
            blocks,
            0,
            indata,
            gpu_ptr<U>(intermediate),
            static_cast<cu::ColReduceArgs>(args),
            out.size() / args.reduction_stride);
      });
    });
  });

  // Prepare the reduction arguments for the 2nd pass
  cu::ColReduceArgs second_args = args;
  second_args.reduction_size = outer;
  second_args.reduction_stride = out.size();
  second_args.ndim = 0;
  second_args.reduce_shape[0] = outer;
  second_args.reduce_strides[0] = out.size();
  second_args.reduce_ndim = 1;
  second_args.non_col_reductions = 1;

  encoder.set_input_array(intermediate);
  encoder.set_output_array(out);
  dispatch_all_types(intermediate.dtype(), [&](auto type_tag) {
    dispatch_reduce_ops(reduce_type, [&](auto reduce_type_tag) {
      dispatch_reduce_ndim(second_args.reduce_ndim, [&](auto reduce_ndim) {
        using OP = MLX_GET_TYPE(reduce_type_tag);
        using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
        using U = typename cu::ReduceResult<OP, T>::type;

        constexpr int N_READS = 4;
        constexpr int BM = 32;
        constexpr int BN = 32;
        dim3 grid = output_grid_for_col_reduce(out, second_args, BN);
        int blocks = BM * BN / N_READS;
        auto kernel =
            cu::col_reduce_looped<T, U, OP, reduce_ndim(), BM, BN, N_READS>;
        encoder.add_kernel_node(
            kernel,
            grid,
            blocks,
            0,
            gpu_ptr<T>(intermediate),
            gpu_ptr<U>(out),
            second_args,
            second_args.reduction_stride);
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
  // - col_reduce_small
  //
  //  It is a column reduce for small columns. Each thread loops over the whole
  //  column without communicating with any other thread.
  //
  // - col_reduce_two_pass
  //
  //  It is a reduce for long columns. To increase parallelism, we split the
  //  reduction in two passes. First we do a column reduce where many
  //  threadblocks operate on different parts of the reduced axis. Then we
  //  perform a final column reduce.
  //
  // Notes: As in row reduce we opt to read as much in order as possible and
  //        leave transpositions as they are (contrary to our Metal backend).
  //
  //        Moreover we need different kernels for short rows and tuning

  // Make the args struct to help route to the best kernel
  cu::ColReduceArgs args(in, plan, axes);

  // Small col reduce with a single or contiguous reduction axis
  if (args.non_col_reductions == 1 && args.reduction_size <= 32 &&
      args.reduction_stride % (16 / in.itemsize()) == 0) {
    col_reduce_small(encoder, in, out, reduce_type, axes, plan, args);
    return;
  }

  // Long column with smallish row
  size_t total_sums = args.non_col_reductions * args.reduction_size;
  size_t approx_threads = out.size();
  if (total_sums / approx_threads > 32) {
    col_reduce_two_pass(encoder, in, out, reduce_type, axes, plan, args);
    return;
  }

  // Fallback col reduce
  col_reduce_looped(encoder, in, out, reduce_type, axes, plan, args);
}

} // namespace mlx::core
