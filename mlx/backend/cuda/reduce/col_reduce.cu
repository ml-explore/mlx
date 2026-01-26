// Copyright © 2025 Apple Inc.

#include <numeric>

#include "mlx/backend/cuda/device.h"
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
    int BLOCKS = 1>
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
          totals[i] = op(totals[i], cast_to<U>(vals[i]));
        }
        loop.next(BM, args.reduce_shape.data(), args.reduce_strides.data());
      }
    } else {
      for (size_t r = start; r < end; r += BM) {
        T vals[N_READS];
        cub::LoadDirectBlocked(thread_x, in + loop.location(), vals);
        for (int i = 0; i < N_READS; i++) {
          totals[i] = op(totals[i], cast_to<U>(vals[i]));
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
        totals[i] = op(totals[i], cast_to<U>(vals[i]));
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

template <typename T, typename U, typename Op, int N_READS = 4>
__global__ void col_reduce_small(
    const T* in,
    U* out,
    const __grid_constant__ ColReduceArgs args,
    size_t total) {
  Op op;
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
      accumulator[j] = op(accumulator[j], cast_to<U>(values[j]));
    }

    in += args.reduction_stride;
  }

  store_vector(out, 0, accumulator);
}

} // namespace cu

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
