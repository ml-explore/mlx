// Copyright © 2025 Apple Inc.

#include <numeric>

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/reduce/reduce.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace mlx::core {

namespace cu {

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

template <typename T, typename U, typename ReduceOp, int N = 4, int M = 1>
__global__ void
row_reduce_simple(const T* in, U* out, size_t n_rows, int size) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  const U init = cu::ReduceInit<ReduceOp, T>::value();
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
        accs[k] = op(accs[k], cast_to<U>(vals[k][j]));
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
        accs[k] = op(accs[k], cast_to<U>(vals[k][j]));
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

template <typename T, typename U, typename Op, int NDIM, int N_READS = 4>
__global__ void row_reduce_looped(
    const T* in,
    U* out,
    const __grid_constant__ RowReduceArgs args) {
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
          total[0] = op(total[0], cast_to<U>(vals[i]));
        }
        inlocal += block.size() * N_READS;
      }

      {
        T vals[N_READS];
        for (int i = 0; i < N_READS; i++) {
          vals[i] = mask[i] ? inlocal[i] : cast_to<T>(init);
        }
        for (int i = 0; i < N_READS; i++) {
          total[0] = op(total[0], cast_to<U>(vals[i]));
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
          total[0] = op(total[0], cast_to<U>(vals[i]));
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

} // namespace cu

void row_reduce_simple(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan) {
  // Allocate data for the output using in's layout to avoid elem_to_loc in the
  // kernel.
  allocate_same_layout(out, in, axes);

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

      T* indata = const_cast<T*>(in.data<T>());
      int size = plan.shape.back();
      encoder.add_kernel_node(
          kernel, grid, block, 0, indata, out.data<U>(), out.size(), size);
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
  allocate_same_layout(out, in, axes);

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
          kernel, grid, block, 0, in.data<T>(), out.data<U>(), args);
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
