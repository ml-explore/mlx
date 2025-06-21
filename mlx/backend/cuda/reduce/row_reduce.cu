// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/cast_op.cuh"
#include "mlx/backend/cuda/reduce/reduce.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>

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
  void convert_shapes_to_contiguous(
      const array& in,
      const std::vector<int>& axes) {
    auto shape_vec = in.shape();
    auto strides_vec = in.strides();
    size_t s = 1;
    for (int i = in.ndim() - 1; i >= 0; i--) {
      strides_vec[i] = s;
      s *= shape_vec[i];
    }
    std::tie(shape_vec, strides_vec) =
        shapes_without_reduction_axes(shape_vec, strides_vec, axes);
    std::tie(shape_vec, strides_vec) =
        collapse_contiguous_dims(shape_vec, strides_vec);
    shape = const_param(shape_vec);
    strides = const_param(strides_vec);
    ndim = shape_vec.size();
  }
};

// template <typename T, typename U, typename Op, int NDIM, int N_READS = 4>
//__global__ void row_reduce_small(
//     const T* in,
//     U* out,
//     size_t out_size,
//     const __grid_constant__ RowReduceArgs args) {
//   size_t out_idx = cg::this_grid().thread_rank();
//   if (out_idx >= out_size) {
//     return;
//   }
//
//   Op op;
//
//   U total_val = ReduceInit<Op, T>::value();
//   LoopedElemToLoc<NDIM, (NDIM > 2)> loop(args.reduce_ndim);
//
//   in += elem_to_loc(out_idx, args.shape.data(), args.strides.data(),
//   args.ndim);
//
//   for (size_t n = 0; n < args.non_row_reductions; n++) {
//     for (int r = 0; r < cuda::ceil_div(args.row_size, N_READS); r++) {
//       U vals[N_READS];
//       cub::LoadDirectBlocked(
//           r,
//           make_cast_iterator<U>(in + loop.location()),
//           vals,
//           args.row_size,
//           ReduceInit<Op, T>::value());
//       total_val = op(total_val, cub::ThreadReduce(vals, op));
//     }
//     loop.next(args.reduce_shape.data(), args.reduce_strides.data());
//   }
//
//   out[out_idx] = total_val;
// }
//
// template <typename T, typename U, typename Op, int NDIM, int N_READS = 4>
//__global__ void row_reduce_small_warp(
//     const T* in,
//     U* out,
//     size_t out_size,
//     const __grid_constant__ RowReduceArgs args) {
//   auto grid = cg::this_grid();
//   auto block = cg::this_thread_block();
//   auto warp = cg::tiled_partition<WARP_SIZE>(block);
//
//   size_t out_idx = grid.thread_rank() / WARP_SIZE;
//   if (out_idx >= out_size) {
//     return;
//   }
//
//   Op op;
//
//   U total_val = ReduceInit<Op, T>::value();
//   LoopedElemToLoc<NDIM, (NDIM > 2)> loop(args.reduce_ndim);
//
//   in += elem_to_loc(out_idx, args.shape.data(), args.strides.data(),
//   args.ndim);
//
//   for (size_t n = warp.thread_rank(); n < args.non_row_reductions;
//        n += WARP_SIZE) {
//     for (int r = 0; r < cuda::ceil_div(args.row_size, N_READS); r++) {
//       U vals[N_READS];
//       cub::LoadDirectBlocked(
//           r,
//           make_cast_iterator<U>(in + loop.location()),
//           vals,
//           args.row_size,
//           ReduceInit<Op, T>::value());
//       total_val = op(total_val, cub::ThreadReduce(vals, op));
//     }
//     loop.next(WARP_SIZE, args.reduce_shape.data(),
//     args.reduce_strides.data());
//   }
//
//   total_val = cg::reduce(warp, total_val, op);
//
//   if (warp.thread_rank() == 0) {
//     out[out_idx] = total_val;
//   }
// }

template <typename T, typename U, typename ReduceOp, int N = 4, int M = 1>
__global__ void row_reduce_simple(T* in, U* out, size_t n_rows, int size) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  const U init = cu::ReduceInit<ReduceOp, T>::value();
  ReduceOp op;

  T vals[M][N];
  U accs[M];
  for (int i = 0; i < M; i++) {
    accs[i] = init;
  }

  const size_t start_row =
      min(n_rows - M, static_cast<size_t>(grid.block_rank() * M));
  const size_t full_blocks = size / (block.size() * N);
  const size_t final_offset = full_blocks * (block.size() * N);
  in += start_row * size;
  out += start_row;

  for (size_t r = 0; r < full_blocks; r++) {
    for (int k = 0; k < M; k++) {
      cub::LoadDirectBlockedVectorized<T, N>(
          block.thread_rank(), in + k * size + r * (block.size() * N), vals[k]);
      for (int j = 0; j < N; j++) {
        accs[k] = op(accs[k], __cast<U, T>(vals[k][j]));
      }
    }
  }

  if (final_offset < size) {
    for (int k = 0; k < M; k++) {
      cub::LoadDirectBlocked(
          block.thread_rank(),
          in + k * size + final_offset,
          vals[k],
          size,
          __cast<T, U>(init));
      for (int j = 0; j < N; j++) {
        accs[k] = op(accs[k], __cast<U, T>(vals[k][j]));
      }
    }
  }

  __shared__ U shared_accumulators[32 * M];
  block_reduce(block, warp, accs, shared_accumulators, op, init);

  if (block.thread_rank() == 0) {
    if (grid.block_rank() * M + M <= n_rows) {
      for (int i = 0; i < M; i++) {
        out[i] = accs[i];
      }
    } else {
      short offset = grid.block_rank() * M + M - n_rows;
      for (int i = offset; i < M; i++) {
        out[i] = accs[i];
      }
    }
  }
}

template <
    typename T,
    typename U,
    typename Op,
    int NDIM,
    int BLOCK_DIM,
    int N_READS = 4>
__global__ void row_reduce_looped(
    T* in,
    U* out,
    size_t out_size,
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
  size_t full_blocks = args.row_size / (BLOCK_DIM * N_READS);
  size_t final_offset = full_blocks * BLOCK_DIM * N_READS;

  in += elem_to_loc(out_idx, args.shape.data(), args.strides.data(), args.ndim);

  for (size_t n = 0; n < args.non_row_reductions; n++) {
    for (size_t r = 0; r < full_blocks; r++) {
      T vals[N_READS];
      cub::LoadDirectBlockedVectorized<T, N_READS>(
          block.thread_rank(),
          in + loop.location() + r * BLOCK_DIM * N_READS,
          vals);
      for (int i = 0; i < N_READS; i++) {
        total[0] = op(total[0], __cast<U, T>(vals[i]));
      }
    }
    if (final_offset < args.row_size) {
      T vals[N_READS];
      cub::LoadDirectBlocked(
          block.thread_rank(),
          in + loop.location() + final_offset,
          vals,
          args.row_size - final_offset,
          __cast<T, U>(init));
      for (int i = 0; i < N_READS; i++) {
        total[0] = op(total[0], __cast<U, T>(vals[i]));
      }
    }
    // TODO: Maybe block.sync() here?
    loop.next(args.reduce_shape.data(), args.reduce_strides.data());
  }

  __shared__ U shared_accumulators[32];
  block_reduce(block, warp, total, shared_accumulators, op, init);

  if (block.thread_rank() == 0) {
    out[out_idx] = total[0];
  }
}

template <typename T, typename U, typename Op, int N = 4>
__global__ void reduce_initialize(U* out, size_t out_size) {
  auto grid = cg::this_grid();
  if (grid.thread_rank() * N + N <= out_size) {
    for (int i = 0; i < N; i++) {
      out[grid.thread_rank() * N + i] = ReduceInit<Op, T>::value();
    }
  } else {
    for (int i = grid.thread_rank() * N; i < out_size; i++) {
      out[i] = ReduceInit<Op, T>::value();
    }
  }
}

template <typename T, typename U, typename Op, int BLOCK_DIM, int N_READS = 4>
__global__ void row_reduce_atomics(
    T* in,
    U* out,
    size_t out_size,
    const __grid_constant__ RowReduceArgs args) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  size_t reduction_idx = grid.block_rank() / out_size;
  size_t out_idx = grid.block_rank() % out_size;

  Op op;

  U total[1];
  U init = ReduceInit<Op, T>::value();
  total[0] = init;
  size_t full_blocks = args.row_size / (BLOCK_DIM * N_READS);
  size_t final_offset = full_blocks * BLOCK_DIM * N_READS;

  in += elem_to_loc(out_idx, args.shape.data(), args.strides.data(), args.ndim);
  in += elem_to_loc(
      reduction_idx,
      args.reduce_shape.data(),
      args.reduce_strides.data(),
      args.reduce_ndim);

  for (size_t r = 0; r < full_blocks; r++) {
    T vals[N_READS];
    cub::LoadDirectBlockedVectorized<T, N_READS>(
        block.thread_rank(), in + r * BLOCK_DIM * N_READS, vals);
    for (int i = 0; i < N_READS; i++) {
      total[0] = op(total[0], __cast<U, T>(vals[i]));
    }
  }
  if (final_offset < args.row_size) {
    T vals[N_READS];
    cub::LoadDirectBlocked(
        block.thread_rank(),
        in + final_offset,
        vals,
        args.row_size - final_offset,
        __cast<T, U>(init));
    for (int i = 0; i < N_READS; i++) {
      total[0] = op(total[0], __cast<U, T>(vals[i]));
    }
  }

  __shared__ U shared_accumulators[32];
  block_reduce(block, warp, total, shared_accumulators, op, init);

  if (block.thread_rank() == 0) {
    op.atomic_update(out + out_idx, total[0]);
  }
}

} // namespace cu

void reduce_initialize(
    cu::CommandEncoder& encoder,
    array& out,
    Reduce::ReduceType reduce_type) {
  constexpr int N_WRITES = 8;
  encoder.set_output_array(out);
  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_ALL_TYPES(out.dtype(), CTYPE, {
      MLX_SWITCH_REDUCE_OPS(reduce_type, OP, {
        using T = cuda_type_t<CTYPE>;
        using U = cu::ReduceResult<OP, T>::type;

        auto kernel = cu::reduce_initialize<T, U, OP, N_WRITES>;
        auto [grid, block] =
            get_launch_args(kernel, out, out.size() >= 1UL << 31, N_WRITES);
        kernel<<<grid, block, 0, stream>>>(out.data<U>(), out.size());
      });
    });
  });
}

void row_reduce_simple(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan) {
  constexpr int N_READS = 8;

  // Allocate data for the output using in's layout to avoid elem_to_loc in the
  // kernel.
  allocate_same_layout(out, in, axes);

  // Just a way to get out of the constness because cub doesn't like it ...
  // (sigh)
  array x = in;

  // TODO: If out.size() < 1024 which will be a common case then write this in
  //       2 passes. Something like 32 * out.size() and then do a warp reduce.
  encoder.set_input_array(x);
  encoder.set_output_array(out);
  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_ALL_TYPES(x.dtype(), CTYPE, {
      MLX_SWITCH_REDUCE_OPS(reduce_type, OP, {
        using T = cuda_type_t<CTYPE>;
        using U = cu::ReduceResult<OP, T>::type;

        // Calculate the grid and block dims
        size_t reductions = plan.shape.back() / N_READS;
        dim3 grid = get_2d_grid_dims(out.shape(), out.strides());
        int threads = std::min(1024UL, reductions);
        threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        dim3 block(threads, 1, 1);

        // Pick the kernel
        auto kernel = cu::row_reduce_simple<T, U, OP, N_READS>;
        if (grid.x >= 1024) {
          grid.x = (grid.x + 1) / 2;
          kernel = cu::row_reduce_simple<T, U, OP, N_READS, 2>;
        }

        // Launch
        kernel<<<grid, block, 0, stream>>>(
            x.data<T>(), out.data<U>(), out.size(), plan.shape.back());
      });
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
  constexpr int N_READS = 8;

  // Allocate data for the output using in's layout to access them as
  // contiguously as possible.
  allocate_same_layout(out, in, axes);

  // Just a way to get out of the constness because cub doesn't like it ...
  // (sigh)
  array x = in;

  encoder.set_input_array(x);
  encoder.set_output_array(out);
  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_ALL_TYPES(x.dtype(), CTYPE, {
      MLX_SWITCH_REDUCE_OPS(reduce_type, OP, {
        using T = cuda_type_t<CTYPE>;
        using U = cu::ReduceResult<OP, T>::type;

        // Calculate the grid and block dims
        args.convert_shapes_to_contiguous(x, axes);
        dim3 grid = get_2d_grid_dims(out.shape(), out.strides());
        size_t reductions = args.row_size / N_READS;
        int threads = std::min(1024UL, reductions);
        threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        dim3 block(threads, 1, 1);

        // Pick the kernel
        auto kernel = cu::row_reduce_looped<T, U, OP, 1, 32, N_READS>;
        MLX_SWITCH_REDUCE_NDIM(args.reduce_ndim, NDIM, {
          MLX_SWITCH_BLOCK_DIM(threads, THREADS, {
            kernel = cu::row_reduce_looped<T, U, OP, NDIM, THREADS, N_READS>;
            block.x = THREADS;
          });
        });

        // Launch
        kernel<<<grid, block, 0, stream>>>(
            x.data<T>(), out.data<U>(), out.size(), args);
      });
    });
  });
}

void row_reduce_atomics(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan,
    cu::RowReduceArgs args) {
  constexpr int N_READS = 8;

  // Allocate data for the output using in's layout to access them as
  // contiguously as possible.
  allocate_same_layout(out, in, axes);

  // Just a way to get out of the constness because cub doesn't like it ...
  // (sigh)
  array x = in;

  // Initialize
  reduce_initialize(encoder, out, reduce_type);

  // Launch the reduction
  encoder.set_input_array(x);
  encoder.set_output_array(out);
  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_ALL_TYPES(x.dtype(), CTYPE, {
      MLX_SWITCH_REDUCE_OPS(reduce_type, OP, {
        using T = cuda_type_t<CTYPE>;
        using U = cu::ReduceResult<OP, T>::type;

        args.convert_shapes_to_contiguous(x, axes);
        dim3 grid = get_2d_grid_dims(out.shape(), out.strides());
        if (grid.x * args.non_row_reductions < INT_MAX) {
          grid.x *= args.non_row_reductions;
        } else if (grid.y * args.non_row_reductions < 65536) {
          grid.y *= args.non_row_reductions;
        } else {
          throw std::runtime_error(
              "[row_reduce_atomics] Non-row reductions need to be factorized which is NYI");
        }
        size_t reductions = args.row_size / N_READS;
        int threads = std::min(1024UL, reductions);
        threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        dim3 block(threads, 1, 1);

        // Pick the kernel
        auto kernel = cu::row_reduce_atomics<T, U, OP, 32, N_READS>;
        MLX_SWITCH_BLOCK_DIM(threads, THREADS, {
          kernel = cu::row_reduce_atomics<T, U, OP, THREADS, N_READS>;
          block.x = THREADS;
        });

        // Launch
        kernel<<<grid, block, 0, stream>>>(
            x.data<T>(), out.data<U>(), out.size(), args);
      });
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
  // Simple row reduce means that we have 1 axis that we are reducing over and
  // it has stride 1.
  if (plan.shape.size() == 1) {
    row_reduce_simple(encoder, in, out, reduce_type, axes, plan);
    return;
  }

  // Make the args struct to help route to the best kernel
  cu::RowReduceArgs args(in, plan, axes);

  // Let's use atomics to increase parallelism
  if (false && args.row_size < 512) {
    row_reduce_atomics(
        encoder, in, out, reduce_type, axes, plan, std::move(args));
  }

  // Fallback row reduce
  row_reduce_looped(encoder, in, out, reduce_type, axes, plan, std::move(args));

  // encoder.launch_kernel([&](cudaStream_t stream) {
  //   MLX_SWITCH_ALL_TYPES(in.dtype(), CTYPE, {
  //     using InType = cuda_type_t<CTYPE>;
  //     MLX_SWITCH_REDUCE_OPS(reduce_type, OP, {
  //       using OutType = cu::ReduceResult<OP, InType>::type;
  //       MLX_SWITCH_REDUCE_NDIM(args.reduce_ndim, NDIM, {
  //         constexpr size_t N_READS = 4;
  //         dim3 out_dims = get_2d_grid_dims(out.shape(), out.strides());
  //         dim3 block_dims, num_blocks;
  //         auto kernel =
  //             cu::row_reduce_small<InType, OutType, OP, NDIM, N_READS>;
  //         if (args.row_size <= 64) {
  //           if ((args.non_row_reductions < 32 && args.row_size <= 8) ||
  //               (args.non_row_reductions <= 8)) {
  //             block_dims.x = std::min(out_dims.x, 1024u);
  //             num_blocks.x = cuda::ceil_div(out_dims.x, block_dims.x);
  //             num_blocks.y = out_dims.y;
  //           } else {
  //             block_dims.x = WARP_SIZE;
  //             num_blocks.y = out_dims.x;
  //             num_blocks.z = out_dims.y;
  //             kernel =
  //                 cu::row_reduce_small_warp<InType, OutType, OP, NDIM,
  //                 N_READS>;
  //           }
  //         } else {
  //           size_t num_threads = cuda::ceil_div(args.row_size, N_READS);
  //           num_threads = cuda::ceil_div(num_threads, WARP_SIZE) * WARP_SIZE;
  //           MLX_SWITCH_BLOCK_DIM(num_threads, BLOCK_DIM_X, {
  //             num_blocks.y = out_dims.x;
  //             num_blocks.z = out_dims.y;
  //             block_dims.x = BLOCK_DIM_X;
  //             kernel = cu::row_reduce_looped<
  //                 InType,
  //                 OutType,
  //                 OP,
  //                 NDIM,
  //                 BLOCK_DIM_X,
  //                 N_READS>;
  //           });
  //         }
  //         kernel<<<num_blocks, block_dims, 0, stream>>>(
  //             in.data<InType>(), out.data<OutType>(), out.size(), args);
  //       });
  //     });
  //   });
  // });
}

} // namespace mlx::core
