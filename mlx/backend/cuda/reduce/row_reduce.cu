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
};

template <typename T, typename U, typename Op, int NDIM, int N_READS = 4>
__global__ void row_reduce_small(
    const T* in,
    U* out,
    size_t out_size,
    const __grid_constant__ RowReduceArgs args) {
  size_t out_idx = cg::this_grid().thread_rank();
  if (out_idx >= out_size) {
    return;
  }

  Op op;

  U total_val = ReduceInit<Op, T>::value();
  LoopedElemToLoc<NDIM, (NDIM > 2)> loop(args.reduce_ndim);

  in += elem_to_loc(out_idx, args.shape.data(), args.strides.data(), args.ndim);

  for (size_t n = 0; n < args.non_row_reductions; n++) {
    for (int r = 0; r < cuda::ceil_div(args.row_size, N_READS); r++) {
      U vals[N_READS];
      cub::LoadDirectBlocked(
          r,
          make_cast_iterator<U>(in + loop.location()),
          vals,
          args.row_size,
          ReduceInit<Op, T>::value());
      total_val = op(total_val, cub::ThreadReduce(vals, op));
    }
    loop.next(args.reduce_shape.data(), args.reduce_strides.data());
  }

  out[out_idx] = total_val;
}

template <typename T, typename U, typename Op, int NDIM, int N_READS = 4>
__global__ void row_reduce_small_warp(
    const T* in,
    U* out,
    size_t out_size,
    const __grid_constant__ RowReduceArgs args) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  size_t out_idx = grid.thread_rank() / WARP_SIZE;
  if (out_idx >= out_size) {
    return;
  }

  Op op;

  U total_val = ReduceInit<Op, T>::value();
  LoopedElemToLoc<NDIM, (NDIM > 2)> loop(args.reduce_ndim);

  in += elem_to_loc(out_idx, args.shape.data(), args.strides.data(), args.ndim);

  for (size_t n = warp.thread_rank(); n < args.non_row_reductions;
       n += WARP_SIZE) {
    for (int r = 0; r < cuda::ceil_div(args.row_size, N_READS); r++) {
      U vals[N_READS];
      cub::LoadDirectBlocked(
          r,
          make_cast_iterator<U>(in + loop.location()),
          vals,
          args.row_size,
          ReduceInit<Op, T>::value());
      total_val = op(total_val, cub::ThreadReduce(vals, op));
    }
    loop.next(WARP_SIZE, args.reduce_shape.data(), args.reduce_strides.data());
  }

  total_val = cg::reduce(warp, total_val, op);

  if (warp.thread_rank() == 0) {
    out[out_idx] = total_val;
  }
}

template <
    typename T,
    typename U,
    typename Op,
    int NDIM,
    int BLOCK_DIM_X,
    int N_READS = 4>
__global__ void row_reduce_looped(
    const T* in,
    U* out,
    size_t out_size,
    const __grid_constant__ RowReduceArgs args) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();

  size_t out_idx = grid.thread_rank() / BLOCK_DIM_X;
  if (out_idx >= out_size) {
    return;
  }

  Op op;

  U total_val = ReduceInit<Op, T>::value();
  LoopedElemToLoc<NDIM, (NDIM > 2)> loop(args.reduce_ndim);

  in += elem_to_loc(out_idx, args.shape.data(), args.strides.data(), args.ndim);

  for (size_t n = 0; n < args.non_row_reductions; n++) {
    for (size_t r = 0; r < cuda::ceil_div(args.row_size, BLOCK_DIM_X * N_READS);
         r++) {
      U vals[N_READS];
      cub::LoadDirectBlocked(
          r * BLOCK_DIM_X + block.thread_index().x,
          make_cast_iterator<U>(in + loop.location()),
          vals,
          args.row_size,
          ReduceInit<Op, T>::value());
      total_val = op(total_val, cub::ThreadReduce(vals, op));
    }
    loop.next(args.reduce_shape.data(), args.reduce_strides.data());
  }

  typedef cub::BlockReduce<U, BLOCK_DIM_X> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp;

  total_val = BlockReduceT(temp).Reduce(total_val, op);

  if (block.thread_rank() == 0) {
    out[out_idx] = total_val;
  }
}

template <typename T, typename U, typename ReduceOp, int N = 4, int M = 1>
__global__ void
row_reduce_per_threadblock(T* in, U* out, size_t n_rows, int size) {
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
  in += start_row * size;
  out += start_row;

  int i = 0;
  for (; i + block.size() * N <= size; i += block.size() * N) {
    for (int k = 0; k < M; k++) {
      cub::LoadDirectBlockedVectorized<T, N>(
          block.thread_rank(), in + k * size + i, vals[k]);
      for (int j = 0; j < N; j++) {
        accs[k] = op(accs[k], __cast<U, T>(vals[k][j]));
      }
    }
  }

  if (size > i) {
    for (int k = 0; k < M; k++) {
      cub::LoadDirectBlocked(
          block.thread_rank(),
          in + k * size + i,
          vals[k],
          size,
          __cast<T, U>(init));
      for (int j = 0; i < N; i++) {
        accs[k] = op(accs[k], __cast<U, T>(vals[k][j]));
      }
    }
  }

  for (int i = 0; i < M; i++) {
    accs[i] = cg::reduce(warp, accs[i], op);
  }

  if (warp.meta_group_size() > 1) {
    __shared__ U shared_accumulators[32 * M];
    if (warp.thread_rank() == 0) {
      for (int i = 0; i < M; i++) {
        shared_accumulators[warp.meta_group_rank() * M + i] = accs[i];
      }
    }
    block.sync();
    if (warp.thread_rank() < warp.meta_group_size()) {
      for (int i = 0; i < M; i++) {
        accs[i] = shared_accumulators[warp.thread_rank() * M + i];
      }
    } else {
      for (int i = 0; i < M; i++) {
        accs[i] = init;
      }
    }
    for (int i = 0; i < M; i++) {
      accs[i] = cg::reduce(warp, accs[i], op);
    }
  }

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

} // namespace cu

void row_reduce_simple(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan) {
  constexpr int N_READS = 8;

  // Initialize out such that its strides match in's layout (except the fastest
  // moving axis)
  auto [_, out_strides] = shapes_without_reduction_axes(in, axes);
  for (auto& s : out_strides) {
    s /= plan.shape.back();
  }
  auto [data_size, rc, cc] = check_contiguity(out.shape(), out_strides);
  auto fl = in.flags();
  fl.row_contiguous = rc;
  fl.col_contiguous = cc;
  fl.contiguous = data_size == out.size();
  out.set_data(
      allocator::malloc(out.nbytes()),
      data_size,
      out_strides,
      fl,
      allocator::free);

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
        dim3 block(threads, 1, 1);
        auto kernel = cu::row_reduce_per_threadblock<T, U, OP, N_READS>;
        if (grid.x >= 1024) {
          grid.x = (grid.x + 1) / 2;
          kernel = cu::row_reduce_per_threadblock<T, U, OP, N_READS, 2>;
        }
        kernel<<<grid, block, 0, stream>>>(
            x.data<T>(), out.data<U>(), out.size(), plan.shape.back());
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
  if (plan.shape.size() == 1) {
    row_reduce_simple(encoder, in, out, reduce_type, axes, plan);
  }
  // cu::RowReduceArgs args(in, plan, axes);

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
