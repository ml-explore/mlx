// Copyright © 2025 Apple Inc.

#include <numeric>

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/reduce/reduce.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/device/utils.hpp"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

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
    
    // Copy to fixed-size arrays
    ndim = shape_vec.size();
    for (int i = 0; i < ndim && i < MAX_NDIM; i++) {
      shape[i] = shape_vec[i];
      strides[i] = strides_vec[i];
    }

    reduce_ndim = plan.shape.size();
    for (int i = 0; i < reduce_ndim && i < MAX_NDIM; i++) {
      reduce_shape[i] = plan.shape[i];
      reduce_strides[i] = plan.strides[i];
    }

    non_col_reductions = 1;
    for (int i = 0; i < reduce_ndim - 1; i++) {
      non_col_reductions *= reduce_shape[i];
    }
  }
};

// Warp reduce helper using runtime warp size
template <typename T, typename Op>
__device__ T warp_reduce_col(T val, Op op) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    T other = __shfl_xor(val, offset);
    val = op(val, other);
  }
  return val;
}

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
    const T* in,
    U* out,
    ColReduceArgs args,
    int64_t out_size) {
  
  constexpr int threads_per_row = BN / N_READS;

  // Compute the indices for the tile
  size_t tile_idx = blockIdx.x + blockIdx.y * gridDim.x;
  size_t tile_x = tile_idx % ((args.reduction_stride + BN - 1) / BN);
  size_t tile_y = tile_idx / ((args.reduction_stride + BN - 1) / BN);
  size_t tile_out = tile_y / out_size;
  tile_y = tile_y % out_size;

  // Compute the indices for the thread within the tile
  short thread_x = threadIdx.x % threads_per_row;
  short thread_y = threadIdx.x / threads_per_row;

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
  
  int remaining = args.reduction_stride - tile_x * BN;
  int base_idx = thread_x * N_READS;
  
  for (size_t r = start; r < end; r += BM) {
    // Load values
    for (int i = 0; i < N_READS; i++) {
      int idx = base_idx + i;
      if (idx < remaining) {
        totals[i] = op(totals[i], static_cast<U>(in[loop.location() + idx]));
      }
    }
    loop.next(BM, args.reduce_shape.data(), args.reduce_strides.data());
  }

  // Do warp reduce for each output.
  constexpr int n_outputs = BN / threads_per_row;
  __shared__ U shared_vals[BM * BN];
  short s_idx = thread_y * BN + thread_x * N_READS;
  for (int i = 0; i < N_READS; i++) {
    shared_vals[s_idx + i] = totals[i];
  }
  __syncthreads();
  
  // Reduce across threads
  if (thread_y == 0) {
    for (int i = 0; i < N_READS; i++) {
      U val = ReduceInit<Op, T>::value();
      for (int j = 0; j < BM; j++) {
        val = op(val, shared_vals[j * BN + thread_x * N_READS + i]);
      }
      totals[i] = val;
    }
  }
  __syncthreads();

  // Write result.
  if (thread_y == 0) {
    if (BLOCKS > 1) {
      out += tile_out * out_size * args.reduction_stride;
    }
    for (int i = 0; i < N_READS; i++) {
      int idx = thread_x * N_READS + i;
      if (tile_x * BN + idx < args.reduction_stride) {
        out[tile_y * args.reduction_stride + tile_x * BN + idx] = totals[i];
      }
    }
  }
}

template <typename T, typename U, typename Op, int N_READS = 4>
__global__ void col_reduce_small(
    const T* in,
    U* out,
    ColReduceArgs args,
    size_t total) {
  Op op;

  const auto idx = (blockIdx.x * blockDim.x + threadIdx.x) * N_READS;
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

  for (size_t i = 0; i < args.reduction_size; i++) {
    auto values = load_vector<N_READS>(in, 0);

    for (int j = 0; j < N_READS; j++) {
      accumulator[j] = op(accumulator[j], static_cast<U>(values[j]));
    }

    in += args.reduction_stride;
  }

  store_vector(out, 0, accumulator);
}

// Simple column reduction kernel for contiguous strided reduce
template <typename T, typename U, typename Op>
__global__ void col_reduce_simple_kernel(
    const T* in,
    U* out,
    int n_rows,
    int n_cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= n_cols) return;
  
  Op op;
  U val = ReduceInit<Op, T>::value();
  
  for (int row = 0; row < n_rows; row++) {
    val = op(val, static_cast<U>(in[row * n_cols + col]));
  }
  
  out[col] = val;
}

} // namespace rocm

inline auto output_grid_for_col_reduce(
    const array& out,
    const rocm::ColReduceArgs& args,
    int bn,
    int outer = 1) {
  int gx, gy = 1;
  size_t n_inner_blocks = ceildiv(args.reduction_stride, (int64_t)bn);
  size_t n_outer_blocks = out.size() / args.reduction_stride;
  size_t n_blocks = n_outer_blocks * n_inner_blocks * outer;
  while (n_blocks / gy > INT32_MAX) {
    gy *= 2;
  }
  gx = ceildiv(n_blocks, (size_t)gy);

  return dim3(gx, gy, 1);
}

// Dispatch helper for reduce operations
template <typename Func>
void dispatch_reduce_ops(Reduce::ReduceType reduce_type, Func&& func) {
  switch (reduce_type) {
    case Reduce::Sum:
      func(std::type_identity<rocm::Sum>{});
      break;
    case Reduce::Prod:
      func(std::type_identity<rocm::Prod>{});
      break;
    case Reduce::Max:
      func(std::type_identity<rocm::Max>{});
      break;
    case Reduce::Min:
      func(std::type_identity<rocm::Min>{});
      break;
    case Reduce::And:
      func(std::type_identity<rocm::And>{});
      break;
    case Reduce::Or:
      func(std::type_identity<rocm::Or>{});
      break;
    default:
      throw std::runtime_error("Unsupported reduce type");
  }
}

// Dispatch helper for reduce ndim
template <typename Func>
void dispatch_reduce_ndim(int ndim, Func&& func) {
  switch (ndim) {
    case 1:
      func(std::integral_constant<int, 1>{});
      break;
    case 2:
      func(std::integral_constant<int, 2>{});
      break;
    case 3:
      func(std::integral_constant<int, 3>{});
      break;
    case 4:
      func(std::integral_constant<int, 4>{});
      break;
    default:
      func(std::integral_constant<int, 5>{});
      break;
  }
}

void col_reduce_looped(
    rocm::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan,
    const rocm::ColReduceArgs& args) {
  // Allocate data for the output
  allocate_same_layout(out, in, axes, encoder);

  encoder.set_input_array(in);
  encoder.set_output_array(out);
  
  dispatch_all_types(in.dtype(), [&](auto type_tag) {
    dispatch_reduce_ops(reduce_type, [&](auto reduce_type_tag) {
      dispatch_reduce_ndim(args.reduce_ndim, [&](auto reduce_ndim) {
        using OP = typename decltype(reduce_type_tag)::type;
        using T = hip_type_t<MLX_GET_TYPE(type_tag)>;
        using U = typename rocm::ReduceResult<OP, T>::type;

        constexpr int N_READS = 4;
        constexpr int BM = 32;
        constexpr int BN = 32;
        dim3 grid = output_grid_for_col_reduce(out, args, BN);
        int blocks = BM * BN / N_READS;
        
        encoder.launch_kernel([&](hipStream_t stream) {
          hipLaunchKernelGGL(
              (rocm::col_reduce_looped<T, U, OP, reduce_ndim(), BM, BN, N_READS>),
              grid, dim3(blocks), 0, stream,
              in.data<T>(),
              out.data<U>(),
              args,
              out.size() / args.reduction_stride);
        });
      });
    });
  });
}

void col_reduce_small(
    rocm::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan,
    const rocm::ColReduceArgs& args) {
  // Allocate data for the output
  allocate_same_layout(out, in, axes, encoder);

  encoder.set_input_array(in);
  encoder.set_output_array(out);
  
  dispatch_all_types(in.dtype(), [&](auto type_tag) {
    dispatch_reduce_ops(reduce_type, [&](auto reduce_type_tag) {
      using OP = typename decltype(reduce_type_tag)::type;
      using T = hip_type_t<MLX_GET_TYPE(type_tag)>;
      using U = typename rocm::ReduceResult<OP, T>::type;

      constexpr int N_READS = 4;
      int block_size = 256;
      int num_blocks = (out.size() + block_size * N_READS - 1) / (block_size * N_READS);
      
      encoder.launch_kernel([&](hipStream_t stream) {
        hipLaunchKernelGGL(
            (rocm::col_reduce_small<T, U, OP, N_READS>),
            dim3(num_blocks), dim3(block_size), 0, stream,
            in.data<T>(),
            out.data<U>(),
            args,
            out.size());
      });
    });
  });
}

void col_reduce(
    rocm::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan) {
  
  // Make the args struct to help route to the best kernel
  rocm::ColReduceArgs args(in, plan, axes);

  // Small col reduce with a single or contiguous reduction axis
  if (args.non_col_reductions == 1 && args.reduction_size <= 32 &&
      args.reduction_stride % 4 == 0) {
    col_reduce_small(encoder, in, out, reduce_type, axes, plan, args);
    return;
  }

  // Fallback col reduce
  col_reduce_looped(encoder, in, out, reduce_type, axes, plan, args);
}

} // namespace mlx::core
