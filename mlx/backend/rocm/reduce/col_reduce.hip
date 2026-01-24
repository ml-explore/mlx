// Copyright © 2025 Apple Inc.

#include <numeric>

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/reduce/reduce.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

struct ColReduceArgs {
  // The size of the contiguous column reduction.
  size_t reduction_size;
  int64_t reduction_stride;

  // Input shape and strides excluding the reduction axes.
  int shape[MAX_NDIM];
  int64_t strides[MAX_NDIM];
  int ndim;

  // Input shape and strides of the reduction axes (including last dimension).
  int reduce_shape[MAX_NDIM];
  int64_t reduce_strides[MAX_NDIM];
  int reduce_ndim;

  // The number of column we are reducing. Namely prod(reduce_shape).
  size_t non_col_reductions;
};

// Warp reduce helper
template <typename T, typename Op>
__device__ T warp_reduce_col(T val, Op op) {
  for (int offset = 32; offset > 0; offset /= 2) {
    T other = __shfl_xor(val, offset);
    val = op(val, other);
  }
  return val;
}

// Element to location helper
__device__ int64_t elem_to_loc_col(
    int64_t elem,
    const int* shape,
    const int64_t* strides,
    int ndim) {
  int64_t loc = 0;
  for (int i = ndim - 1; i >= 0; --i) {
    loc += (elem % shape[i]) * strides[i];
    elem /= shape[i];
  }
  return loc;
}

template <typename T, typename U, typename Op, int BM, int BN, int N_READS = 4>
__global__ void col_reduce_looped_kernel(
    const T* in,
    U* out,
    ColReduceArgs args) {
  // Compute the indices for the tile
  size_t tile_idx = blockIdx.x + blockIdx.y * gridDim.x;
  size_t n_inner_blocks = (args.reduction_stride + BN - 1) / BN;
  size_t tile_x = tile_idx % n_inner_blocks;
  size_t tile_y = tile_idx / n_inner_blocks;

  // Compute the indices for the thread within the tile
  int threads_per_row = BN / N_READS;
  int thread_x = threadIdx.x % threads_per_row;
  int thread_y = threadIdx.x / threads_per_row;

  // Move the input pointer
  int64_t in_offset = elem_to_loc_col(tile_y, args.shape, args.strides, args.ndim);
  in += in_offset + tile_x * BN;

  // Initialize the running totals
  Op op;
  U totals[N_READS];
  for (int i = 0; i < N_READS; i++) {
    totals[i] = ReduceInit<Op, T>::value();
  }

  // Loop over reductions
  size_t total = args.non_col_reductions * args.reduction_size;
  
  int64_t reduce_loc = 0;
  int64_t reduce_idx = thread_y;
  
  // Compute initial reduce location
  {
    int64_t tmp = reduce_idx;
    for (int i = args.reduce_ndim - 1; i >= 0; --i) {
      reduce_loc += (tmp % args.reduce_shape[i]) * args.reduce_strides[i];
      tmp /= args.reduce_shape[i];
    }
  }

  for (size_t r = thread_y; r < total; r += BM) {
    // Load values
    int base_idx = thread_x * N_READS;
    int remaining = args.reduction_stride - tile_x * BN;
    
    for (int i = 0; i < N_READS; i++) {
      int idx = base_idx + i;
      if (idx < remaining) {
        totals[i] = op(totals[i], static_cast<U>(in[reduce_loc + idx]));
      }
    }
    
    // Update reduce location for next iteration
    reduce_idx += BM;
    if (reduce_idx < total) {
      reduce_loc = 0;
      int64_t tmp = reduce_idx;
      for (int i = args.reduce_ndim - 1; i >= 0; --i) {
        reduce_loc += (tmp % args.reduce_shape[i]) * args.reduce_strides[i];
        tmp /= args.reduce_shape[i];
      }
    }
  }

  // Do warp reduce for each output
  constexpr int n_outputs = BN / threads_per_row;
  __shared__ U shared_vals[BM * BN];
  
  int s_idx = thread_y * BN + thread_x * N_READS;
  for (int i = 0; i < N_READS; i++) {
    shared_vals[s_idx + i] = totals[i];
  }
  __syncthreads();
  
  // Reduce across warps
  int lane = threadIdx.x % 64;
  int warp_id = threadIdx.x / 64;
  
  if (warp_id == 0) {
    s_idx = lane * BN / 64;
    for (int i = 0; i < n_outputs; i++) {
      U val = (lane < BM) ? shared_vals[lane * BN + warp_id * n_outputs + i] : ReduceInit<Op, T>::value();
      for (int j = 1; j < BM && j + lane * BM / 64 < BM; j++) {
        int read_idx = (lane + j * 64 / BM) * BN + warp_id * n_outputs + i;
        if (read_idx < BM * BN) {
          val = op(val, shared_vals[read_idx]);
        }
      }
      totals[i] = warp_reduce_col(val, op);
    }
  }
  __syncthreads();

  // Write result
  if (threadIdx.x < BN) {
    int out_idx = tile_y * args.reduction_stride + tile_x * BN + threadIdx.x;
    if (tile_x * BN + threadIdx.x < args.reduction_stride) {
      // Simple version: first thread writes
      if (thread_y == 0) {
        U final_val = ReduceInit<Op, T>::value();
        for (int j = 0; j < BM; j++) {
          final_val = op(final_val, shared_vals[j * BN + threadIdx.x]);
        }
        out[out_idx] = final_val;
      }
    }
  }
}

// Simpler column reduction kernel for contiguous strided reduce
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

void col_reduce(
    rocm::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan) {
  
  // Allocate output
  out.set_data(allocator::malloc(out.nbytes()));
  
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  
  // For simple contiguous strided reduce (most common case in VJP)
  if (plan.type == ReductionOpType::ContiguousStridedReduce && 
      plan.shape.size() == 1) {
    int n_rows = plan.shape[0];
    int n_cols = out.size();
    
    int block_size = 256;
    int num_blocks = (n_cols + block_size - 1) / block_size;
    
    encoder.launch_kernel([&](hipStream_t stream) {
      switch (in.dtype()) {
        case float32:
          switch (reduce_type) {
            case Reduce::Sum:
              hipLaunchKernelGGL(
                  (rocm::col_reduce_simple_kernel<float, float, rocm::Sum>),
                  dim3(num_blocks), dim3(block_size), 0, stream,
                  in.data<float>(), out.data<float>(), n_rows, n_cols);
              break;
            case Reduce::Max:
              hipLaunchKernelGGL(
                  (rocm::col_reduce_simple_kernel<float, float, rocm::Max>),
                  dim3(num_blocks), dim3(block_size), 0, stream,
                  in.data<float>(), out.data<float>(), n_rows, n_cols);
              break;
            case Reduce::Min:
              hipLaunchKernelGGL(
                  (rocm::col_reduce_simple_kernel<float, float, rocm::Min>),
                  dim3(num_blocks), dim3(block_size), 0, stream,
                  in.data<float>(), out.data<float>(), n_rows, n_cols);
              break;
            case Reduce::Prod:
              hipLaunchKernelGGL(
                  (rocm::col_reduce_simple_kernel<float, float, rocm::Prod>),
                  dim3(num_blocks), dim3(block_size), 0, stream,
                  in.data<float>(), out.data<float>(), n_rows, n_cols);
              break;
            default:
              throw std::runtime_error("Unsupported reduce type for col_reduce");
          }
          break;
        case float16:
          switch (reduce_type) {
            case Reduce::Sum:
              hipLaunchKernelGGL(
                  (rocm::col_reduce_simple_kernel<__half, __half, rocm::Sum>),
                  dim3(num_blocks), dim3(block_size), 0, stream,
                  in.data<__half>(), out.data<__half>(), n_rows, n_cols);
              break;
            default:
              throw std::runtime_error("Unsupported reduce type for col_reduce float16");
          }
          break;
        case bfloat16:
          switch (reduce_type) {
            case Reduce::Sum:
              hipLaunchKernelGGL(
                  (rocm::col_reduce_simple_kernel<__hip_bfloat16, __hip_bfloat16, rocm::Sum>),
                  dim3(num_blocks), dim3(block_size), 0, stream,
                  in.data<__hip_bfloat16>(), out.data<__hip_bfloat16>(), n_rows, n_cols);
              break;
            default:
              throw std::runtime_error("Unsupported reduce type for col_reduce bfloat16");
          }
          break;
        default:
          throw std::runtime_error("Unsupported dtype for col_reduce");
      }
    });
    return;
  }
  
  // General case - build args and use looped kernel
  throw std::runtime_error("General col_reduce not yet implemented for ROCm");
}

} // namespace mlx::core
