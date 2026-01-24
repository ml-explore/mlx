// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/cast_op.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>

#include <cassert>

namespace mlx::core {

namespace rocm {

template <typename T>
inline __device__ T logsumexp_exp(T x) {
  return __expf(x);
}

// Warp reduce for max
template <typename T>
__device__ T warp_reduce_max_lse(T val) {
  for (int offset = 32; offset > 0; offset /= 2) {
    T other = __shfl_xor(val, offset);
    val = val > other ? val : other;
  }
  return val;
}

// Warp reduce for sum
template <typename T>
__device__ T warp_reduce_sum_lse(T val) {
  for (int offset = 32; offset > 0; offset /= 2) {
    val += __shfl_xor(val, offset);
  }
  return val;
}

template <typename T, typename AccT, int BLOCK_DIM, int N_READS = 4>
__global__ void logsumexp_kernel(const T* in, T* out, int axis_size) {
  int row = blockIdx.x;
  
  in += row * axis_size;

  // Thread reduce for max
  AccT maxval = -1e38f;
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      AccT val = static_cast<AccT>(in[i + j]);
      maxval = val > maxval ? val : maxval;
    }
  }

  // Block reduce for max
  __shared__ AccT shared_max[BLOCK_DIM / 64 + 1];
  
  AccT warp_max = warp_reduce_max_lse(maxval);
  int lane = threadIdx.x % 64;
  int warp_id = threadIdx.x / 64;
  
  if (lane == 0) {
    shared_max[warp_id] = warp_max;
  }
  __syncthreads();
  
  if (warp_id == 0) {
    maxval = (lane < (BLOCK_DIM + 63) / 64) ? shared_max[lane] : -1e38f;
    maxval = warp_reduce_max_lse(maxval);
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
    shared_max[0] = maxval;
  }
  __syncthreads();
  maxval = shared_max[0];

  // Thread reduce for sum of exp(x - max)
  AccT sumval = 0;
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      sumval += logsumexp_exp(static_cast<AccT>(in[i + j]) - maxval);
    }
  }

  // Block reduce for sum
  __shared__ AccT shared_sum[BLOCK_DIM / 64 + 1];
  
  AccT warp_sum = warp_reduce_sum_lse(sumval);
  
  if (lane == 0) {
    shared_sum[warp_id] = warp_sum;
  }
  __syncthreads();
  
  if (warp_id == 0) {
    sumval = (lane < (BLOCK_DIM + 63) / 64) ? shared_sum[lane] : 0;
    sumval = warp_reduce_sum_lse(sumval);
  }
  __syncthreads();
  
  // Write output
  if (threadIdx.x == 0) {
    if (isinf(maxval)) {
      out[row] = static_cast<T>(maxval);
    } else {
      out[row] = static_cast<T>(logf(sumval) + maxval);
    }
  }
}

} // namespace rocm

void LogSumExp::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  // Make sure that the last dimension is contiguous.
  auto ensure_contiguous = [&s, &encoder](const array& x) {
    if (x.flags().contiguous && x.strides()[x.ndim() - 1] == 1) {
      return x;
    } else {
      array x_copy = contiguous_copy_gpu(x, s);
      encoder.add_temporary(x_copy);
      return x_copy;
    }
  };

  auto in = ensure_contiguous(inputs[0]);
  if (in.flags().row_contiguous) {
    out.set_data(allocator::malloc(out.nbytes()));
  } else {
    auto n = in.shape(-1);
    auto flags = in.flags();
    auto strides = in.strides();
    for (auto& stride : strides) {
      stride /= n;
    }
    bool col_contig = strides[0] == 1;
    for (int i = 1; col_contig && i < strides.size(); ++i) {
      col_contig &=
          (out.shape(i) == 1 || strides[i - 1] == out.shape(i) * strides[i]);
    }
    flags.col_contiguous = col_contig;
    out.set_data(
        allocator::malloc(in.nbytes() / n),
        in.data_size() / n,
        std::move(strides),
        flags);
  }

  int axis_size = in.shape().back();
  int n_rows = in.data_size() / axis_size;

  encoder.set_input_array(in);
  encoder.set_output_array(out);
  
  constexpr int BLOCK_DIM = 256;
  constexpr int N_READS = 4;
  
  encoder.launch_kernel([&](hipStream_t stream) {
    switch (out.dtype()) {
      case float32:
        hipLaunchKernelGGL(
            (rocm::logsumexp_kernel<float, float, BLOCK_DIM, N_READS>),
            dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
            in.data<float>(), out.data<float>(), axis_size);
        break;
      case float16:
        hipLaunchKernelGGL(
            (rocm::logsumexp_kernel<__half, float, BLOCK_DIM, N_READS>),
            dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
            in.data<__half>(), out.data<__half>(), axis_size);
        break;
      case bfloat16:
        hipLaunchKernelGGL(
            (rocm::logsumexp_kernel<__hip_bfloat16, float, BLOCK_DIM, N_READS>),
            dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
            in.data<__hip_bfloat16>(), out.data<__hip_bfloat16>(), axis_size);
        break;
      default:
        throw std::runtime_error("Unsupported type for logsumexp");
    }
  });
}

} // namespace mlx::core
