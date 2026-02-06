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

// Warp reduce for max - use runtime warpSize
template <typename T>
__device__ T warp_reduce_max_lse(T val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    T other = __shfl_xor(val, offset);
    val = val > other ? val : other;
  }
  return val;
}

// Warp reduce for sum - use runtime warpSize
template <typename T>
__device__ T warp_reduce_sum_lse(T val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_xor(val, offset);
  }
  return val;
}

template <typename T, typename AccT, int BLOCK_DIM, int N_READS = 4>
__global__ void logsumexp_kernel(const T* in, T* out, int axis_size) {
  int row = blockIdx.x;
  
  in += row * axis_size;

  // Thread reduce for max
  AccT prevmax;
  AccT maxval = -1e38f;
  AccT normalizer = 0;
  
  for (int r = 0; r < (axis_size + BLOCK_DIM * N_READS - 1) / (BLOCK_DIM * N_READS); r++) {
    int base_idx = r * BLOCK_DIM * N_READS + threadIdx.x * N_READS;
    prevmax = maxval;
    
    #pragma unroll
    for (int j = 0; j < N_READS; ++j) {
      int idx = base_idx + j;
      if (idx < axis_size) {
        AccT val = static_cast<AccT>(in[idx]);
        maxval = val > maxval ? val : maxval;
      }
    }
    
    // Online normalizer calculation
    normalizer = normalizer * logsumexp_exp(prevmax - maxval);
    #pragma unroll
    for (int j = 0; j < N_READS; ++j) {
      int idx = base_idx + j;
      if (idx < axis_size) {
        normalizer += logsumexp_exp(static_cast<AccT>(in[idx]) - maxval);
      }
    }
  }

  // Block reduce for max using shared memory
  __shared__ AccT shared_max[32];  // Max 32 warps
  __shared__ AccT shared_norm[32];
  
  int lane = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  int num_warps = (BLOCK_DIM + warpSize - 1) / warpSize;
  
  // First warp reduce
  prevmax = maxval;
  maxval = warp_reduce_max_lse(maxval);
  normalizer = normalizer * logsumexp_exp(prevmax - maxval);
  normalizer = warp_reduce_sum_lse(normalizer);
  
  if (lane == 0) {
    shared_max[warp_id] = maxval;
    shared_norm[warp_id] = normalizer;
  }
  __syncthreads();
  
  // Second warp reduce (only first warp)
  if (warp_id == 0) {
    prevmax = maxval;
    maxval = (lane < num_warps) ? shared_max[lane] : -1e38f;
    maxval = warp_reduce_max_lse(maxval);
    
    normalizer = (lane < num_warps) ? shared_norm[lane] : 0;
    normalizer = normalizer * logsumexp_exp(prevmax - maxval);
    normalizer = warp_reduce_sum_lse(normalizer);
  }
  
  // Write output
  if (threadIdx.x == 0) {
    if (isinf(maxval)) {
      out[row] = static_cast<T>(maxval);
    } else {
      out[row] = static_cast<T>(logf(normalizer) + maxval);
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
            (rocm::logsumexp_kernel<hip_bfloat16, float, BLOCK_DIM, N_READS>),
            dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
            in.data<hip_bfloat16>(), out.data<hip_bfloat16>(), axis_size);
        break;
      default:
        throw std::runtime_error("Unsupported type for logsumexp");
    }
  });
}

} // namespace mlx::core
 