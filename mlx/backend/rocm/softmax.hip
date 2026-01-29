// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/cast_op.hpp"
#include "mlx/backend/rocm/device/fp16_math.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>

#include <cassert>

namespace mlx::core {

namespace rocm {

template <typename T>
inline __device__ T softmax_exp(T x) {
  // Softmax doesn't need high precision exponential cause x is gonna be in
  // (-oo, 0] anyway and subsequently it will be divided by sum(exp(x_i)).
  if constexpr (std::is_same_v<T, float>) {
    return __expf(x);
  } else {
    return T(expf(static_cast<float>(x)));
  }
}

// Warp reduce for max
template <typename T>
__device__ T warp_reduce_max(T val) {
  for (int offset = 32; offset > 0; offset /= 2) {
    float fval = static_cast<float>(val);
    float other = __shfl_xor(fval, offset);
    val = fval > other ? val : T(other);
  }
  return val;
}

// Warp reduce for sum
template <typename T>
__device__ T warp_reduce_sum(T val) {
  for (int offset = 32; offset > 0; offset /= 2) {
    float fval = static_cast<float>(val);
    float other = __shfl_xor(fval, offset);
    val = T(fval + other);
  }
  return val;
}

template <typename T, typename AccT, int BLOCK_DIM, int N_READS = 4>
__global__ void softmax_kernel(const T* in, T* out, int axis_size) {
  int row = blockIdx.x;
  
  in += row * axis_size;
  out += row * axis_size;

  // Thread reduce for max
  AccT maxval = AccT(-1e38f);  // Very small number
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      AccT val = static_cast<AccT>(in[i + j]);
      maxval = val > maxval ? val : maxval;
    }
  }

  // Block reduce for max
  __shared__ AccT shared_max[BLOCK_DIM / 64 + 1];
  
  AccT warp_max = warp_reduce_max(maxval);
  int lane = threadIdx.x % 64;
  int warp_id = threadIdx.x / 64;
  
  if (lane == 0) {
    shared_max[warp_id] = warp_max;
  }
  __syncthreads();
  
  if (warp_id == 0) {
    maxval = (lane < (BLOCK_DIM + 63) / 64) ? shared_max[lane] : AccT(-1e38f);
    maxval = warp_reduce_max(maxval);
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
    shared_max[0] = maxval;
  }
  __syncthreads();
  maxval = shared_max[0];

  // Thread reduce for sum of exp(x - max)
  AccT sumval = AccT(0);
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      sumval += softmax_exp(static_cast<AccT>(in[i + j]) - maxval);
    }
  }

  // Block reduce for sum
  __shared__ AccT shared_sum[BLOCK_DIM / 64 + 1];
  
  AccT warp_sum = warp_reduce_sum(sumval);
  
  if (lane == 0) {
    shared_sum[warp_id] = warp_sum;
  }
  __syncthreads();
  
  if (warp_id == 0) {
    sumval = (lane < (BLOCK_DIM + 63) / 64) ? shared_sum[lane] : AccT(0);
    sumval = warp_reduce_sum(sumval);
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
    shared_sum[0] = sumval;
  }
  __syncthreads();
  AccT normalizer = AccT(1.0f) / shared_sum[0];

  // Write output
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      out[i + j] = static_cast<T>(softmax_exp(static_cast<AccT>(in[i + j]) - maxval) * normalizer);
    }
  }
}

} // namespace rocm

void Softmax::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& s = stream();

  // Make sure that the last dimension is contiguous.
  auto set_output = [&s, &out](const array& x) {
    if (x.flags().contiguous && x.strides()[x.ndim() - 1] == 1) {
      if (x.is_donatable()) {
        out.copy_shared_buffer(x);
      } else {
        out.set_data(
            allocator::malloc(x.data_size() * x.itemsize()),
            x.data_size(),
            x.strides(),
            x.flags());
      }
      return x;
    } else {
      array x_copy = contiguous_copy_gpu(x, s);
      out.copy_shared_buffer(x_copy);
      return x_copy;
    }
  };

  array in = set_output(inputs[0]);
  bool precise = in.dtype() != float32 && precise_;

  int axis_size = in.shape().back();
  int n_rows = in.data_size() / axis_size;

  auto& encoder = rocm::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  
  constexpr int BLOCK_DIM = 256;
  constexpr int N_READS = 4;
  
  encoder.launch_kernel([&](hipStream_t stream) {
    switch (out.dtype()) {
      case float32:
        hipLaunchKernelGGL(
            (rocm::softmax_kernel<float, float, BLOCK_DIM, N_READS>),
            dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
            in.data<float>(), out.data<float>(), axis_size);
        break;
      case float16:
        if (precise) {
          hipLaunchKernelGGL(
              (rocm::softmax_kernel<__half, float, BLOCK_DIM, N_READS>),
              dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
              in.data<__half>(), out.data<__half>(), axis_size);
        } else {
          hipLaunchKernelGGL(
              (rocm::softmax_kernel<__half, __half, BLOCK_DIM, N_READS>),
              dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
              in.data<__half>(), out.data<__half>(), axis_size);
        }
        break;
      case bfloat16:
        if (precise) {
          hipLaunchKernelGGL(
              (rocm::softmax_kernel<hip_bfloat16, float, BLOCK_DIM, N_READS>),
              dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
              in.data<hip_bfloat16>(), out.data<hip_bfloat16>(), axis_size);
        } else {
          hipLaunchKernelGGL(
              (rocm::softmax_kernel<hip_bfloat16, hip_bfloat16, BLOCK_DIM, N_READS>),
              dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
              in.data<hip_bfloat16>(), out.data<hip_bfloat16>(), axis_size);
        }
        break;
      default:
        throw std::runtime_error("Unsupported type for softmax");
    }
  });
}

} // namespace mlx::core
 