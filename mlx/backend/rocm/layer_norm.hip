// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/reduce/reduce.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/fast_primitives.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

// Warp reduce for sum
__device__ float warp_reduce_sum_f(float val) {
  for (int offset = 32; offset > 0; offset /= 2) {
    val += __shfl_xor(val, offset);
  }
  return val;
}

template <typename T, int BLOCK_DIM, int N_READS = 4>
__global__ void layer_norm_kernel(
    const T* x,
    const T* w,
    const T* b,
    T* out,
    float eps,
    int32_t axis_size,
    int64_t w_stride,
    int64_t b_stride) {
  int row = blockIdx.x;
  
  x += row * axis_size;
  out += row * axis_size;

  // Sum for mean
  float sum = 0;
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      sum += static_cast<float>(x[i + j]);
    }
  }

  // Block reduce for sum
  __shared__ float shared_sum[BLOCK_DIM / 64 + 1];
  
  float warp_sum = warp_reduce_sum_f(sum);
  int lane = threadIdx.x % 64;
  int warp_id = threadIdx.x / 64;
  
  if (lane == 0) {
    shared_sum[warp_id] = warp_sum;
  }
  __syncthreads();
  
  if (warp_id == 0) {
    sum = (lane < (BLOCK_DIM + 63) / 64) ? shared_sum[lane] : 0;
    sum = warp_reduce_sum_f(sum);
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
    shared_sum[0] = sum;
  }
  __syncthreads();
  float mean = shared_sum[0] / axis_size;

  // Compute variance
  float var_sum = 0;
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      float t = static_cast<float>(x[i + j]) - mean;
      var_sum += t * t;
    }
  }

  // Block reduce for variance
  warp_sum = warp_reduce_sum_f(var_sum);
  
  if (lane == 0) {
    shared_sum[warp_id] = warp_sum;
  }
  __syncthreads();
  
  if (warp_id == 0) {
    var_sum = (lane < (BLOCK_DIM + 63) / 64) ? shared_sum[lane] : 0;
    var_sum = warp_reduce_sum_f(var_sum);
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
    shared_sum[0] = var_sum;
  }
  __syncthreads();
  float normalizer = rsqrtf(shared_sum[0] / axis_size + eps);

  // Write output
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      int idx = i + j;
      float norm = (static_cast<float>(x[idx]) - mean) * normalizer;
      float wi = (w_stride == 0) ? static_cast<float>(w[0]) : static_cast<float>(w[idx * w_stride]);
      float bi = (b_stride == 0) ? static_cast<float>(b[0]) : static_cast<float>(b[idx * b_stride]);
      out[idx] = static_cast<T>(wi * norm + bi);
    }
  }
}

} // namespace rocm

namespace fast {

bool LayerNorm::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void LayerNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& out = outputs[0];

  // Make sure that the last dimension is contiguous.
  auto set_output = [&s, &out](const array& x) {
    bool no_copy = x.flags().contiguous && x.strides()[x.ndim() - 1] == 1;
    if (no_copy && x.ndim() > 1) {
      auto s = x.strides()[x.ndim() - 2];
      no_copy &= (s == 0 || s == x.shape().back());
    }
    if (no_copy) {
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

  const array x = set_output(inputs[0]);
  const array& w = inputs[1];
  const array& b = inputs[2];

  int32_t axis_size = x.shape().back();
  int32_t n_rows = x.data_size() / axis_size;
  int64_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;
  int64_t b_stride = (b.ndim() == 1) ? b.strides()[0] : 0;

  auto& encoder = rocm::get_command_encoder(s);
  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(b);
  encoder.set_output_array(out);
  
  constexpr int BLOCK_DIM = 256;
  constexpr int N_READS = 4;
  
  encoder.launch_kernel([&](hipStream_t stream) {
    switch (out.dtype()) {
      case float32:
        hipLaunchKernelGGL(
            (rocm::layer_norm_kernel<float, BLOCK_DIM, N_READS>),
            dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
            x.data<float>(), w.data<float>(), b.data<float>(), out.data<float>(),
            eps_, axis_size, w_stride, b_stride);
        break;
      case float16:
        hipLaunchKernelGGL(
            (rocm::layer_norm_kernel<__half, BLOCK_DIM, N_READS>),
            dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
            x.data<__half>(), w.data<__half>(), b.data<__half>(), out.data<__half>(),
            eps_, axis_size, w_stride, b_stride);
        break;
      case bfloat16:
        hipLaunchKernelGGL(
            (rocm::layer_norm_kernel<__hip_bfloat16, BLOCK_DIM, N_READS>),
            dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
            x.data<__hip_bfloat16>(), w.data<__hip_bfloat16>(), b.data<__hip_bfloat16>(), out.data<__hip_bfloat16>(),
            eps_, axis_size, w_stride, b_stride);
        break;
      default:
        throw std::runtime_error("Unsupported type for layer_norm");
    }
  });
}

void LayerNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // For now, throw an error - VJP requires more complex implementation
  throw std::runtime_error("LayerNormVJP not yet implemented for ROCm");
}

} // namespace fast

} // namespace mlx::core
