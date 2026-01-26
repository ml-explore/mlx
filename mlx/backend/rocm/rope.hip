// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/fast_primitives.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

namespace mlx::core {

namespace rocm {

template <typename T>
__global__ void rope_kernel(
    const T* __restrict__ x,
    const T* __restrict__ cos_freq,
    const T* __restrict__ sin_freq,
    T* __restrict__ out,
    int offset,
    float scale,
    int n_heads,
    int head_dim,
    int seq_len,
    bool forward) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n_heads * seq_len * head_dim;
  
  if (idx >= total) return;
  
  int d = idx % head_dim;
  int s = (idx / head_dim) % seq_len;
  int h = idx / (head_dim * seq_len);
  
  // Only apply RoPE to the first half of dimensions
  int half_dim = head_dim / 2;
  if (d >= half_dim * 2) {
    out[idx] = x[idx];
    return;
  }
  
  int freq_idx = s * half_dim + (d % half_dim);
  float cos_val = static_cast<float>(cos_freq[freq_idx]);
  float sin_val = static_cast<float>(sin_freq[freq_idx]);
  
  float x_val = static_cast<float>(x[idx]);
  float result;
  
  if (d < half_dim) {
    // First half: x * cos - x_pair * sin
    int pair_idx = idx + half_dim;
    float x_pair = static_cast<float>(x[pair_idx]);
    if (forward) {
      result = x_val * cos_val - x_pair * sin_val;
    } else {
      result = x_val * cos_val + x_pair * sin_val;
    }
  } else {
    // Second half: x_pair * sin + x * cos
    int pair_idx = idx - half_dim;
    float x_pair = static_cast<float>(x[pair_idx]);
    if (forward) {
      result = x_pair * sin_val + x_val * cos_val;
    } else {
      result = -x_pair * sin_val + x_val * cos_val;
    }
  }
  
  out[idx] = static_cast<T>(result * scale);
}

} // namespace rocm

namespace fast {

bool RoPE::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void RoPE::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& out = outputs[0];
  
  const array& x = inputs[0];
  const array& cos_freq = inputs[1];
  const array& sin_freq = inputs[2];
  
  out.set_data(allocator::malloc(out.nbytes()));
  
  auto& encoder = rocm::get_command_encoder(s);
  
  int n_heads = x.shape(-3);
  int seq_len = x.shape(-2);
  int head_dim = x.shape(-1);
  int total = n_heads * seq_len * head_dim;
  
  int block_size = 256;
  int num_blocks = (total + block_size - 1) / block_size;
  
  encoder.launch_kernel([&](hipStream_t stream) {
    switch (x.dtype()) {
      case float32:
        hipLaunchKernelGGL(
            rocm::rope_kernel<float>,
            dim3(num_blocks), dim3(block_size), 0, stream,
            x.data<float>(), cos_freq.data<float>(), sin_freq.data<float>(),
            out.data<float>(), 0, scale_, n_heads, head_dim, seq_len, forward_);
        break;
      case float16:
        hipLaunchKernelGGL(
            rocm::rope_kernel<__half>,
            dim3(num_blocks), dim3(block_size), 0, stream,
            x.data<__half>(), cos_freq.data<__half>(), sin_freq.data<__half>(),
            out.data<__half>(), 0, scale_, n_heads, head_dim, seq_len, forward_);
        break;
      case bfloat16:
        hipLaunchKernelGGL(
            rocm::rope_kernel<hip_bfloat16>,
            dim3(num_blocks), dim3(block_size), 0, stream,
            x.data<hip_bfloat16>(), cos_freq.data<hip_bfloat16>(), sin_freq.data<hip_bfloat16>(),
            out.data<hip_bfloat16>(), 0, scale_, n_heads, head_dim, seq_len, forward_);
        break;
      default:
        throw std::runtime_error("Unsupported type for RoPE");
    }
  });
}

} // namespace fast

} // namespace mlx::core
