// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/quantized/quantized.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

template <typename T, typename ScaleT, int GROUP_SIZE, int BITS>
__global__ void affine_quantize_kernel(
    const T* __restrict__ input,
    uint8_t* __restrict__ output,
    ScaleT* __restrict__ scales,
    ScaleT* __restrict__ biases,
    int num_groups,
    int group_size) {
  int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (group_idx >= num_groups) return;
  
  const T* group_input = input + group_idx * group_size;
  
  // Find min and max in group
  T min_val = group_input[0];
  T max_val = group_input[0];
  for (int i = 1; i < group_size; ++i) {
    T val = group_input[i];
    min_val = min(min_val, val);
    max_val = max(max_val, val);
  }
  
  // Compute scale and bias
  T range = max_val - min_val;
  T max_quant = static_cast<T>((1 << BITS) - 1);
  T scale = range / max_quant;
  T bias = min_val;
  
  // Avoid division by zero
  if (scale == T(0)) {
    scale = T(1);
  }
  
  scales[group_idx] = static_cast<ScaleT>(scale);
  biases[group_idx] = static_cast<ScaleT>(bias);
  
  // Quantize values
  int output_idx = group_idx * (group_size * BITS / 8);
  uint8_t packed = 0;
  int bit_offset = 0;
  
  for (int i = 0; i < group_size; ++i) {
    T val = group_input[i];
    int quant_val = static_cast<int>((val - bias) / scale + T(0.5));
    quant_val = max(0, min(static_cast<int>(max_quant), quant_val));
    
    packed |= (quant_val << bit_offset);
    bit_offset += BITS;
    
    if (bit_offset >= 8) {
      output[output_idx++] = packed;
      packed = 0;
      bit_offset = 0;
    }
  }
  
  if (bit_offset > 0) {
    output[output_idx] = packed;
  }
}

template <typename T, typename ScaleT, int GROUP_SIZE, int BITS>
__global__ void affine_dequantize_kernel(
    const uint8_t* __restrict__ input,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ output,
    int num_groups,
    int group_size) {
  int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (group_idx >= num_groups) return;
  
  T scale = static_cast<T>(scales[group_idx]);
  T bias = static_cast<T>(biases[group_idx]);
  
  int input_idx = group_idx * (group_size * BITS / 8);
  T* group_output = output + group_idx * group_size;
  
  uint8_t mask = (1 << BITS) - 1;
  int bit_offset = 0;
  uint8_t packed = input[input_idx];
  
  for (int i = 0; i < group_size; ++i) {
    int quant_val = (packed >> bit_offset) & mask;
    group_output[i] = static_cast<T>(quant_val) * scale + bias;
    
    bit_offset += BITS;
    if (bit_offset >= 8) {
      bit_offset = 0;
      packed = input[++input_idx];
    }
  }
}

} // namespace rocm

void affine_quantize(
    const array& w,
    array& wq,
    array& scales,
    array& biases,
    int group_size,
    int bits,
    rocm::CommandEncoder& enc,
    const Stream& s) {
  int num_elements = w.size();
  int num_groups = num_elements / group_size;
  
  int block_size = 256;
  int num_blocks = (num_groups + block_size - 1) / block_size;
  
  enc.launch_kernel([&](hipStream_t stream) {
    switch (w.dtype()) {
      case float32:
        if (bits == 4) {
          hipLaunchKernelGGL(
              (rocm::affine_quantize_kernel<float, float, 128, 4>),
              dim3(num_blocks), dim3(block_size), 0, stream,
              w.data<float>(), wq.data<uint8_t>(),
              scales.data<float>(), biases.data<float>(),
              num_groups, group_size);
        } else if (bits == 8) {
          hipLaunchKernelGGL(
              (rocm::affine_quantize_kernel<float, float, 128, 8>),
              dim3(num_blocks), dim3(block_size), 0, stream,
              w.data<float>(), wq.data<uint8_t>(),
              scales.data<float>(), biases.data<float>(),
              num_groups, group_size);
        }
        break;
      default:
        throw std::runtime_error("Unsupported dtype for affine_quantize");
    }
  });
}

void affine_dequantize(
    const array& wq,
    const array& scales,
    const array& biases,
    array& w,
    int group_size,
    int bits,
    rocm::CommandEncoder& enc,
    const Stream& s) {
  int num_elements = w.size();
  int num_groups = num_elements / group_size;
  
  int block_size = 256;
  int num_blocks = (num_groups + block_size - 1) / block_size;
  
  enc.launch_kernel([&](hipStream_t stream) {
    switch (w.dtype()) {
      case float32:
        if (bits == 4) {
          hipLaunchKernelGGL(
              (rocm::affine_dequantize_kernel<float, float, 128, 4>),
              dim3(num_blocks), dim3(block_size), 0, stream,
              wq.data<uint8_t>(), scales.data<float>(), biases.data<float>(),
              w.data<float>(), num_groups, group_size);
        } else if (bits == 8) {
          hipLaunchKernelGGL(
              (rocm::affine_dequantize_kernel<float, float, 128, 8>),
              dim3(num_blocks), dim3(block_size), 0, stream,
              wq.data<uint8_t>(), scales.data<float>(), biases.data<float>(),
              w.data<float>(), num_groups, group_size);
        }
        break;
      default:
        throw std::runtime_error("Unsupported dtype for affine_dequantize");
    }
  });
}

} // namespace mlx::core
