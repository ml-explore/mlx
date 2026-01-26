// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/quantized/quantized.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

namespace mlx::core {

namespace rocm {

template <typename T, typename ScaleT, int BITS>
__global__ void fp_quantize_kernel(
    const T* __restrict__ input,
    uint8_t* __restrict__ output,
    ScaleT* __restrict__ scales,
    int num_groups,
    int group_size) {
  int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (group_idx >= num_groups) return;
  
  const T* group_input = input + group_idx * group_size;
  
  // Find max absolute value in group (use float for computation)
  float max_abs = fabsf(static_cast<float>(group_input[0]));
  for (int i = 1; i < group_size; ++i) {
    max_abs = fmaxf(max_abs, fabsf(static_cast<float>(group_input[i])));
  }
  
  // Compute scale (symmetric quantization)
  float max_quant = static_cast<float>((1 << (BITS - 1)) - 1);
  float scale = max_abs / max_quant;
  
  // Avoid division by zero
  if (scale == 0.0f) {
    scale = 1.0f;
  }
  
  scales[group_idx] = static_cast<ScaleT>(scale);
  
  // Quantize values
  int output_idx = group_idx * (group_size * BITS / 8);
  uint8_t packed = 0;
  int bit_offset = 0;
  
  int8_t min_val = -(1 << (BITS - 1));
  int8_t max_val = (1 << (BITS - 1)) - 1;
  
  for (int i = 0; i < group_size; ++i) {
    float val = static_cast<float>(group_input[i]);
    int quant_val = static_cast<int>(roundf(val / scale));
    quant_val = max(static_cast<int>(min_val), min(static_cast<int>(max_val), quant_val));
    
    // Convert to unsigned for packing
    uint8_t uval = static_cast<uint8_t>(quant_val & ((1 << BITS) - 1));
    packed |= (uval << bit_offset);
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

template <typename T, typename ScaleT, int BITS>
__global__ void fp_dequantize_kernel(
    const uint8_t* __restrict__ input,
    const ScaleT* __restrict__ scales,
    T* __restrict__ output,
    int num_groups,
    int group_size) {
  int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (group_idx >= num_groups) return;
  
  float scale = static_cast<float>(scales[group_idx]);
  
  int input_idx = group_idx * (group_size * BITS / 8);
  T* group_output = output + group_idx * group_size;
  
  uint8_t mask = (1 << BITS) - 1;
  int bit_offset = 0;
  uint8_t packed = input[input_idx];
  
  int8_t sign_bit = 1 << (BITS - 1);
  
  for (int i = 0; i < group_size; ++i) {
    uint8_t uval = (packed >> bit_offset) & mask;
    
    // Convert back to signed
    int8_t quant_val;
    if (uval & sign_bit) {
      quant_val = static_cast<int8_t>(uval | ~mask);
    } else {
      quant_val = static_cast<int8_t>(uval);
    }
    
    group_output[i] = static_cast<T>(static_cast<float>(quant_val) * scale);
    
    bit_offset += BITS;
    if (bit_offset >= 8) {
      bit_offset = 0;
      packed = input[++input_idx];
    }
  }
}

// Optimized packed dequantize kernel
template <typename T, int BITS>
__global__ void fp_dequantize_packed_kernel(
    const uint8_t* __restrict__ input,
    const T* __restrict__ scales,
    T* __restrict__ output,
    size_t size,
    int group_size) {
  constexpr int pack_factor = 8 / BITS;
  
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t oindex = idx * pack_factor;
  
  if (oindex >= size) {
    return;
  }
  
  size_t gindex = oindex / group_size;
  float scale = static_cast<float>(scales[gindex]);
  
  uint8_t val = input[idx];
  uint8_t mask = (1 << BITS) - 1;
  uint8_t sign_bit = static_cast<uint8_t>(1 << (BITS - 1));
  
  #pragma unroll
  for (int i = 0; i < pack_factor; ++i) {
    uint8_t uval = (val >> (BITS * i)) & mask;
    
    // Convert to signed
    int8_t quant_val;
    if (uval & sign_bit) {
      quant_val = static_cast<int8_t>(uval | ~mask);
    } else {
      quant_val = static_cast<int8_t>(uval);
    }
    
    output[oindex + i] = static_cast<T>(static_cast<float>(quant_val) * scale);
  }
}

} // namespace rocm

void fp_quantize(
    const array& w,
    array& wq,
    array& scales,
    int group_size,
    int bits,
    rocm::CommandEncoder& enc,
    const Stream& s) {
  int num_elements = w.size();
  int num_groups = num_elements / group_size;
  
  int block_size = 256;
  int num_blocks = (num_groups + block_size - 1) / block_size;
  
  enc.set_input_array(w);
  enc.set_output_array(wq);
  enc.set_output_array(scales);
  
  enc.launch_kernel([&](hipStream_t stream) {
    #define LAUNCH_FP_QUANTIZE(T, ScaleT, BITS) \
      hipLaunchKernelGGL( \
          (rocm::fp_quantize_kernel<T, ScaleT, BITS>), \
          dim3(num_blocks), dim3(block_size), 0, stream, \
          w.data<T>(), wq.data<uint8_t>(), scales.data<ScaleT>(), \
          num_groups, group_size)
    
    #define DISPATCH_BITS(T, ScaleT) \
      switch (bits) { \
        case 2: LAUNCH_FP_QUANTIZE(T, ScaleT, 2); break; \
        case 4: LAUNCH_FP_QUANTIZE(T, ScaleT, 4); break; \
        case 8: LAUNCH_FP_QUANTIZE(T, ScaleT, 8); break; \
        default: throw std::runtime_error("Unsupported bits for fp_quantize"); \
      }
    
    switch (w.dtype()) {
      case float32:
        DISPATCH_BITS(float, float);
        break;
      case float16:
        DISPATCH_BITS(__half, __half);
        break;
      case bfloat16:
        DISPATCH_BITS(hip_bfloat16, hip_bfloat16);
        break;
      default:
        throw std::runtime_error("Unsupported dtype for fp_quantize");
    }
    
    #undef DISPATCH_BITS
    #undef LAUNCH_FP_QUANTIZE
  });
}

void fp_dequantize(
    const array& wq,
    const array& scales,
    array& w,
    int group_size,
    int bits,
    rocm::CommandEncoder& enc,
    const Stream& s) {
  
  enc.set_input_array(wq);
  enc.set_input_array(scales);
  enc.set_output_array(w);
  
  // Use packed kernel for power-of-2 bits
  if (bits == 2 || bits == 4 || bits == 8) {
    int pack_factor = 8 / bits;
    size_t size = w.size() / pack_factor;
    
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    
    enc.launch_kernel([&](hipStream_t stream) {
      #define LAUNCH_FP_DEQUANTIZE_PACKED(T, BITS) \
        hipLaunchKernelGGL( \
            (rocm::fp_dequantize_packed_kernel<T, BITS>), \
            dim3(num_blocks), dim3(block_size), 0, stream, \
            wq.data<uint8_t>(), scales.data<T>(), w.data<T>(), \
            w.size(), group_size)
      
      #define DISPATCH_BITS_PACKED(T) \
        switch (bits) { \
          case 2: LAUNCH_FP_DEQUANTIZE_PACKED(T, 2); break; \
          case 4: LAUNCH_FP_DEQUANTIZE_PACKED(T, 4); break; \
          case 8: LAUNCH_FP_DEQUANTIZE_PACKED(T, 8); break; \
          default: break; \
        }
      
      switch (w.dtype()) {
        case float32:
          DISPATCH_BITS_PACKED(float);
          break;
        case float16:
          DISPATCH_BITS_PACKED(__half);
          break;
        case bfloat16:
          DISPATCH_BITS_PACKED(hip_bfloat16);
          break;
        default:
          throw std::runtime_error("Unsupported dtype for fp_dequantize");
      }
      
      #undef DISPATCH_BITS_PACKED
      #undef LAUNCH_FP_DEQUANTIZE_PACKED
    });
  } else {
    // Fallback for non-power-of-2 bits
    int num_elements = w.size();
    int num_groups = num_elements / group_size;
    
    int block_size = 256;
    int num_blocks = (num_groups + block_size - 1) / block_size;
    
    enc.launch_kernel([&](hipStream_t stream) {
      #define LAUNCH_FP_DEQUANTIZE(T, ScaleT, BITS) \
        hipLaunchKernelGGL( \
            (rocm::fp_dequantize_kernel<T, ScaleT, BITS>), \
            dim3(num_blocks), dim3(block_size), 0, stream, \
            wq.data<uint8_t>(), scales.data<ScaleT>(), w.data<T>(), \
            num_groups, group_size)
      
      #define DISPATCH_BITS(T, ScaleT) \
        switch (bits) { \
          case 3: LAUNCH_FP_DEQUANTIZE(T, ScaleT, 3); break; \
          case 5: LAUNCH_FP_DEQUANTIZE(T, ScaleT, 5); break; \
          case 6: LAUNCH_FP_DEQUANTIZE(T, ScaleT, 6); break; \
          default: throw std::runtime_error("Unsupported bits for fp_dequantize"); \
        }
      
      switch (w.dtype()) {
        case float32:
          DISPATCH_BITS(float, float);
          break;
        case float16:
          DISPATCH_BITS(__half, __half);
          break;
        case bfloat16:
          DISPATCH_BITS(hip_bfloat16, hip_bfloat16);
          break;
        default:
          throw std::runtime_error("Unsupported dtype for fp_dequantize");
      }
      
      #undef DISPATCH_BITS
      #undef LAUNCH_FP_DEQUANTIZE
    });
  }
}

} // namespace mlx::core
