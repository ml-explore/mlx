// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/fast_primitives.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

// FP8 E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits
// Range: [-448, 448], no inf, has NaN

template <typename T>
__device__ uint8_t float_to_fp8_e4m3(T val) {
  float f = static_cast<float>(val);
  
  // Handle special cases
  if (isnan(f)) {
    return 0x7F;  // NaN in E4M3
  }
  
  uint32_t bits = __float_as_uint(f);
  uint32_t sign = (bits >> 31) & 0x1;
  int32_t exp = ((bits >> 23) & 0xFF) - 127;  // Unbias from float
  uint32_t mant = bits & 0x7FFFFF;
  
  // Clamp to E4M3 range
  if (exp < -9) {  // Underflow to zero
    return sign << 7;
  }
  if (exp > 8) {  // Overflow to max
    return (sign << 7) | 0x7E;  // Max normal value
  }
  
  // Rebias for E4M3 (bias = 7)
  int32_t new_exp = exp + 7;
  
  // Round mantissa to 3 bits (round to nearest, ties to even)
  // We're discarding 20 bits, so add 0.5 ULP = 1 << 19 = 0x80000
  uint32_t new_mant = (mant + 0x80000) >> 20;
  if (new_mant > 7) {
    new_mant = 0;
    new_exp++;
    if (new_exp > 15) {
      return (sign << 7) | 0x7E;  // Overflow
    }
  }
  
  if (new_exp <= 0) {
    // Denormal handling
    int shift = 1 - new_exp;
    new_mant = ((mant | 0x800000) >> (20 + shift));
    new_exp = 0;
  }
  
  return (sign << 7) | ((new_exp & 0xF) << 3) | (new_mant & 0x7);
}

template <typename T>
__device__ T fp8_e4m3_to_float(uint8_t val) {
  uint32_t sign = (val >> 7) & 0x1;
  uint32_t exp = (val >> 3) & 0xF;
  uint32_t mant = val & 0x7;
  
  float result;
  if (exp == 0) {
    if (mant == 0) {
      result = 0.0f;
    } else {
      // Denormal: value = mant * 2^(-9)
      result = ldexpf(static_cast<float>(mant), -9);
    }
  } else if (exp == 15 && mant == 7) {
    // NaN
    result = __uint_as_float(0x7FC00000);
  } else {
    // Normal: value = (1 + mant/8) * 2^(exp-7)
    uint32_t float_exp = exp - 7 + 127;
    uint32_t float_mant = mant << 20;
    uint32_t bits = (sign << 31) | (float_exp << 23) | float_mant;
    result = __uint_as_float(bits);
  }
  
  return static_cast<T>(sign ? -fabsf(result) : result);
}

template <typename InT, typename OutT>
__global__ void to_fp8_kernel(const InT* in, OutT* out, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) return;
  
  out[idx] = float_to_fp8_e4m3(in[idx]);
}

template <typename InT, typename OutT>
__global__ void from_fp8_kernel(const InT* in, OutT* out, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) return;
  
  out[idx] = fp8_e4m3_to_float<OutT>(in[idx]);
}

} // namespace rocm

void fast::ConvertFP8::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = rocm::device(s.device);
  auto& enc = d.get_command_encoder(s);
  
  const auto& in = inputs[0];
  auto& out = outputs[0];
  
  out.set_data(allocator::malloc(out.nbytes()));
  
  size_t size = in.size();
  int block_size = 256;
  int num_blocks = (size + block_size - 1) / block_size;
  
  enc.launch_kernel([&](hipStream_t stream) {
    if (to_fp8_) {
      // Convert to FP8
      switch (in.dtype()) {
        case float32:
          hipLaunchKernelGGL(
              (rocm::to_fp8_kernel<float, uint8_t>),
              dim3(num_blocks), dim3(block_size), 0, stream,
              in.data<float>(), out.data<uint8_t>(), size);
          break;
        case float16:
          hipLaunchKernelGGL(
              (rocm::to_fp8_kernel<__half, uint8_t>),
              dim3(num_blocks), dim3(block_size), 0, stream,
              in.data<__half>(), out.data<uint8_t>(), size);
          break;
        case bfloat16:
          hipLaunchKernelGGL(
              (rocm::to_fp8_kernel<hip_bfloat16, uint8_t>),
              dim3(num_blocks), dim3(block_size), 0, stream,
              in.data<hip_bfloat16>(), out.data<uint8_t>(), size);
          break;
        default:
          throw std::runtime_error("Unsupported input type for ConvertFP8 (to_fp8)");
      }
    } else {
      // Convert from FP8
      switch (out.dtype()) {
        case float32:
          hipLaunchKernelGGL(
              (rocm::from_fp8_kernel<uint8_t, float>),
              dim3(num_blocks), dim3(block_size), 0, stream,
              in.data<uint8_t>(), out.data<float>(), size);
          break;
        case float16:
          hipLaunchKernelGGL(
              (rocm::from_fp8_kernel<uint8_t, __half>),
              dim3(num_blocks), dim3(block_size), 0, stream,
              in.data<uint8_t>(), out.data<__half>(), size);
          break;
        case bfloat16:
          hipLaunchKernelGGL(
              (rocm::from_fp8_kernel<uint8_t, hip_bfloat16>),
              dim3(num_blocks), dim3(block_size), 0, stream,
              in.data<uint8_t>(), out.data<hip_bfloat16>(), size);
          break;
        default:
          throw std::runtime_error("Unsupported output type for ConvertFP8 (from_fp8)");
      }
    }
  });
}

} // namespace mlx::core
