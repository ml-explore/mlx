// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/quantized/quantized.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

template <typename T, typename ScaleT, int BITS>
__global__ void affine_quantize_kernel(
    const T* __restrict__ input,
    uint8_t* __restrict__ output,
    ScaleT* __restrict__ scales,
    ScaleT* __restrict__ biases,
    int num_groups,
    int group_size) {
  int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (group_idx >= num_groups)
    return;

  const T* group_input = input + group_idx * group_size;

  // Find min and max in group
  float min_val = static_cast<float>(group_input[0]);
  float max_val = static_cast<float>(group_input[0]);
  for (int i = 1; i < group_size; ++i) {
    float val = static_cast<float>(group_input[i]);
    min_val = fminf(min_val, val);
    max_val = fmaxf(max_val, val);
  }

  // Compute scale and bias
  float range = max_val - min_val;
  float max_quant = static_cast<float>((1 << BITS) - 1);
  float scale = range / max_quant;
  float bias = min_val;

  // Avoid division by zero
  if (scale == 0.0f) {
    scale = 1.0f;
  }

  scales[group_idx] = static_cast<ScaleT>(scale);
  biases[group_idx] = static_cast<ScaleT>(bias);

  // Quantize values
  int output_idx = group_idx * (group_size * BITS / 8);
  int group_bytes = group_size * BITS / 8;
  constexpr uint8_t mask = static_cast<uint8_t>((1u << BITS) - 1u);

  for (int i = 0; i < group_bytes; ++i) {
    output[output_idx + i] = 0;
  }

  for (int i = 0; i < group_size; ++i) {
    float val = static_cast<float>(group_input[i]);
    int quant_val = static_cast<int>((val - bias) / scale + 0.5f);
    quant_val = max(0, min(static_cast<int>(max_quant), quant_val));

    int bit_index = i * BITS;
    int byte_idx = output_idx + (bit_index >> 3);
    int bit_offset = bit_index & 0x7;
    uint32_t shifted =
        static_cast<uint32_t>(static_cast<uint8_t>(quant_val) & mask)
        << bit_offset;

    output[byte_idx] |= static_cast<uint8_t>(shifted & 0xFF);
    if (bit_offset + BITS > 8) {
      output[byte_idx + 1] |= static_cast<uint8_t>((shifted >> 8) & 0xFF);
    }
  }
}

template <typename T, typename ScaleT, int BITS>
__global__ void affine_dequantize_kernel(
    const uint8_t* __restrict__ input,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ output,
    int num_groups,
    int group_size) {
  int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (group_idx >= num_groups)
    return;

  float scale = static_cast<float>(scales[group_idx]);
  float bias = biases ? static_cast<float>(biases[group_idx]) : 0.0f;

  int input_base = group_idx * (group_size * BITS / 8);
  T* group_output = output + group_idx * group_size;
  constexpr uint8_t mask = static_cast<uint8_t>((1u << BITS) - 1u);

  for (int i = 0; i < group_size; ++i) {
    int bit_index = i * BITS;
    int byte_idx = input_base + (bit_index >> 3);
    int bit_offset = bit_index & 0x7;

    uint32_t packed = static_cast<uint32_t>(input[byte_idx]);
    if (bit_offset + BITS > 8) {
      packed |= static_cast<uint32_t>(input[byte_idx + 1]) << 8;
    }

    int quant_val = static_cast<int>((packed >> bit_offset) & mask);
    float dequant_val = static_cast<float>(quant_val) * scale + bias;
    group_output[i] = static_cast<T>(dequant_val);
  }
}

// Optimized dequantize kernel for pack_factor elements at a time.
// RDNA 3.5 (gfx1151) with hipcc 7.13 / LLVM 23: Avoid #pragma unroll — the
// compiler emits incorrectly optimized vectorized stores that corrupt output.
// Use explicit scalar stores instead (same root cause as the uint4 load fix
// in qdequant.hpp).
template <typename T, int BITS>
__global__ void affine_dequantize_packed_kernel(
    const uint8_t* __restrict__ input,
    const T* __restrict__ scales,
    const T* __restrict__ biases,
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
  float bias = biases ? static_cast<float>(biases[gindex]) : 0.0f;

  uint8_t val = input[idx];

  // Manual unroll with explicit scalar stores — avoids LLVM 23 codegen bug
  // on RDNA 3.5 that corrupted #pragma unroll vectorized stores.
  for (int i = 0; i < pack_factor; ++i) {
    if (oindex + i >= size)
      break;
    uint8_t d;
    if constexpr (BITS == 2) {
      d = (val >> (BITS * i)) & 0x03;
    } else if constexpr (BITS == 4) {
      d = (val >> (BITS * i)) & 0x0f;
    } else if constexpr (BITS == 8) {
      d = val;
    }
    output[oindex + i] = static_cast<T>(scale * static_cast<float>(d) + bias);
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

  enc.set_input_array(w);
  enc.set_output_array(wq);
  enc.set_output_array(scales);
  enc.set_output_array(biases);

  enc.launch_kernel([&](hipStream_t stream) {
#define LAUNCH_QUANTIZE(T, ScaleT, BITS)               \
  hipLaunchKernelGGL(                                  \
      (rocm::affine_quantize_kernel<T, ScaleT, BITS>), \
      dim3(num_blocks),                                \
      dim3(block_size),                                \
      0,                                               \
      stream,                                          \
      w.data<T>(),                                     \
      wq.data<uint8_t>(),                              \
      scales.data<ScaleT>(),                           \
      biases.data<ScaleT>(),                           \
      num_groups,                                      \
      group_size)

#define DISPATCH_BITS(T, ScaleT)                                        \
  switch (bits) {                                                       \
    case 2:                                                             \
      LAUNCH_QUANTIZE(T, ScaleT, 2);                                    \
      break;                                                            \
    case 3:                                                             \
      LAUNCH_QUANTIZE(T, ScaleT, 3);                                    \
      break;                                                            \
    case 4:                                                             \
      LAUNCH_QUANTIZE(T, ScaleT, 4);                                    \
      break;                                                            \
    case 5:                                                             \
      LAUNCH_QUANTIZE(T, ScaleT, 5);                                    \
      break;                                                            \
    case 6:                                                             \
      LAUNCH_QUANTIZE(T, ScaleT, 6);                                    \
      break;                                                            \
    case 8:                                                             \
      LAUNCH_QUANTIZE(T, ScaleT, 8);                                    \
      break;                                                            \
    default:                                                            \
      throw std::runtime_error("Unsupported bits for affine_quantize"); \
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
        throw std::runtime_error("Unsupported dtype for affine_quantize");
    }

#undef DISPATCH_BITS
#undef LAUNCH_QUANTIZE
  });
}

void affine_dequantize(
    const array& wq,
    const array& scales,
    const std::optional<array>& biases,
    array& w,
    int group_size,
    int bits,
    rocm::CommandEncoder& enc,
    const Stream& s) {
  enc.set_input_array(wq);
  enc.set_input_array(scales);
  if (biases)
    enc.set_input_array(*biases);
  enc.set_output_array(w);

  // Use packed kernel for power-of-2 bits
  if (bits == 2 || bits == 4 || bits == 8) {
    int pack_factor = 8 / bits;
    size_t size = w.size() / pack_factor;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    enc.launch_kernel([&](hipStream_t stream) {
#define LAUNCH_DEQUANTIZE_PACKED(T, BITS)               \
  hipLaunchKernelGGL(                                   \
      (rocm::affine_dequantize_packed_kernel<T, BITS>), \
      dim3(num_blocks),                                 \
      dim3(block_size),                                 \
      0,                                                \
      stream,                                           \
      wq.data<uint8_t>(),                               \
      scales.data<T>(),                                 \
      biases ? biases->data<T>() : nullptr,             \
      w.data<T>(),                                      \
      w.size(),                                         \
      group_size)

#define DISPATCH_BITS_PACKED(T)       \
  switch (bits) {                     \
    case 2:                           \
      LAUNCH_DEQUANTIZE_PACKED(T, 2); \
      break;                          \
    case 4:                           \
      LAUNCH_DEQUANTIZE_PACKED(T, 4); \
      break;                          \
    case 8:                           \
      LAUNCH_DEQUANTIZE_PACKED(T, 8); \
      break;                          \
    default:                          \
      break;                          \
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
          throw std::runtime_error("Unsupported dtype for affine_dequantize");
      }

#undef DISPATCH_BITS_PACKED
#undef LAUNCH_DEQUANTIZE_PACKED
    });
  } else {
    // Fallback for non-power-of-2 bits (3, 5, 6)
    int num_elements = w.size();
    int num_groups = num_elements / group_size;

    int block_size = 256;
    int num_blocks = (num_groups + block_size - 1) / block_size;

    enc.launch_kernel([&](hipStream_t stream) {
#define LAUNCH_DEQUANTIZE(T, ScaleT, BITS)               \
  hipLaunchKernelGGL(                                    \
      (rocm::affine_dequantize_kernel<T, ScaleT, BITS>), \
      dim3(num_blocks),                                  \
      dim3(block_size),                                  \
      0,                                                 \
      stream,                                            \
      wq.data<uint8_t>(),                                \
      scales.data<ScaleT>(),                             \
      biases ? biases->data<ScaleT>() : nullptr,         \
      w.data<T>(),                                       \
      num_groups,                                        \
      group_size)

#define DISPATCH_BITS(T, ScaleT)                                          \
  switch (bits) {                                                         \
    case 3:                                                               \
      LAUNCH_DEQUANTIZE(T, ScaleT, 3);                                    \
      break;                                                              \
    case 5:                                                               \
      LAUNCH_DEQUANTIZE(T, ScaleT, 5);                                    \
      break;                                                              \
    case 6:                                                               \
      LAUNCH_DEQUANTIZE(T, ScaleT, 6);                                    \
      break;                                                              \
    default:                                                              \
      throw std::runtime_error("Unsupported bits for affine_dequantize"); \
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
          throw std::runtime_error("Unsupported dtype for affine_dequantize");
      }

#undef DISPATCH_BITS
#undef LAUNCH_DEQUANTIZE
    });
  }
}

} // namespace mlx::core
