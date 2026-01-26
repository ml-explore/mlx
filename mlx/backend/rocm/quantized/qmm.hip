// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/quantized/quantized.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <rocblas/rocblas.h>

namespace mlx::core {

namespace {

inline array ensure_row_contiguous(
    const array& x,
    rocm::CommandEncoder& enc,
    const Stream& s) {
  if (!x.flags().row_contiguous) {
    array x_copy = contiguous_copy_gpu(x, s);
    enc.add_temporary(x_copy);
    return x_copy;
  } else {
    return x;
  }
}

inline array ensure_row_contiguous_matrix(
    const array& x,
    rocm::CommandEncoder& enc,
    const Stream& s) {
  if (x.ndim() < 2) {
    if (x.strides()[0] == 1) {
      return x;
    }
  } else {
    auto stride_0 = x.strides()[x.ndim() - 2];
    auto stride_1 = x.strides()[x.ndim() - 1];
    if (stride_0 == x.shape(-1) && stride_1 == 1) {
      return x;
    }
  }
  array x_copy = contiguous_copy_gpu(x, s);
  enc.add_temporary(x_copy);
  return x_copy;
}

} // namespace

namespace rocm {

// Quantized matrix-vector multiply kernel
// Performs: out = x @ dequantize(w, scales, biases)
// where w is quantized weights, scales and biases are per-group parameters
template <typename T, typename ScaleT, int BITS, int GROUP_SIZE>
__global__ void qmv_kernel(
    const T* __restrict__ x,           // [M, K]
    const uint8_t* __restrict__ w,     // [N, K/pack_factor] packed
    const ScaleT* __restrict__ scales, // [N, K/GROUP_SIZE]
    const ScaleT* __restrict__ biases, // [N, K/GROUP_SIZE] or nullptr
    T* __restrict__ out,               // [M, N]
    int M,
    int N,
    int K,
    bool has_bias) {
  
  constexpr int pack_factor = 8 / BITS;
  const int row = blockIdx.x;  // output row (M dimension)
  const int col = blockIdx.y * blockDim.x + threadIdx.x;  // output col (N dimension)
  
  if (row >= M || col >= N) return;
  
  float acc = 0.0f;
  
  int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  
  for (int g = 0; g < num_groups; ++g) {
    float scale = static_cast<float>(scales[col * num_groups + g]);
    float bias = has_bias ? static_cast<float>(biases[col * num_groups + g]) : 0.0f;
    
    int k_start = g * GROUP_SIZE;
    int k_end = min(k_start + GROUP_SIZE, K);
    
    for (int k = k_start; k < k_end; ++k) {
      // Get packed weight
      int pack_idx = k / pack_factor;
      int bit_offset = (k % pack_factor) * BITS;
      uint8_t packed = w[col * (K / pack_factor) + pack_idx];
      uint8_t mask = (1 << BITS) - 1;
      int8_t quant_val = static_cast<int8_t>((packed >> bit_offset) & mask);
      
      // Sign extend if needed
      if (quant_val & (1 << (BITS - 1))) {
        quant_val |= ~mask;
      }
      
      // Dequantize
      float w_val = static_cast<float>(quant_val) * scale + bias;
      
      // Accumulate
      acc += static_cast<float>(x[row * K + k]) * w_val;
    }
  }
  
  out[row * N + col] = static_cast<T>(acc);
}

// Transposed quantized matrix-vector multiply kernel
// Performs: out = x @ dequantize(w, scales, biases).T
template <typename T, typename ScaleT, int BITS, int GROUP_SIZE>
__global__ void qmv_t_kernel(
    const T* __restrict__ x,           // [M, K]
    const uint8_t* __restrict__ w,     // [K, N/pack_factor] packed (stored as [N, K/pack_factor] but accessed transposed)
    const ScaleT* __restrict__ scales, // [N, K/GROUP_SIZE]
    const ScaleT* __restrict__ biases, // [N, K/GROUP_SIZE] or nullptr
    T* __restrict__ out,               // [M, N]
    int M,
    int N,
    int K,
    bool has_bias) {
  
  constexpr int pack_factor = 8 / BITS;
  const int row = blockIdx.x;  // output row (M dimension)
  const int col = blockIdx.y * blockDim.x + threadIdx.x;  // output col (N dimension)
  
  if (row >= M || col >= N) return;
  
  float acc = 0.0f;
  
  int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  
  for (int g = 0; g < num_groups; ++g) {
    float scale = static_cast<float>(scales[col * num_groups + g]);
    float bias = has_bias ? static_cast<float>(biases[col * num_groups + g]) : 0.0f;
    
    int k_start = g * GROUP_SIZE;
    int k_end = min(k_start + GROUP_SIZE, K);
    
    for (int k = k_start; k < k_end; ++k) {
      // Get packed weight - note the transposed access pattern
      int pack_idx = k / pack_factor;
      int bit_offset = (k % pack_factor) * BITS;
      uint8_t packed = w[col * (K / pack_factor) + pack_idx];
      uint8_t mask = (1 << BITS) - 1;
      int8_t quant_val = static_cast<int8_t>((packed >> bit_offset) & mask);
      
      // Sign extend if needed
      if (quant_val & (1 << (BITS - 1))) {
        quant_val |= ~mask;
      }
      
      // Dequantize
      float w_val = static_cast<float>(quant_val) * scale + bias;
      
      // Accumulate
      acc += static_cast<float>(x[row * K + k]) * w_val;
    }
  }
  
  out[row * N + col] = static_cast<T>(acc);
}

} // namespace rocm

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = rocm::device(s.device);
  auto& enc = d.get_command_encoder(s);

  out.set_data(allocator::malloc(out.nbytes()));

  // Make sure the last two dims of x and w, s, b are contiguous
  array x = ensure_row_contiguous_matrix(inputs[0], enc, s);
  array w = ensure_row_contiguous_matrix(inputs[1], enc, s);
  array scales = ensure_row_contiguous_matrix(inputs[2], enc, s);
  std::optional<array> biases = std::nullopt;
  bool has_bias = (mode_ == QuantizationMode::Affine) && (inputs.size() == 4);
  if (has_bias) {
    biases = ensure_row_contiguous_matrix(inputs[3], enc, s);
  }

  enc.set_input_array(x);
  enc.set_input_array(w);
  enc.set_input_array(scales);
  if (has_bias) {
    enc.set_input_array(biases.value());
  }
  enc.set_output_array(out);

  // Extract the matmul shapes
  bool non_batched = w.ndim() == 2 && x.flags().row_contiguous;
  int K = x.shape(-1);
  int M = non_batched ? x.size() / K : x.shape(-2);
  int N = out.shape(-1);

  int block_size = 256;
  dim3 grid((M + 0) / 1, (N + block_size - 1) / block_size);
  grid.x = M;
  
  enc.launch_kernel([&](hipStream_t stream) {
    #define LAUNCH_QMV(T, ScaleT, BITS, GROUP_SIZE) \
      if (transpose_) { \
        hipLaunchKernelGGL( \
            (rocm::qmv_t_kernel<T, ScaleT, BITS, GROUP_SIZE>), \
            grid, dim3(block_size), 0, stream, \
            x.data<T>(), w.data<uint8_t>(), \
            scales.data<ScaleT>(), \
            has_bias ? biases->data<ScaleT>() : nullptr, \
            out.data<T>(), M, N, K, has_bias); \
      } else { \
        hipLaunchKernelGGL( \
            (rocm::qmv_kernel<T, ScaleT, BITS, GROUP_SIZE>), \
            grid, dim3(block_size), 0, stream, \
            x.data<T>(), w.data<uint8_t>(), \
            scales.data<ScaleT>(), \
            has_bias ? biases->data<ScaleT>() : nullptr, \
            out.data<T>(), M, N, K, has_bias); \
      }
    
    #define DISPATCH_GROUP_SIZE(T, ScaleT, BITS) \
      switch (group_size_) { \
        case 32: LAUNCH_QMV(T, ScaleT, BITS, 32); break; \
        case 64: LAUNCH_QMV(T, ScaleT, BITS, 64); break; \
        case 128: LAUNCH_QMV(T, ScaleT, BITS, 128); break; \
        default: throw std::runtime_error("Unsupported group_size for QuantizedMatmul: " + std::to_string(group_size_)); \
      }
    
    #define DISPATCH_BITS(T, ScaleT) \
      switch (bits_) { \
        case 2: DISPATCH_GROUP_SIZE(T, ScaleT, 2); break; \
        case 4: DISPATCH_GROUP_SIZE(T, ScaleT, 4); break; \
        case 8: DISPATCH_GROUP_SIZE(T, ScaleT, 8); break; \
        default: throw std::runtime_error("Unsupported bits for QuantizedMatmul: " + std::to_string(bits_)); \
      }
    
    switch (x.dtype()) {
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
        throw std::runtime_error("Unsupported dtype for QuantizedMatmul");
    }
    
    #undef DISPATCH_BITS
    #undef DISPATCH_GROUP_SIZE
    #undef LAUNCH_QMV
  });
}

// GatherQMM kernel - gather-based quantized matrix multiply
namespace rocm {

template <typename T, typename ScaleT, int BITS, int GROUP_SIZE>
__global__ void gather_qmv_kernel(
    const T* __restrict__ x,           // [B, M, K]
    const uint8_t* __restrict__ w,     // [E, N, K/pack_factor] packed
    const ScaleT* __restrict__ scales, // [E, N, K/GROUP_SIZE]
    const ScaleT* __restrict__ biases, // [E, N, K/GROUP_SIZE] or nullptr
    const uint32_t* __restrict__ lhs_indices, // [B]
    const uint32_t* __restrict__ rhs_indices, // [B]
    T* __restrict__ out,               // [B, M, N]
    int B,
    int M,
    int N,
    int K,
    int E,
    bool has_bias) {
  
  constexpr int pack_factor = 8 / BITS;
  
  int batch = blockIdx.z;
  int row = blockIdx.x;  // output row (M dimension)
  int col = blockIdx.y * blockDim.x + threadIdx.x;  // output col (N dimension)
  
  if (batch >= B || row >= M || col >= N) return;
  
  uint32_t lhs_idx = lhs_indices[batch];
  uint32_t rhs_idx = rhs_indices[batch];
  
  const T* x_ptr = x + lhs_idx * M * K + row * K;
  const uint8_t* w_ptr = w + rhs_idx * N * (K / pack_factor) + col * (K / pack_factor);
  const ScaleT* scales_ptr = scales + rhs_idx * N * ((K + GROUP_SIZE - 1) / GROUP_SIZE) + col * ((K + GROUP_SIZE - 1) / GROUP_SIZE);
  const ScaleT* biases_ptr = has_bias ? biases + rhs_idx * N * ((K + GROUP_SIZE - 1) / GROUP_SIZE) + col * ((K + GROUP_SIZE - 1) / GROUP_SIZE) : nullptr;
  
  float acc = 0.0f;
  
  int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  
  for (int g = 0; g < num_groups; ++g) {
    float scale = static_cast<float>(scales_ptr[g]);
    float bias = has_bias ? static_cast<float>(biases_ptr[g]) : 0.0f;
    
    int k_start = g * GROUP_SIZE;
    int k_end = min(k_start + GROUP_SIZE, K);
    
    for (int k = k_start; k < k_end; ++k) {
      // Get packed weight
      int pack_idx = k / pack_factor;
      int bit_offset = (k % pack_factor) * BITS;
      uint8_t packed = w_ptr[pack_idx];
      uint8_t mask = (1 << BITS) - 1;
      int8_t quant_val = static_cast<int8_t>((packed >> bit_offset) & mask);
      
      // Sign extend if needed
      if (quant_val & (1 << (BITS - 1))) {
        quant_val |= ~mask;
      }
      
      // Dequantize
      float w_val = static_cast<float>(quant_val) * scale + bias;
      
      // Accumulate
      acc += static_cast<float>(x_ptr[k]) * w_val;
    }
  }
  
  out[batch * M * N + row * N + col] = static_cast<T>(acc);
}

} // namespace rocm

void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = rocm::device(s.device);
  auto& enc = d.get_command_encoder(s);

  out.set_data(allocator::malloc(out.nbytes()));

  // Make sure the last two dims of x and w, s, b are contiguous
  array x = ensure_row_contiguous_matrix(inputs[0], enc, s);
  array w = ensure_row_contiguous_matrix(inputs[1], enc, s);
  array scales = ensure_row_contiguous_matrix(inputs[2], enc, s);
  std::optional<array> biases = std::nullopt;
  bool has_bias = (mode_ == QuantizationMode::Affine) && (inputs.size() == 6);
  if (has_bias) {
    biases = ensure_row_contiguous_matrix(inputs[3], enc, s);
  }
  const array& lhs_indices = inputs[inputs.size() - 2];
  const array& rhs_indices = inputs[inputs.size() - 1];

  enc.set_input_array(x);
  enc.set_input_array(w);
  enc.set_input_array(scales);
  if (has_bias) {
    enc.set_input_array(biases.value());
  }
  enc.set_input_array(lhs_indices);
  enc.set_input_array(rhs_indices);
  enc.set_output_array(out);

  // Extract the matmul shapes
  int K = x.shape(-1);
  int M = x.shape(-2);
  int N = out.shape(-1);
  int B = out.size() / M / N;
  int E = w.size() / w.shape(-1) / w.shape(-2);

  int block_size = 256;
  dim3 grid(M, (N + block_size - 1) / block_size, B);
  
  enc.launch_kernel([&](hipStream_t stream) {
    #define LAUNCH_GATHER_QMV(T, ScaleT, BITS, GROUP_SIZE) \
      hipLaunchKernelGGL( \
          (rocm::gather_qmv_kernel<T, ScaleT, BITS, GROUP_SIZE>), \
          grid, dim3(block_size), 0, stream, \
          x.data<T>(), w.data<uint8_t>(), \
          scales.data<ScaleT>(), \
          has_bias ? biases->data<ScaleT>() : nullptr, \
          lhs_indices.data<uint32_t>(), rhs_indices.data<uint32_t>(), \
          out.data<T>(), B, M, N, K, E, has_bias)
    
    #define DISPATCH_GROUP_SIZE_GATHER(T, ScaleT, BITS) \
      switch (group_size_) { \
        case 32: LAUNCH_GATHER_QMV(T, ScaleT, BITS, 32); break; \
        case 64: LAUNCH_GATHER_QMV(T, ScaleT, BITS, 64); break; \
        case 128: LAUNCH_GATHER_QMV(T, ScaleT, BITS, 128); break; \
        default: throw std::runtime_error("Unsupported group_size for GatherQMM: " + std::to_string(group_size_)); \
      }
    
    #define DISPATCH_BITS_GATHER(T, ScaleT) \
      switch (bits_) { \
        case 2: DISPATCH_GROUP_SIZE_GATHER(T, ScaleT, 2); break; \
        case 4: DISPATCH_GROUP_SIZE_GATHER(T, ScaleT, 4); break; \
        case 8: DISPATCH_GROUP_SIZE_GATHER(T, ScaleT, 8); break; \
        default: throw std::runtime_error("Unsupported bits for GatherQMM: " + std::to_string(bits_)); \
      }
    
    switch (x.dtype()) {
      case float32:
        DISPATCH_BITS_GATHER(float, float);
        break;
      case float16:
        DISPATCH_BITS_GATHER(__half, __half);
        break;
      case bfloat16:
        DISPATCH_BITS_GATHER(hip_bfloat16, hip_bfloat16);
        break;
      default:
        throw std::runtime_error("Unsupported dtype for GatherQMM");
    }
    
    #undef DISPATCH_BITS_GATHER
    #undef DISPATCH_GROUP_SIZE_GATHER
    #undef LAUNCH_GATHER_QMV
  });
}

} // namespace mlx::core
