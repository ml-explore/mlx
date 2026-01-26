// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/quantized.h"
#include "mlx/backend/cuda/quantized/quantized_utils.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/primitives.h"

#include <cooperative_groups.h>
#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace {

inline array ensure_row_contiguous_matrix(
    const array& x,
    cu::CommandEncoder& enc,
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

namespace cu {

namespace cg = cooperative_groups;

// Transposed quantized matrix-vector multiply kernel for affine quantization
// Performs: out = x @ dequantize(w, scales, biases).T
// where w is stored as [N, K/pack_factor] (weights stored with output dim
// first) scales/biases are stored as [N, K/GROUP_SIZE]
template <typename T, int BITS, int GROUP_SIZE, bool HAS_BIAS>
__global__ void qmv_t_kernel(
    const T* __restrict__ x, // [M, K]
    const uint8_t* __restrict__ w, // [N, K/pack_factor] packed
    const T* __restrict__ scales, // [N, K/GROUP_SIZE]
    const T* __restrict__ biases, // [N, K/GROUP_SIZE] or nullptr
    T* __restrict__ out, // [M, N]
    int M,
    int N,
    int K) {
  constexpr int pack_factor = get_pack_factor<BITS, 8>();
  constexpr uint8_t mask = (1 << BITS) - 1;

  const int row = blockIdx.x; // output row (M dimension)
  const int col =
      blockIdx.y * blockDim.x + threadIdx.x; // output col (N dimension)

  if (row >= M || col >= N)
    return;

  float acc = 0.0f;

  int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  int packed_K = (K + pack_factor - 1) / pack_factor;

  for (int g = 0; g < num_groups; ++g) {
    float scale = static_cast<float>(scales[col * num_groups + g]);
    float bias =
        HAS_BIAS ? static_cast<float>(biases[col * num_groups + g]) : 0.0f;

    int k_start = g * GROUP_SIZE;
    int k_end = min(k_start + GROUP_SIZE, K);

    for (int k = k_start; k < k_end; ++k) {
      // Get packed weight
      int pack_idx = k / pack_factor;
      int bit_offset = (k % pack_factor) * BITS;
      uint8_t packed = w[col * packed_K + pack_idx];
      uint8_t quant_val = (packed >> bit_offset) & mask;

      // Dequantize (affine: val * scale + bias)
      float w_val = static_cast<float>(quant_val) * scale + bias;

      // Accumulate
      acc += static_cast<float>(x[row * K + k]) * w_val;
    }
  }

  out[row * N + col] = static_cast<T>(acc);
}

// Non-transposed quantized matrix-vector multiply kernel
// Performs: out = x @ dequantize(w, scales, biases)
// For transpose=false, quantization packs along the N (output) dimension:
// - w is stored as [K, N_packed] where N_packed = N * bits / 8
// - scales/biases are stored as [K, N/GROUP_SIZE]
// - Groups are along the N dimension, not K
template <typename T, int BITS, int GROUP_SIZE, bool HAS_BIAS>
__global__ void qmv_n_kernel(
    const T* __restrict__ x, // [M, K]
    const uint8_t* __restrict__ w, // [K, N_packed] packed along N
    const T* __restrict__ scales, // [K, N/GROUP_SIZE]
    const T* __restrict__ biases, // [K, N/GROUP_SIZE] or nullptr
    T* __restrict__ out, // [M, N]
    int M,
    int N,
    int K) {
  constexpr int pack_factor = get_pack_factor<BITS, 8>();
  constexpr uint8_t mask = (1 << BITS) - 1;

  const int row = blockIdx.x; // output row (M dimension)
  const int col =
      blockIdx.y * blockDim.x + threadIdx.x; // output col (N dimension)

  if (row >= M || col >= N)
    return;

  float acc = 0.0f;

  // For non-transposed, groups are along N, not K
  int num_n_groups = (N + GROUP_SIZE - 1) / GROUP_SIZE;
  int n_group = col / GROUP_SIZE;
  int packed_N = (N + pack_factor - 1) / pack_factor;

  for (int k = 0; k < K; ++k) {
    // scales/biases are [K, N/GROUP_SIZE], indexed by [k, n_group]
    float scale = static_cast<float>(scales[k * num_n_groups + n_group]);
    float bias =
        HAS_BIAS ? static_cast<float>(biases[k * num_n_groups + n_group]) : 0.0f;

    // w is [K, N_packed], packed along N
    int pack_idx = col / pack_factor;
    int bit_offset = (col % pack_factor) * BITS;
    uint8_t packed = w[k * packed_N + pack_idx];
    uint8_t quant_val = (packed >> bit_offset) & mask;

    // Dequantize
    float w_val = static_cast<float>(quant_val) * scale + bias;

    // Accumulate
    acc += static_cast<float>(x[row * K + k]) * w_val;
  }

  out[row * N + col] = static_cast<T>(acc);
}

} // namespace cu

namespace {

// Local dispatch for power-of-2 bits only (2, 4, 8)
// Non-power-of-2 bits (3, 5, 6) require multi-byte unpacking which is not yet
// implemented
template <typename F>
void dispatch_qmm_bits(int bits, F&& f) {
  switch (bits) {
    case 2:
      f(std::integral_constant<int, 2>{});
      break;
    case 4:
      f(std::integral_constant<int, 4>{});
      break;
    case 8:
      f(std::integral_constant<int, 8>{});
      break;
    default:
      throw std::runtime_error(
          "[QuantizedMatmul::eval_gpu] bits must be 2, 4, or 8 (got " +
          std::to_string(bits) + ")");
  }
}

} // namespace

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("QuantizedMatmul::eval_gpu");
  auto& s = stream();
  auto& d = cu::device(s.device);
  auto& enc = d.get_command_encoder(s);

  // Validate: only affine quantization mode is supported
  if (mode_ != QuantizationMode::Affine) {
    throw std::runtime_error(
        "[QuantizedMatmul::eval_gpu] Only affine quantization mode is "
        "supported on CUDA. Non-affine modes (e.g., fp quantization with "
        "uint8 scales) are not yet implemented.");
  }

  // Validate: batched weights are not supported
  if (inputs[1].ndim() > 2) {
    throw std::runtime_error(
        "[QuantizedMatmul::eval_gpu] Batched weights (w.ndim() > 2) are not "
        "supported on CUDA. Please use 2D weight matrices.");
  }

  out.set_data(cu::malloc_async(out.nbytes(), enc));

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

  // Validate: only supported group sizes (32, 64, 128)
  if (group_size_ != 32 && group_size_ != 64 && group_size_ != 128) {
    throw std::runtime_error(
        "[QuantizedMatmul::eval_gpu] group_size must be 32, 64, or 128 (got " +
        std::to_string(group_size_) + ")");
  }

  // Extract the matmul shapes
  // Flatten all leading dimensions of x into the row dimension M.
  // This matches the kernel's assumption that x is laid out as a single [M, K] matrix.
  int K = x.shape(-1);
  int M = x.size() / K;
  int N = out.shape(-1);

  int block_size = 256;
  dim3 grid(M, (N + block_size - 1) / block_size);

  dispatch_float_types(x.dtype(), "QuantizedMatmul", [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    dispatch_groups(group_size_, [&](auto group_size) {
      dispatch_qmm_bits(bits_, [&](auto bits) {
        constexpr int GROUP_SIZE = group_size.value;
        constexpr int BITS = bits.value;

        if (has_bias) {
          if (transpose_) {
            auto kernel = cu::qmv_t_kernel<T, BITS, GROUP_SIZE, true>;
            enc.add_kernel_node(
                kernel,
                grid,
                dim3(block_size),
                0,
                gpu_ptr<T>(x),
                gpu_ptr<uint8_t>(w),
                gpu_ptr<T>(scales),
                gpu_ptr<T>(biases.value()),
                gpu_ptr<T>(out),
                M,
                N,
                K);
          } else {
            auto kernel = cu::qmv_n_kernel<T, BITS, GROUP_SIZE, true>;
            enc.add_kernel_node(
                kernel,
                grid,
                dim3(block_size),
                0,
                gpu_ptr<T>(x),
                gpu_ptr<uint8_t>(w),
                gpu_ptr<T>(scales),
                gpu_ptr<T>(biases.value()),
                gpu_ptr<T>(out),
                M,
                N,
                K);
          }
        } else {
          if (transpose_) {
            auto kernel = cu::qmv_t_kernel<T, BITS, GROUP_SIZE, false>;
            enc.add_kernel_node(
                kernel,
                grid,
                dim3(block_size),
                0,
                gpu_ptr<T>(x),
                gpu_ptr<uint8_t>(w),
                gpu_ptr<T>(scales),
                static_cast<T*>(nullptr),
                gpu_ptr<T>(out),
                M,
                N,
                K);
          } else {
            auto kernel = cu::qmv_n_kernel<T, BITS, GROUP_SIZE, false>;
            enc.add_kernel_node(
                kernel,
                grid,
                dim3(block_size),
                0,
                gpu_ptr<T>(x),
                gpu_ptr<uint8_t>(w),
                gpu_ptr<T>(scales),
                static_cast<T*>(nullptr),
                gpu_ptr<T>(out),
                M,
                N,
                K);
          }
        }
      });
    });
  });
}

} // namespace mlx::core
