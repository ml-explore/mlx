// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/quantized.h"
#include "mlx/backend/cuda/quantized/quantized_utils.cuh"
#include "mlx/dtype_utils.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace mlx::core {
namespace cu {

namespace cg = cooperative_groups;

template <typename T, int group_size>
__global__ void
ternary_quantize(const T* w, uint32_t* out, T* scales, size_t size) {
  // Quantize {-1,0,1} to 2-bit codes q = round(w/scale)+1 with per-group
  // scale=max|w|.
  constexpr int bits = 2;

  auto block_size = cg::this_thread_block().dim_threads();
  auto block_idx = cg::this_thread_block().group_index();
  auto idx_in_block = cg::this_thread_block().thread_index();

  auto tidx = block_idx.x * block_size.x + idx_in_block.x;
  auto tidy = block_idx.y * block_size.y + idx_in_block.y;

  auto grid_dim_x = cg::this_grid().dim_blocks().x * block_size.x;

  constexpr float eps = 1e-7;
  constexpr int simd_size = WARP_SIZE;
  constexpr int elements_per_uint = 32 / bits; // values packed into one uint32
  constexpr int values_per_reduce = group_size / simd_size;
  constexpr int threads_per_pack = elements_per_uint / values_per_reduce;

  static_assert(
      group_size % simd_size == 0,
      "Group size must be divisible by simd size.");
  static_assert(threads_per_pack > 0, "Threads per pack must be positive.");
  static_assert(
      elements_per_uint % values_per_reduce == 0,
      "elements_per_uint must be divisible by values_per_reduce.");
  static_assert(
      (threads_per_pack & (threads_per_pack - 1)) == 0,
      "Threads per pack must be power of 2.");

  const size_t offset = tidx + grid_dim_x * size_t(tidy);
  const size_t in_index = offset * values_per_reduce;
  if (in_index >= size) {
    return;
  }

  float w_thread[values_per_reduce];
  float w_max = 0.0f;

#pragma clang loop unroll(full)
  for (int i = 0; i < values_per_reduce; i++) {
    const float val = static_cast<float>(w[in_index + i]);
    w_thread[i] = val;
    w_max = max(w_max, abs(val));
  }

  // Group max and scale
  auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
  w_max = cg::reduce(warp, w_max, cg::greater<float>{});
  const float scale = max(w_max, eps);

  const auto lane = warp.thread_rank();
  if (lane == 0) {
    scales[in_index / group_size] = static_cast<T>(scale);
  }

  uint32_t packed = 0;

  const uint32_t start_bit =
      (lane % threads_per_pack) * (values_per_reduce * bits);

#pragma clang loop unroll(full)
  for (int i = 0; i < values_per_reduce; i++) {
    const uint32_t q =
        static_cast<uint32_t>(roundf(w_thread[i] / scale) + 1.0f);
    packed |= uq << (start_bit + (bits * i));
  }

#pragma clang loop unroll(full)
  for (uint32_t stride = 1; stride < threads_per_pack; stride <<= 1) {
    packed |= warp.shfl_xor(packed, stride);
  }

  if (lane % threads_per_pack == 0) {
    const size_t out_index = in_index / elements_per_uint;
    out[out_index] = packed;
  }
}

template <typename OutT, typename ScaleT, int group_size>
__global__ void ternary_dequantize(
    const uint8_t* w,
    const ScaleT* scales,
    OutT* out,
    size_t size) {
  constexpr int bits = 2;
  constexpr int pack_factor = 8 / bits; // values per packed byte

  auto block_size = cg::this_thread_block().dim_threads();
  auto block_idx = cg::this_thread_block().group_index();
  auto idx_in_block = cg::this_thread_block().thread_index();

  auto tidx = block_idx.x * block_size.x + idx_in_block.x;
  auto tidy = block_idx.y * block_size.y + idx_in_block.y;

  auto grid_dim_x = cg::this_grid().dim_blocks().x * block_size.x;

  const size_t offset = tidx + grid_dim_x * size_t(tidy);
  const size_t out_index = offset * pack_factor;
  if (out_index >= size) {
    return;
  }

  const float scale = static_cast<float>(scales[out_index / group_size]);
  const uint32_t val = w[offset];
  out += out_index;

#pragma clang loop unroll(full)
  for (int i = 0; i < pack_factor; i++) {
    const uint8_t d = (val >> (bits * i)) & 0x03u;
    out[i] = static_cast<OutT>(scale * (int(d) - 1));
  }
}

} // namespace cu

void ternary_quantize(
    const array& w,
    array& wq,
    array& scales,
    int group_size,
    int bits,
    cu::CommandEncoder& enc,
    const Stream& s) {
  if (group_size != 32 && group_size != 64 && group_size != 128) {
    throw std::runtime_error(
        "[Quantize::eval_gpu] CUDA ternary quantize only supports group_size in {32, 64, 128}.");
  }
  if (bits != 2) {
    throw std::runtime_error(
        "[Quantize::eval_gpu] CUDA ternary quantize only supports bits=2.");
  }

  enc.set_input_array(w);
  enc.set_output_array(wq);
  enc.set_output_array(scales);
  dispatch_float_types(w.dtype(), "ternary_quantize", [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    if constexpr (!std::is_same_v<T, double>) {
      dispatch_groups(group_size, [&](auto group_size_) {
        int per_thread = group_size_.value / WARP_SIZE;
        size_t nthreads = w.size() / per_thread;
        bool large = nthreads > UINT_MAX;

        auto grid_shape = w.shape();
        grid_shape.back() /= per_thread;

        auto [num_blocks, block_dims] =
            get_launch_args(nthreads, grid_shape, w.strides(), large);

        auto kernel = cu::ternary_quantize<T, group_size_.value>;
        enc.add_kernel_node(
            kernel,
            num_blocks,
            block_dims,
            0,
            gpu_ptr<T>(w),
            gpu_ptr<uint32_t>(wq),
            gpu_ptr<T>(scales),
            w.size());
      });
    } else {
      throw std::runtime_error(
          "[Quantize::eval_gpu] Can not quantize input with type float64.");
    }
  });
}

void ternary_dequantize(
    const array& wq,
    const array& scales,
    array& w,
    int group_size,
    int bits,
    cu::CommandEncoder& enc,
    const Stream& s) {
  if (group_size != 32 && group_size != 64 && group_size != 128) {
    throw std::runtime_error(
        "[Quantize::eval_gpu] CUDA ternary dequantize only supports group_size in {32, 64, 128}.");
  }
  if (bits != 2) {
    throw std::runtime_error(
        "[Quantize::eval_gpu] CUDA ternary dequantize only supports bits=2.");
  }

  constexpr int uint8_per_uint32 = 4;
  constexpr int packs_per_int = 4;

  size_t expected_wq_bytes = w.size() / packs_per_int;
  if (wq.nbytes() != expected_wq_bytes) {
    throw std::runtime_error(
        "[Quantize::eval_gpu] CUDA ternary dequantize expected wq to contain packed 2-bit values for the output size.");
  }

  size_t size = w.size() / packs_per_int;
  bool large = size > UINT_MAX;
  auto grid_shape = w.shape();
  grid_shape.back() *= uint8_per_uint32;

  enc.set_input_array(wq);
  enc.set_input_array(scales);
  enc.set_output_array(w);
  dispatch_float_types(w.dtype(), "ternary_dequantize", [&](auto out_tag) {
    using OutT = cuda_type_t<MLX_GET_TYPE(out_tag)>;
    if constexpr (!std::is_same_v<OutT, double>) {
      dispatch_float_types(scales.dtype(), "ternary_dequantize", [&](auto s_tag) {
        using ScaleT = cuda_type_t<MLX_GET_TYPE(s_tag)>;
        if constexpr (!std::is_same_v<ScaleT, double>) {
          dispatch_groups(group_size, [&](auto group_size_) {
            auto kernel =
                cu::ternary_dequantize<OutT, ScaleT, group_size_.value>;
            auto [num_blocks, block_dims] =
                get_launch_args(size, grid_shape, w.strides(), large);
            enc.add_kernel_node(
                kernel,
                num_blocks,
                block_dims,
                0,
                gpu_ptr<uint8_t>(wq),
                gpu_ptr<ScaleT>(scales),
                gpu_ptr<OutT>(w),
                w.size());
          });
        } else {
          throw std::runtime_error(
              "[Quantize::eval_gpu] Can not use float64 scales for ternary dequantize.");
        }
      });
    } else {
      throw std::runtime_error(
          "[Quantize::eval_gpu] Can not dequantize to output with type float64.");
    }
  });
}

} // namespace mlx::core
