// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/quantized.h"
#include "mlx/dtype_utils.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

namespace mlx::core {
namespace cu {

template <int bits>
struct Quantize {
  __device__ uint8_t operator()(float x) {
    if constexpr (bits == 8) {
      return __nv_fp8_e4m3(x).__x;
    } else {
      return __nv_fp4_e2m1(x).__x;
    }
  }
};

template <int bits>
struct Dequantize {
  __device__ float operator()(uint8_t x) {
    if constexpr (bits == 8) {
      return float(*(__nv_fp8_e4m3*)(&x));
    } else {
      return float(*(__nv_fp4_e2m1*)(&x));
    }
  }
};

namespace cg = cooperative_groups;

template <typename T, int group_size, int bits, bool use_mx_scale>
__global__ void
fp_quantize(const T* w, uint8_t* out, uint8_t* scales, size_t size) {
  auto block_size = cg::this_thread_block().dim_threads();
  auto block_idx = cg::this_thread_block().group_index();
  auto idx_in_block = cg::this_thread_block().thread_index();

  auto tidx = block_idx.x * block_size.x + idx_in_block.x;
  auto tidy = block_idx.y * block_size.y + idx_in_block.y;

  auto grid_dim_x =
      cg::this_grid().dim_blocks().x * cg::this_grid().block_index().x;
  size_t index = tidx + grid_dim_x * size_t(tidy);
  if (index >= size) {
    return;
  }

  float w_thread = w[index];

  cg::greater<float> max_op;
  auto warp = cg::tiled_partition<group_size>(cg::this_thread_block());

  float scale = cg::reduce(warp, abs(w_thread), max_op);
  scale /= bits == 4 ? 6.0f : 448.0f;
  // Convert to mx scale or nv scale
  using ScaleType =
      std::conditional_t<use_mx_scale, __nv_fp8_e8m0, __nv_fp8_e4m3>;
  auto s = ScaleType(scale);
  uint8_t q_scale = s.__x;
  scale = float(s);

  // Write out the scales
  size_t gindex = index / group_size;
  if (index % group_size == 0) {
    scales[gindex] = q_scale;
  }

  uint8_t output = Quantize<bits>{}(scale == 0 ? 0.0f : w_thread / scale);
  if (bits == 4) {
    uint8_t sval = warp.shfl_down(output, 1);
    output |= sval << bits;
  }
  constexpr int pack_factor = bits == 8 ? 1 : 2;
  if (index % pack_factor == 0) {
    out[index / pack_factor] = output;
  }
}

template <typename T, int group_size, int bits, bool use_mx_scale>
__global__ void
fp_dequantize(const uint8_t* w, const uint8_t* scales, T* out, size_t size) {
  auto block_size = cg::this_thread_block().dim_threads();
  auto block_idx = cg::this_thread_block().group_index();
  auto idx_in_block = cg::this_thread_block().thread_index();

  auto tidx = block_idx.x * block_size.x + idx_in_block.x;
  auto tidy = block_idx.y * block_size.y + idx_in_block.y;

  auto grid_dim_x =
      cg::this_grid().dim_blocks().x * cg::this_grid().block_index().x;

  constexpr int pack_factor = bits == 8 ? 1 : 2;
  size_t offset = tidx + grid_dim_x * size_t(tidy);
  size_t oindex = offset * pack_factor;

  if (oindex >= size) {
    return;
  }

  size_t gindex = oindex / group_size;
  using ScaleType =
      std::conditional_t<use_mx_scale, __nv_fp8_e8m0, __nv_fp8_e4m3>;
  auto scale = float(((ScaleType*)(scales))[gindex]);

  out += oindex;

  uint val = w[offset];
#pragma clang loop unroll(full)
  for (int i = 0; i < pack_factor; i++) {
    uint8_t d;
    if (bits == 4) {
      d = (val >> (bits * i)) & 0x0f;
    } else if (bits == 8) {
      d = val;
    }
    out[i] = static_cast<T>(scale * Dequantize<bits>{}(d));
  }
}

} // namespace cu

void fp_quantize(
    const array& w,
    array& wq,
    array& scales,
    int group_size,
    int bits,
    cu::CommandEncoder& enc,
    const Stream& s) {
  enc.set_input_array(w);
  enc.set_output_array(wq);
  enc.set_output_array(scales);
  dispatch_float_types(w.dtype(), "fp_quantize", [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    if constexpr (!std::is_same_v<T, double>) {
      auto kernel = cu::fp_quantize<T, 32, 4, true>;
      if (bits == 8) {
        kernel = cu::fp_quantize<T, 32, 8, true>;
      } else if (group_size == 16) {
        kernel = cu::fp_quantize<T, 16, 4, false>;
      }
      bool large = w.size() > UINT_MAX;
      auto [num_blocks, block_dims] =
          get_launch_args(w.size(), w.shape(), w.strides(), large);
      enc.add_kernel_node(
          kernel,
          num_blocks,
          block_dims,
          0,
          gpu_ptr<T>(w),
          gpu_ptr<uint8_t>(wq),
          gpu_ptr<uint8_t>(scales),
          w.size());
    } else {
      throw std::runtime_error(
          "[Quantize::eval_gpu] Can not quantize input with type float64.");
    }
  });
}

void fp_dequantize(
    const array& wq,
    const array& scales,
    array& w,
    int group_size,
    int bits,
    cu::CommandEncoder& enc,
    const Stream& s) {
  constexpr int uint8_per_uint32 = 4;
  int packs_per_int = 8 / bits;

  size_t size = w.size() / packs_per_int;
  bool large = size > UINT_MAX;
  auto grid_shape = w.shape();
  grid_shape.back() *= uint8_per_uint32;

  enc.set_input_array(wq);
  enc.set_input_array(scales);
  enc.set_output_array(w);
  dispatch_float_types(w.dtype(), "fp_dequantize", [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    if constexpr (!std::is_same_v<T, double>) {
      auto kernel = cu::fp_dequantize<T, 32, 4, true>;
      if (bits == 8) {
        kernel = cu::fp_dequantize<T, 32, 8, true>;
      } else if (group_size == 16) {
        kernel = cu::fp_dequantize<T, 16, 4, false>;
      }
      auto [num_blocks, block_dims] =
          get_launch_args(size, grid_shape, w.strides(), large);
      enc.add_kernel_node(
          kernel,
          num_blocks,
          block_dims,
          0,
          gpu_ptr<uint8_t>(wq),
          gpu_ptr<uint8_t>(scales),
          gpu_ptr<T>(w),
          w.size());
    } else {
      throw std::runtime_error(
          "[Quantize::eval_gpu] Can not dequantize to output with type float64.");
    }
  });
}

} // namespace mlx::core
