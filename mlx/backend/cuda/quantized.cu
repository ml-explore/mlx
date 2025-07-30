// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/fast_primitives.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <nvtx3/nvtx3.hpp>

namespace mlx::core {
namespace cu {

namespace cg = cooperative_groups;

template <int bits, int wsize = 8>
inline constexpr __device__ short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr __device__ short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

template <typename T, int group_size, int bits>
__global__ void
affine_quantize(const T* w, uint8_t* out, T* scales, T* biases, size_t size) {
  auto block_size = cg::this_thread_block().dim_threads();
  auto block_idx = cg::this_thread_block().group_index();
  auto idx_in_block = cg::this_thread_block().thread_index();

  auto tidx = block_idx.x * block_size.x + idx_in_block.x;
  auto tidy = block_idx.y * block_size.y + idx_in_block.y;

  auto grid_dim_x =
      cg::this_grid().dim_blocks().x * cg::this_grid().block_index().x;
  constexpr float eps = 1e-7;
  constexpr int simd_size = WARP_SIZE;
  constexpr float n_bins = (1 << bits) - 1;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int values_per_reduce = group_size / simd_size;
  constexpr int writes_per_reduce = pack_factor / values_per_reduce;
  constexpr int writes_per_pack =
      writes_per_reduce > 1 ? 1 : values_per_reduce / pack_factor;
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;

  size_t offset = tidx + grid_dim_x * size_t(tidy);
  size_t in_index = offset * values_per_reduce;
  if (in_index >= size) {
    return;
  }
  size_t out_index = power_of_2_bits
      ? offset * writes_per_pack
      : offset * bytes_per_pack / writes_per_reduce;

  float w_thread[values_per_reduce];
  float w_min = Limits<float>::max();
  float w_max = 0;

#pragma clang loop unroll(full)
  for (int i = 0; i < values_per_reduce; i++) {
    float val = w[in_index + i];
    w_thread[i] = val;
    w_min = min(w_min, val);
    w_max = max(w_max, val);
  }

  cg::greater<float> max_op;
  cg::less<float> min_op;
  auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());

  w_min = cg::reduce(warp, w_min, min_op);
  w_max = cg::reduce(warp, w_max, max_op);

  float scale = max((w_max - w_min) / n_bins, eps);
  bool side = abs(w_min) > abs(w_max);
  scale = side ? scale : -scale;
  float edge = side ? w_min : w_max;
  float q0 = round(edge / scale);
  bool at_zero = q0 == 0.0f;
  scale = at_zero ? scale : edge / q0;
  float bias = at_zero ? 0 : edge;

  // Write out the scales and biases
  size_t gindex = in_index / group_size;
  if (in_index % group_size == 0) {
    scales[gindex] = static_cast<T>(scale);
    biases[gindex] = static_cast<T>(bias);
  }

  using OutType = std::conditional_t<bits == 5, uint64_t, uint32_t>;
  OutType output = 0;

#pragma clang loop unroll(full)
  for (int i = 0; i < values_per_reduce; i++) {
    uint8_t val = min(round((w_thread[i] - bias) / scale), n_bins);
    if (bits == 8) {
      output = val;
    } else {
      output |= val << (bits * (i % pack_factor));
    }

    if (pack_factor < values_per_reduce && i % pack_factor == pack_factor - 1) {
      out[out_index + i / pack_factor] = output;
      output = 0;
    } else {
#pragma clang loop unroll(full)
      for (int j = 1; j < writes_per_reduce; j++) {
        uint8_t sval = warp.shfl_down(val, j);
        output |= static_cast<OutType>(sval)
            << (bits * (j * values_per_reduce + i));
      }
    }
  }
  if constexpr (bits == 3 || bits == 6) {
    if (in_index % pack_factor == 0 && out_index % bytes_per_pack == 0) {
      out[out_index] = output & 0xff;
      out[out_index + 1] = (output & 0xff00) >> 8;
      out[out_index + 2] = (output & 0xff0000) >> 16;
    }
  } else if constexpr (bits == 5) {
    if (in_index % pack_factor == 0 && out_index % bytes_per_pack == 0) {
      out[out_index] = output & 0xff;
      out[out_index + 1] = (output & 0xff00) >> 8;
      out[out_index + 2] = (output & 0xff0000) >> 16;
      out[out_index + 3] = (output & 0xff000000) >> 24;
      out[out_index + 4] = (output & 0xff00000000) >> 32;
    }
  } else {
    if constexpr (writes_per_reduce > 0) {
      if (out_index % writes_per_reduce == 0) {
        out[out_index / writes_per_reduce] = output;
      }
    }
  }
}

template <typename T, int group_size, int bits>
__global__ void affine_dequantize(
    const uint8_t* w,
    const T* scales,
    const T* biases,
    T* out,
    size_t size) {
  auto block_size = cg::this_thread_block().dim_threads();
  auto block_idx = cg::this_thread_block().group_index();
  auto idx_in_block = cg::this_thread_block().thread_index();

  auto tidx = block_idx.x * block_size.x + idx_in_block.x;
  auto tidy = block_idx.y * block_size.y + idx_in_block.y;

  auto grid_dim_x =
      cg::this_grid().dim_blocks().x * cg::this_grid().block_index().x;

  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  size_t offset = tidx + grid_dim_x * size_t(tidy);
  size_t oindex = offset * pack_factor;

  if (oindex >= size) {
    return;
  }

  size_t gindex = oindex / group_size;
  T scale = scales[gindex];
  T bias = biases[gindex];
  out += oindex;

  if constexpr (bits == 3) {
    w += offset * bytes_per_pack;
    out[0] = static_cast<T>(w[0] & 0x7) * scale + bias;
    out[1] = static_cast<T>((w[0] & 0x38) >> 3) * scale + bias;
    out[2] = (static_cast<T>((w[0] & 0xc0) >> 6) +
              static_cast<T>((w[1] & 0x1) << 2)) *
            scale +
        bias;
    out[3] = static_cast<T>((w[1] & 0xe) >> 1) * scale + bias;
    out[4] = static_cast<T>((w[1] & 0x70) >> 4) * scale + bias;
    out[5] = (static_cast<T>((w[1] & 0x80) >> 7) +
              static_cast<T>((w[2] & 0x3) << 1)) *
            scale +
        bias;
    out[6] = static_cast<T>((w[2] & 0x1c) >> 2) * scale + bias;
    out[7] = static_cast<T>((w[2] & 0xe0) >> 5) * scale + bias;
  } else if constexpr (bits == 5) {
    w += offset * bytes_per_pack;
    out[0] = static_cast<T>(w[0] & 0x1f) * scale + bias;
    out[1] = (static_cast<T>((w[0] & 0xe0) >> 5) +
              static_cast<T>((w[1] & 0x3) << 3)) *
            scale +
        bias;
    out[2] = static_cast<T>((w[1] & 0x7c) >> 2) * scale + bias;
    out[3] = (static_cast<T>((w[1] & 0x80) >> 7) +
              static_cast<T>((w[2] & 0xf) << 1)) *
            scale +
        bias;
    out[4] = (static_cast<T>((w[2] & 0xf0) >> 4) +
              static_cast<T>((w[3] & 0x1) << 4)) *
            scale +
        bias;
    out[5] = static_cast<T>((w[3] & 0x3e) >> 1) * scale + bias;
    out[6] = (static_cast<T>((w[3] & 0xc0) >> 6) +
              static_cast<T>((w[4] & 0x7) << 2)) *
            scale +
        bias;
    out[7] = static_cast<T>((w[4] & 0xf8) >> 3) * scale + bias;
  } else if constexpr (bits == 6) {
    w += offset * bytes_per_pack;
    out[0] = static_cast<T>(w[0] & 0x3f) * scale + bias;
    out[1] = (static_cast<T>((w[0] >> 6) & 0x03) +
              static_cast<T>((w[1] & 0x0f) << 2)) *
            scale +
        bias;
    out[2] = (static_cast<T>((w[1] >> 4) & 0x0f) +
              static_cast<T>((w[2] & 0x03) << 4)) *
            scale +
        bias;
    out[3] = static_cast<T>((w[2] >> 2) & 0x3f) * scale + bias;
  } else {
    uint val = w[offset];
#pragma clang loop unroll(full)
    for (int i = 0; i < pack_factor; i++) {
      uint8_t d;
      if (bits == 2) {
        d = (val >> (bits * i)) & 0x03;
      } else if (bits == 4) {
        d = (val >> (bits * i)) & 0x0f;
      } else if (bits == 8) {
        d = val;
      }
      out[i] = scale * static_cast<T>(d) + bias;
    }
  }
}

} // namespace cu
namespace {

inline array ensure_row_contiguous(
    const array& x,
    cu::CommandEncoder& enc,
    const Stream& s) {
  if (!x.flags().row_contiguous) {
    array x_copy = contiguous_copy_gpu(x, s);
    enc.add_temporary(x_copy);
    return x_copy;
  } else {
    return x;
  }
}

} // namespace

template <typename F>
void dispatch_groups(int group_size, F&& f) {
  switch (group_size) {
    case 32:
      f(std::integral_constant<int, 32>{});
      break;
    case 64:
      f(std::integral_constant<int, 64>{});
      break;
    case 128:
      f(std::integral_constant<int, 128>{});
      break;
  }
}

template <typename F>
void dispatch_bits(int bits, F&& f) {
  switch (bits) {
    case 2:
      f(std::integral_constant<int, 2>{});
      break;
    case 3:
      f(std::integral_constant<int, 3>{});
      break;
    case 4:
      f(std::integral_constant<int, 4>{});
      break;
    case 5:
      f(std::integral_constant<int, 5>{});
      break;
    case 6:
      f(std::integral_constant<int, 6>{});
      break;
    case 8:
      f(std::integral_constant<int, 8>{});
      break;
  }
}

void fast::AffineQuantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& w_pre = inputs[0];
  auto& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));

  auto& s = stream();
  auto& d = cu::device(s.device);
  auto& enc = d.get_command_encoder(s);

  auto w = ensure_row_contiguous(w_pre, enc, s);
  enc.set_input_array(w);
  if (dequantize_) {
    auto scales = ensure_row_contiguous(inputs[1], enc, s);
    auto biases = ensure_row_contiguous(inputs[2], enc, s);
    enc.set_input_array(scales);
    enc.set_input_array(biases);
    enc.set_output_array(out);
  } else {
    auto& scales = outputs[1];
    auto& biases = outputs[2];
    scales.set_data(allocator::malloc(scales.nbytes()));
    biases.set_data(allocator::malloc(biases.nbytes()));
    enc.set_output_array(out);
    enc.set_output_array(scales);
    enc.set_output_array(biases);
  }

  auto dtype = dequantize_ ? outputs[0].dtype() : inputs[0].dtype();

  // Treat uint32 as uint8 in kernel
  int uint8_per_uint32 = 4;
  int packs_per_int = (bits_ == 3 || bits_ == 5) ? 8
      : bits_ == 6                               ? 4
                                                 : 8 / bits_;
  int per_thread = dequantize_ ? packs_per_int : group_size_ / WARP_SIZE;
  size_t size =
      dequantize_ ? out.size() / packs_per_int : w.size() / per_thread;

  bool large = size > UINT_MAX;
  auto grid_shape = w.shape();

  if (dequantize_) {
    grid_shape.back() *= uint8_per_uint32;
  } else {
    grid_shape.back() /= per_thread;
  }

  dispatch_float_types(dtype, "affine_quantize", [&](auto type_tag) {
    dispatch_groups(group_size_, [&](auto group_size) {
      dispatch_bits(bits_, [&](auto bits) {
        using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;
        if (dequantize_) {
          auto [num_blocks, block_dims] =
              get_launch_args(size, grid_shape, w.strides(), large);
          enc.add_kernel_node(
              cu::affine_dequantize<DataType, group_size.value, bits.value>,
              num_blocks,
              block_dims,
              w.data<uint8_t>(),
              inputs[1].data<DataType>(),
              inputs[2].data<DataType>(),
              out.data<DataType>(),
              out.size());
        } else {
          auto [num_blocks, block_dims] =
              get_launch_args(size, grid_shape, w.strides(), large);
          enc.add_kernel_node(
              cu::affine_quantize<DataType, group_size.value, bits.value>,
              num_blocks,
              block_dims,
              w.data<DataType>(),
              out.data<uint8_t>(),
              outputs[1].data<DataType>(),
              outputs[2].data<DataType>(),
              w.size());
        }
      });
    });
  });
}

} // namespace mlx::core
