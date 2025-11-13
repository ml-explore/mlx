// ABOUTME: Metal kernel for chunked KV cache writes.
// ABOUTME: Copies prompt chunks into paged blocks on device.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

struct PagedKVWriteParams {
  uint head_dim;
  uint block_size;
  uint chunk_tokens;
  uint num_kv_heads;
  uint chunk_token_stride;
  uint chunk_head_stride;
  uint kv_head_stride;
  uint block_stride;
  uint row_stride;
};

struct PagedKVQuantParams {
  uint group_size;
  uint groups_per_head;
  uint bits;
  uint values_per_byte;
  uint bytes_per_token;
  uint symmetric;
  uint vq_head_stride;
  uint vq_block_stride;
  uint vq_row_stride;
  uint scale_head_stride;
  uint scale_block_stride;
  uint scale_row_stride;
};

template <typename T>
kernel void paged_kv_write(
    device T* k_cache [[buffer(0)]],
    device T* v_cache [[buffer(1)]],
    device const int* block_row [[buffer(2)]],
    constant uint& start_pos [[buffer(3)]],
    device const T* k_src [[buffer(4)]],
    device const T* v_src [[buffer(5)]],
    constant PagedKVWriteParams& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  const uint token_idx = gid / params.num_kv_heads;
  const uint kv_idx = gid % params.num_kv_heads;
  if (token_idx >= params.chunk_tokens) {
    return;
  }

  const uint logical_pos = start_pos + token_idx;
  const uint block_idx = logical_pos / params.block_size;
  const uint row = logical_pos % params.block_size;

  const int block_id = block_row[block_idx];
  if (block_id < 0) {
    return;
  }

  device T* k_dst = k_cache + kv_idx * params.kv_head_stride +
      block_id * params.block_stride + row * params.row_stride;
  device T* v_dst = v_cache + kv_idx * params.kv_head_stride +
      block_id * params.block_stride + row * params.row_stride;

  const device T* k_ptr = k_src + token_idx * params.chunk_token_stride +
      kv_idx * params.chunk_head_stride;
  const device T* v_ptr = v_src + token_idx * params.chunk_token_stride +
      kv_idx * params.chunk_head_stride;

  for (uint d = 0; d < params.head_dim; ++d) {
    k_dst[d] = k_ptr[d];
    v_dst[d] = v_ptr[d];
  }
}

instantiate_kernel("paged_kv_write_float16", paged_kv_write, float16_t);
instantiate_kernel("paged_kv_write_bfloat16", paged_kv_write, bfloat16_t);
instantiate_kernel("paged_kv_write_float32", paged_kv_write, float);

template <typename T, typename ScaleT>
kernel void paged_kv_write_quantized(
    device T* k_cache [[buffer(0)]],
    device T* v_cache [[buffer(1)]],
    device uchar* vq_cache [[buffer(2)]],
    device ScaleT* scale_cache [[buffer(3)]],
    device ScaleT* zero_cache [[buffer(4)]],
    device const int* block_row [[buffer(5)]],
    constant uint& start_pos [[buffer(6)]],
    device const T* k_src [[buffer(7)]],
    device const T* v_src [[buffer(8)]],
    constant PagedKVWriteParams& params [[buffer(9)]],
    constant PagedKVQuantParams& quant_params [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
  const uint token_idx = gid / params.num_kv_heads;
  const uint kv_idx = gid % params.num_kv_heads;
  if (token_idx >= params.chunk_tokens) {
    return;
  }

  const uint logical_pos = start_pos + token_idx;
  const uint block_idx = logical_pos / params.block_size;
  const uint row = logical_pos % params.block_size;

  const int block_id = block_row[block_idx];
  if (block_id < 0) {
    return;
  }

  device T* k_dst = k_cache + kv_idx * params.kv_head_stride +
      block_id * params.block_stride + row * params.row_stride;
  device T* v_dst = v_cache + kv_idx * params.kv_head_stride +
      block_id * params.block_stride + row * params.row_stride;

  const device T* k_ptr = k_src + token_idx * params.chunk_token_stride +
      kv_idx * params.chunk_head_stride;
  const device T* v_ptr = v_src + token_idx * params.chunk_token_stride +
      kv_idx * params.chunk_head_stride;

  for (uint d = 0; d < params.head_dim; ++d) {
    k_dst[d] = k_ptr[d];
    v_dst[d] = v_ptr[d];
  }

  device uchar* vq_dst = vq_cache + kv_idx * quant_params.vq_head_stride +
      block_id * quant_params.vq_block_stride +
      row * quant_params.vq_row_stride;
  device ScaleT* scale_dst = scale_cache +
      kv_idx * quant_params.scale_head_stride +
      block_id * quant_params.scale_block_stride +
      row * quant_params.scale_row_stride;
  device ScaleT* zero_dst = zero_cache +
      kv_idx * quant_params.scale_head_stride +
      block_id * quant_params.scale_block_stride +
      row * quant_params.scale_row_stride;

  const float levels = float((1u << quant_params.bits) - 1u);
  const float offset = float(1u << (quant_params.bits - 1u));
  uint byte_index = 0;

  for (uint group = 0; group < quant_params.groups_per_head; ++group) {
    const uint group_offset = group * quant_params.group_size;
    float min_val = Limits<float>::max;
    float max_val = Limits<float>::min;

    for (uint i = 0; i < quant_params.group_size; ++i) {
      const uint feature_idx = group_offset + i;
      float value = 0.0f;
      if (feature_idx < params.head_dim) {
        value = float(v_ptr[feature_idx]);
      }
      min_val = metal::fmin(min_val, value);
      max_val = metal::fmax(max_val, value);
    }

    float scale;
    float zero;
    if (quant_params.symmetric) {
      const float max_abs =
          metal::fmax(metal::fabs(min_val), metal::fabs(max_val));
      if (max_abs > 0.0f) {
        scale = max_abs / offset;
      } else {
        scale = 1.0f;
      }
      zero = offset;
    } else {
      const float span = max_val - min_val;
      if (span > 0.0f) {
        scale = span / levels;
        zero = metal::rint(-min_val / scale);
      } else {
        scale = 1.0f;
        zero = -min_val;
      }
    }

    scale_dst[group] = ScaleT(scale);
    zero_dst[group] = ScaleT(zero);

    const uint bytes_per_group =
        (quant_params.group_size * quant_params.bits + 7u) / 8u;
    uint local_byte_index = 0;
    uchar current_byte = 0;

    for (uint i = 0; i < quant_params.group_size; ++i) {
      const uint feature_idx = group_offset + i;
      float value = 0.0f;
      if (feature_idx < params.head_dim) {
        value = float(v_ptr[feature_idx]);
      }
      float q_val = scale > 0.0f ? (value / scale) + zero : zero;
      float q_round = metal::rint(q_val);
      q_round = metal::clamp(q_round, 0.0f, levels);
      const uint q_int = static_cast<uint>(q_round);

      if (quant_params.bits == 8) {
        vq_dst[byte_index + local_byte_index] = uchar(q_int & 0xFFu);
        local_byte_index += 1;
      } else {
        if ((i & 1u) == 0u) {
          current_byte = uchar(q_int & 0x0Fu);
          if (i == quant_params.group_size - 1u) {
            vq_dst[byte_index + local_byte_index] = current_byte;
            local_byte_index += 1;
          }
        } else {
          current_byte |= uchar((q_int & 0x0Fu) << 4u);
          vq_dst[byte_index + local_byte_index] = current_byte;
          local_byte_index += 1;
        }
      }
    }

    for (; local_byte_index < bytes_per_group; ++local_byte_index) {
      vq_dst[byte_index + local_byte_index] = uchar(0);
    }
    byte_index += bytes_per_group;
  }
}

instantiate_kernel(
    "paged_kv_write_quantized_float16_float16",
    paged_kv_write_quantized,
    float16_t,
    float16_t);
instantiate_kernel(
    "paged_kv_write_quantized_bfloat16_float16",
    paged_kv_write_quantized,
    bfloat16_t,
    float16_t);
instantiate_kernel(
    "paged_kv_write_quantized_float32_float16",
    paged_kv_write_quantized,
    float,
    float16_t);
