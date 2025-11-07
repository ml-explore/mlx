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

instantiate_kernel(
    "paged_kv_write_float16", paged_kv_write, float16_t);
instantiate_kernel(
    "paged_kv_write_bfloat16", paged_kv_write, bfloat16_t);
instantiate_kernel("paged_kv_write_float32", paged_kv_write, float);
