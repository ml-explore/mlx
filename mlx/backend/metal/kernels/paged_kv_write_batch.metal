// ABOUTME: Metal kernel that writes a batch of single-token KV pairs.
// ABOUTME: Appends decode tokens for multiple sequences in parallel.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

struct PagedKVWriteBatchParams {
  uint batch_size;
  uint layers;
  uint head_dim;
  uint block_size;
  uint max_blocks;
  uint max_blocks_per_seq;
  uint num_kv_heads;
  uint kv_head_stride;
  uint layer_stride;
  uint block_stride;
  uint row_stride;
  uint batch_head_stride;
  uint batch_seq_stride;
  uint batch_layer_stride;
  uint table_stride;
};

struct PagedKVWriteLayersTokensParams {
  uint batch_size;
  uint layers;
  uint head_dim;
  uint block_size;
  uint max_blocks;
  uint max_blocks_per_seq;
  uint num_kv_heads;
  uint kv_head_stride;
  uint layer_stride;
  uint block_stride;
  uint row_stride;
  uint tokens;
  uint token_layer_stride;
  uint token_step_stride;
  uint token_batch_stride;
  uint token_head_stride;
  uint token_dim_stride;
  uint table_stride;
};

template <typename T>
kernel void paged_kv_write_batch(
    device T* k_cache [[buffer(0)]],
    device T* v_cache [[buffer(1)]],
    device const int* block_tables [[buffer(2)]],
    device const int* context_lens [[buffer(3)]],
    device const T* k_src [[buffer(4)]],
    device const T* v_src [[buffer(5)]],
    constant PagedKVWriteBatchParams& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  const uint dim_idx = gid % params.head_dim;
  const uint kv_idx = (gid / params.head_dim) % params.num_kv_heads;
  const uint seq_idx = gid / (params.head_dim * params.num_kv_heads);
  if (seq_idx >= params.batch_size) {
    return;
  }

  const uint logical_pos = static_cast<uint>(context_lens[seq_idx]);
  const uint block_idx = logical_pos / params.block_size;
  const uint row = logical_pos % params.block_size;
  if (block_idx >= params.max_blocks_per_seq) {
    return;
  }

  const device int* table = block_tables + seq_idx * params.table_stride;
  const int block_id = table[block_idx];
  if (block_id < 0) {
    return;
  }

  device T* k_dst = k_cache + kv_idx * params.kv_head_stride +
      block_id * params.block_stride + row * params.row_stride;
  device T* v_dst = v_cache + kv_idx * params.kv_head_stride +
      block_id * params.block_stride + row * params.row_stride;

  const device T* k_ptr = k_src + seq_idx * params.batch_seq_stride +
      kv_idx * params.batch_head_stride;
  const device T* v_ptr = v_src + seq_idx * params.batch_seq_stride +
      kv_idx * params.batch_head_stride;

  k_dst[dim_idx] = k_ptr[dim_idx];
  v_dst[dim_idx] = v_ptr[dim_idx];
}

instantiate_kernel(
    "paged_kv_write_batch_float16_t",
    paged_kv_write_batch,
    float16_t);
instantiate_kernel(
    "paged_kv_write_batch_bfloat16_t",
    paged_kv_write_batch,
    bfloat16_t);
instantiate_kernel("paged_kv_write_batch_float", paged_kv_write_batch, float);

template <typename T>
kernel void paged_kv_write_layers_batch(
    device T* k_cache [[buffer(0)]],
    device T* v_cache [[buffer(1)]],
    device const int* block_tables [[buffer(2)]],
    device const int* context_lens [[buffer(3)]],
    device const T* k_src [[buffer(4)]],
    device const T* v_src [[buffer(5)]],
    constant PagedKVWriteBatchParams& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  const uint dim_idx = gid % params.head_dim;
  const uint kv_idx = (gid / params.head_dim) % params.num_kv_heads;
  const uint seq_idx =
      (gid / (params.head_dim * params.num_kv_heads)) % params.batch_size;
  const uint layer_idx =
      gid / (params.head_dim * params.num_kv_heads * params.batch_size);
  if (layer_idx >= params.layers || seq_idx >= params.batch_size) {
    return;
  }

  const uint logical_pos = static_cast<uint>(context_lens[seq_idx]);
  const uint block_idx = logical_pos / params.block_size;
  const uint row = logical_pos % params.block_size;
  if (block_idx >= params.max_blocks_per_seq) {
    return;
  }

  const device int* table = block_tables + seq_idx * params.table_stride;
  const int block_id = table[block_idx];
  if (block_id < 0) {
    return;
  }

  device T* k_layer = k_cache + layer_idx * params.layer_stride;
  device T* v_layer = v_cache + layer_idx * params.layer_stride;
  device T* k_dst = k_layer + kv_idx * params.kv_head_stride +
      block_id * params.block_stride + row * params.row_stride;
  device T* v_dst = v_layer + kv_idx * params.kv_head_stride +
      block_id * params.block_stride + row * params.row_stride;

  const device T* k_layer_src = k_src + layer_idx * params.batch_layer_stride;
  const device T* v_layer_src = v_src + layer_idx * params.batch_layer_stride;
  const device T* k_ptr = k_layer_src + seq_idx * params.batch_seq_stride +
      kv_idx * params.batch_head_stride;
  const device T* v_ptr = v_layer_src + seq_idx * params.batch_seq_stride +
      kv_idx * params.batch_head_stride;

  k_dst[dim_idx] = k_ptr[dim_idx];
  v_dst[dim_idx] = v_ptr[dim_idx];
}

instantiate_kernel(
    "paged_kv_write_layers_batch_float16_t",
    paged_kv_write_layers_batch,
    float16_t);
instantiate_kernel(
    "paged_kv_write_layers_batch_bfloat16_t",
    paged_kv_write_layers_batch,
    bfloat16_t);
instantiate_kernel(
    "paged_kv_write_layers_batch_float",
    paged_kv_write_layers_batch,
    float);

template <typename T>
kernel void paged_kv_write_layers_tokens(
    device T* k_cache [[buffer(0)]],
    device T* v_cache [[buffer(1)]],
    device const int* block_tables [[buffer(2)]],
    device const int* context_lens [[buffer(3)]],
    device const T* k_src [[buffer(4)]],
    device const T* v_src [[buffer(5)]],
    constant PagedKVWriteLayersTokensParams& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  const uint dim_idx = gid % params.head_dim;
  const uint kv_idx = (gid / params.head_dim) % params.num_kv_heads;
  const uint seq_idx =
      (gid / (params.head_dim * params.num_kv_heads)) % params.batch_size;
  const uint token_idx =
      (gid / (params.head_dim * params.num_kv_heads * params.batch_size)) %
      params.tokens;
  const uint layer_idx = gid /
      (params.head_dim * params.num_kv_heads * params.batch_size *
       params.tokens);
  if (layer_idx >= params.layers || seq_idx >= params.batch_size) {
    return;
  }

  int base_pos = context_lens[seq_idx];
  if (base_pos < 0) {
    return;
  }
  uint logical_pos = static_cast<uint>(base_pos) + token_idx;
  const uint block_idx = logical_pos / params.block_size;
  const uint row = logical_pos % params.block_size;
  if (block_idx >= params.max_blocks_per_seq) {
    return;
  }
  const device int* table = block_tables + seq_idx * params.table_stride;
  const int block_id = table[block_idx];
  if (block_id < 0) {
    return;
  }

  device T* layer_cache = k_cache + layer_idx * params.layer_stride;
  device T* value_cache = v_cache + layer_idx * params.layer_stride;
  device T* k_dst = layer_cache + kv_idx * params.kv_head_stride +
      block_id * params.block_stride + row * params.row_stride;
  device T* v_dst = value_cache + kv_idx * params.kv_head_stride +
      block_id * params.block_stride + row * params.row_stride;

  const device T* layer_src = k_src + layer_idx * params.token_layer_stride;
  const device T* token_src = layer_src + token_idx * params.token_step_stride +
      seq_idx * params.token_batch_stride + kv_idx * params.token_head_stride +
      dim_idx * params.token_dim_stride;
  const device T* layer_v_src = v_src + layer_idx * params.token_layer_stride;
  const device T* token_v_src = layer_v_src +
      token_idx * params.token_step_stride +
      seq_idx * params.token_batch_stride + kv_idx * params.token_head_stride +
      dim_idx * params.token_dim_stride;

  k_dst[dim_idx] = token_src[0];
  v_dst[dim_idx] = token_v_src[0];
}

instantiate_kernel(
    "paged_kv_write_layers_tokens_float16_t",
    paged_kv_write_layers_tokens,
    float16_t);
instantiate_kernel(
    "paged_kv_write_layers_tokens_bfloat16_t",
    paged_kv_write_layers_tokens,
    bfloat16_t);
instantiate_kernel(
    "paged_kv_write_layers_tokens_float",
    paged_kv_write_layers_tokens,
    float);
