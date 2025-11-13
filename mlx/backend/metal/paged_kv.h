// ABOUTME: Declares Metal helpers for paged KV cache writes.
// ABOUTME: Exposes dispatch utilities for chunked KV copy kernels.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/metal/device.h"
#include "mlx/stream.h"

namespace mlx::core::fast {

struct PagedKVQuantConfig {
  int bits;
  int group_size;
  int bytes_per_token;
  int groups_per_head;
  bool symmetric;
};

bool paged_kv_write_use_fallback(
    const array& k_cache,
    const array& v_cache,
    const array& block_row,
    const array& k_chunk,
    const array& v_chunk,
    array* vq_cache,
    array* v_scale_cache,
    array* v_zero_cache,
    const PagedKVQuantConfig* quant,
    Stream s);

bool paged_kv_write_batch_use_fallback(
    const array& k_cache,
    const array& v_cache,
    const array& block_ids,
    const array& tok_offsets,
    const array& k_batch,
    const array& v_batch,
    Stream s);

void paged_kv_write_batch(
    const Stream& s,
    metal::Device& device,
    const array& k_cache,
    const array& v_cache,
    const array& block_ids,
    const array& tok_offsets,
    const array& k_batch,
    const array& v_batch);

void paged_kv_write_layers_batch(
    const Stream& s,
    metal::Device& device,
    const array& k_cache,
    const array& v_cache,
    const array& block_tables,
    const array& context_lens,
    const array& k_batch,
    const array& v_batch);

void paged_kv_write_layers_tokens(
    const Stream& s,
    metal::Device& device,
    const array& k_cache,
    const array& v_cache,
    const array& block_tables,
    const array& context_lens,
    const array& k_tokens,
    const array& v_tokens);

void paged_kv_write(
    const Stream& s,
    metal::Device& device,
    const array& k_cache,
    const array& v_cache,
    const array& block_row,
    uint32_t start_pos,
    const array& k_chunk,
    const array& v_chunk,
    array* vq_cache,
    array* v_scale_cache,
    array* v_zero_cache,
    const PagedKVQuantConfig* quant);

} // namespace mlx::core::fast
