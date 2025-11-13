#pragma once

#include <optional>

#include "mlx/array.h"
#include "mlx/backend/metal/device.h"
#include "mlx/stream.h"

namespace mlx::core::fast {

struct PagedAttentionQuantConfig {
  int bits;
  int group_size;
  int bytes_per_group;
  int groups_per_head;
  bool symmetric;
};

bool paged_attention_use_fallback(
    const array& q,
    const array& k,
    const array& v,
    const array& block_tables,
    const array& context_lens,
    const array* vq_cache,
    const array* v_scale_cache,
    const array* v_zero_cache,
    const PagedAttentionQuantConfig* quant,
    Stream s);

void paged_attention(
    const Stream& s,
    metal::Device& device,
    const array& q,
    const array& k,
    const array& v,
    const array& block_tables,
    const array& context_lens,
    const std::optional<array>& kv_mapping,
    float scale,
    const array* vq_cache,
    const array* v_scale_cache,
    const array* v_zero_cache,
    const PagedAttentionQuantConfig* quant,
    array& out);

void paged_attention_with_overlay(
    const Stream& s,
    metal::Device& device,
    const array& q,
    const array& k,
    const array& v,
    const array& block_tables,
    const array& context_lens,
    const std::optional<array>& kv_mapping,
    float scale,
    const array* vq_cache,
    const array* v_scale_cache,
    const array* v_zero_cache,
    const PagedAttentionQuantConfig* quant,
    const array& k_overlay,
    const array& v_overlay,
    array& out,
    std::optional<uint32_t> overlay_len_override = std::nullopt);

void paged_attention_prewarm_kernel(
    metal::Device& device,
    Dtype dtype,
    uint32_t block_size,
    uint32_t threads_per_head = 0,
    uint32_t vec_width = 0);

double paged_attention_last_time_ms();

void paged_prefill(
    const Stream& s,
    metal::Device& device,
    const array& q,
    const array& k,
    const array& v,
    const array& base_lens,
    const array& block_tables,
    const array& context_lens,
    const std::optional<array>& kv_mapping,
    float scale,
    const array* vq_cache,
    const array* v_scale_cache,
    const array* v_zero_cache,
    const PagedAttentionQuantConfig* quant,
    array& out);

void paged_prefill_prewarm_kernel(
    metal::Device& device,
    Dtype dtype,
    uint32_t block_size,
    uint32_t threads_per_head = 0,
    uint32_t vec_width = 0);

double paged_prefill_last_time_ms();

} // namespace mlx::core::fast
