#pragma once

#include <optional>

#include "mlx/array.h"
#include "mlx/backend/metal/device.h"
#include "mlx/stream.h"

namespace mlx::core::fast {

bool paged_attention_use_fallback(
    const array& q,
    const array& k,
    const array& v,
    const array& block_tables,
    const array& context_lens,
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
    array& out);

void paged_attention_prewarm_kernel(
    metal::Device& device,
    Dtype dtype,
    uint32_t block_size,
    uint32_t threads_per_head = 0,
    uint32_t vec_width = 0);

double paged_attention_last_time_ms();

} // namespace mlx::core::fast
