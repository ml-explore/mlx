// ABOUTME: Implements Metal paged attention decode kernel using streaming
// softmax. ABOUTME: Operates on paged KV blocks to accelerate per-token
// decoding on Apple GPUs.
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

} // namespace mlx::core::fast
