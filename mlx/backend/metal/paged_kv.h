// ABOUTME: Declares Metal helpers for paged KV cache writes.
// ABOUTME: Exposes dispatch utilities for chunked KV copy kernels.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/metal/device.h"
#include "mlx/stream.h"

namespace mlx::core::fast {

bool paged_kv_write_use_fallback(
    const array& k_cache,
    const array& v_cache,
    const array& block_row,
    const array& k_chunk,
    const array& v_chunk,
    Stream s);

void paged_kv_write(
    const Stream& s,
    metal::Device& device,
    const array& k_cache,
    const array& v_cache,
    const array& block_row,
    uint32_t start_pos,
    const array& k_chunk,
    const array& v_chunk);

} // namespace mlx::core::fast
