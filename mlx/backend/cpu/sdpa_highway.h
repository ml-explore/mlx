// Copyright © 2026 Apple Inc.

#pragma once

#include <cstdint>

namespace mlx::core::fast {

enum class SdpaHighwayDType : uint8_t {
  Float32,
  Float16,
  BFloat16,
};

void sdpa_highway(
    void* output,
    const void* queries,
    const void* keys,
    const void* values,
    SdpaHighwayDType dtype,
    int B,
    int n_q_heads,
    int n_kv_heads,
    int M,
    int seq_len,
    int head_dim,
    float scale,
    bool do_causal,
    const void* mask,
    bool has_mask,
    const void* sinks,
    bool has_sinks);

} // namespace mlx::core::fast
