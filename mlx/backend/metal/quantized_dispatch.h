// Copyright © 2026 Apple Inc.

#pragma once

namespace mlx::core::metal {

// The #4198 measurements cover the applegpu_g17s M5 Max class's two-way split-K
// region between these output widths. Keep this deliberately narrow until a
// wider device and shape sweep establishes a better crossover.
constexpr int kQmmTSplitKNaxMinN = 6656;
constexpr int kQmmTSplitKNaxMaxN = 8192;

constexpr bool qmm_t_splitk_should_use_nax(
    bool fallback_qmm_is_nax_eligible,
    int architecture_generation,
    char architecture_size,
    bool transpose,
    bool single_batch,
    bool affine,
    int group_size,
    int bits,
    int m_tiles,
    int N,
    int K,
    int split_k) {
  return fallback_qmm_is_nax_eligible && architecture_generation == 17 &&
      architecture_size == 's' && transpose && single_batch && affine &&
      group_size == 64 && bits == 4 && m_tiles == 1 &&
      N >= kQmmTSplitKNaxMinN && N <= kQmmTSplitKNaxMaxN && K % 128 == 0 &&
      split_k == 2;
}

} // namespace mlx::core::metal
