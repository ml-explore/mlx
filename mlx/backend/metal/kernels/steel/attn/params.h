// Copyright © 2024 Apple Inc.

#pragma once

///////////////////////////////////////////////////////////////////////////////
// Attn param classes
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

struct AttnParams {
  int B; ///< Batch Size
  int H; ///< Heads
  int L; ///< Sequence Length
  int D; ///< Head Dim

  int gqa_factor; ///< Group Query factor
  float scale; ///< Attention scale

  int NQ; ///< Number of query blocks
  int NK; ///< Number of key/value blocks

  size_t Q_strides[3]; ///< Query  strides (B, H, L, D = 1)
  size_t K_strides[3]; ///< Key    strides (B, H, L, D = 1)
  size_t V_strides[3]; ///< Value  strides (B, H, L, D = 1)
  size_t O_strides[3]; ///< Output strides (B, H, L, D = 1)
};

} // namespace steel
} // namespace mlx
