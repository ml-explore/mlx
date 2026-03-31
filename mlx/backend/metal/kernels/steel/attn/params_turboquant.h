// Copyright © 2024-25 Apple Inc.
// TurboQuant attention parameters for compressed KV cache

#pragma once

///////////////////////////////////////////////////////////////////////////////
// TurboQuant Attn param classes
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

struct TurboQuantAttnParams {
  int N; ///< KV sequence length
  int gqa_factor; ///< Group Query Attention factor (H_q / H_kv)
  float scale; ///< Attention scale (1/sqrt(D))
  float qjl_scale; ///< QJL correction scale (sqrt(pi/2) / D)

  int packed_d_mse; ///< Bytes per token for MSE indices
  int packed_d_signs; ///< Bytes per token for QJL sign bits
  int packed_d_v; ///< Bytes per token for quantized values
  int n_groups; ///< Number of value quantization groups (D / group_size)
  int group_size; ///< Value quantization group size
  int n_centroids; ///< Number of MSE centroids (2^mse_bits)
  int mse_bits; ///< Bits per MSE index (2 or 4)
  int v_bits; ///< Bits per value index (2 or 4)
};

} // namespace steel
} // namespace mlx
