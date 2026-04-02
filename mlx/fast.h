// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <optional>
#include <variant>

#include "mlx/api.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

MLX_API array rms_norm(
    const array& x,
    const std::optional<array>& weight,
    float eps,
    StreamOrDevice s = {});

MLX_API array layer_norm(
    const array& x,
    const std::optional<array>& weight,
    const std::optional<array>& bias,
    float eps,
    StreamOrDevice s = {});

MLX_API array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    int offset,
    const std::optional<array>& freqs = std::nullopt,
    StreamOrDevice s = {});

MLX_API array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    const array& offset,
    const std::optional<array>& freqs = std::nullopt,
    StreamOrDevice s = {});

/** Computes: O = softmax(Q @ K.T) @ V **/
MLX_API array scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& values,
    const float scale,
    const std::string& mask_mode = "",
    std::optional<array> mask_arr = {},
    const std::optional<array>& sinks = {},
    StreamOrDevice s = {});

using TemplateArg = std::variant<int, bool, Dtype>;
using ScalarArg = std::variant<bool, int, float>;

using CustomKernelFunction = std::function<std::vector<array>(
    const std::vector<array>&,
    const std::vector<Shape>&,
    const std::vector<Dtype>&,
    std::tuple<int, int, int>,
    std::tuple<int, int, int>,
    std::vector<std::pair<std::string, TemplateArg>>,
    std::optional<float>,
    bool,
    StreamOrDevice)>;

MLX_API CustomKernelFunction metal_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header = "",
    bool ensure_row_contiguous = true,
    bool atomic_outputs = false);

MLX_API CustomKernelFunction cuda_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header = "",
    bool ensure_row_contiguous = true,
    int shared_memory = 0);

MLX_API std::vector<array> precompiled_cuda_kernel(
    const std::string& name,
    const std::string& compiled_source,
    const std::vector<array>& inputs,
    const std::vector<Shape>& output_shapes,
    const std::vector<Dtype>& output_dtypes,
    const std::vector<ScalarArg>& scalars,
    std::tuple<int, int, int> grid,
    std::tuple<int, int, int> threadgroup,
    int shared_memory = 0,
    std::optional<float> init_value = std::nullopt,
    bool ensure_row_contiguous = false,
    StreamOrDevice s = {});

/** MLA shared-latent nope scoring — first-class MLA primitive.
 *
 * Computes: scores[b,h,s] = scale * dot(q_nope[b,h,:], dequant(latent[b,s,:]))
 * Latent is shared across all heads (no broadcast).
 * INT4 affine dequant in-kernel.
 */
MLX_API array mla_nope_scores(
    const array& q_nope,    // [B, H, 256] float16/bfloat16
    const array& k_packed,  // [B, S, 32]  uint32 (INT4 packed)
    const array& k_scales,  // [B, S, 4]   float32
    const array& k_biases,  // [B, S, 4]   float32
    float scale,
    StreamOrDevice s = {});

/** Fused quantized MLA SDPA for decode (L==1).
 *
 * Single kernel fusing: INT4 dequant + split nope/rope scoring +
 * online softmax + value accumulation. Replaces 5+ separate dispatches.
 * Output is latent attention result (pre-unembed).
 */
MLX_API array mla_fused_sdpa(
    const array& q_nope,      // [B, H, 256] pre-scaled, post-embed_q
    const array& q_pe,        // [B, H, 64]  pre-scaled
    const array& lat_packed,  // [B, S, 32]  uint32 INT4 packed latent
    const array& lat_scales,  // [B, S, 4]   fp16 scales
    const array& lat_biases,  // [B, S, 4]   fp16 biases
    const array& k_pe,        // [B, S, 64]  fp16 RoPE keys
    float scale,
    StreamOrDevice s = {});

/**
 * Fused INT4 affine quantization for MLA latent cache.
 * Single kernel replacing mx.quantize multi-dispatch overhead
 * for MLA dimensions (256 latent, group_size=64, 4-bit).
 */
MLX_API std::vector<array> mla_quantize_store(
    const array& input,  // [..., 256] fp16 latent
    StreamOrDevice s = {});

} // namespace mlx::core::fast
