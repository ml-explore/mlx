// Copyright © 2024 Apple Inc.
// Fused GRU cell for RNN on Metal. See Apple Metal docs:
// https://developer.apple.com/documentation/metal

#include <metal_math>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// Numerically stable sigmoid; one branch for speed.
inline float fast_sigmoid(float x) {
  float y = 1.0f / (1.0f + metal::fast::exp(-metal::abs(x)));
  return (x < 0.0f) ? 1.0f - y : y;
}

inline float4 fast_sigmoid4(float4 x) {
  return float4(
      fast_sigmoid(x.x),
      fast_sigmoid(x.y),
      fast_sigmoid(x.z),
      fast_sigmoid(x.w));
}

// Fused GRU cell: one kernel for gating. Inputs are pre-computed (e.g. by
// GEMM). Optimized: vectorized float4 path when 4 consecutive h fit, else
// scalar.
[[host_name("gru_cell_fused_float")]] [[kernel]] void gru_cell_fused_float(
    const device float* input_proj, // [B, 3H] input projection
    const device float* hidden_proj, // [B, 3H] hidden projection
    const device float* hidden_prev, // [B, H] previous hidden state
    device float* output, // [B, H] output hidden state
    constant uint& batch_size,
    constant uint& hidden_size,
    uint idx [[thread_position_in_grid]]) {
  uint stride_3h = 3u * hidden_size;
  uint h_quads = (hidden_size + 3u) / 4u;
  uint total_quads = batch_size * h_quads;
  if (idx >= total_quads)
    return;

  uint batch_idx = idx / h_quads;
  uint h_base = (idx % h_quads) * 4u;
  uint base = batch_idx * stride_3h + h_base;
  uint prev_base = batch_idx * hidden_size + h_base;

  // Vectorized path: 4 consecutive elements, coalesced float4 load/store
  // GRU: n = tanh(x_n + r * h_proj_n), out = (1-z)*n + z*h_prev (h_proj_n =
  // hidden_proj[2*H:])
  if (h_base + 4u <= hidden_size) {
    float4 x_r4 = *reinterpret_cast<const device float4*>(input_proj + base) +
        *reinterpret_cast<const device float4*>(hidden_proj + base);
    float4 x_z4 = *reinterpret_cast<const device float4*>(
                      input_proj + base + hidden_size) +
        *reinterpret_cast<const device float4*>(
            hidden_proj + base + hidden_size);
    float4 x_n4 = *reinterpret_cast<const device float4*>(
        input_proj + base + 2u * hidden_size);
    float4 h_proj_n4 = *reinterpret_cast<const device float4*>(
        hidden_proj + base + 2u * hidden_size);
    float4 h_prev4 =
        *reinterpret_cast<const device float4*>(hidden_prev + prev_base);

    float4 r4 = fast_sigmoid4(x_r4);
    float4 z4 = fast_sigmoid4(x_z4);
    float4 n4 = metal::fast::tanh(x_n4 + r4 * h_proj_n4);
    *reinterpret_cast<device float4*>(output + prev_base) =
        (1.0f - z4) * n4 + z4 * h_prev4;
    return;
  }

  // Scalar tail (last partial quad per row)
  for (uint i = 0u; i < 4u && (h_base + i) < hidden_size; i++) {
    uint b = base + i;
    uint pb = prev_base + i;
    float x_r = input_proj[b] + hidden_proj[b];
    float x_z = input_proj[b + hidden_size] + hidden_proj[b + hidden_size];
    float x_n = input_proj[b + 2u * hidden_size];
    float h_proj_n = hidden_proj[b + 2u * hidden_size];
    float h_prev = hidden_prev[pb];
    float r = fast_sigmoid(x_r);
    float z = fast_sigmoid(x_z);
    float n = metal::fast::tanh(x_n + r * h_proj_n);
    output[pb] = (1.0f - z) * n + z * h_prev;
  }
}

// Same as above but with recurrent bias bhn [H] for n-gate; avoids per-step add
// in Python.
[[host_name("gru_cell_fused_float_bias")]] [[kernel]] void
gru_cell_fused_float_bias(
    const device float* input_proj,
    const device float* hidden_proj,
    const device float* hidden_prev,
    const device float* bhn,
    device float* output,
    constant uint& batch_size,
    constant uint& hidden_size,
    uint idx [[thread_position_in_grid]]) {
  uint stride_3h = 3u * hidden_size;
  uint h_quads = (hidden_size + 3u) / 4u;
  uint total_quads = batch_size * h_quads;
  if (idx >= total_quads)
    return;

  uint batch_idx = idx / h_quads;
  uint h_base = (idx % h_quads) * 4u;
  uint base = batch_idx * stride_3h + h_base;
  uint prev_base = batch_idx * hidden_size + h_base;

  if (h_base + 4u <= hidden_size) {
    float4 x_r4 = *reinterpret_cast<const device float4*>(input_proj + base) +
        *reinterpret_cast<const device float4*>(hidden_proj + base);
    float4 x_z4 = *reinterpret_cast<const device float4*>(
                      input_proj + base + hidden_size) +
        *reinterpret_cast<const device float4*>(
            hidden_proj + base + hidden_size);
    float4 x_n4 = *reinterpret_cast<const device float4*>(
        input_proj + base + 2u * hidden_size);
    float4 h_proj_n4 = *reinterpret_cast<const device float4*>(
                           hidden_proj + base + 2u * hidden_size) +
        *reinterpret_cast<const device float4*>(bhn + h_base);
    float4 h_prev4 =
        *reinterpret_cast<const device float4*>(hidden_prev + prev_base);

    float4 r4 = fast_sigmoid4(x_r4);
    float4 z4 = fast_sigmoid4(x_z4);
    float4 n4 = metal::fast::tanh(x_n4 + r4 * h_proj_n4);
    *reinterpret_cast<device float4*>(output + prev_base) =
        (1.0f - z4) * n4 + z4 * h_prev4;
    return;
  }

  for (uint i = 0u; i < 4u && (h_base + i) < hidden_size; i++) {
    uint b = base + i;
    uint pb = prev_base + i;
    float x_r = input_proj[b] + hidden_proj[b];
    float x_z = input_proj[b + hidden_size] + hidden_proj[b + hidden_size];
    float x_n = input_proj[b + 2u * hidden_size];
    float h_proj_n = hidden_proj[b + 2u * hidden_size] + bhn[h_base + i];
    float h_prev = hidden_prev[pb];
    float r = fast_sigmoid(x_r);
    float z = fast_sigmoid(x_z);
    float n = metal::fast::tanh(x_n + r * h_proj_n);
    output[pb] = (1.0f - z) * n + z * h_prev;
  }
}

// BFloat16 path: vectorized float4 when possible, else scalar; math in float,
// bf16 on write.
[[host_name("gru_cell_fused_bfloat16")]] [[kernel]] void
gru_cell_fused_bfloat16(
    const device bfloat16_t* input_proj,
    const device bfloat16_t* hidden_proj,
    const device bfloat16_t* hidden_prev,
    device bfloat16_t* output,
    constant uint& batch_size,
    constant uint& hidden_size,
    uint idx [[thread_position_in_grid]]) {
  uint stride_3h = 3u * hidden_size;
  uint h_quads = (hidden_size + 3u) / 4u;
  uint total_quads = batch_size * h_quads;
  if (idx >= total_quads)
    return;

  uint batch_idx = idx / h_quads;
  uint h_base = (idx % h_quads) * 4u;
  uint base = batch_idx * stride_3h + h_base;
  uint prev_base = batch_idx * hidden_size + h_base;

  if (h_base + 4u <= hidden_size) {
    float4 x_r4 = float4(
                      input_proj[base],
                      input_proj[base + 1],
                      input_proj[base + 2],
                      input_proj[base + 3]) +
        float4(hidden_proj[base],
               hidden_proj[base + 1],
               hidden_proj[base + 2],
               hidden_proj[base + 3]);
    float4 x_z4 = float4(
                      input_proj[base + hidden_size],
                      input_proj[base + hidden_size + 1],
                      input_proj[base + hidden_size + 2],
                      input_proj[base + hidden_size + 3]) +
        float4(hidden_proj[base + hidden_size],
               hidden_proj[base + hidden_size + 1],
               hidden_proj[base + hidden_size + 2],
               hidden_proj[base + hidden_size + 3]);
    float4 x_n4 = float4(
        input_proj[base + 2u * hidden_size],
        input_proj[base + 2u * hidden_size + 1],
        input_proj[base + 2u * hidden_size + 2],
        input_proj[base + 2u * hidden_size + 3]);
    float4 h_proj_n4 = float4(
        hidden_proj[base + 2u * hidden_size],
        hidden_proj[base + 2u * hidden_size + 1],
        hidden_proj[base + 2u * hidden_size + 2],
        hidden_proj[base + 2u * hidden_size + 3]);
    float4 h_prev4 = float4(
        hidden_prev[prev_base],
        hidden_prev[prev_base + 1],
        hidden_prev[prev_base + 2],
        hidden_prev[prev_base + 3]);

    float4 r4 = fast_sigmoid4(x_r4);
    float4 z4 = fast_sigmoid4(x_z4);
    float4 n4 = metal::fast::tanh(x_n4 + r4 * h_proj_n4);
    float4 out4 = (1.0f - z4) * n4 + z4 * h_prev4;
    output[prev_base] = bfloat16_t(out4.x);
    output[prev_base + 1] = bfloat16_t(out4.y);
    output[prev_base + 2] = bfloat16_t(out4.z);
    output[prev_base + 3] = bfloat16_t(out4.w);
    return;
  }

  for (uint i = 0u; i < 4u && (h_base + i) < hidden_size; i++) {
    uint b = base + i;
    uint pb = prev_base + i;
    float x_r = float(input_proj[b]) + float(hidden_proj[b]);
    float x_z = float(input_proj[b + hidden_size]) +
        float(hidden_proj[b + hidden_size]);
    float x_n = float(input_proj[b + 2u * hidden_size]);
    float h_proj_n = float(hidden_proj[b + 2u * hidden_size]);
    float h_prev = float(hidden_prev[pb]);
    float r = fast_sigmoid(x_r);
    float z = fast_sigmoid(x_z);
    float n = metal::fast::tanh(x_n + r * h_proj_n);
    output[pb] = bfloat16_t((1.0f - z) * n + z * h_prev);
  }
}

// BFloat16 path with bhn
[[host_name("gru_cell_fused_bfloat16_bias")]] [[kernel]] void
gru_cell_fused_bfloat16_bias(
    const device bfloat16_t* input_proj,
    const device bfloat16_t* hidden_proj,
    const device bfloat16_t* hidden_prev,
    const device bfloat16_t* bhn,
    device bfloat16_t* output,
    constant uint& batch_size,
    constant uint& hidden_size,
    uint idx [[thread_position_in_grid]]) {
  uint stride_3h = 3u * hidden_size;
  uint h_quads = (hidden_size + 3u) / 4u;
  uint total_quads = batch_size * h_quads;
  if (idx >= total_quads)
    return;

  uint batch_idx = idx / h_quads;
  uint h_base = (idx % h_quads) * 4u;
  uint base = batch_idx * stride_3h + h_base;
  uint prev_base = batch_idx * hidden_size + h_base;

  if (h_base + 4u <= hidden_size) {
    float4 x_r4 = float4(
                      input_proj[base],
                      input_proj[base + 1],
                      input_proj[base + 2],
                      input_proj[base + 3]) +
        float4(hidden_proj[base],
               hidden_proj[base + 1],
               hidden_proj[base + 2],
               hidden_proj[base + 3]);
    float4 x_z4 = float4(
                      input_proj[base + hidden_size],
                      input_proj[base + hidden_size + 1],
                      input_proj[base + hidden_size + 2],
                      input_proj[base + hidden_size + 3]) +
        float4(hidden_proj[base + hidden_size],
               hidden_proj[base + hidden_size + 1],
               hidden_proj[base + hidden_size + 2],
               hidden_proj[base + hidden_size + 3]);
    float4 x_n4 = float4(
        input_proj[base + 2u * hidden_size],
        input_proj[base + 2u * hidden_size + 1],
        input_proj[base + 2u * hidden_size + 2],
        input_proj[base + 2u * hidden_size + 3]);
    float4 h_proj_n4 = float4(
                           hidden_proj[base + 2u * hidden_size],
                           hidden_proj[base + 2u * hidden_size + 1],
                           hidden_proj[base + 2u * hidden_size + 2],
                           hidden_proj[base + 2u * hidden_size + 3]) +
        float4(bhn[h_base], bhn[h_base + 1], bhn[h_base + 2], bhn[h_base + 3]);
    float4 h_prev4 = float4(
        hidden_prev[prev_base],
        hidden_prev[prev_base + 1],
        hidden_prev[prev_base + 2],
        hidden_prev[prev_base + 3]);

    float4 r4 = fast_sigmoid4(x_r4);
    float4 z4 = fast_sigmoid4(x_z4);
    float4 n4 = metal::fast::tanh(x_n4 + r4 * h_proj_n4);
    float4 out4 = (1.0f - z4) * n4 + z4 * h_prev4;
    output[prev_base] = bfloat16_t(out4.x);
    output[prev_base + 1] = bfloat16_t(out4.y);
    output[prev_base + 2] = bfloat16_t(out4.z);
    output[prev_base + 3] = bfloat16_t(out4.w);
    return;
  }

  for (uint i = 0u; i < 4u && (h_base + i) < hidden_size; i++) {
    uint b = base + i;
    uint pb = prev_base + i;
    float x_r = float(input_proj[b]) + float(hidden_proj[b]);
    float x_z = float(input_proj[b + hidden_size]) +
        float(hidden_proj[b + hidden_size]);
    float x_n = float(input_proj[b + 2u * hidden_size]);
    float h_proj_n =
        float(hidden_proj[b + 2u * hidden_size]) + float(bhn[h_base + i]);
    float h_prev = float(hidden_prev[pb]);
    float r = fast_sigmoid(x_r);
    float z = fast_sigmoid(x_z);
    float n = metal::fast::tanh(x_n + r * h_proj_n);
    output[pb] = bfloat16_t((1.0f - z) * n + z * h_prev);
  }
}