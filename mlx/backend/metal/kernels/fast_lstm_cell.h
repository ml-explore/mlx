// Copyright © 2024 Apple Inc.
// Fused LSTM cell for RNN on Metal. Same pattern as fast_gru_cell.

#include <metal_stdlib>
#include <metal_math>

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

inline float fast_sigmoid(float x) {
  float y = 1.0f / (1.0f + metal::fast::exp(-metal::abs(x)));
  return (x < 0.0f) ? 1.0f - y : y;
}

inline float4 fast_sigmoid4(float4 x) {
  return float4(fast_sigmoid(x.x), fast_sigmoid(x.y), fast_sigmoid(x.z), fast_sigmoid(x.w));
}

// LSTM: i,f,g,o = gates; cell_new = f*cell_prev + i*g; hidden_new = o*tanh(cell_new)
[[host_name("lstm_cell_fused_float")]] [[kernel]] void lstm_cell_fused_float(
    const device float* input_proj,   // [B, 4H]
    const device float* hidden_proj, // [B, 4H]
    const device float* cell_prev,   // [B, H]
    const device float* hidden_prev, // [B, H] (unused; for API consistency)
    device float* output_cell,       // [B, H]
    device float* output_hidden,     // [B, H]
    constant uint& batch_size,
    constant uint& hidden_size,
    uint idx [[thread_position_in_grid]]) {

  uint stride_4h = 4u * hidden_size;
  uint h_quads = (hidden_size + 3u) / 4u;
  uint total_quads = batch_size * h_quads;
  if (idx >= total_quads) return;

  uint batch_idx = idx / h_quads;
  uint h_base = (idx % h_quads) * 4u;
  uint base = batch_idx * stride_4h + h_base;
  uint prev_base = batch_idx * hidden_size + h_base;

  if (h_base + 4u <= hidden_size) {
    float4 i4 = fast_sigmoid4(*reinterpret_cast<const device float4*>(input_proj + base) +
                             *reinterpret_cast<const device float4*>(hidden_proj + base));
    float4 f4 = fast_sigmoid4(*reinterpret_cast<const device float4*>(input_proj + base + hidden_size) +
                             *reinterpret_cast<const device float4*>(hidden_proj + base + hidden_size));
    float4 g4 = metal::fast::tanh(*reinterpret_cast<const device float4*>(input_proj + base + 2u * hidden_size) +
                                   *reinterpret_cast<const device float4*>(hidden_proj + base + 2u * hidden_size));
    float4 o4 = fast_sigmoid4(*reinterpret_cast<const device float4*>(input_proj + base + 3u * hidden_size) +
                             *reinterpret_cast<const device float4*>(hidden_proj + base + 3u * hidden_size));
    float4 c_prev4 = *reinterpret_cast<const device float4*>(cell_prev + prev_base);

    float4 c_new4 = f4 * c_prev4 + i4 * g4;
    *reinterpret_cast<device float4*>(output_cell + prev_base) = c_new4;
    *reinterpret_cast<device float4*>(output_hidden + prev_base) = o4 * metal::fast::tanh(c_new4);
    return;
  }

  for (uint i = 0u; i < 4u && (h_base + i) < hidden_size; i++) {
    uint b = base + i;
    uint pb = prev_base + i;
    float i_g = fast_sigmoid(input_proj[b] + hidden_proj[b]);
    float f_g = fast_sigmoid(input_proj[b + hidden_size] + hidden_proj[b + hidden_size]);
    float g_g = metal::fast::tanh(input_proj[b + 2u * hidden_size] + hidden_proj[b + 2u * hidden_size]);
    float o_g = fast_sigmoid(input_proj[b + 3u * hidden_size] + hidden_proj[b + 3u * hidden_size]);
    float c_prev = cell_prev[pb];
    float c_new = f_g * c_prev + i_g * g_g;
    output_cell[pb] = c_new;
    output_hidden[pb] = o_g * metal::fast::tanh(c_new);
  }
}

// BFloat16 path
[[host_name("lstm_cell_fused_bfloat16")]] [[kernel]] void lstm_cell_fused_bfloat16(
    const device bfloat16_t* input_proj,
    const device bfloat16_t* hidden_proj,
    const device bfloat16_t* cell_prev,
    const device bfloat16_t* hidden_prev,
    device bfloat16_t* output_cell,
    device bfloat16_t* output_hidden,
    constant uint& batch_size,
    constant uint& hidden_size,
    uint idx [[thread_position_in_grid]]) {

  uint stride_4h = 4u * hidden_size;
  uint h_quads = (hidden_size + 3u) / 4u;
  uint total_quads = batch_size * h_quads;
  if (idx >= total_quads) return;

  uint batch_idx = idx / h_quads;
  uint h_base = (idx % h_quads) * 4u;
  uint base = batch_idx * stride_4h + h_base;
  uint prev_base = batch_idx * hidden_size + h_base;

  if (h_base + 4u <= hidden_size) {
    float4 i4 = fast_sigmoid4(float4(input_proj[base], input_proj[base+1], input_proj[base+2], input_proj[base+3]) +
                             float4(hidden_proj[base], hidden_proj[base+1], hidden_proj[base+2], hidden_proj[base+3]));
    float4 f4 = fast_sigmoid4(float4(input_proj[base+hidden_size], input_proj[base+hidden_size+1], input_proj[base+hidden_size+2], input_proj[base+hidden_size+3]) +
                             float4(hidden_proj[base+hidden_size], hidden_proj[base+hidden_size+1], hidden_proj[base+hidden_size+2], hidden_proj[base+hidden_size+3]));
    float4 g4 = metal::fast::tanh(float4(input_proj[base+2u*hidden_size], input_proj[base+2u*hidden_size+1], input_proj[base+2u*hidden_size+2], input_proj[base+2u*hidden_size+3]) +
                                   float4(hidden_proj[base+2u*hidden_size], hidden_proj[base+2u*hidden_size+1], hidden_proj[base+2u*hidden_size+2], hidden_proj[base+2u*hidden_size+3]));
    float4 o4 = fast_sigmoid4(float4(input_proj[base+3u*hidden_size], input_proj[base+3u*hidden_size+1], input_proj[base+3u*hidden_size+2], input_proj[base+3u*hidden_size+3]) +
                             float4(hidden_proj[base+3u*hidden_size], hidden_proj[base+3u*hidden_size+1], hidden_proj[base+3u*hidden_size+2], hidden_proj[base+3u*hidden_size+3]));
    float4 c_prev4 = float4(cell_prev[prev_base], cell_prev[prev_base+1], cell_prev[prev_base+2], cell_prev[prev_base+3]);

    float4 c_new4 = f4 * c_prev4 + i4 * g4;
    float4 h_new4 = o4 * metal::fast::tanh(c_new4);
    output_cell[prev_base] = bfloat16_t(c_new4.x); output_cell[prev_base+1] = bfloat16_t(c_new4.y);
    output_cell[prev_base+2] = bfloat16_t(c_new4.z); output_cell[prev_base+3] = bfloat16_t(c_new4.w);
    output_hidden[prev_base] = bfloat16_t(h_new4.x); output_hidden[prev_base+1] = bfloat16_t(h_new4.y);
    output_hidden[prev_base+2] = bfloat16_t(h_new4.z); output_hidden[prev_base+3] = bfloat16_t(h_new4.w);
    return;
  }

  for (uint i = 0u; i < 4u && (h_base + i) < hidden_size; i++) {
    uint b = base + i;
    uint pb = prev_base + i;
    float i_g = fast_sigmoid(float(input_proj[b]) + float(hidden_proj[b]));
    float f_g = fast_sigmoid(float(input_proj[b + hidden_size]) + float(hidden_proj[b + hidden_size]));
    float g_g = metal::fast::tanh(float(input_proj[b + 2u * hidden_size]) + float(hidden_proj[b + 2u * hidden_size]));
    float o_g = fast_sigmoid(float(input_proj[b + 3u * hidden_size]) + float(hidden_proj[b + 3u * hidden_size]));
    float c_prev = float(cell_prev[pb]);
    float c_new = f_g * c_prev + i_g * g_g;
    output_cell[pb] = bfloat16_t(c_new);
    output_hidden[pb] = bfloat16_t(o_g * metal::fast::tanh(c_new));
  }
}
