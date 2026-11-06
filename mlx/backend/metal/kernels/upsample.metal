// Copyright © 2026 Apple Inc.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

// Sizes are baked in as function constants so index math (div/mod/mul) is
// resolved at pipeline-build time instead of costing a runtime op per thread.
constant int C [[function_constant(0)]];
constant int H_in [[function_constant(1)]];
constant int W_in [[function_constant(2)]];
constant int H_out [[function_constant(3)]];
constant int W_out [[function_constant(4)]];

// One thread per output value. y_l/y_r/x_l/x_r are each output index's
// neighbour input indices, y_w/x_w the blend weight toward the right one.
template <typename T>
[[kernel]] void upsample_bilinear(
    const device T* x [[buffer(0)]],
    device T* out [[buffer(1)]],
    const device uint* y_l [[buffer(2)]],
    const device uint* y_r [[buffer(3)]],
    const device float* y_w [[buffer(4)]],
    const device uint* x_l [[buffer(5)]],
    const device uint* x_r [[buffer(6)]],
    const device float* x_w [[buffer(7)]],
    uint3 pos [[thread_position_in_grid]]) {
  uint c = pos.x;
  uint ox = pos.y;
  uint oy = pos.z % uint(H_out);
  uint b = pos.z / uint(H_out);
  if (c >= uint(C) || ox >= uint(W_out)) {
    return;
  }

  uint y0 = y_l[oy], y1 = y_r[oy], x0 = x_l[ox], x1 = x_r[ox];
  float ty = y_w[oy], tx = x_w[ox];

  uint base = b * uint(H_in) * uint(W_in) * uint(C);
  float q00 = static_cast<float>(x[base + (y0 * uint(W_in) + x0) * uint(C) + c]);
  float q10 = static_cast<float>(x[base + (y0 * uint(W_in) + x1) * uint(C) + c]);
  float q01 = static_cast<float>(x[base + (y1 * uint(W_in) + x0) * uint(C) + c]);
  float q11 = static_cast<float>(x[base + (y1 * uint(W_in) + x1) * uint(C) + c]);

  float r0 = (1 - tx) * q00 + tx * q10;
  float r1 = (1 - tx) * q01 + tx * q11;
  out[((b * uint(H_out) + oy) * uint(W_out) + ox) * uint(C) + c] =
      static_cast<T>((1 - ty) * r0 + ty * r1);
}

// One thread per input value, gathering its share of every output index
// that reads it (no atomics needed); y_lo/y_hi (x_lo/x_hi) give that range.
template <typename T>
[[kernel]] void upsample_bilinear_vjp(
    const device T* g [[buffer(0)]],
    device T* gx [[buffer(1)]],
    const device uint* y_l [[buffer(2)]],
    const device uint* y_r [[buffer(3)]],
    const device float* y_w [[buffer(4)]],
    const device uint* x_l [[buffer(5)]],
    const device uint* x_r [[buffer(6)]],
    const device float* x_w [[buffer(7)]],
    const device uint* y_lo [[buffer(8)]],
    const device uint* y_hi [[buffer(9)]],
    const device uint* x_lo [[buffer(10)]],
    const device uint* x_hi [[buffer(11)]],
    uint3 pos [[thread_position_in_grid]]) {
  uint c = pos.x;
  uint ix = pos.y;
  uint iy = pos.z % uint(H_in);
  uint b = pos.z / uint(H_in);
  if (c >= uint(C) || ix >= uint(W_in)) {
    return;
  }

  uint y_lo_i = y_lo[iy], y_hi_i = y_hi[iy];
  uint x_lo_i = x_lo[ix], x_hi_i = x_hi[ix];

  uint base = b * uint(H_out) * uint(W_out) * uint(C);

  float sum = 0;
  for (uint oy = y_lo_i; oy < y_hi_i; oy++) {
    // Hoist the y-axis weight out of the x loop: it does not depend on ox.
    float ty = y_w[oy];
    float wy = 0;
    if (y_l[oy] == iy) {
      wy += 1 - ty;
    }
    if (y_r[oy] == iy) {
      wy += ty;
    }
    uint row = base + oy * uint(W_out) * uint(C);
    for (uint ox = x_lo_i; ox < x_hi_i; ox++) {
      float tx = x_w[ox];
      float wx = 0;
      if (x_l[ox] == ix) {
        wx += 1 - tx;
      }
      if (x_r[ox] == ix) {
        wx += tx;
      }
      sum += wy * wx * static_cast<float>(g[row + ox * uint(C) + c]);
    }
  }

  gx[((b * uint(H_in) + iy) * uint(W_in) + ix) * uint(C) + c] =
      static_cast<T>(sum);
}

// clang-format off
#define instantiate_upsample_bilinear(name, type)           \
  instantiate_kernel("upsample_bilinear_" #name, upsample_bilinear, type) \
  instantiate_kernel("upsample_bilinear_vjp_" #name, upsample_bilinear_vjp, type)

instantiate_upsample_bilinear(float16, half)
instantiate_upsample_bilinear(bfloat16, bfloat16_t)
instantiate_upsample_bilinear(float32, float) // clang-format on
