// Copyright © 2025 Apple Inc.

#pragma once

#include <metal_stdlib>

using namespace metal;

template <typename T>
[[kernel]] void sdpa_chunked_reduce(
    const device T* chunk_outs [[buffer(0)]],
    const device float* chunk_lses [[buffer(1)]],
    device T* output [[buffer(2)]],
    const constant int& n_chunks [[buffer(3)]],
    const constant int& D [[buffer(4)]],
    const constant int& qL [[buffer(5)]],
    const constant int& H [[buffer(6)]],
    const constant int64_t* O_strides [[buffer(7)]],
    const constant int& BHqL [[buffer(8)]],
    uint3 tid [[thread_position_in_grid]]) {
  const int d = tid.x;
  const int q = tid.y;
  const int bh = tid.z;

  if (d >= D || q >= qL) {
    return;
  }

  const int h = bh % H;
  const int b = bh / H;
  const int64_t bhq = int64_t(bh) * int64_t(qL) + int64_t(q);

  float max_lse = -INFINITY;
  for (int c = 0; c < n_chunks; c++) {
    int64_t lse_idx = int64_t(c) * int64_t(BHqL) + bhq;
    max_lse = max(max_lse, chunk_lses[lse_idx]);
  }

  float acc = 0.0f;
  float sum_w = 0.0f;
  for (int c = 0; c < n_chunks; c++) {
    int64_t lse_idx = int64_t(c) * int64_t(BHqL) + bhq;
    float w = metal::exp(chunk_lses[lse_idx] - max_lse);
    if (w > 0.0f) {
      sum_w += w;
      int64_t out_idx = int64_t(c) * int64_t(BHqL) * int64_t(D) +
          bhq * int64_t(D) + int64_t(d);
      acc += w * float(chunk_outs[out_idx]);
    }
  }

  float result = (sum_w > 0.0f) ? (acc / sum_w) : 0.0f;
  int64_t o_idx = int64_t(b) * O_strides[0] +
      int64_t(h) * O_strides[1] + int64_t(q) * O_strides[2] + int64_t(d);
  output[o_idx] = T(result);
}
