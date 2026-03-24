// Copyright © 2025 Apple Inc.

#pragma once

#include <metal_stdlib>

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// Reduction kernel for chunked SDPA
//
// Combines N per-chunk outputs using logsumexp-weighted averaging:
//
//   max_lse = max(lse_1, ..., lse_N)
//   w_c     = exp(lse_c - max_lse)
//   out     = sum(w_c * out_c) / sum(w_c)
//
// Grid: (D, qL, B*H) — one thread per output element, dispatch_threads
///////////////////////////////////////////////////////////////////////////////

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
  // tid.x = d (head dimension index)
  // tid.y = q (query sequence index)
  // tid.z = bh (batch*head linear index)

  const int d = tid.x;
  const int q = tid.y;
  const int bh = tid.z;

  if (d >= D || q >= qL)
    return;

  // Decompose bh into batch and head indices
  const int h = bh % H;
  const int b = bh / H;

  // Linear index within the BHqL plane (for chunk_outs and chunk_lses)
  const int64_t bhq = int64_t(bh) * int64_t(qL) + int64_t(q);

  // --- Pass 1: find max logsumexp across chunks ---
  float max_lse = -INFINITY;
  for (int c = 0; c < n_chunks; c++) {
    int64_t lse_idx = int64_t(c) * int64_t(BHqL) + bhq;
    float lse_val = chunk_lses[lse_idx];
    max_lse = max(max_lse, lse_val);
  }

  // --- Pass 2: accumulate weighted sum and total weight ---
  // Guard: when all keys in a chunk are causally masked, the kernel output
  // is NaN (0/0 in softmax) and lse is -inf.  exp(-inf - max_lse) = 0,
  // but 0 * NaN = NaN in IEEE 754.  Skip zero-weight chunks to avoid this.
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

  // Normalize
  float result = (sum_w > 0.0f) ? (acc / sum_w) : 0.0f;

  // Write to strided output: O_strides = [batch_stride, head_stride, seq_stride]
  // D dimension stride is 1 (innermost, contiguous)
  int64_t o_idx = int64_t(b) * O_strides[0] +
                  int64_t(h) * O_strides[1] +
                  int64_t(q) * O_strides[2] +
                  int64_t(d);

  output[o_idx] = T(result);
}
