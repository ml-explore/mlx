// Copyright © 2026 Apple Inc.

#pragma once

// MoE Expert Parallelism Metal Kernels
//
// These kernels accelerate the scatter/gather operations in
// MoeDispatchExchange and MoeCombineExchange primitives.

// Kernel 1: moe_dispatch_local
// Scatters LOCAL tokens into the dispatched buffer using a precomputed
// slot_map. slot_map[nk] = flat_idx into dispatched[E_local, cap_total, D], or
// -1 to skip.
//
// Grid: (D_ceil, valid_count, 1)  where D_ceil = ceil(D / ELEM_PER_THREAD)
// Group: (min(D_ceil, 256), 1, 1)
//
// Note: valid_count = number of (n,k) pairs with slot_map >= 0
// nk_indices[i] = original n*top_k+k index for the i-th valid entry
template <typename T>
[[kernel]] void moe_dispatch_local(
    const device T* tokens [[buffer(0)]], // [N, D]
    device T* dispatched [[buffer(1)]], // [E_local * cap_total * D] flat
    const device int* slot_map [[buffer(2)]], // [valid_count] flat_idx values
    const device int* nk_indices
    [[buffer(3)]], // [valid_count] n*top_k+k originals
    constant int& D [[buffer(4)]],
    constant int& top_k [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  int d = gid.x;
  int i = gid.y;
  if (d >= D)
    return;

  int flat_idx = slot_map[i];
  int nk = nk_indices[i];
  int n = nk / top_k;

  dispatched[static_cast<long>(flat_idx) * D + d] =
      tokens[static_cast<long>(n) * D + d];
}

// Kernel 2: moe_dispatch_scatter_remote
// Scatters received remote tokens (from RDMA exchange) into dispatched buffer.
// recv_meta[i] = (local_expert, pos) packed as meta32
// recv_payload is contiguous [cnt, D] after meta extraction on CPU.
//
// Grid: (D_ceil, cnt, 1)
// Group: (min(D_ceil, 256), 1, 1)
template <typename T>
[[kernel]] void moe_dispatch_scatter_remote(
    const device T* recv_payload [[buffer(0)]], // [cnt, D]
    device T* dispatched [[buffer(1)]], // [E_local * cap_total * D] flat
    const device int* recv_flat_idx
    [[buffer(2)]], // [cnt] precomputed flat indices
    constant int& D [[buffer(3)]],
    constant int& cnt [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  int d = gid.x;
  int i = gid.y;
  if (d >= D || i >= cnt)
    return;

  int flat_idx = recv_flat_idx[i];
  dispatched[static_cast<long>(flat_idx) * D + d] =
      recv_payload[static_cast<long>(i) * D + d];
}

// Kernel 3: moe_combine_gather_remote
// Gathers expert outputs for peer's requested tokens into a send buffer.
// For each request i, lookup expert_out at flat index and copy to send_results.
//
// Grid: (D_ceil, cnt, 1)
// Group: (min(D_ceil, 256), 1, 1)
template <typename T>
[[kernel]] void moe_combine_gather_remote(
    const device T* expert_out [[buffer(0)]], // [E_local * cap_total * D] flat
    device T* send_results [[buffer(1)]], // [cnt, D]
    const device int* eo_flat_idx
    [[buffer(2)]], // [cnt] precomputed flat indices
    constant int& D [[buffer(3)]],
    constant int& cnt [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  int d = gid.x;
  int i = gid.y;
  if (d >= D || i >= cnt)
    return;

  int flat_idx = eo_flat_idx[i];
  send_results[static_cast<long>(i) * D + d] =
      expert_out[static_cast<long>(flat_idx) * D + d];
}

// Kernel 4: moe_combine_weighted_sum
// Performs weighted accumulation of expert outputs per token.
// For each token n, sums weights[n,k] * src[k_data_idx, d] for k=0..top_k-1.
// Uses float32 accumulation for precision regardless of input dtype.
//
// data_src contains interleaved local and remote results indexed by src_idx.
// src_idx[n * top_k + k] = index into data_src for that (n,k) pair, or -1 to
// skip.
//
// Grid: (D_ceil, N, 1)
// Group: (min(D_ceil, 256), 1, 1)
template <typename T>
[[kernel]] void moe_combine_weighted_sum(
    const device T* data_src [[buffer(0)]], // [total_entries, D]
    device T* output [[buffer(1)]], // [N, D]
    const device T* original [[buffer(2)]], // [N, D] fallback
    const device float* weights [[buffer(3)]], // [N, top_k]
    const device int* src_idx [[buffer(4)]], // [N * top_k]
    constant int& D [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant int& top_k [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
  int d = gid.x;
  int n = gid.y;
  if (d >= D || n >= N)
    return;

  float accum = 0.0f;
  bool has_valid = false;

  for (int k = 0; k < top_k; k++) {
    int idx = src_idx[n * top_k + k];
    if (idx >= 0) {
      has_valid = true;
      float w = weights[n * top_k + k];
      accum += w * static_cast<float>(data_src[static_cast<long>(idx) * D + d]);
    }
  }

  if (has_valid) {
    output[static_cast<long>(n) * D + d] = static_cast<T>(accum);
  } else {
    output[static_cast<long>(n) * D + d] =
        original[static_cast<long>(n) * D + d];
  }
}

// Kernel 5: moe_packet_gather
// Gathers rows from a source buffer into packet format with 16B headers.
// Each packet row = [header(16B) | payload(D*sizeof(T)) | pad] aligned to
// row_stride. Used for dispatch remote pack and combine response pack.
//
// Grid: (D, cnt, 1)
// Group: (min(D, 256), 1, 1)
template <typename T>
[[kernel]] void moe_packet_gather(
    const device T* source [[buffer(0)]], // flat source [rows, D]
    device uint8_t* packet [[buffer(1)]], // [cnt, row_stride]
    const device int* src_idx [[buffer(2)]], // [cnt] source row indices
    const device uint32_t* headers [[buffer(3)]], // [cnt] header values
    constant int& D [[buffer(4)]],
    constant int& cnt [[buffer(5)]],
    constant int& row_stride [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
  int d = gid.x;
  int i = gid.y;
  if (d >= D || i >= cnt)
    return;

  long pkt_base = (long)i * row_stride;

  // Write header into 16B-aligned region (first thread per row only)
  if (d == 0) {
    *reinterpret_cast<device uint32_t*>(packet + pkt_base) = headers[i];
  }

  // Write payload at offset 16 (aligned for vectorized access)
  int row = src_idx[i];
  device T* payload = reinterpret_cast<device T*>(packet + pkt_base + 16);
  payload[d] = source[(long)row * D + d];
}

// Kernel 6: moe_packet_scatter
// Scatters payload from packet format into a target buffer.
// Each packet row = [header(16B) | payload(D*sizeof(T)) | pad].
// flat_idx provides the destination row index in the target buffer.
//
// Grid: (D, cnt, 1)
// Group: (min(D, 256), 1, 1)
template <typename T>
[[kernel]] void moe_packet_scatter(
    const device uint8_t* packet [[buffer(0)]], // [cnt, row_stride]
    device T* target [[buffer(1)]], // flat target buffer
    const device int* flat_idx [[buffer(2)]], // [cnt] target row indices
    constant int& D [[buffer(3)]],
    constant int& cnt [[buffer(4)]],
    constant int& row_stride [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  int d = gid.x;
  int i = gid.y;
  if (d >= D || i >= cnt)
    return;

  long pkt_base = (long)i * row_stride;
  // Read payload at offset 16 (aligned)
  const device T* payload =
      reinterpret_cast<const device T*>(packet + pkt_base + 16);
  int out_idx = flat_idx[i];
  target[(long)out_idx * D + d] = payload[d];
}
