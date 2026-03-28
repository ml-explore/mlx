// Copyright © 2024-25 Apple Inc.
// TurboQuant fused attention: computes attention directly from compressed KV
// cache data (MSE quantized keys + QJL sign correction + quantized values).
//
// Follows the sdpa_vector pattern: SIMD groups stride over KV tokens, threads
// within each SIMD group split the head dimension D.

#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/steel/attn/params_turboquant.h"

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// TurboQuant vector attention kernel (decode path, qL <= 8)
///////////////////////////////////////////////////////////////////////////////

template <typename T, int D>
[[kernel]] void sdpa_vector_turboquant(
    const device T* q_rot [[buffer(0)]],
    const device T* q_sketch [[buffer(1)]],
    const device uint8_t* k_packed [[buffer(2)]],
    const device uint8_t* k_signs [[buffer(3)]],
    const device float* k_norms [[buffer(4)]],
    const device float* k_res_norms [[buffer(5)]],
    const device float* centroids [[buffer(6)]],
    const device uint8_t* v_packed [[buffer(7)]],
    const device float* v_scales [[buffer(8)]],
    const device float* v_zeros [[buffer(9)]],
    device T* out [[buffer(10)]],
    device float* out_m [[buffer(11)]],
    device float* out_l [[buffer(12)]],
    const constant mlx::steel::TurboQuantAttnParams& params [[buffer(13)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  // Thread/SIMD layout: 1024 threads = 32 SIMD groups × 32 threads
  constexpr int BN = 32;    // Number of SIMD groups (KV token stride)
  constexpr int BD = 32;    // Threads per SIMD group (dimension stride)
  constexpr int per_thread = D / BD; // Coordinates per thread

  // MSE unpacking constants (2-bit: 4 values per byte)
  constexpr int mse_bits = 2;
  constexpr int mse_vpb = 8 / mse_bits; // values per byte = 4
  constexpr uint mse_mask = (1u << mse_bits) - 1u;

  // Value unpacking constants (2-bit: 4 values per byte)
  constexpr int v_bits = 2;
  constexpr int v_vpb = 8 / v_bits;
  constexpr uint v_mask = (1u << v_bits) - 1u;

  typedef float U;

  // Thread-private storage
  thread U q_r[per_thread]; // Rotated query coordinates
  thread U q_s[per_thread]; // Sketched query coordinates
  thread U o[per_thread];   // Output accumulator

  // Threadgroup memory for cross-SIMD-group reduction
  threadgroup U tg_outputs[BN * BD];
  threadgroup U tg_max_scores[BN];
  threadgroup U tg_sum_scores[BN];

  // --- Position computation ---
  const int q_batch_head_idx = tid.x; // [0, B*H_q)
  const int q_seq_idx = tid.y;        // [0, qL)
  const int kv_head_idx = q_batch_head_idx / params.gqa_factor;

  // Offset into pre-rotated/sketched query arrays (B*H_q, qL, D) layout
  const int q_offset =
      (q_batch_head_idx * int(tpg.y) + q_seq_idx) * D +
      simd_lid * per_thread;

  // Load query coordinates for this thread (pre-scaled by attention scale)
  for (int i = 0; i < per_thread; i++) {
    q_r[i] = static_cast<U>(params.scale) * static_cast<U>(q_rot[q_offset + i]);
    q_s[i] = static_cast<U>(params.scale) * static_cast<U>(q_sketch[q_offset + i]);
    o[i] = U(0);
  }

  // Cache centroids in registers (only 4 values for 2-bit MSE)
  thread U c[4];
  for (int i = 0; i < params.n_centroids && i < 4; i++) {
    c[i] = centroids[i];
  }

  // --- KV base offsets (contiguous B*H_kv, N, packed_dim layout) ---
  const long kv_packed_base =
      long(kv_head_idx) * long(params.N) * long(params.packed_d_mse);
  const long kv_signs_base =
      long(kv_head_idx) * long(params.N) * long(params.packed_d_signs);
  const long kv_norms_base =
      long(kv_head_idx) * long(params.N);
  const long kv_v_packed_base =
      long(kv_head_idx) * long(params.N) * long(params.packed_d_v);
  const long kv_v_sg_base =
      long(kv_head_idx) * long(params.N) * long(params.n_groups);

  U max_score = -INFINITY;
  U sum_exp_score = U(0);

  // Coordinate range for this thread
  const int coord_start = simd_lid * per_thread;

  // MSE byte index for this thread's coordinates
  // For 2-bit MSE with per_thread=4: exactly 1 byte per thread
  const int mse_byte_for_thread = coord_start / mse_vpb;

  // QJL sign byte and bit offset for this thread's coordinates
  const int sign_byte_for_thread = coord_start / 8;
  const int sign_bit_offset = coord_start % 8;

  // Value byte index (same as MSE for 2-bit)
  const int v_byte_for_thread = coord_start / v_vpb;

  // --- Main loop: stride over KV tokens ---
  for (int n = simd_gid; n < params.N; n += BN) {

    // === MSE SCORE ===
    U mse_partial = U(0);
    if (mse_byte_for_thread < params.packed_d_mse) {
      const uint8_t packed =
          k_packed[kv_packed_base + long(n) * long(params.packed_d_mse) +
                   mse_byte_for_thread];
      for (int sub = 0; sub < per_thread; sub++) {
        const uint idx = (uint(packed) >> (sub * mse_bits)) & mse_mask;
        mse_partial += q_r[sub] * c[idx];
      }
    }
    U mse_score = simd_sum(mse_partial);
    mse_score *= k_norms[kv_norms_base + n];

    // === QJL CORRECTION ===
    U qjl_partial = U(0);
    if (sign_byte_for_thread < params.packed_d_signs) {
      const uint8_t packed_signs =
          k_signs[kv_signs_base + long(n) * long(params.packed_d_signs) +
                  sign_byte_for_thread];
      for (int sub = 0; sub < per_thread; sub++) {
        const int bit_pos = sign_bit_offset + sub;
        const U sign_val =
            ((uint(packed_signs) >> bit_pos) & 1u) ? U(1.0) : U(-1.0);
        qjl_partial += q_s[sub] * sign_val;
      }
    }
    U qjl_score = simd_sum(qjl_partial);
    qjl_score *= k_res_norms[kv_norms_base + n] * params.qjl_scale;

    // Combined score (scale already baked into q_r and q_s at load time)
    const U score = mse_score + qjl_score;

    // === ONLINE SOFTMAX UPDATE ===
    const U new_max = max(max_score, score);
    const U factor = fast::exp(max_score - new_max);
    const U exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    // === VALUE DEQUANT + WEIGHTED ACCUMULATE ===
    if (v_byte_for_thread < params.packed_d_v) {
      const uint8_t packed_v =
          v_packed[kv_v_packed_base + long(n) * long(params.packed_d_v) +
                   v_byte_for_thread];
      // Hoist scale/zero loads (all per_thread coords share same group)
      const int group_idx = coord_start / params.group_size;
      const long sg_offset = kv_v_sg_base + long(n) * long(params.n_groups);
      const U scale_val = v_scales[sg_offset + group_idx];
      const U zero_val = v_zeros[sg_offset + group_idx];
      for (int sub = 0; sub < per_thread; sub++) {
        const uint qval = (uint(packed_v) >> (sub * v_bits)) & v_mask;
        const U val = U(qval) * scale_val + zero_val;
        o[sub] = o[sub] * factor + exp_score * val;
      }
    } else {
      // Thread handles no value coordinates (D not multiple of BD)
      for (int sub = 0; sub < per_thread; sub++) {
        o[sub] *= factor;
      }
    }
  }

  // === CROSS-SIMD-GROUP REDUCTION ===
  // Each SIMD group has processed a different subset of KV tokens.
  // Merge their online softmax states.

  // 1. Communicate per-SIMD-group max and sum
  if (simd_lid == 0) {
    tg_max_scores[simd_gid] = max_score;
    tg_sum_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 2. Compute global max and rescaling factors
  // Each thread reads a different SIMD group's max (simd_lid → group index)
  max_score = tg_max_scores[simd_lid];
  const U global_max = simd_max(max_score);
  const U factor = fast::exp(max_score - global_max);
  sum_exp_score = simd_sum(tg_sum_scores[simd_lid] * factor);

  // 3. Aggregate outputs across SIMD groups via transpose pattern
  for (int i = 0; i < per_thread; i++) {
    // Write: row=simd_lid (within-group thread), col=simd_gid (group)
    tg_outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Read transposed: row=simd_gid, col=simd_lid
    // factor holds rescaling for SIMD group simd_lid
    // NOTE: do NOT normalize — output is unnormalized (acc, m, l) for merge
    o[i] = simd_sum(tg_outputs[simd_gid * BD + simd_lid] * factor);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // === WRITE OUTPUT (unnormalized acc + softmax state) ===
  // After transpose reduction, SIMD group simd_gid owns output coords
  // [simd_gid * per_thread, (simd_gid+1) * per_thread)
  const int tg_idx = q_batch_head_idx * int(tpg.y) + q_seq_idx;
  if (simd_lid == 0) {
    const int out_offset = tg_idx * D + simd_gid * per_thread;
    for (int i = 0; i < per_thread; i++) {
      out[out_offset + i] = static_cast<T>(o[i]);
    }
    // Write m and l once per threadgroup (only first SIMD group)
    if (simd_gid == 0) {
      out_m[tg_idx] = global_max;
      out_l[tg_idx] = sum_exp_score;
    }
  }
}
