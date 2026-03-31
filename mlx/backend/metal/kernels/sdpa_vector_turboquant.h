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

template <typename T, int D, int MSE_BITS = 2, int V_BITS = 2>
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
  constexpr int BN = 32; // Number of SIMD groups (KV token stride)
  constexpr int BD = 32; // Threads per SIMD group (dimension stride)
  constexpr int per_thread = D / BD; // Coordinates per thread

  // MSE unpacking constants (derived from template params)
  constexpr int mse_bits = MSE_BITS;
  constexpr int mse_vpb = 8 / mse_bits; // values per byte
  constexpr uint mse_mask = (1u << mse_bits) - 1u;

  // Value unpacking constants (derived from template params)
  constexpr int v_bits = V_BITS;
  constexpr int v_vpb = 8 / v_bits;
  constexpr uint v_mask = (1u << v_bits) - 1u;

  typedef float U;

  // Thread-private storage
  thread U q_r[per_thread]; // Rotated query coordinates
  thread U q_s[per_thread]; // Sketched query coordinates
  thread U o[per_thread]; // Output accumulator

  // Threadgroup memory for cross-SIMD-group reduction
  threadgroup U tg_outputs[BN * BD];
  threadgroup U tg_max_scores[BN];
  threadgroup U tg_sum_scores[BN];

  // --- Position computation ---
  const int q_batch_head_idx = tid.x; // [0, B*H_q)
  const int q_seq_idx = tid.y; // [0, qL)
  const int kv_head_idx = q_batch_head_idx / params.gqa_factor;

  // Offset into pre-rotated/sketched query arrays (B*H_q, qL, D) layout
  const int q_offset =
      (q_batch_head_idx * int(tpg.y) + q_seq_idx) * D + simd_lid * per_thread;

  // Load query coordinates for this thread (pre-scaled by attention scale)
  for (int i = 0; i < per_thread; i++) {
    q_r[i] = static_cast<U>(params.scale) * static_cast<U>(q_rot[q_offset + i]);
    q_s[i] =
        static_cast<U>(params.scale) * static_cast<U>(q_sketch[q_offset + i]);
    o[i] = U(0);
  }

  // Cache centroids in registers (2^MSE_BITS values)
  constexpr int n_cent = 1 << MSE_BITS;
  thread U c[n_cent];
  for (int i = 0; i < n_cent; i++) {
    c[i] = centroids[i];
  }

  // --- KV base offsets (contiguous B*H_kv, N, packed_dim layout) ---
  const long kv_packed_base =
      long(kv_head_idx) * long(params.N) * long(params.packed_d_mse);
  const long kv_signs_base =
      long(kv_head_idx) * long(params.N) * long(params.packed_d_signs);
  const long kv_norms_base = long(kv_head_idx) * long(params.N);
  const long kv_v_packed_base =
      long(kv_head_idx) * long(params.N) * long(params.packed_d_v);
  const long kv_v_sg_base =
      long(kv_head_idx) * long(params.N) * long(params.n_groups);

  U max_score = -INFINITY;
  U sum_exp_score = U(0);

  // Coordinate range for this thread
  const int coord_start = simd_lid * per_thread;

  // QJL sign byte and bit offset for this thread's coordinates
  const int sign_byte_for_thread = coord_start / 8;
  const int sign_bit_offset = coord_start % 8;

  // --- Main loop: stride over KV tokens ---
  for (int n = simd_gid; n < params.N; n += BN) {
    // === MSE SCORE ===
    U mse_partial = U(0);
    {
      const long mse_row_base =
          kv_packed_base + long(n) * long(params.packed_d_mse);
      for (int sub = 0; sub < per_thread; sub++) {
        const int global_coord = coord_start + sub;
        const int byte_idx = global_coord / mse_vpb;
        const int sub_idx = global_coord % mse_vpb;
        if (byte_idx < params.packed_d_mse) {
          const uint8_t packed = k_packed[mse_row_base + byte_idx];
          const uint idx = (uint(packed) >> (sub_idx * mse_bits)) & mse_mask;
          mse_partial += q_r[sub] * c[idx];
        }
      }
    }
    U mse_score = simd_sum(mse_partial);
    mse_score *= k_norms[kv_norms_base + n];

    // === QJL CORRECTION ===
    U qjl_partial = U(0);
    if (sign_byte_for_thread < params.packed_d_signs) {
      const uint8_t packed_signs = k_signs
          [kv_signs_base + long(n) * long(params.packed_d_signs) +
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
    {
      const long v_row_base =
          kv_v_packed_base + long(n) * long(params.packed_d_v);
      const int group_idx = coord_start / params.group_size;
      const long sg_offset = kv_v_sg_base + long(n) * long(params.n_groups);
      const U scale_val = v_scales[sg_offset + group_idx];
      const U zero_val = v_zeros[sg_offset + group_idx];
      for (int sub = 0; sub < per_thread; sub++) {
        const int global_coord = coord_start + sub;
        const int byte_idx = global_coord / v_vpb;
        const int sub_idx = global_coord % v_vpb;
        if (byte_idx < params.packed_d_v) {
          const uint8_t packed_v = v_packed[v_row_base + byte_idx];
          const uint qval = (uint(packed_v) >> (sub_idx * v_bits)) & v_mask;
          const U val = U(qval) * scale_val + zero_val;
          o[sub] = o[sub] * factor + exp_score * val;
        } else {
          o[sub] *= factor;
        }
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

///////////////////////////////////////////////////////////////////////////////
// TurboQuant 2-pass attention (long sequences, N >= 1024)
///////////////////////////////////////////////////////////////////////////////

// Pass 1: Each threadgroup handles a BLOCK of KV tokens (stride by blocks
// count). Single SIMD group (32 threads) per threadgroup, same D-splitting as
// 1-pass. Grid: (H_kv, B, blocks). Threadgroup: (32, gqa_factor, 1).
template <typename T, int D, int MSE_BITS = 2, int V_BITS = 2>
[[kernel]] void sdpa_vector_turboquant_2pass_1(
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
    device float* out_sums [[buffer(11)]],
    device float* out_maxs [[buffer(12)]],
    const constant mlx::steel::TurboQuantAttnParams& params [[buffer(13)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int per_thread = D / BD;
  constexpr int mse_bits = MSE_BITS;
  constexpr int mse_vpb = 8 / mse_bits;
  constexpr uint mse_mask = (1u << mse_bits) - 1u;
  constexpr int v_bits = V_BITS;
  constexpr int v_vpb = 8 / v_bits;
  constexpr uint v_mask = (1u << v_bits) - 1u;

  typedef float U;

  thread U q_r[per_thread];
  thread U q_s[per_thread];
  thread U o[per_thread] = {0};

  // Grid positions
  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * gqa_factor;
  const int blocks = tpg.z;

  const int q_batch_head_idx = batch_idx * num_q_heads + q_head_idx;

  // Load query (pre-scaled)
  const int q_offset = q_batch_head_idx * D + simd_lid * per_thread;
  for (int i = 0; i < per_thread; i++) {
    q_r[i] = static_cast<U>(params.scale) * static_cast<U>(q_rot[q_offset + i]);
    q_s[i] =
        static_cast<U>(params.scale) * static_cast<U>(q_sketch[q_offset + i]);
  }

  // Cache centroids
  constexpr int n_cent = 1 << MSE_BITS;
  thread U c[n_cent];
  for (int i = 0; i < n_cent; i++) {
    c[i] = centroids[i];
  }

  // KV base offsets — include batch dimension (data layout: B, H_kv, N,
  // packed_d)
  const int kv_batch_head = batch_idx * num_kv_heads + kv_head_idx;
  const long kv_packed_base =
      long(kv_batch_head) * long(params.N) * long(params.packed_d_mse);
  const long kv_signs_base =
      long(kv_batch_head) * long(params.N) * long(params.packed_d_signs);
  const long kv_norms_base = long(kv_batch_head) * long(params.N);
  const long kv_v_packed_base =
      long(kv_batch_head) * long(params.N) * long(params.packed_d_v);
  const long kv_v_sg_base =
      long(kv_batch_head) * long(params.N) * long(params.n_groups);

  const int coord_start = simd_lid * per_thread;
  const int sign_byte = coord_start / 8;
  const int sign_bit_off = coord_start % 8;

  U max_score = -INFINITY;
  U sum_exp_score = U(0);

  // Block-strided loop over KV tokens
  for (int n = block_idx; n < params.N; n += blocks) {
    // MSE score (per-coordinate byte indexing for any bit width)
    U mse_partial = U(0);
    {
      const long mse_row_base =
          kv_packed_base + long(n) * long(params.packed_d_mse);
      for (int sub = 0; sub < per_thread; sub++) {
        const int global_coord = coord_start + sub;
        const int byte_idx = global_coord / mse_vpb;
        const int sub_idx = global_coord % mse_vpb;
        if (byte_idx < params.packed_d_mse) {
          const uint8_t packed = k_packed[mse_row_base + byte_idx];
          mse_partial +=
              q_r[sub] * c[(uint(packed) >> (sub_idx * mse_bits)) & mse_mask];
        }
      }
    }
    U mse_score = simd_sum(mse_partial) * k_norms[kv_norms_base + n];

    // QJL correction
    U qjl_partial = U(0);
    if (sign_byte < params.packed_d_signs) {
      const uint8_t ps = k_signs
          [kv_signs_base + long(n) * long(params.packed_d_signs) + sign_byte];
      for (int sub = 0; sub < per_thread; sub++) {
        U sv = ((uint(ps) >> (sign_bit_off + sub)) & 1u) ? U(1.0) : U(-1.0);
        qjl_partial += q_s[sub] * sv;
      }
    }
    U score = mse_score +
        simd_sum(qjl_partial) * k_res_norms[kv_norms_base + n] *
            params.qjl_scale;

    // Online softmax
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    // Value dequant + accumulate (per-coordinate byte indexing)
    {
      const long v_row_base =
          kv_v_packed_base + long(n) * long(params.packed_d_v);
      const int gi = coord_start / params.group_size;
      const long sg_off = kv_v_sg_base + long(n) * long(params.n_groups);
      const U sv = v_scales[sg_off + gi];
      const U zv = v_zeros[sg_off + gi];
      for (int sub = 0; sub < per_thread; sub++) {
        const int global_coord = coord_start + sub;
        const int byte_idx = global_coord / v_vpb;
        const int sub_idx = global_coord % v_vpb;
        if (byte_idx < params.packed_d_v) {
          const uint8_t pv = v_packed[v_row_base + byte_idx];
          U val = U((uint(pv) >> (sub_idx * v_bits)) & v_mask) * sv + zv;
          o[sub] = o[sub] * factor + exp_score * val;
        } else {
          o[sub] *= factor;
        }
      }
    }
  }

  // Write partial results for this block
  const int out_idx = q_batch_head_idx;
  if (simd_lid == 0) {
    out_sums[out_idx * blocks + block_idx] = sum_exp_score;
    out_maxs[out_idx * blocks + block_idx] = max_score;
  }
  // Each thread writes its per_thread output coords
  const int out_base =
      out_idx * blocks * D + block_idx * D + simd_lid * per_thread;
  for (int i = 0; i < per_thread; i++) {
    out[out_base + i] = static_cast<T>(o[i]);
  }
}

// Pass 2: Merge partial results across blocks. Outputs UNNORMALIZED (acc, m,
// l). Grid: (B*H_q, 1, 1). Threadgroup: (1024, 1, 1) = 32 SIMD groups × 32
// threads.
template <typename T, int D>
[[kernel]] void sdpa_vector_turboquant_2pass_2(
    const device T* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    device T* out [[buffer(3)]],
    device float* out_m [[buffer(4)]],
    device float* out_l [[buffer(5)]],
    const constant int& blocks [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;

  typedef float U;

  thread U o[elem_per_thread] = {0};
  threadgroup U tg_outputs[BN * BD];

  const int head_idx = tid.x;
  const device T* p = partials + head_idx * blocks * D + simd_gid * D +
      simd_lid * elem_per_thread;
  const device float* s = sums + head_idx * blocks;
  const device float* m = maxs + head_idx * blocks;

  // Find global max across all blocks
  U max_score = -INFINITY;
  U sum_exp_score = U(0);
  for (int b = 0; b < blocks / BN; ++b) {
    max_score = max(max_score, m[simd_lid + BN * b]);
  }
  max_score = simd_max(max_score);

  // Compute global sum with rescaling
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(m[simd_lid + BN * b] - max_score);
    sum_exp_score += factor * s[simd_lid + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);

  // Accumulate rescaled partials
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(m[simd_gid + BN * b] - max_score);
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] += factor * static_cast<U>(p[i]);
    }
    p += BN * D;
  }

  // Transpose reduction across SIMD groups (same as 1-pass)
  for (int i = 0; i < elem_per_thread; i++) {
    tg_outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(tg_outputs[simd_gid * BD + simd_lid]);
    // NOT normalizing — output is unnormalized for log-sum-exp merge
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write unnormalized output + softmax state
  if (simd_lid == 0) {
    const int out_base = head_idx * D + simd_gid * elem_per_thread;
    for (int i = 0; i < elem_per_thread; i++) {
      out[out_base + i] = static_cast<T>(o[i]);
    }
    if (simd_gid == 0) {
      out_m[head_idx] = max_score;
      out_l[head_idx] = sum_exp_score;
    }
  }
}
