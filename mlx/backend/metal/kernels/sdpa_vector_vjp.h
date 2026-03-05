// Copyright © 2024 Apple Inc.

#pragma once

#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/atomic.h"

using namespace metal;

// Note: Function constants (has_mask, query_transposed, do_causal, bool_mask,
// float_mask, has_sinks) are defined in sdpa_vector.h with indices 20-25.
// blocks is at index 26. This header assumes sdpa_vector.h is included first.
constant bool direct_write [[function_constant(27)]];
constant bool dq_only [[function_constant(29)]];

///////////////////////////////////////////////////////////////////////////////
// SDPA Vector VJP Kernel
//
// Computes gradients dQ, dK, dV for scaled dot-product attention backward pass.
//
// Forward: O = softmax(scale * Q @ K^T) @ V
//
// Backward (VJP):
//   P = softmax(scale * Q @ K^T)  [reconstructed from logsumexp]
//   dV = P^T @ dO
//   dP = dO @ V^T
//   dS = P * (dP - sum(dP * P))  [softmax gradient]
//   dQ = scale * dS @ K
//   dK = scale * dS^T @ Q
//
// This kernel handles the "vector" case where Q_seq is small (<=8).
// Each threadgroup processes one (batch, head, q_seq) position.
//
// IMPORTANT: Stride Assumption
// This kernel uses input strides (k_head_stride, k_seq_stride, v_head_stride,
// v_seq_stride) for output array (d_keys, d_values) pointer arithmetic.
// The dispatch code must ensure that output arrays have matching strides:
//   - d_k.strides() must match k.strides() for head and seq dimensions
//   - d_v.strides() must match v.strides() for head and seq dimensions
// Failure to maintain this invariant will cause memory corruption.
///////////////////////////////////////////////////////////////////////////////

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector_vjp(
    // Forward inputs
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    // Forward output and upstream gradient
    const device T* out [[buffer(3)]],
    const device T* d_out [[buffer(4)]],
    // Logsumexp from forward (for numerically stable softmax reconstruction)
    const device float* logsumexp [[buffer(5)]],
    // Gradient outputs
    device T* d_queries [[buffer(6)]],
    device T* d_keys [[buffer(7)]],
    device T* d_values [[buffer(8)]],
    // Attention parameters
    const constant int& gqa_factor [[buffer(9)]],
    const constant int& N [[buffer(10)]], // KV sequence length
    const constant size_t& k_head_stride [[buffer(11)]],
    const constant size_t& k_seq_stride [[buffer(12)]],
    const constant size_t& v_head_stride [[buffer(13)]],
    const constant size_t& v_seq_stride [[buffer(14)]],
    const constant float& scale [[buffer(15)]],
    // Output (O/dO) stride parameters - STEEL forward may produce non-row-major
    // layout Physical layout can be BLHV (strides [L*H*V, V, H*V, 1]) vs
    // logical BHLV
    const constant int& num_q_heads [[buffer(16)]],
    const constant size_t& o_batch_stride [[buffer(17)]],
    const constant size_t& o_head_stride [[buffer(18)]],
    const constant size_t& o_seq_stride [[buffer(19)]],
    // Optional mask inputs
    const device bool* bmask [[buffer(20), function_constant(bool_mask)]],
    const device T* fmask [[buffer(21), function_constant(float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(22), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(23), function_constant(has_mask)]],
    const constant int& mask_head_stride
    [[buffer(24), function_constant(has_mask)]],
    // Optional attention sinks
    const device T* sinks [[buffer(25), function_constant(has_sinks)]],
    // Delta output for 2-kernel split (dQ-only pass writes delta for dK/dV
    // pass)
    device float* delta_out [[buffer(26), function_constant(dq_only)]],
    // Thread position info
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  // Block sizes matching forward kernel
  constexpr int BN = 32; // Number of simdgroups (parallel KV positions)
  constexpr int BD = 32; // Simdgroup width (threads per simdgroup)
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U; // Accumulator type for numerical stability

  // Thread-local storage for queries, keys, values, gradients
  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U v[v_per_thread];
  thread U dq[qk_per_thread]; // Gradient w.r.t. query
  thread U d_o[v_per_thread]; // Upstream gradient
  thread U o[v_per_thread]; // Forward output (for delta computation)

  // Threadgroup memory for reductions and communication
  threadgroup U shared_delta[1]; // delta = sum(dO * O)
  // For D=256, shared_dQ[BN * 256] = 32KB which hits Metal's limit with delta_smem.
  // Use BN * min(D, 128) and do two passes when D > 128.
  constexpr int D_TILE = D > 128 ? 128 : D;
  threadgroup U shared_dQ[BN * D_TILE]; // For dQ reduction across simdgroups

  // Compute positions (same as forward)
  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  // Decompose batch_head index into batch and head for stride-based O/dO access
  // STEEL forward produces BLHV physical layout, so we need explicit strides
  const int batch_idx = q_batch_head_idx / num_q_heads;
  const int head_idx = q_batch_head_idx % num_q_heads;
  // LSE is row-major [B*H, L], so keep the combined index for LSE access
  const int lse_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  const int q_offset =
      query_transposed ? tpg.x * q_seq_idx + q_batch_head_idx : lse_offset;

  // Set up input pointers
  queries += q_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;

  // Set up mask pointers if needed
  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }

  // Set up output/gradient pointers
  // Use explicit strides for O/dO to handle BLHV physical layout from STEEL
  // For BLHV strides: o_batch_stride = L*H*V, o_head_stride = V, o_seq_stride =
  // H*V
  out += batch_idx * o_batch_stride + head_idx * o_head_stride +
      q_seq_idx * o_seq_stride + simd_lid * v_per_thread;
  d_out += batch_idx * o_batch_stride + head_idx * o_head_stride +
      q_seq_idx * o_seq_stride + simd_lid * v_per_thread;
  // LSE is row-major [B*H, L] - no stride adjustment needed
  logsumexp += lse_offset;

  d_queries += lse_offset * D + simd_lid * qk_per_thread;
  d_keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  d_values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;

  // Load query (scaled by M_LOG2E_F to match exp2 domain)
  const U log2e_scale = static_cast<U>(scale * M_LOG2E_F);
  const U inv_log2e = static_cast<U>(1.0f / M_LOG2E_F);
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = log2e_scale * queries[i];
  }

  // Initialize dQ accumulator to zero
  for (int i = 0; i < qk_per_thread; i++) {
    dq[i] = 0;
  }

  // Load forward output O and upstream gradient dO
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = out[i];
    d_o[i] = d_out[i];
  }

  // Compute delta = sum(dO * O) - needed for softmax gradient
  // This is invariant across all KV positions for this query
  U local_delta = 0;
  for (int i = 0; i < v_per_thread; i++) {
    local_delta += d_o[i] * o[i];
  }
  // Sum across simdgroup
  local_delta = simd_sum(local_delta);

  // First simdgroup stores delta to shared memory
  if (simd_gid == 0 && simd_lid == 0) {
    shared_delta[0] = local_delta;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  U delta = shared_delta[0];

  // In dq_only mode, write delta for the dK/dV kernel
  if (dq_only) {
    if (simd_gid == 0 && simd_lid == 0) {
      delta_out[lse_offset] = delta;
    }
  }

  // Load logsumexp for this query position
  U lse = logsumexp[0];

  // shared_dQ is initialized per-pass in the dQ write-back section below

  // Main loop over KV sequence
  for (int kv_idx = simd_gid; kv_idx < N; kv_idx += BN) {
    bool use_key = true;

    // Apply causal or explicit mask
    if (do_causal) {
      use_key = kv_idx <= (N - int(tpg.y) + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }

    if (use_key) {
      // Load key for this position
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = keys[j];
      }

      // Load value for this position
      for (int j = 0; j < v_per_thread; j++) {
        v[j] = values[j];
      }

      // Reconstruct attention score: S = scale * Q @ K^T
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);

      // Add float mask if present (scaled by M_LOG2E_F to match log2 domain)
      if (float_mask) {
        score += static_cast<U>(M_LOG2E_F) * static_cast<U>(fmask[0]);
      }

      // Reconstruct attention probability: P = exp2(S - logsumexp)
      // Using exp2 to match STEEL attention domain (logsumexp is in log2
      // domain)
      U prob = fast::exp2(score - lse);

      // Compute dP = dO @ V^T for this KV position
      U dP = 0;
      for (int j = 0; j < v_per_thread; j++) {
        dP += d_o[j] * v[j];
      }
      dP = simd_sum(dP);

      // Compute dS = P * (dP - delta) [softmax gradient]
      U dS = prob * (dP - delta);

      // Accumulate dQ += scale * dS @ K
      // Note: Although Q was scaled by M_LOG2E_F internally, the softmax
      // gradient dS compensates for this because the overall softmax(S') =
      // softmax(S). The gradient dQ = scale * dS @ K matches the reference.
      for (int j = 0; j < qk_per_thread; j++) {
        dq[j] += static_cast<U>(scale) * dS * k[j];
      }

      // Compute dK and dV (skip in dq_only mode for 2-kernel split)
      if (!dq_only) {
        // dK = dS @ Q * scale
        // Note: q[j] = scale * M_LOG2E_F * Q[j], so dS * q gives:
        // dK = scale * M_LOG2E_F * dS @ Q
        // Reference expects: dK = scale * dS @ Q
        // So we multiply by inv_log2e to cancel the M_LOG2E_F
        if (direct_write) {
          // No contention (L=1, gqa_factor=1): write T directly
          for (int j = 0; j < qk_per_thread; j++) {
            d_keys[j] = static_cast<T>(inv_log2e * dS * q[j]);
          }
        } else {
          for (int j = 0; j < qk_per_thread; j++) {
            U dk_val = inv_log2e * dS * q[j];
            mlx_atomic_fetch_add_explicit(
                reinterpret_cast<device mlx_atomic<float>*>(d_keys),
                static_cast<float>(dk_val),
                j);
          }
        }

        // Accumulate dV += P^T @ dO = P * dO (for this KV position)
        if (direct_write) {
          for (int j = 0; j < v_per_thread; j++) {
            d_values[j] = static_cast<T>(prob * d_o[j]);
          }
        } else {
          for (int j = 0; j < v_per_thread; j++) {
            mlx_atomic_fetch_add_explicit(
                reinterpret_cast<device mlx_atomic<float>*>(d_values),
                static_cast<float>(prob * d_o[j]),
                j);
          }
        }
      }
    }

    // Move to next KV block
    keys += inner_k_stride;
    values += inner_v_stride;
    if (!dq_only) {
      d_keys += inner_k_stride;
      d_values += inner_v_stride;
    }

    if (bool_mask) {
      bmask += BN * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += BN * mask_kv_seq_stride;
    }
  }

  // Write accumulated dQ gradient
  // Need to reduce across simdgroups since each processed different KV
  // positions but they all contribute to the same query gradient

  // For D <= 128: single pass, shared_dQ[BN * D] fits in threadgroup memory.
  // For D > 128 (e.g., D=256): two passes over 128-wide slices.
  //   Pass 0: reduce dQ[0:128], write to output offset 0
  //   Pass 1: reduce dQ[128:256], write to output offset 128
  // KV loads are NOT replayed (dQ is fully accumulated in registers from the
  // main loop above). Only the shared memory reduction is split.

  constexpr int num_d_passes = (D + D_TILE - 1) / D_TILE;
  constexpr int elems_per_tile = D_TILE / BD; // Elements per thread in each tile

  for (int d_pass = 0; d_pass < num_d_passes; d_pass++) {
    // Initialize shared_dQ to zero for this pass
    for (int idx = simd_gid * BD + simd_lid; idx < BN * D_TILE; idx += BN * BD) {
      shared_dQ[idx] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Store each simdgroup's partial dQ to shared memory for this D_TILE slice
    // Map thread-local dq indices to the current pass's range
    for (int i = 0; i < elems_per_tile; i++) {
      int local_idx = d_pass * elems_per_tile + i; // Index into dq[] register array
      shared_dQ[simd_gid * D_TILE + simd_lid * elems_per_tile + i] = dq[local_idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce dQ across simdgroups for this D_TILE slice
    if (simd_gid == 0) {
      for (int i = 0; i < elems_per_tile; i++) {
        U sum_dq = 0;
        for (int sg = 0; sg < BN; sg++) {
          sum_dq += shared_dQ[sg * D_TILE + simd_lid * elems_per_tile + i];
        }
        int out_idx = d_pass * elems_per_tile + i;
        d_queries[out_idx] = static_cast<T>(sum_dq);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

///////////////////////////////////////////////////////////////////////////////
// SDPA Vector VJP dK/dV Kernel - Atomic-free second pass
//
// Computes dK and dV gradients without atomic operations by parallelizing
// over KV positions instead of query positions.
//
// Grid: (B * n_kv_heads, ceil(N / BN), 1)
// Group: (1024, 1, 1) = 32 simdgroups × 32 threads
//
// Each simdgroup exclusively owns one KV position and iterates over ALL
// query positions (L) and GQA heads (gqa_factor), accumulating dK/dV locally.
// No atomics needed because there's no write contention.
//
// Requires delta[B*H_q, L] precomputed by the dQ kernel (dq_only mode).
///////////////////////////////////////////////////////////////////////////////

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector_vjp_dkdv(
    // Forward inputs
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    // Upstream gradient
    const device T* d_out [[buffer(3)]],
    // Logsumexp from forward
    const device float* logsumexp [[buffer(4)]],
    // Delta from dQ kernel
    const device float* delta_in [[buffer(5)]],
    // Gradient outputs
    device T* d_keys [[buffer(6)]],
    device T* d_values [[buffer(7)]],
    // Parameters
    const constant int& gqa_factor [[buffer(8)]],
    const constant int& N [[buffer(9)]],
    const constant int& L [[buffer(10)]],
    const constant int& num_q_heads [[buffer(11)]],
    const constant size_t& k_head_stride [[buffer(12)]],
    const constant size_t& k_seq_stride [[buffer(13)]],
    const constant size_t& v_head_stride [[buffer(14)]],
    const constant size_t& v_seq_stride [[buffer(15)]],
    const constant float& scale [[buffer(16)]],
    // Output (dO) stride parameters
    const constant size_t& o_batch_stride [[buffer(17)]],
    const constant size_t& o_head_stride [[buffer(18)]],
    const constant size_t& o_seq_stride [[buffer(19)]],
    // Optional mask inputs
    const device bool* bmask [[buffer(20), function_constant(bool_mask)]],
    const device T* fmask [[buffer(21), function_constant(float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(22), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(23), function_constant(has_mask)]],
    const constant int& mask_head_stride
    [[buffer(24), function_constant(has_mask)]],
    // Thread position info
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;

  typedef float U;

  // Grid layout: tid.x = batch_idx * n_kv_heads + kv_head_idx
  //              tid.y = kv_block_idx
  int n_kv_heads = num_q_heads / gqa_factor;
  int batch_idx = tid.x / n_kv_heads;
  int kv_head_idx = tid.x % n_kv_heads;

  // Each simdgroup handles one KV position
  int kv_idx = int(tid.y) * BN + simd_gid;
  if (kv_idx >= N)
    return;

  // Load K and V for this position (persistent across query iterations)
  const device T* k_ptr = keys + tid.x * k_head_stride + kv_idx * k_seq_stride +
      simd_lid * qk_per_thread;
  const device T* v_ptr = values + tid.x * v_head_stride +
      kv_idx * v_seq_stride + simd_lid * v_per_thread;

  thread U k[qk_per_thread];
  thread U v[v_per_thread];
  for (int j = 0; j < qk_per_thread; j++)
    k[j] = k_ptr[j];
  for (int j = 0; j < v_per_thread; j++)
    v[j] = v_ptr[j];

  // Initialize dK, dV accumulators
  thread U dk[qk_per_thread];
  thread U dv[v_per_thread];
  for (int j = 0; j < qk_per_thread; j++)
    dk[j] = 0;
  for (int j = 0; j < v_per_thread; j++)
    dv[j] = 0;

  // Precompute for Q access
  // total_batch_heads = B * num_q_heads = tpg.x * gqa_factor
  int total_bh = int(tpg.x) * gqa_factor;
  U log2e_scale = static_cast<U>(scale * M_LOG2E_F);

  // Iterate over all GQA heads sharing this KV head
  int first_q_head = kv_head_idx * gqa_factor;
  for (int g = 0; g < gqa_factor; g++) {
    int q_head_idx = first_q_head + g;
    int q_batch_head_idx = batch_idx * num_q_heads + q_head_idx;

    // Iterate over all query positions
    for (int l = 0; l < L; l++) {
      // Apply mask
      bool use_key = true;
      if (do_causal) {
        use_key = kv_idx <= (N - L + l);
      } else if (bool_mask) {
        use_key = bmask
            [q_batch_head_idx * mask_head_stride + l * mask_q_seq_stride +
             kv_idx * mask_kv_seq_stride];
      } else if (float_mask) {
        T fmask_val = fmask
            [q_batch_head_idx * mask_head_stride + l * mask_q_seq_stride +
             kv_idx * mask_kv_seq_stride];
        use_key = (fmask_val >= Limits<T>::finite_min);
      }

      if (!use_key)
        continue;

      // Load Q for this query position
      int q_offset = query_transposed ? total_bh * l + q_batch_head_idx
                                      : q_batch_head_idx * L + l;
      const device T* q_ptr = queries + q_offset * D + simd_lid * qk_per_thread;

      // Load dO using explicit strides
      const device T* do_ptr = d_out + batch_idx * o_batch_stride +
          q_head_idx * o_head_stride + l * o_seq_stride +
          simd_lid * v_per_thread;

      // Load LSE and delta
      int lse_idx = q_batch_head_idx * L + l;
      U lse = logsumexp[lse_idx];
      U delta_val = delta_in[lse_idx];

      // Load Q
      thread U q_local[qk_per_thread];
      for (int j = 0; j < qk_per_thread; j++)
        q_local[j] = q_ptr[j];

      // Load dO
      thread U d_o[v_per_thread];
      for (int j = 0; j < v_per_thread; j++)
        d_o[j] = do_ptr[j];

      // Compute attention score: S = scale * Q @ K^T (in log2 domain)
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += (log2e_scale * q_local[j]) * k[j];
      }
      score = simd_sum(score);

      // Apply float mask
      if (float_mask) {
        T fmask_val = fmask
            [q_batch_head_idx * mask_head_stride + l * mask_q_seq_stride +
             kv_idx * mask_kv_seq_stride];
        score += static_cast<U>(M_LOG2E_F) * static_cast<U>(fmask_val);
      }

      // Reconstruct probability
      U prob = fast::exp2(score - lse);

      // Compute dP = dO @ V^T
      U dP = 0;
      for (int j = 0; j < v_per_thread; j++) {
        dP += d_o[j] * v[j];
      }
      dP = simd_sum(dP);

      // Compute dS = prob * (dP - delta)
      U dS = prob * (dP - delta_val);

      // Accumulate dK += scale * dS * Q
      for (int j = 0; j < qk_per_thread; j++) {
        dk[j] += static_cast<U>(scale) * dS * q_local[j];
      }

      // Accumulate dV += prob * dO
      for (int j = 0; j < v_per_thread; j++) {
        dv[j] += prob * d_o[j];
      }
    }
  }

  // Write dK, dV (no atomics - each simdgroup owns its KV position)
  device T* dk_ptr = d_keys + tid.x * k_head_stride + kv_idx * k_seq_stride +
      simd_lid * qk_per_thread;
  device T* dv_ptr = d_values + tid.x * v_head_stride + kv_idx * v_seq_stride +
      simd_lid * v_per_thread;

  for (int j = 0; j < qk_per_thread; j++) {
    dk_ptr[j] = static_cast<T>(dk[j]);
  }
  for (int j = 0; j < v_per_thread; j++) {
    dv_ptr[j] = static_cast<T>(dv[j]);
  }
}
