// Copyright © 2024 Apple Inc.

#include <metal_simdgroup>

using namespace metal;

constant bool has_mask [[function_constant(20)]];
constant bool query_transposed [[function_constant(21)]];
constant bool do_causal [[function_constant(22)]];
constant bool bool_mask [[function_constant(23)]];
constant bool float_mask [[function_constant(24)]];
constant bool has_sinks [[function_constant(25)]];
constant int blocks [[function_constant(26)]];
constant int n_per_simd [[function_constant(27)]];

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    const device bool* bmask [[buffer(11), function_constant(bool_mask)]],
    const device T* fmask [[buffer(12), function_constant(float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(13), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(14), function_constant(has_mask)]],
    const constant int& mask_head_stride
    [[buffer(15), function_constant(has_mask)]],
    const device T* sinks [[buffer(16), function_constant(has_sinks)]],
    const constant int& num_q_heads
    [[buffer(17), function_constant(has_sinks)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  const int q_offset =
      query_transposed ? tpg.x * q_seq_idx + q_batch_head_idx : o_offset;
  queries += q_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }

  out += o_offset * V + simd_gid * v_per_thread;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (has_sinks && simd_gid == 0) {
    max_score = static_cast<U>(sinks[q_batch_head_idx % num_q_heads]);
    sum_exp_score = 1;
  }

  // For each key
  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }
    if (use_key) {
      // Read the key
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = keys[j];
      }

      // Compute the i-th score
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);
      if (float_mask) {
        score += static_cast<U>(fmask[0]);
      }

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }

    // Move the pointers to the next kv
    keys += inner_k_stride;
    values += inner_v_stride;
    if (bool_mask) {
      bmask += BN * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += BN * mask_kv_seq_stride;
    }
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Now we need to aggregate all the outputs
  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& q_seq_len [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant size_t& k_head_stride [[buffer(8)]],
    const constant size_t& k_seq_stride [[buffer(9)]],
    const constant size_t& v_head_stride [[buffer(10)]],
    const constant size_t& v_seq_stride [[buffer(11)]],
    const constant float& scale [[buffer(12)]],
    const device bool* bmask [[buffer(13), function_constant(bool_mask)]],
    const device T* fmask [[buffer(14), function_constant(float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(15), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(16), function_constant(has_mask)]],
    const constant int& mask_head_stride
    [[buffer(17), function_constant(has_mask)]],
    const device T* sinks [[buffer(18), function_constant(has_sinks)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  // Upper bound on query rows per simdgroup, sized so the per-lane
  // accumulator state stays register-resident.
  constexpr int MAX_NQ = D <= 128 ? 4 : 2;

  typedef float U;

  // Adjust positions
  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * gqa_factor;
  const int q_batch_head_idx = (batch_idx * num_q_heads + q_head_idx);
  const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;

  keys += kv_batch_head_idx * k_head_stride + block_idx * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_batch_head_idx * v_head_stride + block_idx * v_seq_stride +
      simd_lid * v_per_thread;

  if (n_per_simd == 1) {
    // One query row per simdgroup (decode shapes). Same as the original
    // single-row kernel.
    const int q_seq_idx = tidtg.z;
    const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;
    const int q_offset = query_transposed
        ? num_q_heads * q_seq_idx + q_batch_head_idx
        : o_offset;

    thread U q[qk_per_thread];
    thread U o[v_per_thread] = {0};

    queries += q_offset * D + simd_lid * qk_per_thread;
    out += o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
    if (bool_mask) {
      bmask += q_batch_head_idx * mask_head_stride +
          block_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
    }
    if (float_mask) {
      fmask += q_batch_head_idx * mask_head_stride +
          block_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
    }
    sums += o_offset * blocks + block_idx;
    maxs += o_offset * blocks + block_idx;

    // Read the query
    for (int i = 0; i < qk_per_thread; i++) {
      q[i] = static_cast<U>(scale) * queries[i];
    }

    U max_score = Limits<U>::finite_min;
    U sum_exp_score = 0;
    if (has_sinks && block_idx == 0) {
      max_score = static_cast<U>(sinks[q_head_idx]);
      sum_exp_score = 1;
    }

    // For each key
    for (int i = block_idx; i < N; i += blocks) {
      bool use_key = true;
      if (do_causal) {
        use_key = i <= (N - q_seq_len + int(q_seq_idx));
      } else if (bool_mask) {
        use_key = bmask[0];
      } else if (float_mask) {
        use_key = (fmask[0] >= Limits<T>::finite_min);
      }
      if (use_key) {
        // Compute the i-th score
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
          score += q[j] * keys[j];
        }
        score = simd_sum(score);

        if (float_mask) {
          score += fmask[0];
        }

        // Update the accumulators
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        // Update the output accumulator
        for (int j = 0; j < v_per_thread; j++) {
          o[j] = o[j] * factor + exp_score * values[j];
        }
      }

      // Move the pointers to the next kv
      keys += blocks * int(k_seq_stride);
      values += blocks * int(v_seq_stride);
      if (bool_mask) {
        bmask += blocks * mask_kv_seq_stride;
      }
      if (float_mask) {
        fmask += blocks * mask_kv_seq_stride;
      }
    }

    // Write the sum and max and outputs
    if (simd_lid == 0) {
      sums[0] = sum_exp_score;
      maxs[0] = max_score;
    }

    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  } else {
    // Several query rows per simdgroup (8 < qL <= 16, e.g. speculative
    // decoding verify shapes). The K/V rows are read once per threadgroup
    // and reused across the rows, keeping the block-parallel reduction
    // over the key sequence that a single-row-per-simdgroup layout would
    // lose to the threadgroup size limit.
    const int nq = min(n_per_simd, MAX_NQ);
    const int q_seq0 = tidtg.z * nq;

    thread U q[MAX_NQ][qk_per_thread];
    thread U o[MAX_NQ][v_per_thread];
    thread U kk[qk_per_thread];
    thread U vv[v_per_thread];
    thread U max_score[MAX_NQ];
    thread U sum_exp_score[MAX_NQ];

    if (bool_mask) {
      bmask += q_batch_head_idx * mask_head_stride +
          block_idx * mask_kv_seq_stride + q_seq0 * mask_q_seq_stride;
    }
    if (float_mask) {
      fmask += q_batch_head_idx * mask_head_stride +
          block_idx * mask_kv_seq_stride + q_seq0 * mask_q_seq_stride;
    }

    // Read the queries and initialize the accumulators
    for (int r = 0; r < nq; r++) {
      const int q_seq_idx = q_seq0 + r;
      max_score[r] = Limits<U>::finite_min;
      sum_exp_score[r] = 0;
      for (int i = 0; i < v_per_thread; i++) {
        o[r][i] = 0;
      }
      if (q_seq_idx < q_seq_len) {
        const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;
        const int q_offset = query_transposed
            ? num_q_heads * q_seq_idx + q_batch_head_idx
            : o_offset;
        const device T* qr = queries + q_offset * D + simd_lid * qk_per_thread;
        for (int i = 0; i < qk_per_thread; i++) {
          q[r][i] = static_cast<U>(scale) * qr[i];
        }
        if (has_sinks && block_idx == 0) {
          max_score[r] = static_cast<U>(sinks[q_head_idx]);
          sum_exp_score[r] = 1;
        }
      }
    }

    // For each key
    for (int i = block_idx; i < N; i += blocks) {
      bool use_key[MAX_NQ];
      bool any_use_key = false;
      for (int r = 0; r < nq; r++) {
        const int q_seq_idx = q_seq0 + r;
        bool use = q_seq_idx < q_seq_len;
        if (use) {
          if (do_causal) {
            use = i <= (N - q_seq_len + q_seq_idx);
          } else if (bool_mask) {
            use = bmask[r * mask_q_seq_stride];
          } else if (float_mask) {
            use = (fmask[r * mask_q_seq_stride] >= Limits<T>::finite_min);
          }
        }
        use_key[r] = use;
        any_use_key |= use;
      }

      if (any_use_key) {
        // Read the key and value once and reuse across the query rows
        for (int j = 0; j < qk_per_thread; j++) {
          kk[j] = keys[j];
        }
        for (int j = 0; j < v_per_thread; j++) {
          vv[j] = values[j];
        }

        for (int r = 0; r < nq; r++) {
          if (!use_key[r]) {
            continue;
          }
          // Compute the i-th score for the r-th row
          U score = 0;
          for (int j = 0; j < qk_per_thread; j++) {
            score += q[r][j] * kk[j];
          }
          score = simd_sum(score);
          if (float_mask) {
            score += static_cast<U>(fmask[r * mask_q_seq_stride]);
          }

          // Update the accumulators
          U new_max = max(max_score[r], score);
          U factor = fast::exp(max_score[r] - new_max);
          U exp_score = fast::exp(score - new_max);

          max_score[r] = new_max;
          sum_exp_score[r] = sum_exp_score[r] * factor + exp_score;

          // Update the output accumulator
          for (int j = 0; j < v_per_thread; j++) {
            o[r][j] = o[r][j] * factor + exp_score * vv[j];
          }
        }
      }

      // Move the pointers to the next kv
      keys += blocks * int(k_seq_stride);
      values += blocks * int(v_seq_stride);
      if (bool_mask) {
        bmask += blocks * mask_kv_seq_stride;
      }
      if (float_mask) {
        fmask += blocks * mask_kv_seq_stride;
      }
    }

    // Write the sums, maxes and partial outputs
    for (int r = 0; r < nq; r++) {
      const int q_seq_idx = q_seq0 + r;
      if (q_seq_idx >= q_seq_len) {
        continue;
      }
      const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;
      if (simd_lid == 0) {
        sums[o_offset * blocks + block_idx] = sum_exp_score[r];
        maxs[o_offset * blocks + block_idx] = max_score[r];
      }
      device T* outr =
          out + o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
      for (int j = 0; j < v_per_thread; j++) {
        outr[j] = static_cast<T>(o[r][j]);
      }
    }
  }
}

template <typename T, int D>
[[kernel]] void sdpa_vector_2pass_2(
    const device T* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& blocks [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;

  typedef float U;

  thread U o[elem_per_thread] = {0};
  threadgroup U outputs[BN * BD];

  // Adjust positions
  const int head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int q_offset = head_idx * tpg.y + q_seq_idx;
  partials += q_offset * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
  sums += q_offset * blocks;
  maxs += q_offset * blocks;
  out += q_offset * D + simd_gid * elem_per_thread;

  // Set defaults
  U sum_exp_score = 0.0;
  U max_score = Limits<U>::finite_min;

  // Reduce the max
  for (int b = 0; b < blocks / BN; ++b) {
    max_score = max(max_score, maxs[simd_lid + BN * b]);
  }
  max_score = simd_max(max_score);

  // Reduce the d
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_lid + BN * b] - max_score);
    sum_exp_score += factor * sums[simd_lid + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);

  // Reduce the sum exp and partials
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_gid] - max_score);

    // Update the output accumulator
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] += factor * static_cast<U>(partials[i]);
    }
    maxs += BN;
    sums += BN;
    partials += BN * D;
  }

  // Use shared memory to transpose and reduce the final block
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}
