// TurboQuant SDPA vector kernel: decode with pre-rotated queries and
// bit-packed KV cache. Reads 3-bit packed indices + norms + codebook,
// computes attention without materializing dequantized KV vectors.
//
// Pre-rotated query: Q_rot = WHT(signs * Q), computed once per head.
// Score: dot(Q_rot, codebook[K_indices]) * norm / sqrt(dim)
// No WHT butterfly in the inner loop.

// NOTE: function_constants and metal includes are provided by the
// parent .metal file that includes this header.

// TurboQuant SDPA: packed K/V with codebook dequantization
// K is stored as bit-packed uint32 indices + float32 norms
// V is stored as pre-dequantized fp16 (via incremental decode buffer)
//
// Template params:
//   T: output type (float16/bfloat16)
//   D: head dimension (64, 128)
//   V_DIM: value dimension (usually == D)
//   BITS: quantization bits (2, 3, 4)
//   VPW: values per uint32 word (16, 10, 8 for 2, 3, 4 bits)
template <typename T, int D, int V_DIM = D, int BITS = 3, int VPW = 10>
[[kernel]] void sdpa_vector_turbo(
    const device T* queries [[buffer(0)]], // pre-rotated queries
    const device uint32_t* k_packed [[buffer(1)]], // packed K indices
    const device T* values [[buffer(2)]], // dequantized V (from decode buffer)
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]], // in uint32 words
    const constant size_t& k_seq_stride [[buffer(7)]], // in uint32 words
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
    const device float* k_norms [[buffer(18)]], // per-vector norms
    const constant size_t& k_norm_head_stride [[buffer(19)]],
    const device float* codebook [[buffer(20)]], // 2^BITS centroids
    const constant float& inv_sqrt_dim [[buffer(21)]], // 1/sqrt(dim)
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V_DIM / BD;
  constexpr int PACKED_DIM = (D + VPW - 1) / VPW;
  constexpr uint BIT_MASK = (1u << BITS) - 1u;

  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
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

  // Query pointer (pre-rotated)
  queries += q_offset * D + simd_lid * qk_per_thread;

  // K packed pointer: navigate to correct head, then stride by simd_gid
  k_packed += kv_head_idx * k_head_stride + simd_gid * k_seq_stride;
  k_norms += kv_head_idx * k_norm_head_stride + simd_gid;

  // V pointer (dequantized fp16 from decode buffer)
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

  out += o_offset * V_DIM + simd_gid * v_per_thread;

  // Read pre-rotated query (already scaled)
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -1e9f;
  U sum_exp_score = 0;
  if (has_sinks && simd_gid == 0) {
    max_score = static_cast<U>(sinks[q_batch_head_idx % num_q_heads]);
    sum_exp_score = 1;
  }

  // For each key position
  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= -1e9f);
    }
    if (use_key) {
      // --- TurboQuant: read packed K indices, codebook lookup ---
      // Each thread handles qk_per_thread = D/32 elements
      // Thread simd_lid handles elements [simd_lid*qk_per_thread,
      // (simd_lid+1)*qk_per_thread)
      U score = 0;
      int elem_start = simd_lid * qk_per_thread;
      for (int j = 0; j < qk_per_thread; j++) {
        int elem = elem_start + j;
        int word_idx = elem / VPW;
        int pos_in_word = elem % VPW;
        uint word = k_packed[word_idx];
        uint idx = (word >> (pos_in_word * BITS)) & BIT_MASK;
        U k_val = codebook[idx];
        score += q[j] * k_val;
      }

      // Apply norm and scale: score = dot(q_rot, codebook[indices]) * norm *
      // inv_sqrt_dim
      U norm_val = k_norms[0];
      score = simd_sum(score) * norm_val * inv_sqrt_dim;

      if (float_mask) {
        score += static_cast<U>(fmask[0]);
      }

      // Update the accumulators (same as standard sdpa_vector)
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update output with dequantized V (from decode buffer, already fp16)
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * static_cast<U>(values[j]);
      }
    }

    // Advance K packed pointer by BN positions
    k_packed += BN * k_seq_stride;
    k_norms += BN;
    values += inner_v_stride;
    if (bool_mask) {
      bmask += BN * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += BN * mask_kv_seq_stride;
    }
  }

  // Reduction across SIMD groups (same as standard sdpa_vector)
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write output
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}
