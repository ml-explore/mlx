// Copyright © 2024 Apple Inc.

#include <metal_simdgroup>

using namespace metal;

template <typename T, int D>
[[kernel]] void sdpa_vector(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor,
    const constant int& N,
    const constant size_t& k_stride,
    const constant size_t& v_stride,
    const constant float& scale,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;
  constexpr int stride = BN * D;

  typedef float U;

  thread U q[elem_per_thread];
  thread U k[elem_per_thread];
  thread U o[elem_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int head_idx = tid.y;
  const int kv_head_idx = head_idx / gqa_factor;
  queries += head_idx * D + simd_lid * elem_per_thread;
  keys += kv_head_idx * k_stride + simd_gid * D + simd_lid * elem_per_thread;
  values += kv_head_idx * v_stride + simd_gid * D + simd_lid * elem_per_thread;
  out += head_idx * D + simd_gid * elem_per_thread;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < elem_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < elem_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;

  // For each key
  for (int i = simd_gid; i < N; i += BN) {
    // Read the key
    for (int i = 0; i < elem_per_thread; i++) {
      k[i] = keys[i];
    }

    // Compute the i-th score
    U score = 0;
    for (int i = 0; i < elem_per_thread; i++) {
      score += q[i] * k[i];
    }
    score = simd_sum(score);

    // Update the accumulators
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    // Update the output accumulator
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] = o[i] * factor + exp_score * values[i];
    }

    // Move the pointers to the next kv
    keys += stride;
    values += stride;
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
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor) / sum_exp_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

template <typename T, typename U, int elem_per_thread, int bits>
METAL_FUNC U load_queries(const device T* queries, thread U* q, U scale) {
  U query_sum = 0;
  if (bits == 4) {
    for (int i = 0; i < elem_per_thread; i += 4) {
      q[i] = scale * queries[i];
      q[i + 1] = scale * queries[i + 1];
      q[i + 2] = scale * queries[i + 2];
      q[i + 3] = scale * queries[i + 3];
      query_sum += q[i] + q[i + 1] + q[i + 2] + q[i + 3];
      q[i + 1] /= 16.0f;
      q[i + 2] /= 256.0f;
      q[i + 3] /= 4096.0f;
    }
  } else if (bits == 8) {
    for (int i = 0; i < elem_per_thread; i++) {
      q[i] = scale * queries[i];
      query_sum += q[i];
    }
  }
  return query_sum;
}

template <typename U, int elem_per_thread, int bits>
METAL_FUNC void load_keys(const device uint32_t* keys, thread U* k) {
  if (bits == 4) {
    auto ks = (const device uint16_t*)keys;
    for (int i = 0; i < elem_per_thread / 4; i++) {
      k[4 * i] = ks[i] & 0x000f;
      k[4 * i + 1] = ks[i] & 0x00f0;
      k[4 * i + 2] = ks[i] & 0x0f00;
      k[4 * i + 3] = ks[i] & 0xf000;
    }
  } else if (bits == 8) {
    auto ks = (const device uint8_t*)keys;
    for (int i = 0; i < elem_per_thread; i++) {
      k[i] = ks[i];
    }
  }
}

template <typename U, int elem_per_thread, int bits>
METAL_FUNC void load_values(
    const device uint32_t* values,
    thread U* v,
    U value_scale,
    U value_bias) {
  auto vs = (const device uint8_t*)values;
  if (bits == 4) {
    U s[2] = {value_scale, value_scale / 16.0f};
    for (int i = 0; i < elem_per_thread / 2; i++) {
      v[2 * i] = s[0] * (vs[i] & 0x0f) + value_bias;
      v[2 * i + 1] = s[1] * (vs[i] & 0xf0) + value_bias;
    }
  } else if (bits == 8) {
    for (int i = 0; i < elem_per_thread; i++) {
      v[i] = value_scale * vs[i] + value_bias;
    }
  }
}

template <typename T, int D>
[[kernel]] void sdpa_vector_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device float* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& gqa_factor,
    const constant int& N,
    const constant size_t& k_stride,
    const constant size_t& v_stride,
    const constant float& scale,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 8;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;
  constexpr int stride = BN * D;
  constexpr int blocks = 32;

  typedef float U;

  thread U q[elem_per_thread];
  thread U k[elem_per_thread];
  thread U o[elem_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int block_idx = tid.z;
  const int head_idx = tid.y;
  const int kv_head_idx = head_idx / gqa_factor;
  queries += head_idx * D + simd_lid * elem_per_thread;
  keys += kv_head_idx * k_stride + (block_idx * BN + simd_gid) * D +
      simd_lid * elem_per_thread;
  values += kv_head_idx * v_stride + (block_idx * BN + simd_gid) * D +
      simd_lid * elem_per_thread;
  out += head_idx * blocks * D + block_idx * D + simd_lid * elem_per_thread;
  sums += head_idx * blocks + block_idx;
  maxs += head_idx * blocks + block_idx;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < elem_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < elem_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -1e9;
  U sum_exp_score = 0;

  // For each key
  for (int i = block_idx * BN + simd_gid; i < N; i += blocks * BN) {
    // Read the key
    for (int i = 0; i < elem_per_thread; i++) {
      k[i] = keys[i];
    }

    // Compute the i-th score
    U score = 0;
    for (int i = 0; i < elem_per_thread; i++) {
      score += q[i] * k[i];
    }
    score = simd_sum(score);

    // Update the accumulators
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    // Update the output accumulator
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] = o[i] * factor + exp_score * values[i];
    }

    // Move the pointers to the next kv
    keys += blocks * stride;
    values += blocks * stride;
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = (simd_lid < BN) ? max_scores[simd_lid] : -1e9;
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = (simd_lid < BN) ? sum_exp_scores[simd_lid] : 0;
  sum_exp_score = simd_sum(sum_exp_score * factor);

  // Write the sum and new max
  if (simd_gid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = new_max;
  }

  // Now we need to aggregate all the outputs
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BN + simd_gid] =
        o[i] * fast::exp(max_scores[simd_gid] - new_max);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // And write the output
    if (simd_gid == 0) {
      U output = outputs[simd_lid * BN];
      for (int j = 1; j < BN; j++) {
        output += outputs[simd_lid * BN + j];
      }
      out[i] = static_cast<T>(output);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <typename T, int D>
[[kernel]] void sdpa_vector_2pass_2(
    const device float* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    device T* out [[buffer(3)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;
  constexpr int blocks = 32;

  typedef float U;

  thread U o[elem_per_thread];
  threadgroup U outputs[BN * BD];

  // Adjust positions
  const int head_idx = tid.y;
  partials += head_idx * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
  sums += head_idx * blocks;
  maxs += head_idx * blocks;
  out += head_idx * D + simd_gid * elem_per_thread;

  // First everybody reads the max and sum_exp
  U max_score = maxs[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  U sum_exp_score = simd_sum(sums[simd_lid] * factor);

  // Now read the block into registers and then use shared memory to transpose
  // it
  for (int i = 0; i < elem_per_thread; i++) {
    o[i] = partials[i];
  }
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor) / sum_exp_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

template <typename T, int D, int group_size, int bits>
[[kernel]] void quant_sdpa_vector_2pass_1(
    const device T* queries [[buffer(0)]],
    const device uint32_t* keys [[buffer(1)]],
    const device T* key_scales [[buffer(2)]],
    const device T* key_biases [[buffer(3)]],
    const device uint32_t* values [[buffer(4)]],
    const device T* value_scales [[buffer(5)]],
    const device T* value_biases [[buffer(6)]],
    device float* out [[buffer(7)]],
    device float* sums [[buffer(8)]],
    device float* maxs [[buffer(9)]],
    const constant int& gqa_factor,
    const constant int& N,
    const constant size_t& k_stride,
    const constant size_t& v_stride,
    const constant size_t& k_group_stride,
    const constant size_t& v_group_stride,
    const constant float& scale,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint quad_gid [[quadgroup_index_in_threadgroup]],
    uint quad_lid [[thread_index_in_quadgroup]]) {
  constexpr int BN = 8;
  constexpr int BD = 4;
  constexpr int elem_per_thread = D / BD;
  const int stride = BN * D;
  constexpr int blocks = 32;
  constexpr int pack_factor = 32 / bits;

  typedef float U;

  thread U q[elem_per_thread];
  thread U k[elem_per_thread];
  thread U v[elem_per_thread];
  thread U o[elem_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int block_idx = tid.z;
  const int head_idx = tid.y;
  const int kv_head_idx = head_idx / gqa_factor;
  queries += head_idx * D + quad_lid * elem_per_thread;

  const int kv_idx =
      (block_idx * BN + quad_gid) * D + quad_lid * elem_per_thread;
  const int packed_idx = kv_idx / pack_factor;
  const int k_group_idx = kv_head_idx * k_group_stride + kv_idx / group_size;
  const int v_group_idx = kv_head_idx * v_group_stride + kv_idx / group_size;

  keys += kv_head_idx * k_stride + packed_idx;
  key_scales += k_group_idx;
  key_biases += k_group_idx;
  values += kv_head_idx * v_stride + packed_idx;
  value_scales += v_group_idx;
  value_biases += v_group_idx;

  out += head_idx * blocks * D + block_idx * D + quad_lid * elem_per_thread;
  sums += head_idx * blocks + block_idx;
  maxs += head_idx * blocks + block_idx;

  // Read the query and 0 the output accumulator
  U query_sum = load_queries<T, U, elem_per_thread, bits>(
      queries, q, static_cast<U>(scale));
  for (int i = 0; i < elem_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -1e9;
  U sum_exp_score = 0;

  // For each key
  for (int i = block_idx * BN + quad_gid; i < N; i += blocks * BN) {
    // Read the key
    load_keys<U, elem_per_thread, bits>(keys, k);

    // Assume D % group_size == 0 so all the keys are in the same group
    U key_scale = key_scales[0];
    U key_bias = key_biases[0];

    // Compute the i-th score
    U score = 0;
    for (int i = 0; i < elem_per_thread; i++) {
      score += q[i] * k[i];
    }
    score = score * key_scale + query_sum * key_bias;
    score = quad_sum(score);

    // Update the accumulators
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    U value_scale = value_scales[0];
    U value_bias = value_biases[0];
    load_values<U, elem_per_thread, bits>(values, v, value_scale, value_bias);

    // Update the output accumulator
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] = o[i] * factor + exp_score * v[i];
    }

    // Move the pointers to the next kv
    keys += blocks * stride / pack_factor;
    key_scales += blocks * stride / group_size;
    key_biases += blocks * stride / group_size;
    values += blocks * stride / pack_factor;
    value_scales += blocks * stride / group_size;
    value_biases += blocks * stride / group_size;
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (quad_lid == 0) {
    max_scores[quad_gid] = max_score;
    sum_exp_scores[quad_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = (simd_lid < BN) ? max_scores[simd_lid] : -1e9;
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = (simd_lid < BN) ? sum_exp_scores[simd_lid] : 0;
  sum_exp_score = simd_sum(sum_exp_score * factor);

  // Write the sum and new max
  if (simd_gid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = new_max;
  }

  // Now we need to aggregate all the outputs
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[quad_lid * BN + quad_gid] =
        o[i] * fast::exp(max_scores[quad_gid] - new_max);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (quad_gid == 0) {
      U output = outputs[quad_lid * BN];
      for (int j = 1; j < BN; j++) {
        output += outputs[quad_lid * BN + j];
      }
      out[i] = static_cast<T>(output);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
