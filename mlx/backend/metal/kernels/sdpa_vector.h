// Copyright © 2024 Apple Inc.

#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/fp_quantized.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"

using namespace metal;

constant bool has_mask [[function_constant(20)]];
constant bool query_transposed [[function_constant(21)]];
constant bool do_causal [[function_constant(22)]];
constant bool bool_mask [[function_constant(23)]];
constant bool float_mask [[function_constant(24)]];
constant bool has_sinks [[function_constant(25)]];
constant int blocks [[function_constant(26)]];

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

template <typename T, typename U, int elem_per_thread>
METAL_FUNC void load_queries(const device T* queries, thread U* q, U scale) {
  for (int i = 0; i < elem_per_thread; i++) {
    q[i] = scale * queries[i];
  }
}

constant bool has_affine_bias [[function_constant(27)]];

template <QuantMode mode, int bits>
struct QuantFastOps {
  using Cfg = QuantConfig<mode>;
  using load_t = metal::conditional_t<bits == 4, uint16_t, uint32_t>;
  static_assert(
      bits == 4 || bits == 8,
      "QuantFastOps only supports 4/8-bit packing");
  static constant constexpr uint32_t mask = (1u << bits) - 1;

  template <typename U, int elem_per_thread>
  [[clang::always_inline]] static U
  dot(const thread U* q, const device uint32_t* keys, U scale, U bias) {
    Dequant<mode, U> dequant;
    auto ks = reinterpret_cast<const device load_t*>(keys);
    U score = 0;
    [[maybe_unused]] U bias_acc = 0;

#pragma clang loop unroll(full)
    for (int j = 0; j < elem_per_thread / 4; j++) {
      load_t p = ks[j];
      U v0 = dequant.raw(p & mask);
      U v1 = dequant.raw((p >> (bits * 1)) & mask);
      U v2 = dequant.raw((p >> (bits * 2)) & mask);
      U v3 = dequant.raw((p >> (bits * 3)) & mask);

      score += q[4 * j + 0] * v0;
      score += q[4 * j + 1] * v1;
      score += q[4 * j + 2] * v2;
      score += q[4 * j + 3] * v3;

      if constexpr (Cfg::has_bias) {
        bias_acc += (q[4 * j + 0] + q[4 * j + 1] + q[4 * j + 2] + q[4 * j + 3]);
      }
    }

    if constexpr (Cfg::has_bias) {
      return fma(scale, score, bias * bias_acc);
    } else {
      return scale * score;
    }
  }

  template <typename U, int elem_per_thread>
  [[clang::always_inline]] static void accumulate(
      thread U* o,
      const device uint32_t* values,
      U factor,
      U w_scale,
      U bias) {
    Dequant<mode, U> dequant;
    auto vs = reinterpret_cast<const device load_t*>(values);
#pragma clang loop unroll(full)
    for (int j = 0; j < elem_per_thread / 4; j++) {
      load_t p = vs[j];
      U v0 = dequant.raw(p & mask);
      U v1 = dequant.raw((p >> (bits * 1)) & mask);
      U v2 = dequant.raw((p >> (bits * 2)) & mask);
      U v3 = dequant.raw((p >> (bits * 3)) & mask);

      if constexpr (Cfg::has_bias) {
        o[4 * j + 0] = fma(o[4 * j + 0], factor, fma(w_scale, v0, bias));
        o[4 * j + 1] = fma(o[4 * j + 1], factor, fma(w_scale, v1, bias));
        o[4 * j + 2] = fma(o[4 * j + 2], factor, fma(w_scale, v2, bias));
        o[4 * j + 3] = fma(o[4 * j + 3], factor, fma(w_scale, v3, bias));
      } else {
        o[4 * j + 0] = fma(o[4 * j + 0], factor, v0 * w_scale);
        o[4 * j + 1] = fma(o[4 * j + 1], factor, v1 * w_scale);
        o[4 * j + 2] = fma(o[4 * j + 2], factor, v2 * w_scale);
        o[4 * j + 3] = fma(o[4 * j + 3], factor, v3 * w_scale);
      }
    }
  }
};

// (Generic Path for 2, 3, 5, 6 bits)
template <QuantMode mode, int bits>
struct QuantOps {
  using Cfg = QuantConfig<mode>;
  static constant constexpr int pack_factor = PackInfo<bits>::pack_factor;
  static constant constexpr int bytes_per_pack = PackInfo<bits>::bytes_per_pack;

  template <typename U, int elem_per_thread>
  [[clang::always_inline]] static U
  dot(const thread U* q, const device uint32_t* keys, U scale, U bias) {
    Dequant<mode, U> dequant;
    auto ks = reinterpret_cast<const device uint8_t*>(keys);
    thread uint8_t raw[pack_factor];
    U score = 0;
    [[maybe_unused]] U bias_acc = 0;

#pragma clang loop unroll(full)
    for (int j = 0; j < elem_per_thread; j += pack_factor) {
      PackReader<bits>::load(ks, raw);
#pragma clang loop unroll(full)
      for (int t = 0; t < pack_factor; ++t) {
        U decoded = dequant.raw(raw[t]);
        score += q[j + t] * decoded;
        if constexpr (Cfg::has_bias)
          bias_acc += q[j + t];
      }
      ks += bytes_per_pack;
    }
    return Cfg::has_bias ? fma(scale, score, bias * bias_acc) : scale * score;
  }

  template <typename U, int elem_per_thread>
  [[clang::always_inline]] static void accumulate(
      thread U* o,
      const device uint32_t* values,
      U factor,
      U w_scale,
      U bias) {
    Dequant<mode, U> dequant;
    auto vs = reinterpret_cast<const device uint8_t*>(values);
    thread uint8_t raw[pack_factor];
#pragma clang loop unroll(full)
    for (int j = 0; j < elem_per_thread; j += pack_factor) {
      PackReader<bits>::load(vs, raw);
#pragma clang loop unroll(full)
      for (int t = 0; t < pack_factor; ++t) {
        U decoded = dequant.raw(raw[t]);
        if constexpr (Cfg::has_bias) {
          o[j + t] = fma(o[j + t], factor, fma(w_scale, decoded, bias));
        } else {
          o[j + t] = fma(o[j + t], factor, decoded * w_scale);
        }
      }
      vs += bytes_per_pack;
    }
  }
};

template <QuantMode mode>
struct QuantOps<mode, 4> {
  template <typename U, int elem_per_thread>
  [[clang::always_inline]] static U
  dot(const thread U* q, const device uint32_t* keys, U scale, U bias) {
    return QuantFastOps<mode, 4>::template dot<U, elem_per_thread>(
        q, keys, scale, bias);
  }
  template <typename U, int elem_per_thread>
  [[clang::always_inline]] static void accumulate(
      thread U* o,
      const device uint32_t* values,
      U factor,
      U w_scale,
      U bias) {
    QuantFastOps<mode, 4>::template accumulate<U, elem_per_thread>(
        o, values, factor, w_scale, bias);
  }
};

template <QuantMode mode>
struct QuantOps<mode, 8> {
  template <typename U, int elem_per_thread>
  [[clang::always_inline]] static U
  dot(const thread U* q, const device uint32_t* keys, U scale, U bias) {
    return QuantFastOps<mode, 8>::template dot<U, elem_per_thread>(
        q, keys, scale, bias);
  }
  template <typename U, int elem_per_thread>
  [[clang::always_inline]] static void accumulate(
      thread U* o,
      const device uint32_t* values,
      U factor,
      U w_scale,
      U bias) {
    QuantFastOps<mode, 8>::template accumulate<U, elem_per_thread>(
        o, values, factor, w_scale, bias);
  }
};
template <QuantMode mode, typename T>
using ScaleTypeT = typename QuantConfig<mode>::template scale_storage_t<T>;

template <typename T, int D, QuantMode mode, int group_size, int bits>
[[kernel]] void quant_sdpa_vector_2pass_1(
    const device T* queries [[buffer(0)]],
    const device uint32_t* keys [[buffer(1)]],
    const device ScaleTypeT<mode, T>* key_scales [[buffer(2)]],
    const device uint32_t* values [[buffer(3)]],
    const device ScaleTypeT<mode, T>* value_scales [[buffer(4)]],
    device T* out [[buffer(5)]],
    device float* sums [[buffer(6)]],
    device float* maxs [[buffer(7)]],
    const constant int& gqa_factor [[buffer(8)]],
    const constant int& N [[buffer(9)]],
    const constant size_t& k_stride [[buffer(10)]],
    const constant size_t& v_stride [[buffer(11)]],
    const constant size_t& k_group_stride [[buffer(12)]],
    const constant size_t& v_group_stride [[buffer(13)]],
    const constant float& scale [[buffer(14)]],
    const device bool* bmask [[buffer(15), function_constant(bool_mask)]],
    const device T* fmask [[buffer(16), function_constant(float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(17), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(18), function_constant(has_mask)]],
    const constant int& mask_head_stride
    [[buffer(19), function_constant(has_mask)]],
    const device T* key_biases
    [[buffer(20), function_constant(has_affine_bias)]],
    const device T* value_biases
    [[buffer(21), function_constant(has_affine_bias)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint quad_gid [[quadgroup_index_in_threadgroup]],
    uint quad_lid [[thread_index_in_quadgroup]]) {
  // Quadgroup approach: BN=8 quads × BD=4 lanes = 32 threads = 1 simdgroup
  // Each quad processes one key, lanes split D dimension.
  // elem_per_thread=D/4 is large enough for all pack_factors (max 8).
  using Cfg = QuantConfig<mode>;

  constexpr int BN = 8;
  constexpr int BD = 4;
  constexpr int elem_per_thread = D / BD;

  typedef float U;
  [[maybe_unused]] Dequant<mode, U> dequant;

  thread U q[elem_per_thread];
  thread U o[elem_per_thread] = {0};

  const int block_idx = tid.z;
  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  const int q_offset =
      query_transposed ? tpg.x * q_seq_idx + q_batch_head_idx : o_offset;

  queries += q_offset * D + quad_lid * elem_per_thread;

  const int kv_idx =
      (block_idx * BN + quad_gid) * D + quad_lid * elem_per_thread;
  const int k_group_idx = kv_head_idx * k_group_stride + kv_idx / group_size;
  const int v_group_idx = kv_head_idx * v_group_stride + kv_idx / group_size;

  QuantDataPtr<bits> key_ptr(keys, k_stride, kv_head_idx, kv_idx);
  QuantDataPtr<bits> value_ptr(values, v_stride, kv_head_idx, kv_idx);

  key_scales += k_group_idx;
  value_scales += v_group_idx;
  if constexpr (Cfg::has_bias) {
    key_biases += k_group_idx;
    value_biases += v_group_idx;
  }

  out += o_offset * blocks * D + block_idx * D + quad_lid * elem_per_thread;
  sums += o_offset * blocks + block_idx;
  maxs += o_offset * blocks + block_idx;

  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        (block_idx * BN + quad_gid) * mask_kv_seq_stride +
        q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        (block_idx * BN + quad_gid) * mask_kv_seq_stride +
        q_seq_idx * mask_q_seq_stride;
  }

  load_queries<T, U, elem_per_thread>(queries, q, static_cast<U>(scale));

  constexpr int stride = BN * D;
  const int data_step = blocks * stride;
  const int scale_step = data_step / group_size;
  const int mask_step = BN * blocks * mask_kv_seq_stride;

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;

  // Main loop: each quad processes one key at a time
  for (int i = block_idx * BN + quad_gid; i < N; i += blocks * BN) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }

    if (use_key) {
      U key_scale, key_bias = 0;
      if constexpr (Cfg::has_bias) {
        key_scale = U(key_scales[0]);
        key_bias = U(key_biases[0]);
      } else {
        key_scale = dequant.scale(key_scales[0]);
      }

      U score = QuantOps<mode, bits>::template dot<U, elem_per_thread>(
          q, key_ptr.ptr(), key_scale, key_bias);
      score = quad_sum(score);

      if (float_mask) {
        score += static_cast<U>(fmask[0]);
      }

      // Online softmax update
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      U value_scale, value_bias = 0;
      if constexpr (Cfg::has_bias) {
        value_scale = U(value_scales[0]);
        value_bias = U(value_biases[0]);
      } else {
        value_scale = dequant.scale(value_scales[0]);
      }

      QuantOps<mode, bits>::template accumulate<U, elem_per_thread>(
          o,
          value_ptr.ptr(),
          factor,
          exp_score * value_scale,
          exp_score * value_bias);
    }

    // Advance pointers
    key_ptr.advance(data_step);
    value_ptr.advance(data_step);
    key_scales += scale_step;
    value_scales += scale_step;
    if constexpr (Cfg::has_bias) {
      key_biases += scale_step;
      value_biases += scale_step;
    }
    if (bool_mask) {
      bmask += mask_step;
    }
    if (float_mask) {
      fmask += mask_step;
    }
  }

  U sg_max = (quad_lid == 0) ? max_score : Limits<U>::finite_min;
  U global_max = simd_max(sg_max);

  U sg_sum =
      (quad_lid == 0) ? sum_exp_score * fast::exp(max_score - global_max) : 0;
  U global_sum = simd_sum(sg_sum);

  if (simd_lid == 0) {
    sums[0] = global_sum;
    maxs[0] = global_max;
  }

  // Output reduction: sum across quads (same quad_lid only)
  U rescale = fast::exp(max_score - global_max);
  for (int i = 0; i < elem_per_thread; i++) {
    U val = o[i] * rescale;
    val += simd_shuffle_xor(val, 4); // sum quads 0+1, 2+3, 4+5, 6+7
    val += simd_shuffle_xor(val, 8); // sum quads 0-3, 4-7
    val += simd_shuffle_xor(val, 16); // sum quads 0-7
    // All lanes with same quad_lid now have the full sum; quad_gid=0 writes
    if (quad_gid == 0) {
      out[i] = static_cast<T>(val);
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

  typedef float U;

  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};

  // Adjust positions
  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_seq_len = tptg.z;
  const int q_seq_idx = tidtg.z;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * gqa_factor;
  const int q_batch_head_idx = (batch_idx * num_q_heads + q_head_idx);
  const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;
  const int q_offset =
      query_transposed ? num_q_heads * q_seq_idx + q_batch_head_idx : o_offset;

  queries += q_offset * D + simd_lid * qk_per_thread;

  const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;
  keys += kv_batch_head_idx * k_head_stride + block_idx * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_batch_head_idx * v_head_stride + block_idx * v_seq_stride +
      simd_lid * v_per_thread;
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
      for (int i = 0; i < qk_per_thread; i++) {
        score += q[i] * keys[i];
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
      for (int i = 0; i < v_per_thread; i++) {
        o[i] = o[i] * factor + exp_score * values[i];
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
