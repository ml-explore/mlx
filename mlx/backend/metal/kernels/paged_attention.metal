#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

constant uint MAX_HEAD_DIM = 512;
constant ushort FC_BLOCK_SIZE [[function_constant(0)]];
constant ushort FC_THREADGROUP_WIDTH [[function_constant(1)]];
constant ushort FC_VEC_WIDTH [[function_constant(2)]];
constant uint MAX_THREADGROUP_WIDTH = 256;

struct PagedParams {
  uint head_dim;
  uint block_size;
  uint max_blocks_per_seq;
  uint num_q_heads;
  uint num_kv_heads;
  uint has_kv_mapping;
  float scale;
  uint q_batch_stride;
  uint q_head_stride;
  uint kv_head_stride;
  uint block_stride;
  uint row_stride;
  uint out_batch_stride;
  uint out_head_stride;
};

template <typename T>
inline float to_float(T value) {
  return static_cast<float>(value);
}

template <>
inline float to_float<bfloat16_t>(bfloat16_t value) {
  return static_cast<float>(value);
}

template <typename T>
inline T from_float(float value) {
  return static_cast<T>(value);
}

template <>
inline bfloat16_t from_float(float value) {
  return bfloat16_t(value);
}

template <typename T>
kernel void paged_attention_decode(
    const device T* q [[buffer(0)]],
    const device T* k [[buffer(1)]],
    const device T* v [[buffer(2)]],
    const device int* block_tables [[buffer(3)]],
    const device int* context_lens [[buffer(4)]],
    const device int* kv_mapping [[buffer(5)]],
    device T* out [[buffer(6)]],
    const constant PagedParams& params [[buffer(7)]],
    ushort3 tg_pos [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
  const uint batch_idx = tg_pos.y;
  const uint head_idx = tg_pos.x;
  const uint head_dim = params.head_dim;
  const uint block_size_const =
      FC_BLOCK_SIZE > 0 ? (uint)FC_BLOCK_SIZE : params.block_size;
  const uint threadgroup_width =
      metal::min(
          FC_THREADGROUP_WIDTH > 0 ? (uint)FC_THREADGROUP_WIDTH : 32u,
          MAX_THREADGROUP_WIDTH);
  const uint vec_width = FC_VEC_WIDTH > 0 ? (uint)FC_VEC_WIDTH : 1u;

  device T* out_ptr = out + batch_idx * params.out_batch_stride +
      head_idx * params.out_head_stride;

  if (head_dim == 0) {
    return;
  }
  if (head_dim > MAX_HEAD_DIM) {
    for (uint base = tid * vec_width; base < head_dim;
         base += threadgroup_width * vec_width) {
      for (uint lane = 0; lane < vec_width; ++lane) {
        uint idx = base + lane;
        if (idx < head_dim) {
          out_ptr[idx] = from_float<T>(0.0f);
        }
      }
    }
    return;
  }

  const device int* block_row =
      block_tables + batch_idx * params.max_blocks_per_seq;
  const uint seq_len =
      context_lens[batch_idx] > 0 ? static_cast<uint>(context_lens[batch_idx])
                                  : 0u;
  if (seq_len == 0) {
    for (uint base = tid * vec_width; base < head_dim;
         base += threadgroup_width * vec_width) {
      for (uint lane = 0; lane < vec_width; ++lane) {
        uint idx = base + lane;
        if (idx < head_dim) {
          out_ptr[idx] = from_float<T>(0.0f);
        }
      }
    }
    return;
  }

  const uint kv_head = params.has_kv_mapping
      ? static_cast<uint>(kv_mapping[head_idx])
      : (params.num_kv_heads == 1 ? 0u
                                  : (head_idx * params.num_kv_heads) /
              params.num_q_heads);

  const device T* q_ptr = q + batch_idx * params.q_batch_stride +
      head_idx * params.q_head_stride;
  const device T* k_head = k + kv_head * params.kv_head_stride;
  const device T* v_head = v + kv_head * params.kv_head_stride;

  threadgroup float q_shared[MAX_HEAD_DIM];
  threadgroup float accum_shared[MAX_HEAD_DIM];
  threadgroup float partial_sums[MAX_THREADGROUP_WIDTH];
  threadgroup float shared_alpha;
  threadgroup float shared_weight;
  threadgroup float shared_inv_l;
  threadgroup float shared_m_val;

  for (uint base = tid * vec_width; base < head_dim;
       base += threadgroup_width * vec_width) {
    for (uint lane = 0; lane < vec_width; ++lane) {
      uint idx = base + lane;
      if (idx < head_dim) {
        q_shared[idx] = to_float(q_ptr[idx]) * params.scale;
        accum_shared[idx] = 0.0f;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float m = -INFINITY;
  float l = 0.0f;
  uint tokens_done = 0;

  for (uint blk = 0; blk < params.max_blocks_per_seq &&
          tokens_done < seq_len;
       ++blk) {
    const int block_id = block_row[blk];
    if (block_id < 0) {
      continue;
    }

    const device T* block_k = k_head + block_id * params.block_stride;
    const device T* block_v = v_head + block_id * params.block_stride;

    uint rows = block_size_const;
    if (tokens_done + rows > seq_len) {
      rows = seq_len - tokens_done;
    }

    for (uint row = 0; row < rows; ++row) {
      const device T* k_row = block_k + row * params.row_stride;
      const device T* v_row = block_v + row * params.row_stride;

      float partial = 0.0f;
      for (uint base = tid * vec_width; base < head_dim;
           base += threadgroup_width * vec_width) {
        for (uint lane = 0; lane < vec_width; ++lane) {
          uint idx = base + lane;
          if (idx < head_dim) {
            partial += q_shared[idx] * to_float(k_row[idx]);
          }
        }
      }
      partial_sums[tid] = partial;
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (tid == 0) {
        float score = 0.0f;
        for (uint i = 0; i < threadgroup_width; ++i) {
          score += partial_sums[i];
        }
        float m_new = m > score ? m : score;
        float alpha = (m == -INFINITY) ? 0.0f : metal::exp(m - m_new);
        float weight = metal::exp(score - m_new);
        shared_alpha = alpha;
        shared_weight = weight;
        shared_m_val = m_new;
        l = l * alpha + weight;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float alpha = shared_alpha;
      float weight = shared_weight;
      float m_new = shared_m_val;

      for (uint base = tid * vec_width; base < head_dim;
           base += threadgroup_width * vec_width) {
        for (uint lane = 0; lane < vec_width; ++lane) {
          uint idx = base + lane;
          if (idx < head_dim) {
            float v_val = to_float(v_row[idx]);
            accum_shared[idx] = accum_shared[idx] * alpha + weight * v_val;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      m = m_new;
      tokens_done += 1;
    }
  }

  if (tid == 0) {
    shared_inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv_l = shared_inv_l;
  for (uint base = tid * vec_width; base < head_dim;
       base += threadgroup_width * vec_width) {
    for (uint lane = 0; lane < vec_width; ++lane) {
      uint idx = base + lane;
      if (idx < head_dim) {
        out_ptr[idx] = from_float<T>(accum_shared[idx] * inv_l);
      }
    }
  }
}

instantiate_kernel(
    "paged_attention_decode_float16", paged_attention_decode, float16_t);
instantiate_kernel(
    "paged_attention_decode_bfloat16", paged_attention_decode, bfloat16_t);
instantiate_kernel(
    "paged_attention_decode_float32", paged_attention_decode, float);
