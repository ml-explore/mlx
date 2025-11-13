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
  uint q_query_stride;
  uint q_len;
  uint kv_head_stride;
  uint block_stride;
  uint row_stride;
  uint out_batch_stride;
  uint out_head_stride;
  uint out_query_stride;
  uint overlay_batch_stride;
  uint overlay_head_stride;
  uint overlay_seq_stride;
  uint overlay_len;
  uint overlay_valid;
};

struct PagedQuantParams {
  uint enabled;
  uint bits;
  uint group_size;
  uint groups_per_head;
  uint bytes_per_group;
  uint vq_head_stride;
  uint vq_block_stride;
  uint vq_row_stride;
  uint scale_head_stride;
  uint scale_block_stride;
  uint scale_row_stride;
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

inline float dequant_value(
    const device uchar* vq_row,
    const device half* scale_row,
    const device half* zero_row,
    const constant PagedQuantParams& quant,
    uint feature_idx) {
  uint group = feature_idx / quant.group_size;
  if (group >= quant.groups_per_head) {
    return 0.0f;
  }
  uint offset = feature_idx - group * quant.group_size;
  uint byte_index = group * quant.bytes_per_group;
  uint q_value = 0;

  if (quant.bits == 8) {
    byte_index += offset;
    q_value = static_cast<uint>(vq_row[byte_index]);
  } else {
    byte_index += offset >> 1;
    uchar packed = vq_row[byte_index];
    if ((offset & 1u) == 0u) {
      q_value = static_cast<uint>(packed & 0x0Fu);
    } else {
      q_value = static_cast<uint>((packed >> 4u) & 0x0Fu);
    }
  }

  float scale = static_cast<float>(scale_row[group]);
  float zero = static_cast<float>(zero_row[group]);
  return (static_cast<float>(q_value) - zero) * scale;
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
    const device uchar* v_q [[buffer(8)]],
    const device half* v_scale [[buffer(9)]],
    const device half* v_zero [[buffer(10)]],
    const constant PagedQuantParams& quant_params [[buffer(11)]],
    ushort3 tg_pos [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
  const uint batch_idx = tg_pos.y;
  const uint head_idx = tg_pos.x;
  const uint head_dim = params.head_dim;
  const uint block_size_const =
      FC_BLOCK_SIZE > 0 ? (uint)FC_BLOCK_SIZE : params.block_size;
  const uint threadgroup_width = metal::min(
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
  const uint seq_len = context_lens[batch_idx] > 0
      ? static_cast<uint>(context_lens[batch_idx])
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
      : (params.num_kv_heads == 1
             ? 0u
             : (head_idx * params.num_kv_heads) / params.num_q_heads);

  const device T* q_ptr =
      q + batch_idx * params.q_batch_stride + head_idx * params.q_head_stride;
  const device T* k_head = k + kv_head * params.kv_head_stride;
  const device T* v_head = v + kv_head * params.kv_head_stride;
  const bool use_quant = quant_params.enabled != 0;
  const device uchar* vq_head = v_q + kv_head * quant_params.vq_head_stride;
  const device half* scale_head =
      v_scale + kv_head * quant_params.scale_head_stride;
  const device half* zero_head =
      v_zero + kv_head * quant_params.scale_head_stride;

  threadgroup float q_shared[MAX_HEAD_DIM];
  threadgroup float accum_shared[MAX_HEAD_DIM];
  threadgroup float partial_sums[MAX_THREADGROUP_WIDTH];
  threadgroup float shared_alpha;
  threadgroup float shared_weight;
  threadgroup float shared_inv_l = 0.0f;
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

  for (uint blk = 0; blk < params.max_blocks_per_seq && tokens_done < seq_len;
       ++blk) {
    const int block_id = block_row[blk];
    if (block_id < 0) {
      continue;
    }

    const device T* block_k = k_head + block_id * params.block_stride;
    const device T* block_v = v_head + block_id * params.block_stride;
    const device uchar* block_vq =
        vq_head + block_id * quant_params.vq_block_stride;
    const device half* block_scale =
        scale_head + block_id * quant_params.scale_block_stride;
    const device half* block_zero =
        zero_head + block_id * quant_params.scale_block_stride;

    uint rows = block_size_const;
    if (tokens_done + rows > seq_len) {
      rows = seq_len - tokens_done;
    }

    for (uint row = 0; row < rows; ++row) {
      const device T* k_row = block_k + row * params.row_stride;
      const device T* v_row = block_v + row * params.row_stride;
      const device uchar* vq_row = block_vq + row * quant_params.vq_row_stride;
      const device half* scale_row =
          block_scale + row * quant_params.scale_row_stride;
      const device half* zero_row =
          block_zero + row * quant_params.scale_row_stride;

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
        float alpha = (m == -INFINITY) ? 0.0f : metal::precise::exp(m - m_new);
        float weight = metal::precise::exp(score - m_new);
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
            float v_val = use_quant
                ? dequant_value(vq_row, scale_row, zero_row, quant_params, idx)
                : to_float(v_row[idx]);
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
    "paged_attention_decode_float16",
    paged_attention_decode,
    float16_t);
instantiate_kernel(
    "paged_attention_decode_bfloat16",
    paged_attention_decode,
    bfloat16_t);
instantiate_kernel(
    "paged_attention_decode_float32",
    paged_attention_decode,
    float);

template <typename T>
kernel void paged_attention_decode_overlay(
    const device T* q [[buffer(0)]],
    const device T* k [[buffer(1)]],
    const device T* v [[buffer(2)]],
    const device int* block_tables [[buffer(3)]],
    const device int* context_lens [[buffer(4)]],
    const device int* kv_mapping [[buffer(5)]],
    device T* out [[buffer(6)]],
    const constant PagedParams& params [[buffer(7)]],
    const device uchar* v_q [[buffer(8)]],
    const device half* v_scale [[buffer(9)]],
    const device half* v_zero [[buffer(10)]],
    const constant PagedQuantParams& quant_params [[buffer(11)]],
    const device T* k_overlay [[buffer(12)]],
    const device T* v_overlay [[buffer(13)]],
    ushort3 tg_pos [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
  const uint batch_idx = tg_pos.y;
  const uint head_idx = tg_pos.x;
  const uint head_dim = params.head_dim;
  const uint block_size_const =
      FC_BLOCK_SIZE > 0 ? (uint)FC_BLOCK_SIZE : params.block_size;
  const uint threadgroup_width = metal::min(
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
  const uint seq_len = context_lens[batch_idx] > 0
      ? static_cast<uint>(context_lens[batch_idx])
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
      : (params.num_kv_heads == 1
             ? 0u
             : (head_idx * params.num_kv_heads) / params.num_q_heads);

  const device T* q_ptr =
      q + batch_idx * params.q_batch_stride + head_idx * params.q_head_stride;
  const device T* k_head = k + kv_head * params.kv_head_stride;
  const device T* v_head = v + kv_head * params.kv_head_stride;
  const bool use_quant = quant_params.enabled != 0;
  const device uchar* vq_head = v_q + kv_head * quant_params.vq_head_stride;
  const device half* scale_head =
      v_scale + kv_head * quant_params.scale_head_stride;
  const device half* zero_head =
      v_zero + kv_head * quant_params.scale_head_stride;

  threadgroup float q_shared[MAX_HEAD_DIM];
  threadgroup float accum_shared[MAX_HEAD_DIM];
  threadgroup float partial_sums[MAX_THREADGROUP_WIDTH];
  threadgroup float shared_alpha;
  threadgroup float shared_weight;
  threadgroup float shared_inv_l = 0.0f;
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

  for (uint blk = 0; blk < params.max_blocks_per_seq && tokens_done < seq_len;
       ++blk) {
    const int block_id = block_row[blk];
    if (block_id < 0) {
      continue;
    }

    const device T* block_k = k_head + block_id * params.block_stride;
    const device T* block_v = v_head + block_id * params.block_stride;
    const device uchar* block_vq =
        vq_head + block_id * quant_params.vq_block_stride;
    const device half* block_scale =
        scale_head + block_id * quant_params.scale_block_stride;
    const device half* block_zero =
        zero_head + block_id * quant_params.scale_block_stride;

    uint rows = block_size_const;
    if (tokens_done + rows > seq_len) {
      rows = seq_len - tokens_done;
    }

    for (uint row = 0; row < rows; ++row) {
      const device T* k_row = block_k + row * params.row_stride;
      const device T* v_row = block_v + row * params.row_stride;
      const device uchar* vq_row = block_vq + row * quant_params.vq_row_stride;
      const device half* scale_row =
          block_scale + row * quant_params.scale_row_stride;
      const device half* zero_row =
          block_zero + row * quant_params.scale_row_stride;

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
        float alpha = (m == -INFINITY) ? 0.0f : metal::precise::exp(m - m_new);
        float weight = metal::precise::exp(score - m_new);
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
            float v_val = use_quant
                ? dequant_value(vq_row, scale_row, zero_row, quant_params, idx)
                : to_float(v_row[idx]);
            accum_shared[idx] = accum_shared[idx] * alpha + weight * v_val;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      m = m_new;
      tokens_done += 1;
    }
  }

  uint overlay_total = params.overlay_len;
  if (params.overlay_valid > 0 && params.overlay_valid < overlay_total) {
    overlay_total = params.overlay_valid;
  }
  if (overlay_total > 0 && params.overlay_head_stride > 0) {
    for (uint step = 0; step < overlay_total; ++step) {
      uint overlay_offset = batch_idx * params.overlay_batch_stride +
          kv_head * params.overlay_head_stride;
      if (params.overlay_seq_stride > 0) {
        overlay_offset += step * params.overlay_seq_stride;
      }
      const device T* overlay_k_row = k_overlay + overlay_offset;
      const device T* overlay_v_row = v_overlay + overlay_offset;

      float partial = 0.0f;
      for (uint base = tid * vec_width; base < head_dim;
           base += threadgroup_width * vec_width) {
        for (uint lane = 0; lane < vec_width; ++lane) {
          uint idx = base + lane;
          if (idx < head_dim) {
            partial += q_shared[idx] * to_float(overlay_k_row[idx]);
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
        float alpha = (m == -INFINITY) ? 0.0f : metal::precise::exp(m - m_new);
        float weight = metal::precise::exp(score - m_new);
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
            float v_val = to_float(overlay_v_row[idx]);
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
    "paged_attention_decode_overlay_float16",
    paged_attention_decode_overlay,
    float16_t);
instantiate_kernel(
    "paged_attention_decode_overlay_bfloat16",
    paged_attention_decode_overlay,
    bfloat16_t);
instantiate_kernel(
    "paged_attention_decode_overlay_float32",
    paged_attention_decode_overlay,
    float);

template <typename T>
kernel void paged_prefill_kernel(
    const device T* q [[buffer(0)]],
    const device T* k [[buffer(1)]],
    const device T* v [[buffer(2)]],
    const device int* block_tables [[buffer(3)]],
    const device int* context_lens [[buffer(4)]],
    const device int* base_lens [[buffer(5)]],
    const device int* kv_mapping [[buffer(6)]],
    device T* out [[buffer(7)]],
    const constant PagedParams& params [[buffer(8)]],
    const device uchar* v_q [[buffer(9)]],
    const device half* v_scale [[buffer(10)]],
    const device half* v_zero [[buffer(11)]],
    const constant PagedQuantParams& quant_params [[buffer(12)]],
    ushort3 tg_pos [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
  const uint batch_idx = tg_pos.y;
  const uint head_idx = tg_pos.x;
  const uint head_dim = params.head_dim;
  const uint block_size_const =
      FC_BLOCK_SIZE > 0 ? (uint)FC_BLOCK_SIZE : params.block_size;
  const uint threadgroup_width = metal::min(
      FC_THREADGROUP_WIDTH > 0 ? (uint)FC_THREADGROUP_WIDTH : 32u,
      MAX_THREADGROUP_WIDTH);
  const uint vec_width = FC_VEC_WIDTH > 0 ? (uint)FC_VEC_WIDTH : 1u;

  const device int* block_row =
      block_tables + batch_idx * params.max_blocks_per_seq;
  const uint seq_cap = context_lens[batch_idx] > 0
      ? static_cast<uint>(context_lens[batch_idx])
      : 0u;
  const uint base_len =
      base_lens[batch_idx] > 0 ? static_cast<uint>(base_lens[batch_idx]) : 0u;

  const uint kv_head = params.has_kv_mapping
      ? static_cast<uint>(kv_mapping[head_idx])
      : (params.num_kv_heads == 1
             ? 0u
             : (head_idx * params.num_kv_heads) / params.num_q_heads);

  const device T* q_head_base =
      q + batch_idx * params.q_batch_stride + head_idx * params.q_head_stride;
  const device T* k_head = k + kv_head * params.kv_head_stride;
  const device T* v_head = v + kv_head * params.kv_head_stride;
  const bool use_quant = quant_params.enabled != 0;
  const device uchar* vq_head = v_q + kv_head * quant_params.vq_head_stride;
  const device half* scale_head =
      v_scale + kv_head * quant_params.scale_head_stride;
  const device half* zero_head =
      v_zero + kv_head * quant_params.scale_head_stride;

  device T* out_head_base = out + batch_idx * params.out_batch_stride +
      head_idx * params.out_head_stride;

  threadgroup float q_shared[MAX_HEAD_DIM];
  threadgroup float accum_shared[MAX_HEAD_DIM];
  threadgroup float partial_sums[MAX_THREADGROUP_WIDTH];
  threadgroup float shared_alpha;
  threadgroup float shared_weight;
  threadgroup float shared_inv_l = 0.0f;
  threadgroup float shared_m_val;

  for (uint q_idx = 0; q_idx < params.q_len; ++q_idx) {
    const device T* q_ptr = q_head_base + q_idx * params.q_query_stride;
    device T* out_ptr = out_head_base + q_idx * params.out_query_stride;

    uint limit = base_len + q_idx + 1;
    const uint seq_len = seq_cap < limit ? seq_cap : limit;
    if (head_dim == 0 || seq_len == 0 || head_dim > MAX_HEAD_DIM) {
      for (uint base = tid * vec_width; base < head_dim;
           base += threadgroup_width * vec_width) {
        for (uint lane = 0; lane < vec_width; ++lane) {
          uint idx = base + lane;
          if (idx < head_dim) {
            out_ptr[idx] = from_float<T>(0.0f);
          }
        }
      }
      continue;
    }

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

    for (uint blk = 0; blk < params.max_blocks_per_seq && tokens_done < seq_len;
         ++blk) {
      const int block_id = block_row[blk];
      if (block_id < 0) {
        continue;
      }

      const device T* block_k = k_head + block_id * params.block_stride;
      const device T* block_v = v_head + block_id * params.block_stride;
      const device uchar* block_vq =
          vq_head + block_id * quant_params.vq_block_stride;
      const device half* block_scale =
          scale_head + block_id * quant_params.scale_block_stride;
      const device half* block_zero =
          zero_head + block_id * quant_params.scale_block_stride;

      uint rows = block_size_const;
      if (tokens_done + rows > seq_len) {
        rows = seq_len - tokens_done;
      }

      for (uint row = 0; row < rows; ++row) {
        const device T* k_row = block_k + row * params.row_stride;
        const device T* v_row = block_v + row * params.row_stride;
        const device uchar* vq_row =
            block_vq + row * quant_params.vq_row_stride;
        const device half* scale_row =
            block_scale + row * quant_params.scale_row_stride;
        const device half* zero_row =
            block_zero + row * quant_params.scale_row_stride;

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
          float alpha =
              (m == -INFINITY) ? 0.0f : metal::precise::exp(m - m_new);
          float weight = metal::precise::exp(score - m_new);
          shared_alpha = alpha;
          shared_weight = weight;
          shared_m_val = m_new;
          l = l * alpha + weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float alpha = shared_alpha;
        float weight = shared_weight;
        float m_new = shared_m_val;
        if (tid == 0) {
          m = m_new;
        }

        for (uint base = tid * vec_width; base < head_dim;
             base += threadgroup_width * vec_width) {
          for (uint lane = 0; lane < vec_width; ++lane) {
            uint idx = base + lane;
            if (idx < head_dim) {
              float v_value;
              if (use_quant) {
                v_value = dequant_value(
                    vq_row, scale_row, zero_row, quant_params, idx);
              } else {
                v_value = to_float(v_row[idx]);
              }
              accum_shared[idx] = accum_shared[idx] * alpha + weight * v_value;
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      tokens_done += rows;
    }

    float inv_l = (l == 0.0f) ? 0.0f : (1.0f / l);
    if (tid == 0) {
      shared_inv_l = inv_l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    inv_l = shared_inv_l;

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
}

instantiate_kernel("paged_prefill_float16", paged_prefill_kernel, float16_t);
instantiate_kernel("paged_prefill_bfloat16", paged_prefill_kernel, bfloat16_t);
instantiate_kernel("paged_prefill_float32", paged_prefill_kernel, float);
