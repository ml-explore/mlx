// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/paged_attention_primitives.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core::paged_attention {

void paged_attention_v1(
    const array& q,
    const array& k_cache,
    const array& v_cache,
    const array& block_tables,
    const array& context_lens,
    const int head_size,
    const int block_size,
    const int num_kv_heads,
    const float scale,
    const float softcapping,
    const int max_context_len,
    const int max_num_blocks_per_seq,
    const std::optional<array> alibi,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const int num_heads,
    const int num_seqs,
    array& out,
    metal::Device& d,
    const Stream& s) {
  const int partition_size = 0;
  const int num_threads = 256;
  const int num_simd_lanes = 32;
  const bool use_partitioning = false;
  const bool use_alibi = alibi.has_value();

  std::string type_string = get_type_string(q.dtype());
  std::string kname;
  kname.reserve(64);
  concatenate(
      kname,
      "paged_attention_",
      type_string,
      "_hs",
      head_size,
      "_bs",
      block_size,
      "_nt",
      num_threads,
      "_nsl",
      num_simd_lanes,
      "_ps",
      partition_size);

  auto template_def = get_template_definition(
      kname,
      "paged_attention",
      type_string,
      head_size,
      block_size,
      num_threads,
      num_simd_lanes,
      partition_size);

  // Encode and dispatch kernel
  metal::MTLFCList func_consts = {
      {use_partitioning, MTL::DataType::DataTypeBool, 10},
      {use_alibi, MTL::DataType::DataTypeBool, 20},
  };

  std::string hash_name = kname;
  auto kernel = get_paged_attention_kernel(
      d, kname, hash_name, func_consts, template_def);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  auto num_simds = num_threads / num_simd_lanes;
  auto max_num_partitions =
      (max_context_len + partition_size - 1) / partition_size;
  auto logits_size = partition_size * size_of(float32);
  auto outputs_size = (num_simds / 2) * head_size * size_of(float32);
  auto shared_mem_size = std::max(logits_size, outputs_size);
  compute_encoder.set_threadgroup_memory_length(shared_mem_size, 0);

  compute_encoder.set_output_array(out, 2);
  compute_encoder.set_input_array(q, 3);
  compute_encoder.set_input_array(k_cache, 4);
  compute_encoder.set_input_array(v_cache, 5);

  compute_encoder.set_bytes(num_kv_heads, 6);
  compute_encoder.set_bytes(scale, 7);
  compute_encoder.set_bytes(softcapping, 8);

  compute_encoder.set_input_array(block_tables, 9);
  compute_encoder.set_input_array(context_lens, 10);

  compute_encoder.set_bytes(max_num_blocks_per_seq, 11);

  if (use_alibi) {
    compute_encoder.set_input_array(alibi.value(), 12);
  }

  compute_encoder.set_bytes(q_stride, 13);
  compute_encoder.set_bytes(kv_block_stride, 14);
  compute_encoder.set_bytes(kv_head_stride, 15);

  MTL::Size grid_dims(num_heads, num_seqs, 1);
  MTL::Size group_dims(num_threads, 1, 1);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  return;
}

void paged_attention_v2(
    const array& q,
    const array& k_cache,
    const array& v_cache,
    const array& block_tables,
    const array& context_lens,
    const int head_size,
    const int block_size,
    const int num_kv_heads,
    const float scale,
    const float softcapping,
    const int max_context_len,
    const int max_num_blocks_per_seq,
    const int max_num_partitions,
    const std::optional<array> alibi,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const int num_heads,
    const int num_seqs,
    array& out,
    metal::Device& d,
    const Stream& s) {
  const int partition_size = 512;
  const int num_threads = 256;
  const int num_simd_lanes = 32;
  const bool use_partitioning = true;
  const bool use_alibi = alibi.has_value();

  std::string type_string = get_type_string(q.dtype());
  std::string kname;
  kname.reserve(64);
  concatenate(
      kname,
      "paged_attention_",
      type_string,
      "_hs",
      head_size,
      "_bs",
      block_size,
      "_nt",
      num_threads,
      "_nsl",
      num_simd_lanes,
      "_ps",
      partition_size);

  auto template_def = get_template_definition(
      kname,
      "paged_attention",
      type_string,
      head_size,
      block_size,
      num_threads,
      num_simd_lanes,
      partition_size);

  // Encode and dispatch kernel
  metal::MTLFCList func_consts = {
      {use_partitioning, MTL::DataType::DataTypeBool, 10},
      {use_alibi, MTL::DataType::DataTypeBool, 20},
  };

  std::string hash_name = kname;
  auto kernel = get_paged_attention_kernel(
      d, kname, hash_name, func_consts, template_def);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  auto tmp_out =
      array({num_seqs, num_heads, max_num_partitions, head_size}, float32);
  tmp_out.set_data(allocator::malloc(tmp_out.nbytes()));
  auto exp_sums = array({num_seqs, num_heads, max_num_partitions}, float32);
  exp_sums.set_data(allocator::malloc(exp_sums.nbytes()));

  std::vector<array> copies = {tmp_out, exp_sums};

  auto num_simds = num_threads / num_simd_lanes;
  auto max_num_partitions =
      (max_context_len + partition_size - 1) / partition_size;
  auto logits_size = partition_size * size_of(float32);
  auto outputs_size = (num_simds / 2) * head_size * size_of(float32);
  auto shared_mem_size = std::max(logits_size, outputs_size);
  compute_encoder.set_threadgroup_memory_length(shared_mem_size, 0);

  compute_encoder.set_output_array(tmp_out, 0);
  compute_encoder.set_output_array(exp_sums, 1);
  compute_encoder.set_output_array(out, 2);
  compute_encoder.set_input_array(q, 3);
  compute_encoder.set_input_array(k_cache, 4);
  compute_encoder.set_input_array(v_cache, 5);

  compute_encoder.set_bytes(num_kv_heads, 6);
  compute_encoder.set_bytes(scale, 7);
  compute_encoder.set_bytes(softcapping, 8);

  compute_encoder.set_input_array(block_tables, 9);
  compute_encoder.set_input_array(context_lens, 10);

  compute_encoder.set_bytes(max_num_blocks_per_seq, 11);

  if (use_alibi) {
    compute_encoder.set_input_array(alibi.value(), 12);
  }

  compute_encoder.set_bytes(q_stride, 13);
  compute_encoder.set_bytes(kv_block_stride, 14);
  compute_encoder.set_bytes(kv_head_stride, 15);

  MTL::Size grid_dims(num_heads, num_seqs, max_num_partitions);
  MTL::Size group_dims(num_threads, 1, 1);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  d.add_temporaries(std::move(copies), s.index);
  return;
}

void PagedAttention::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc(out.nbytes()));

  auto& q = inputs[0];
  auto& k_cache = inputs[1];
  auto& v_cache = inputs[2];
  auto& block_tables = inputs[3];
  auto& context_lens = inputs[4];
  const auto alibi_slopes =
      inputs.size() == 6 ? std::optional{inputs[5]} : std::nullopt;

  if (use_v1_) {
    paged_attention_v1(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        head_size_,
        block_size_,
        num_kv_heads_,
        softmax_scale_,
        softcapping_.value_or(0.),
        max_context_len_,
        max_num_blocks_per_seq_,
        alibi_slopes,
        q_stride_,
        kv_block_stride_,
        kv_head_stride_,
        num_heads_,
        num_seqs_,
        out,
        d,
        s);
  } else {
    paged_attention_v2(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        head_size_,
        block_size_,
        num_kv_heads_,
        softmax_scale_,
        softcapping_.value_or(0.),
        max_context_len_,
        max_num_blocks_per_seq_,
        max_num_partitions_,
        alibi_slopes,
        q_stride_,
        kv_block_stride_,
        kv_head_stride_,
        num_heads_,
        num_seqs_,
        out,
        d,
        s);
  }
}
} // namespace mlx::core::paged_attention