// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/metal/kernels/paged_attention.h"
#include "mlx/backend/metal/kernels/utils.h"

#define instantiate_paged_attention_inner(                                     \
    type, head_size, block_size, num_threads, num_simd_lanes, partition_size)  \
  template                                                                     \
      [[host_name("paged_attention_" #type "_hs" #head_size "_bs" #block_size  \
                  "_nt" #num_threads "_nsl" #num_simd_lanes                    \
                  "_ps" #partition_size)]] [[kernel]] void                     \
      paged_attention<                                                         \
          type,                                                                \
          head_size,                                                           \
          block_size,                                                          \
          num_threads,                                                         \
          num_simd_lanes,                                                      \
          partition_size>(                                                     \
          device float* exp_sums                                               \
          [[buffer(0), function_constant(use_partitioning)]],                  \
          device float* max_logits                                             \
          [[buffer(1), function_constant(use_partitioning)]],                  \
          device type* out [[buffer(2)]],                                      \
          device const type* q [[buffer(3)]],                                  \
          device const type* k_cache [[buffer(4)]],                            \
          device const type* v_cache [[buffer(5)]],                            \
          const constant int& num_kv_heads [[buffer(6)]],                      \
          const constant float& scale [[buffer(7)]],                           \
          const constant float& softcapping [[buffer(8)]],                     \
          device const uint32_t* block_tables [[buffer(9)]],                   \
          device const uint32_t* context_lens [[buffer(10)]],                  \
          const constant int& max_num_blocks_per_seq [[buffer(11)]],           \
          device const float* alibi_slopes                                     \
          [[buffer(12), function_constant(use_alibi)]],                        \
          const constant int& q_stride [[buffer(13)]],                         \
          const constant int& kv_block_stride [[buffer(14)]],                  \
          const constant int& kv_head_stride [[buffer(15)]],                   \
          threadgroup char* shared_mem [[threadgroup(0)]],                     \
          uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], \
          uint3 threadgroups_per_grid [[threadgroups_per_grid]],               \
          uint3 thread_position_in_threadgroup                                 \
          [[thread_position_in_threadgroup]],                                  \
          uint simd_tid [[simdgroup_index_in_threadgroup]],                    \
          uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_paged_attention_v2_reduce_inner(                           \
    type, head_size, num_threads, num_simd_lanes, partition_size)              \
  template [[host_name("paged_attention_v2_reduce_" #type "_hs" #head_size     \
                       "_nt" #num_threads "_nsl" #num_simd_lanes               \
                       "_ps" #partition_size)]] [[kernel]] void                \
  paged_attention_v2_reduce<                                                   \
      type,                                                                    \
      head_size,                                                               \
      num_threads,                                                             \
      num_simd_lanes,                                                          \
      partition_size>(                                                         \
      device type * out [[buffer(0)]],                                         \
      const device float* exp_sums [[buffer(1)]],                              \
      const device float* max_logits [[buffer(2)]],                            \
      const device type* tmp_out [[buffer(3)]],                                \
      device uint32_t* context_lens [[buffer(4)]],                             \
      const constant int& max_num_partitions [[buffer(5)]],                    \
      threadgroup char* shared_mem [[threadgroup(0)]],                         \
      uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],     \
      uint3 threadgroups_per_grid [[threadgroups_per_grid]],                   \
      uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], \
      uint3 threads_per_threadgroup [[threads_per_threadgroup]],               \
      uint simd_tid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_paged_attention_heads(                                 \
    type, block_size, num_threads, num_simd_lanes, partition_size)         \
  instantiate_paged_attention_inner(                                       \
      type, 64, block_size, num_threads, num_simd_lanes, partition_size);  \
  instantiate_paged_attention_inner(                                       \
      type, 80, block_size, num_threads, num_simd_lanes, partition_size);  \
  instantiate_paged_attention_inner(                                       \
      type, 96, block_size, num_threads, num_simd_lanes, partition_size);  \
  instantiate_paged_attention_inner(                                       \
      type, 112, block_size, num_threads, num_simd_lanes, partition_size); \
  instantiate_paged_attention_inner(                                       \
      type, 128, block_size, num_threads, num_simd_lanes, partition_size); \
  instantiate_paged_attention_inner(                                       \
      type, 192, block_size, num_threads, num_simd_lanes, partition_size); \
  instantiate_paged_attention_inner(                                       \
      type, 256, block_size, num_threads, num_simd_lanes, partition_size);

#define instantiate_paged_attention_v2_reduce_heads(           \
    type, num_threads, num_simd_lanes, partition_size)         \
  instantiate_paged_attention_v2_reduce_inner(                 \
      type, 64, num_threads, num_simd_lanes, partition_size);  \
  instantiate_paged_attention_v2_reduce_inner(                 \
      type, 80, num_threads, num_simd_lanes, partition_size);  \
  instantiate_paged_attention_v2_reduce_inner(                 \
      type, 96, num_threads, num_simd_lanes, partition_size);  \
  instantiate_paged_attention_v2_reduce_inner(                 \
      type, 112, num_threads, num_simd_lanes, partition_size); \
  instantiate_paged_attention_v2_reduce_inner(                 \
      type, 128, num_threads, num_simd_lanes, partition_size); \
  instantiate_paged_attention_v2_reduce_inner(                 \
      type, 192, num_threads, num_simd_lanes, partition_size); \
  instantiate_paged_attention_v2_reduce_inner(                 \
      type, 256, num_threads, num_simd_lanes, partition_size);

#define instantiate_paged_attention_block_size(               \
    type, num_threads, num_simd_lanes, partition_size)        \
  instantiate_paged_attention_heads(                          \
      type, 8, num_threads, num_simd_lanes, partition_size);  \
  instantiate_paged_attention_heads(                          \
      type, 16, num_threads, num_simd_lanes, partition_size); \
  instantiate_paged_attention_heads(                          \
      type, 32, num_threads, num_simd_lanes, partition_size);

// TODO: tune num_threads = 256
// NOTE: partition_size = 0
#define instantiate_paged_attention_v1(type, num_simd_lanes) \
  instantiate_paged_attention_block_size(type, 256, num_simd_lanes, 0);

// TODO: tune num_threads = 256
// NOTE: partition_size = 512
#define instantiate_paged_attention_v2(type, num_simd_lanes)              \
  instantiate_paged_attention_block_size(type, 256, num_simd_lanes, 512); \
  instantiate_paged_attention_v2_reduce_heads(type, 256, num_simd_lanes, 512);

instantiate_paged_attention_v1(float, 32);
instantiate_paged_attention_v1(bfloat16_t, 32);
instantiate_paged_attention_v1(half, 32);

instantiate_paged_attention_v2(float, 32);
instantiate_paged_attention_v2(bfloat16_t, 32);
instantiate_paged_attention_v2(half, 32);
