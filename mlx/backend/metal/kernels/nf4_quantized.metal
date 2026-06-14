// Copyright © 2025 Apple Inc.
// NF4 quantized kernel instantiations

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/nf4_quantized.h"

// NF4 is always 4-bit with group_size=64.
// We instantiate for all three compute types.

#define instantiate_nf4(name, type, group_size) \
  instantiate_kernel( \
      "nf4_" #name "_" #type "_gs_" #group_size "_b_4", \
      nf4_ ## name, \
      type, \
      group_size)

#define instantiate_nf4_batched(name, type, batched, group_size) \
  instantiate_kernel( \
      "nf4_" #name "_" #type "_gs_" #group_size "_b_4_batch_" #batched, \
      nf4_ ## name, \
      type, \
      group_size, \
      batched)

#define instantiate_nf4_aligned(name, type, aligned, group_size) \
  instantiate_kernel( \
      "nf4_" #name "_" #type "_gs_" #group_size "_b_4_alN_" #aligned, \
      nf4_ ## name, \
      type, \
      group_size, \
      aligned)

#define instantiate_nf4_aligned_batched(name, type, aligned, batched, group_size) \
  instantiate_kernel( \
      "nf4_" #name "_" #type "_gs_" #group_size "_b_4_alN_" #aligned "_batch_" #batched, \
      nf4_ ## name, \
      type, \
      group_size, \
      aligned, \
      batched)

#define instantiate_nf4_quad(name, type, D, batched, group_size) \
  instantiate_kernel( \
      "nf4_" #name "_" #type "_gs_" #group_size "_b_4_d_" #D "_batch_" #batched, \
      nf4_ ## name, \
      type, \
      group_size, \
      D, \
      batched)

#define instantiate_nf4_split_k(name, type, split_k, group_size) \
  instantiate_kernel( \
      "nf4_" #name "_" #type "_gs_" #group_size "_b_4_spk_" #split_k, \
      nf4_ ## name, \
      type, \
      group_size, \
      split_k)

#define instantiate_nf4_batched_wrap(name, type, group_size) \
  instantiate_nf4_batched(name, type, 1, group_size) \
  instantiate_nf4_batched(name, type, 0, group_size)

#define instantiate_nf4_all_batched(type, group_size) \
  instantiate_nf4_batched_wrap(qmv_fast, type, group_size) \
  instantiate_nf4_batched_wrap(qmv, type, group_size) \
  instantiate_nf4_batched_wrap(qvm, type, group_size) \
  instantiate_nf4_batched_wrap(qmm_n, type, group_size)

#define instantiate_nf4_all_aligned(type, group_size) \
  instantiate_nf4_aligned_batched(qmm_t, type, true, 1, group_size) \
  instantiate_nf4_aligned_batched(qmm_t, type, true, 0, group_size) \
  instantiate_nf4_aligned_batched(qmm_t, type, false, 1, group_size) \
  instantiate_nf4_aligned_batched(qmm_t, type, false, 0, group_size)

#define instantiate_nf4_all_quad(type, group_size) \
  instantiate_nf4_quad(qmv_quad, type, 64, 1, group_size) \
  instantiate_nf4_quad(qmv_quad, type, 64, 0, group_size) \
  instantiate_nf4_quad(qmv_quad, type, 128, 1, group_size) \
  instantiate_nf4_quad(qmv_quad, type, 128, 0, group_size)

#define instantiate_nf4_all_splitk(type, group_size) \
  instantiate_nf4_split_k(qvm_split_k, type, 8, group_size) \
  instantiate_nf4_split_k(qvm_split_k, type, 32, group_size) \
  instantiate_nf4_aligned(qmm_t_splitk, type, true, group_size) \
  instantiate_nf4_aligned(qmm_t_splitk, type, false, group_size)

#define instantiate_nf4_quantize_dequantize(type, group_size) \
  instantiate_kernel( \
    "nf4_quantize_" #type "_gs_" #group_size "_b_4", \
    nf4_quantize, \
    type, \
    group_size) \
  instantiate_kernel( \
    "nf4_dequantize_" #type "_gs_" #group_size "_b_4", \
    nf4_dequantize, \
    type, \
    group_size)

#define instantiate_nf4_all(type, group_size) \
  instantiate_nf4_all_batched(type, group_size) \
  instantiate_nf4_all_quad(type, group_size) \
  instantiate_nf4_all_splitk(type, group_size) \
  instantiate_nf4_all_aligned(type, group_size) \
  instantiate_nf4_quantize_dequantize(type, group_size)

// NF4 with group_size=64 (the bitsandbytes default)
instantiate_nf4_all(float, 64)
instantiate_nf4_all(bfloat16_t, 64)
instantiate_nf4_all(float16_t, 64)

// Also support group_size=32 for flexibility
instantiate_nf4_all(float, 32)
instantiate_nf4_all(bfloat16_t, 32)
instantiate_nf4_all(float16_t, 32)

// And group_size=128 (ROCm default)
instantiate_nf4_all(float, 128)
instantiate_nf4_all(bfloat16_t, 128)
instantiate_nf4_all(float16_t, 128)
// clang-format on
