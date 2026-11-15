// Copyright © 2023-2024 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

///////////////////////////////////////////////////////////////////////////////
// affine_gather_qmv, affine_gather_qmv_fast, affine_gather_qvm,
// affine_quantize, affine_dequantize
///////////////////////////////////////////////////////////////////////////////

#define instantiate_simple(name, type, scales_type, suffix, group_size, bits) \
  instantiate_kernel(   \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs_0" suffix, \
      name,             \
      type,       \
      scales_type,             \
      group_size,       \
      bits,             \
      false)

#define instantiate_simple_all(type, scales_type, suffix, group_size, bits)               \
  instantiate_simple(affine_gather_qmv, type, scales_type, suffix, group_size, bits)      \
  instantiate_simple(affine_gather_qmv_fast, type, scales_type, suffix, group_size, bits) \
  instantiate_simple(affine_gather_qvm, type, scales_type, suffix, group_size, bits)

///////////////////////////////////////////////////////////////////////////////
// affine_qmm_t, affine_qmm_t_splitk, affine_gather_qmm_t, affine_qmm_n,
// affine_gather_qmm_n
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qmm(name, type, scales_type, suffix, group_size, bits, aligned_n, batched) \
  instantiate_kernel( \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs_0_alN_" #aligned_n "_batch_" #batched "_bm_32_bn_32" suffix, \
      name,       \
      type,       \
      scales_type,       \
      group_size, \
      bits,       \
      false,      \
      aligned_n,  \
      batched,    \
      32,         \
      32,         \
      32)

#define instantiate_qmm_aligned(name, type, scales_type, suffix, group_size, bits) \
  instantiate_qmm(name, type, scales_type, suffix, group_size, bits, 0, 0)         \
  instantiate_qmm(name, type, scales_type, suffix, group_size, bits, 1, 0)

#define instantiate_qmm_batched(name, type, scales_type, suffix, group_size, bits) \
  instantiate_qmm(name, type, scales_type, suffix, group_size, bits, 0, 0)         \
  instantiate_qmm(name, type, scales_type, suffix, group_size, bits, 0, 1)

#define instantiate_qmm_aligned_batched(name, type, scales_type, suffix, group_size, bits) \
  instantiate_qmm(name, type, scales_type, suffix, group_size, bits, 0, 0)                 \
  instantiate_qmm(name, type, scales_type, suffix, group_size, bits, 0, 1)                 \
  instantiate_qmm(name, type, scales_type, suffix, group_size, bits, 1, 0)                 \
  instantiate_qmm(name, type, scales_type, suffix, group_size, bits, 1, 1)

#define instantiate_qmm_all(type, scales_type, suffix, group_size, bits) \
  instantiate_qmm_aligned_batched(affine_qmm_t, type, scales_type, suffix, group_size, bits) \
  instantiate_qmm_aligned(affine_qmm_t_splitk, type, scales_type, suffix, group_size, bits)  \
  instantiate_qmm_aligned(affine_gather_qmm_t, type, scales_type, suffix, group_size, bits)  \
  instantiate_qmm_batched(affine_qmm_n, type, scales_type, suffix, group_size, bits)         \
  instantiate_qmm(affine_gather_qmm_n, type, scales_type, suffix, group_size, bits, 0, 0)

///////////////////////////////////////////////////////////////////////////////
// affine_qmv, affine_qmv_fast
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qmv(name, type, scales_type, suffix, group_size, bits, batched, partial_rows) \
  instantiate_kernel( \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs_0_batch_" #batched "_pr_" #partial_rows "_r_4" suffix, \
      name,           \
      type,       \
      scales_type,           \
      group_size,     \
      bits,           \
      false,          \
      batched,        \
      partial_rows,   \
      4)

#define instantiate_qmv_batched(name, type, scales_type, suffix, group_size, bits, partial_rows) \
  instantiate_qmv(name, type, scales_type, suffix, group_size, bits, 0, partial_rows)            \
  instantiate_qmv(name, type, scales_type, suffix, group_size, bits, 1, partial_rows)

#define instantiate_qmv_all(type, scales_type, suffix, group_size, bits)                   \
  instantiate_qmv_batched(affine_qmv, type, scales_type, suffix, group_size, bits, 0)      \
  instantiate_qmv_batched(affine_qmv_fast, type, scales_type, suffix, group_size, bits, 0) \
  instantiate_qmv_batched(affine_qmv_fast, type, scales_type, suffix, group_size, bits, 1)

///////////////////////////////////////////////////////////////////////////////
// affine_qvm
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qvm(name, type, scales_type, suffix, group_size, bits, batched) \
  instantiate_kernel( \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs_0_batch_" #batched suffix, \
      name,           \
      type,       \
      scales_type,           \
      group_size,     \
      bits,           \
      false,          \
      batched)

#define instantiate_qvm_all(type, scales_type, suffix, group_size, bits)       \
  instantiate_qvm(affine_qvm, type, scales_type, suffix, group_size, bits, 0)  \
  instantiate_qvm(affine_qvm, type, scales_type, suffix, group_size, bits, 1)

///////////////////////////////////////////////////////////////////////////////
// affine_qvm_splitk
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qvm_splitk(name, type, scales_type, suffix, group_size, bits, split_k) \
  instantiate_kernel( \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_spk_" #split_k suffix, \
      name,           \
      type,       \
      scales_type,           \
      group_size,     \
      bits,           \
      split_k)

#define instantiate_qvm_splitk_all(type, scales_type, suffix, group_size, bits)              \
  instantiate_qvm_splitk(affine_qvm_splitk, type, scales_type, suffix, group_size, bits, 8)  \
  instantiate_qvm_splitk(affine_qvm_splitk, type, scales_type, suffix, group_size, bits, 32)

///////////////////////////////////////////////////////////////////////////////
// affine_gather_qmm_rhs
///////////////////////////////////////////////////////////////////////////////

#define instantiate_grouped_qmm(name, type, scales_type, suffix, group_size, bits, transpose) \
  instantiate_kernel( \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs_0_t_" #transpose suffix, \
      name,           \
      type,       \
      scales_type,           \
      group_size,     \
      bits,           \
      false,          \
      transpose)

#define instantiate_grouped_qmm_all(type, scales_type, suffix, group_size, bits)                 \
  instantiate_grouped_qmm(affine_gather_qmm_rhs, type, scales_type, suffix, group_size, bits, 0) \
  instantiate_grouped_qmm(affine_gather_qmm_rhs, type, scales_type, suffix, group_size, bits, 1)

///////////////////////////////////////////////////////////////////////////////
// affine_qmv_quad
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qmv_quad(name, type, scales_type, suffix, group_size, bits, d, batched) \
  instantiate_kernel( \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_d_" #d "_batch_" #batched suffix, \
      name,           \
      type,       \
      scales_type,           \
      group_size,     \
      bits,           \
      d,              \
      batched)

#define instantiate_qmv_quad_batched(name, type, scales_type, suffix, group_size, bits, d) \
  instantiate_qmv_quad(name, type, scales_type, suffix, group_size, bits, d, 0)            \
  instantiate_qmv_quad(name, type, scales_type, suffix, group_size, bits, d, 1)

#define instantiate_qmv_quad_all(type, scales_type, suffix, group_size, bits)                    \
  instantiate_qmv_quad_batched(affine_qmv_quad, type, scales_type, suffix, group_size, bits, 64) \
  instantiate_qmv_quad_batched(affine_qmv_quad, type, scales_type, suffix, group_size, bits, 128)

///////////////////////////////////////////////////////////////////////////////
// affine_qmv_wide
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qmv_wide(name, type, scales_type, suffix, group_size, bits, vecs_per_tg, k_lanes, batched) \
  instantiate_kernel( \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_nv_" #vecs_per_tg "_kl_" #k_lanes "_batch_" #batched suffix, \
      name,           \
      type,       \
      scales_type,           \
      group_size,     \
      bits,           \
      vecs_per_tg,    \
      k_lanes,        \
      batched)

#define instantiate_qmv_wide_batched(name, type, scales_type, suffix, group_size, bits, vecs_per_tg, k_lanes) \
  instantiate_qmv_wide(name, type, scales_type, suffix, group_size, bits, vecs_per_tg, k_lanes, 0)            \
  instantiate_qmv_wide(name, type, scales_type, suffix, group_size, bits, vecs_per_tg, k_lanes, 1)

#define instantiate_qmv_wide_all(type, scales_type, suffix, group_size, bits)                      \
  instantiate_qmv_wide_batched(affine_qmv_wide, type, scales_type, suffix, group_size, bits, 2, 8) \
  instantiate_qmv_wide_batched(affine_qmv_wide, type, scales_type, suffix, group_size, bits, 3, 8) \
  instantiate_qmv_wide_batched(affine_qmv_wide, type, scales_type, suffix, group_size, bits, 4, 8) \
  instantiate_qmv_wide_batched(affine_qmv_wide, type, scales_type, suffix, group_size, bits, 5, 8) \
  instantiate_qmv_wide_batched(affine_qmv_wide, type, scales_type, suffix, group_size, bits, 6, 8) \
  instantiate_qmv_wide_batched(affine_qmv_wide, type, scales_type, suffix, group_size, bits, 7, 8) \
  instantiate_qmv_wide_batched(affine_qmv_wide, type, scales_type, suffix, group_size, bits, 8, 8)

///////////////////////////////////////////////////////////////////////////////
// affine_qmv_mma
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qmv_mma(name, type, scales_type, suffix, group_size, bits, wk, bn, tm) \
  instantiate_kernel( \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_wk_" #wk "_bn_" #bn "_tm_" #tm suffix, \
      name,           \
      type,       \
      scales_type,           \
      group_size,     \
      bits,           \
      wk,             \
      bn,             \
      tm)

#define instantiate_qmv_mma_all(type, scales_type, suffix, group_size, bits)               \
  instantiate_qmv_mma(affine_qmv_mma, type, scales_type, suffix, group_size, bits, 4, 16, 1) \
  instantiate_qmv_mma(affine_qmv_mma, type, scales_type, suffix, group_size, bits, 4, 16, 2) \
  instantiate_qmv_mma(affine_qmv_mma, type, scales_type, suffix, group_size, bits, 4, 32, 1)

///////////////////////////////////////////////////////////////////////////////
// Instantiation
///////////////////////////////////////////////////////////////////////////////

#define instantiate_quantized_all(type, scales_type, suffix, group_size, bits) \
  instantiate_qmm_all(type, scales_type, suffix, group_size, bits)             \
  instantiate_qmv_all(type, scales_type, suffix, group_size, bits)             \
  instantiate_qvm_all(type, scales_type, suffix, group_size, bits)             \
  instantiate_qvm_splitk_all(type, scales_type, suffix, group_size, bits)      \
  instantiate_simple_all(type, scales_type, suffix, group_size, bits)          \
  instantiate_grouped_qmm_all(type, scales_type, suffix, group_size, bits)     \
  instantiate_qmv_quad_all(type, scales_type, suffix, group_size, bits)        \
  instantiate_qmv_wide_all(type, scales_type, suffix, group_size, bits)

#define instantiate_quantized_all_bits(type, scales_type, suffix, group_size) \
  instantiate_quantized_all(type, scales_type, suffix, group_size, 1)         \
  instantiate_quantized_all(type, scales_type, suffix, group_size, 2)         \
  instantiate_quantized_all(type, scales_type, suffix, group_size, 3)         \
  instantiate_quantized_all(type, scales_type, suffix, group_size, 4)         \
  instantiate_quantized_all(type, scales_type, suffix, group_size, 5)         \
  instantiate_quantized_all(type, scales_type, suffix, group_size, 6)         \
  instantiate_quantized_all(type, scales_type, suffix, group_size, 8)         \
  instantiate_qmv_mma_all(type, scales_type, suffix, group_size, 4)           \
  instantiate_qmv_mma_all(type, scales_type, suffix, group_size, 8)

#define instantiate_quantized_all_groups(type, scales_type, suffix) \
  instantiate_quantized_all_bits(type, scales_type, suffix, 32)     \
  instantiate_quantized_all_bits(type, scales_type, suffix, 64)     \
  instantiate_quantized_all_bits(type, scales_type, suffix, 128)

instantiate_quantized_all_groups(float, float, "")
instantiate_quantized_all_groups(float16_t, float16_t, "")
instantiate_quantized_all_groups(bfloat16_t, bfloat16_t, "")

#define instantiate_quantization(type, group_size, bits) \
  instantiate_kernel("affine_quantize_" #type "_gs_" #group_size "_b_" #bits "_hgs_0", affine_quantize, type, group_size, bits, false) \
  instantiate_kernel("affine_dequantize_" #type "_gs_" #group_size "_b_" #bits "_hgs_0", affine_dequantize, type, group_size, bits, false)
#define instantiate_quantization_bits(type, group_size) \
  instantiate_quantization(type, group_size, 1) \
  instantiate_quantization(type, group_size, 2) \
  instantiate_quantization(type, group_size, 3) \
  instantiate_quantization(type, group_size, 4) \
  instantiate_quantization(type, group_size, 5) \
  instantiate_quantization(type, group_size, 6) \
  instantiate_quantization(type, group_size, 8)
#define instantiate_quantization_groups(type) \
  instantiate_quantization_bits(type, 32) \
  instantiate_quantization_bits(type, 64) \
  instantiate_quantization_bits(type, 128)
instantiate_quantization_groups(float)
instantiate_quantization_groups(float16_t)
instantiate_quantization_groups(bfloat16_t)
instantiate_quantized_all_groups(float, float16_t, "_scales_float16_t")
instantiate_quantized_all_groups(float, bfloat16_t, "_scales_bfloat16_t")
// clang-format on
