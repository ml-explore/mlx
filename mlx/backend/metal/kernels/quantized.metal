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

#define instantiate_simple(name, type, group_size, bits) \
  instantiate_kernel(   \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs_0", \
      name,             \
      type,             \
      group_size,       \
      bits,             \
      false)

#define instantiate_simple_all(type, group_size, bits)               \
  instantiate_simple(affine_gather_qmv, type, group_size, bits)      \
  instantiate_simple(affine_gather_qmv_fast, type, group_size, bits) \
  instantiate_simple(affine_gather_qvm, type, group_size, bits)      \
  instantiate_simple(affine_quantize, type, group_size, bits)        \
  instantiate_simple(affine_dequantize, type, group_size, bits)

///////////////////////////////////////////////////////////////////////////////
// affine_qmm_t, affine_qmm_t_splitk, affine_gather_qmm_t, affine_qmm_n,
// affine_gather_qmm_n
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qmm(name, type, group_size, bits, aligned_n, batched) \
  instantiate_kernel( \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs_0_alN_" #aligned_n "_batch_" #batched "_bm_32_bn_32", \
      name,       \
      type,       \
      group_size, \
      bits,       \
      false,      \
      aligned_n,  \
      batched,    \
      32,         \
      32,         \
      32)

#define instantiate_qmm_aligned(name, type, group_size, bits) \
  instantiate_qmm(name, type, group_size, bits, 0, 0)         \
  instantiate_qmm(name, type, group_size, bits, 1, 0)

#define instantiate_qmm_batched(name, type, group_size, bits) \
  instantiate_qmm(name, type, group_size, bits, 0, 0)         \
  instantiate_qmm(name, type, group_size, bits, 0, 1)

#define instantiate_qmm_aligned_batched(name, type, group_size, bits) \
  instantiate_qmm(name, type, group_size, bits, 0, 0)                 \
  instantiate_qmm(name, type, group_size, bits, 0, 1)                 \
  instantiate_qmm(name, type, group_size, bits, 1, 0)                 \
  instantiate_qmm(name, type, group_size, bits, 1, 1)

#define instantiate_qmm_all(type, group_size, bits) \
  instantiate_qmm_aligned_batched(affine_qmm_t, type, group_size, bits) \
  instantiate_qmm_aligned(affine_qmm_t_splitk, type, group_size, bits)  \
  instantiate_qmm_aligned(affine_gather_qmm_t, type, group_size, bits)  \
  instantiate_qmm_batched(affine_qmm_n, type, group_size, bits)         \
  instantiate_qmm(affine_gather_qmm_n, type, group_size, bits, 0, 0)

///////////////////////////////////////////////////////////////////////////////
// affine_qmv, affine_qmv_fast
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qmv(name, type, group_size, bits, batched, partial_rows) \
  instantiate_kernel( \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs_0_batch_" #batched "_pr_" #partial_rows "_r_4", \
      name,           \
      type,           \
      group_size,     \
      bits,           \
      false,          \
      batched,        \
      partial_rows,   \
      4)

#define instantiate_qmv_batched(name, type, group_size, bits, partial_rows) \
  instantiate_qmv(name, type, group_size, bits, 0, partial_rows)            \
  instantiate_qmv(name, type, group_size, bits, 1, partial_rows)

#define instantiate_qmv_all(type, group_size, bits)                   \
  instantiate_qmv_batched(affine_qmv, type, group_size, bits, 0)      \
  instantiate_qmv_batched(affine_qmv_fast, type, group_size, bits, 0) \
  instantiate_qmv_batched(affine_qmv_fast, type, group_size, bits, 1)

///////////////////////////////////////////////////////////////////////////////
// affine_qvm
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qvm(name, type, group_size, bits, batched) \
  instantiate_kernel( \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs_0_batch_" #batched, \
      name,           \
      type,           \
      group_size,     \
      bits,           \
      false,          \
      batched)

#define instantiate_qvm_all(type, group_size, bits)       \
  instantiate_qvm(affine_qvm, type, group_size, bits, 0)  \
  instantiate_qvm(affine_qvm, type, group_size, bits, 1)

///////////////////////////////////////////////////////////////////////////////
// affine_qvm_splitk
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qvm_splitk(name, type, group_size, bits, split_k) \
  instantiate_kernel( \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_spk_" #split_k, \
      name,           \
      type,           \
      group_size,     \
      bits,           \
      split_k)

#define instantiate_qvm_splitk_all(type, group_size, bits)              \
  instantiate_qvm_splitk(affine_qvm_splitk, type, group_size, bits, 8)  \
  instantiate_qvm_splitk(affine_qvm_splitk, type, group_size, bits, 32)

///////////////////////////////////////////////////////////////////////////////
// affine_gather_qmm_rhs
///////////////////////////////////////////////////////////////////////////////

#define instantiate_grouped_qmm(name, type, group_size, bits, transpose) \
  instantiate_kernel( \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs_0_t_" #transpose, \
      name,           \
      type,           \
      group_size,     \
      bits,           \
      false,          \
      transpose)

#define instantiate_grouped_qmm_all(type, group_size, bits)                 \
  instantiate_grouped_qmm(affine_gather_qmm_rhs, type, group_size, bits, 0) \
  instantiate_grouped_qmm(affine_gather_qmm_rhs, type, group_size, bits, 1)

///////////////////////////////////////////////////////////////////////////////
// affine_qmv_quad
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qmv_quad(name, type, group_size, bits, d, batched) \
  instantiate_kernel( \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_d_" #d "_batch_" #batched, \
      name,           \
      type,           \
      group_size,     \
      bits,           \
      d,              \
      batched)

#define instantiate_qmv_quad_batched(name, type, group_size, bits, d) \
  instantiate_qmv_quad(name, type, group_size, bits, d, 0)            \
  instantiate_qmv_quad(name, type, group_size, bits, d, 1)

#define instantiate_qmv_quad_all(type, group_size, bits)                    \
  instantiate_qmv_quad_batched(affine_qmv_quad, type, group_size, bits, 64) \
  instantiate_qmv_quad_batched(affine_qmv_quad, type, group_size, bits, 128)

///////////////////////////////////////////////////////////////////////////////
// affine_qmv_wide
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qmv_wide(name, type, group_size, bits, vecs_per_tg, k_lanes, batched) \
  instantiate_kernel( \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_nv_" #vecs_per_tg "_kl_" #k_lanes "_batch_" #batched, \
      name,           \
      type,           \
      group_size,     \
      bits,           \
      vecs_per_tg,    \
      k_lanes,        \
      batched)

#define instantiate_qmv_wide_batched(name, type, group_size, bits, vecs_per_tg, k_lanes) \
  instantiate_qmv_wide(name, type, group_size, bits, vecs_per_tg, k_lanes, 0)            \
  instantiate_qmv_wide(name, type, group_size, bits, vecs_per_tg, k_lanes, 1)

#define instantiate_qmv_wide_all(type, group_size, bits)                      \
  instantiate_qmv_wide_batched(affine_qmv_wide, type, group_size, bits, 2, 8) \
  instantiate_qmv_wide_batched(affine_qmv_wide, type, group_size, bits, 3, 8) \
  instantiate_qmv_wide_batched(affine_qmv_wide, type, group_size, bits, 4, 8) \
  instantiate_qmv_wide_batched(affine_qmv_wide, type, group_size, bits, 5, 8) \
  instantiate_qmv_wide_batched(affine_qmv_wide, type, group_size, bits, 6, 8) \
  instantiate_qmv_wide_batched(affine_qmv_wide, type, group_size, bits, 7, 8) \
  instantiate_qmv_wide_batched(affine_qmv_wide, type, group_size, bits, 8, 8)

///////////////////////////////////////////////////////////////////////////////
// Instantiation
///////////////////////////////////////////////////////////////////////////////

#define instantiate_quantized_all(type, group_size, bits) \
  instantiate_qmm_all(type, group_size, bits)             \
  instantiate_qmv_all(type, group_size, bits)             \
  instantiate_qvm_all(type, group_size, bits)             \
  instantiate_qvm_splitk_all(type, group_size, bits)      \
  instantiate_simple_all(type, group_size, bits)          \
  instantiate_grouped_qmm_all(type, group_size, bits)     \
  instantiate_qmv_quad_all(type, group_size, bits)        \
  instantiate_qmv_wide_all(type, group_size, bits)

#define instantiate_quantized_all_bits(type, group_size) \
  instantiate_quantized_all(type, group_size, 1)         \
  instantiate_quantized_all(type, group_size, 2)         \
  instantiate_quantized_all(type, group_size, 3)         \
  instantiate_quantized_all(type, group_size, 4)         \
  instantiate_quantized_all(type, group_size, 5)         \
  instantiate_quantized_all(type, group_size, 6)         \
  instantiate_quantized_all(type, group_size, 8)

#define instantiate_quantized_all_groups(type) \
  instantiate_quantized_all_bits(type, 32)     \
  instantiate_quantized_all_bits(type, 64)     \
  instantiate_quantized_all_bits(type, 128)

instantiate_quantized_all_groups(float)
instantiate_quantized_all_groups(float16_t)
instantiate_quantized_all_groups(bfloat16_t)
// clang-format on
