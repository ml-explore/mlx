// Copyright © 2025-2026 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/fp_quantized.h"

///////////////////////////////////////////////////////////////////////////////
// fp_gather_qmv, fp_gather_qmv_fast, fp_gather_qvm, fp_quantize, fp_dequantize,
// fp_quantize_dequantize
///////////////////////////////////////////////////////////////////////////////

#define instantiate_simple(name, type, mode, group_size, bits, has_global_scale) \
  instantiate_kernel(   \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs_" #has_global_scale, \
      fp_ ## name,      \
      type,             \
      group_size,       \
      bits,             \
      has_global_scale)

#define instantiate_simple_all(type, mode, group_size, bits, has_global_scale)        \
  instantiate_simple(gather_qmv, type, mode, group_size, bits, has_global_scale)      \
  instantiate_simple(gather_qmv_fast, type, mode, group_size, bits, has_global_scale) \
  instantiate_simple(gather_qvm, type, mode, group_size, bits, has_global_scale)      \
  instantiate_simple(quantize, type, mode, group_size, bits, has_global_scale)        \
  instantiate_simple(dequantize, type, mode, group_size, bits, has_global_scale)      \
  instantiate_simple(quantize_dequantize, type, mode, group_size, bits, has_global_scale)

///////////////////////////////////////////////////////////////////////////////
// fp_qmm_t, fp_qmm_t_splitk, fp_gather_qmm_t, fp_qmm_n, fp_gather_qmm_n
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qmm(name, type, mode, group_size, bits, has_global_scale, aligned_n, batched, bm, bn) \
  instantiate_kernel(   \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs_" #has_global_scale "_alN_" #aligned_n "_batch_" #batched "_bm_" #bm "_bn_" #bn, \
      fp_ ##name,       \
      type,             \
      group_size,       \
      bits,             \
      has_global_scale, \
      aligned_n,        \
      batched,          \
      bm,               \
      32,               \
      bn)

#define instantiate_qmm_aligned(name, type, mode, group_size, bits, has_global_scale, bm, bn) \
  instantiate_qmm(name, type, mode, group_size, bits, has_global_scale, 0, 0, bm, bn)         \
  instantiate_qmm(name, type, mode, group_size, bits, has_global_scale, 1, 0, bm, bn)

#define instantiate_qmm_batched(name, type, mode, group_size, bits, has_global_scale, bm, bn) \
  instantiate_qmm(name, type, mode, group_size, bits, has_global_scale, 0, 0, bm, bn)         \
  instantiate_qmm(name, type, mode, group_size, bits, has_global_scale, 0, 1, bm, bn)

#define instantiate_qmm_aligned_batched(name, type, mode, group_size, bits, has_global_scale, bm, bn) \
  instantiate_qmm(name, type, mode, group_size, bits, has_global_scale, 0, 0, bm, bn)                 \
  instantiate_qmm(name, type, mode, group_size, bits, has_global_scale, 0, 1, bm, bn)                 \
  instantiate_qmm(name, type, mode, group_size, bits, has_global_scale, 1, 0, bm, bn)                 \
  instantiate_qmm(name, type, mode, group_size, bits, has_global_scale, 1, 1, bm, bn)

#define instantiate_qmm_all(type, mode, group_size, bits, has_global_scale)                      \
  instantiate_qmm_aligned_batched(qmm_t, type, mode, group_size, bits, has_global_scale, 32, 32) \
  instantiate_qmm_aligned(gather_qmm_t, type, mode, group_size, bits, has_global_scale, 32, 32)  \
  instantiate_qmm(gather_qmm_n, type, mode, group_size, bits, has_global_scale, 0, 0, 32, 32)

#define instantiate_qmm_all_missing_global_scale(type, mode, group_size, bits)   \
  instantiate_qmm_aligned(qmm_t_splitk, type, mode, group_size, bits, 0, 32, 32) \
  instantiate_qmm_batched(qmm_n, type, mode, group_size, bits, 0, 32, 32)

// Specializations for Fp4HalfLoader with 64x64 tiles.
instantiate_qmm_aligned_batched(qmm_t, float16_t, mxfp4, 32, 4, 0, 64, 64)
instantiate_qmm_aligned_batched(qmm_t, float16_t, nvfp4, 16, 4, 0, 64, 64)
instantiate_qmm_aligned_batched(qmm_t, bfloat16_t, nvfp4, 16, 4, 0, 64, 64)

///////////////////////////////////////////////////////////////////////////////
// fp_qmv, fp_qmv_fast
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qmv(name, type, mode, group_size, bits, has_global_scale, batched, partial_rows, results) \
  instantiate_kernel(   \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs_" #has_global_scale "_batch_" #batched "_pr_" #partial_rows "_r_" #results, \
      fp_ ##name,       \
      type,             \
      group_size,       \
      bits,             \
      has_global_scale, \
      batched,          \
      partial_rows,     \
      results)

#define instantiate_qmv_batched(name, type, mode, group_size, bits, has_global_scale, partial_rows, results) \
  instantiate_qmv(name, type, mode, group_size, bits, has_global_scale, 0, partial_rows, results)            \
  instantiate_qmv(name, type, mode, group_size, bits, has_global_scale, 1, partial_rows, results)

#define instantiate_qmv_all(type, mode, group_size, bits, has_global_scale)               \
  instantiate_qmv_batched(qmv, type, mode, group_size, bits, has_global_scale, 0, 4)      \
  instantiate_qmv_batched(qmv_fast, type, mode, group_size, bits, has_global_scale, 0, 4) \
  instantiate_qmv_batched(qmv_fast, type, mode, group_size, bits, has_global_scale, 1, 4)

// Specializations for narrow qmv_fast with results_per_simdgroup=2.
#define instantiate_qmv_fast_narrow(type)                        \
  instantiate_qmv_batched(qmv_fast, type, nvfp4, 16, 4, 0, 0, 2) \
  instantiate_qmv_batched(qmv_fast, type, nvfp4, 16, 4, 0, 1, 2) \
  instantiate_qmv_batched(qmv_fast, type, nvfp4, 16, 4, 1, 0, 2) \
  instantiate_qmv_batched(qmv_fast, type, nvfp4, 16, 4, 1, 1, 2)

instantiate_qmv_fast_narrow(float)
instantiate_qmv_fast_narrow(float16_t)
instantiate_qmv_fast_narrow(bfloat16_t)

///////////////////////////////////////////////////////////////////////////////
// fp_qvm
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qvm(name, type, mode, group_size, bits, has_global_scale, batched) \
  instantiate_kernel(   \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs_" #has_global_scale "_batch_" #batched, \
      fp_ ## name,      \
      type,             \
      group_size,       \
      bits,             \
      has_global_scale, \
      batched)

#define instantiate_qvm_all(type, mode, group_size, bits, has_global_scale) \
  instantiate_qvm(qvm, type, mode, group_size, bits, has_global_scale, 0)        \
  instantiate_qvm(qvm, type, mode, group_size, bits, has_global_scale, 1)

///////////////////////////////////////////////////////////////////////////////
// fp_qvm_splitk
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qvm_splitk(name, type, mode, group_size, bits, has_global_scale, split_k) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_spk_" #split_k, \
      fp_ ## name,    \
      type,           \
      group_size,     \
      bits,           \
      split_k)

#define instantiate_qvm_splitk_all_missing_global_scale(type, mode, group_size, bits) \
  instantiate_qvm_splitk(qvm_splitk, type, mode, group_size, bits, 0, 8)              \
  instantiate_qvm_splitk(qvm_splitk, type, mode, group_size, bits, 0, 32)

///////////////////////////////////////////////////////////////////////////////
// fp_gather_qmm_rhs
///////////////////////////////////////////////////////////////////////////////

#define instantiate_grouped_qmm(name, type, mode, group_size, bits, has_global_scale, transpose) \
  instantiate_kernel(   \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs_" #has_global_scale "_t_" #transpose, \
      fp_ ## name,      \
      type,             \
      group_size,       \
      bits,             \
      has_global_scale, \
      transpose)

#define instantiate_grouped_qmm_all(type, mode, group_size, bits, has_global_scale)          \
  instantiate_grouped_qmm(gather_qmm_rhs, type, mode, group_size, bits, has_global_scale, 0) \
  instantiate_grouped_qmm(gather_qmm_rhs, type, mode, group_size, bits, has_global_scale, 1)

///////////////////////////////////////////////////////////////////////////////
// fp_qmv_quad
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qmv_quad(name, type, mode, group_size, bits, has_global_scale, d, batched) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_d_" #d "_batch_" #batched, \
      fp_ ## name,    \
      type,           \
      group_size,     \
      bits,           \
      d,              \
      batched)

#define instantiate_qmv_quad_batched(name, type, mode, group_size, bits, has_global_scale, d) \
  instantiate_qmv_quad(name, type, mode, group_size, bits, has_global_scale, d, 0)            \
  instantiate_qmv_quad(name, type, mode, group_size, bits, has_global_scale, d, 1)

#define instantiate_qmv_quad_all_missing_global_scale(type, mode, group_size, bits) \
  instantiate_qmv_quad_batched(qmv_quad, type, mode, group_size, bits, 0, 64)       \
  instantiate_qmv_quad_batched(qmv_quad, type, mode, group_size, bits, 0, 128)

///////////////////////////////////////////////////////////////////////////////
// fp_qmv_wide
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qmv_wide(name, type, mode, group_size, bits, has_global_scale, vecs_per_tg, k_lanes, batched) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_nv_" #vecs_per_tg "_kl_" #k_lanes "_batch_" #batched, \
      fp_ ## name,    \
      type,           \
      group_size,     \
      bits,           \
      vecs_per_tg,    \
      k_lanes,        \
      batched)

#define instantiate_qmv_wide_batched(name, type, mode, group_size, bits, has_global_scale, vecs_per_tg, k_lanes) \
  instantiate_qmv_wide(name, type, mode, group_size, bits, has_global_scale, vecs_per_tg, k_lanes, 0)            \
  instantiate_qmv_wide(name, type, mode, group_size, bits, has_global_scale, vecs_per_tg, k_lanes, 1)

#define instantiate_qmv_wide_all_missing_global_scale(type, mode, group_size, bits) \
  instantiate_qmv_wide_batched(qmv_wide, type, mode, group_size, bits, 0, 2, 16)    \
  instantiate_qmv_wide_batched(qmv_wide, type, mode, group_size, bits, 0, 3, 16)    \
  instantiate_qmv_wide_batched(qmv_wide, type, mode, group_size, bits, 0, 4, 16)    \
  instantiate_qmv_wide_batched(qmv_wide, type, mode, group_size, bits, 0, 5, 16)

///////////////////////////////////////////////////////////////////////////////
// fp_qmv_mma
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qmv_mma(name, type, mode, group_size, bits, has_global_scale, wk, bn, tm) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_wk_" #wk "_bn_" #bn "_tm_" #tm, \
      fp_ ## name,    \
      type,           \
      group_size,     \
      bits,           \
      wk,             \
      bn,             \
      tm)

#define instantiate_qmv_mma_all_missing_global_scale(type, mode, group_size, bits) \
  instantiate_qmv_mma(qmv_mma, type, mode, group_size, bits, 0, 4, 16, 1)          \
  instantiate_qmv_mma(qmv_mma, type, mode, group_size, bits, 0, 4, 16, 2)          \
  instantiate_qmv_mma(qmv_mma, type, mode, group_size, bits, 0, 8, 16, 1)          \
  instantiate_qmv_mma(qmv_mma, type, mode, group_size, bits, 0, 8, 16, 2)

///////////////////////////////////////////////////////////////////////////////
// Instantiation
///////////////////////////////////////////////////////////////////////////////

#define instantiate_quantized_all(type, mode, group_size, bits, has_global_scale) \
  instantiate_simple_all(type, mode, group_size, bits, has_global_scale)          \
  instantiate_qmm_all(type, mode, group_size, bits, has_global_scale)             \
  instantiate_qmv_all(type, mode, group_size, bits, has_global_scale)             \
  instantiate_qvm_all(type, mode, group_size, bits, has_global_scale)             \
  instantiate_grouped_qmm_all(type, mode, group_size, bits, has_global_scale)

#define instantiate_quantized_all_missing_global_scale(type, mode, group_size, bits) \
  instantiate_qmm_all_missing_global_scale(type, mode, group_size, bits)             \
  instantiate_qvm_splitk_all_missing_global_scale(type, mode, group_size, bits)      \
  instantiate_qmv_quad_all_missing_global_scale(type, mode, group_size, bits)        \
  instantiate_qmv_wide_all_missing_global_scale(type, mode, group_size, bits)        \
  instantiate_qmv_mma_all_missing_global_scale(type, mode, group_size, bits)

#define instantiate_quantized_all_quants(type)     \
  instantiate_quantized_all(type, mxfp4, 32, 4, 0) \
  instantiate_quantized_all(type, mxfp8, 32, 8, 0) \
  instantiate_quantized_all(type, nvfp4, 16, 4, 0) \
  instantiate_quantized_all(type, nvfp4, 16, 4, 1)

#define instantiate_quantized_all_quants_no_global_scale(type)       \
  instantiate_quantized_all_missing_global_scale(type, mxfp4, 32, 4) \
  instantiate_quantized_all_missing_global_scale(type, mxfp8, 32, 8) \
  instantiate_quantized_all_missing_global_scale(type, nvfp4, 16, 4)

instantiate_quantized_all_quants(float)
instantiate_quantized_all_quants(float16_t)
instantiate_quantized_all_quants(bfloat16_t)
instantiate_quantized_all_quants_no_global_scale(float)
instantiate_quantized_all_quants_no_global_scale(float16_t)
instantiate_quantized_all_quants_no_global_scale(bfloat16_t)
// clang-format on
