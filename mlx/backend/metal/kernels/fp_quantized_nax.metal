// Copyright © 2025-2026 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/fp_quantized_nax.h"

///////////////////////////////////////////////////////////////////////////////
// fp_qmm_t_nax, fp_gather_qmm_t_nax
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qmm(name, type, mode, group_size, bits, has_global_scale, aligned_n, batch, bm, bn) \
  instantiate_kernel(   \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs_" #has_global_scale "_alN_" #aligned_n "_batch_" #batch "_bm_" #bm "_bn_" #bn "_wm_2_wn_2", \
      fp_ ##name,       \
      type,             \
      group_size,       \
      bits,             \
      has_global_scale, \
      aligned_n,        \
      batch,            \
      bm,               \
      64,               \
      bn,               \
      2,                \
      2)

#define instantiate_qmm_aligned(name, type, mode, group_size, bits, has_global_scale, bm, bn) \
  instantiate_qmm(name, type, mode, group_size, bits, has_global_scale, 0, 0, bm, bn)         \
  instantiate_qmm(name, type, mode, group_size, bits, has_global_scale, 1, 0, bm, bn)

#define instantiate_qmm_aligned_batched(name, type, mode, group_size, bits, has_global_scale, bm, bn) \
  instantiate_qmm(name, type, mode, group_size, bits, has_global_scale, 0, 0, bm, bn)                 \
  instantiate_qmm(name, type, mode, group_size, bits, has_global_scale, 0, 1, bm, bn)                 \
  instantiate_qmm(name, type, mode, group_size, bits, has_global_scale, 1, 0, bm, bn)                 \
  instantiate_qmm(name, type, mode, group_size, bits, has_global_scale, 1, 1, bm, bn)

#define instantiate_qmm_all(type, mode, group_size, bits, has_global_scale)                          \
  instantiate_qmm_aligned_batched(qmm_t_nax, type, mode, group_size, bits, has_global_scale, 64, 64) \
  instantiate_qmm_aligned_batched(qmm_t_nax, type, mode, group_size, bits, has_global_scale, 32, 64) \
  instantiate_qmm_aligned(gather_qmm_t_nax, type, mode, group_size, bits, has_global_scale, 64, 64)

///////////////////////////////////////////////////////////////////////////////
// fp_gather_qmm_rhs_nax
///////////////////////////////////////////////////////////////////////////////

#define instantiate_grouped_qmm(name, type, mode, group_size, bits, has_global_scale, transpose, bm, bn) \
  instantiate_kernel(   \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs_" #has_global_scale "_t_" #transpose "_bm_" #bm "_bn_" #bn, \
      fp_ ## name,      \
      type,             \
      group_size,       \
      bits,             \
      has_global_scale, \
      transpose,        \
      bm,               \
      bn,               \
      64,               \
      2,                \
      2)

#define instantiate_grouped_qmm_all(type, mode, group_size, bits, has_global_scale)                      \
  instantiate_grouped_qmm(gather_qmm_rhs_nax, type, mode, group_size, bits, has_global_scale, 0, 32, 64) \
  instantiate_grouped_qmm(gather_qmm_rhs_nax, type, mode, group_size, bits, has_global_scale, 0, 64, 64) \
  instantiate_grouped_qmm(gather_qmm_rhs_nax, type, mode, group_size, bits, has_global_scale, 1, 32, 64) \
  instantiate_grouped_qmm(gather_qmm_rhs_nax, type, mode, group_size, bits, has_global_scale, 1, 64, 64)

///////////////////////////////////////////////////////////////////////////////
// fp_qmv_nax
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qmv_nax(name, type, mode, group_size, bits, has_global_scale, wk, bn) \
  instantiate_kernel(   \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_wk_" #wk "_bn_" #bn, \
      fp_ ## name,      \
      type,             \
      group_size,       \
      bits,             \
      wk,               \
      bn)

#define instantiate_qmv_nax_all_missing_global_scale(type, mode, group_size, bits) \
  instantiate_qmv_nax(qmv_nax, type, mode, group_size, bits, 0, 8, 32)

///////////////////////////////////////////////////////////////////////////////
// Instantiation
///////////////////////////////////////////////////////////////////////////////

#define instantiate_quantized_all(type, mode, group_size, bits, has_global_scale) \
  instantiate_qmm_all(type, mode, group_size, bits, has_global_scale)             \
  instantiate_grouped_qmm_all(type, mode, group_size, bits, has_global_scale)     \

#define instantiate_quantized_all_missing_global_scale(type, mode, group_size, bits) \
  instantiate_qmv_nax_all_missing_global_scale(type, mode, group_size, bits)

#define instantiate_quantized_all_quants(type)            \
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
