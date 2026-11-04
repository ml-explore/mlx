// Copyright © 2025-2026 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/fp_quantized_nax.h"

///////////////////////////////////////////////////////////////////////////////
// fp_qmm_t_nax, fp_gather_qmm_t_nax, fp_qmm_t_nax_splitk
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qmm(name, type, mode, group_size, bits, has_global_scale, aligned_n, batch, bm, bn, wm, wn) \
  instantiate_kernel(   \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs_" #has_global_scale "_alN_" #aligned_n "_batch_" #batch "_bm_" #bm "_bn_" #bn "_wm_" #wm "_wn_" #wn, \
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
      wm,               \
      wn)

#define instantiate_qmm_wm2_wn2(name, type, mode, group_size, bits, has_global_scale, aligned_n, batch, bm, bn) \
  instantiate_qmm(name, type, mode, group_size, bits, has_global_scale, aligned_n, batch, bm, bn, 2, 2)

#define instantiate_qmm_aligned(name, type, mode, group_size, bits, has_global_scale, bm, bn) \
  instantiate_qmm_wm2_wn2(name, type, mode, group_size, bits, has_global_scale, 0, 0, bm, bn) \
  instantiate_qmm_wm2_wn2(name, type, mode, group_size, bits, has_global_scale, 1, 0, bm, bn)

#define instantiate_qmm_aligned_batched(name, type, mode, group_size, bits, has_global_scale, bm, bn) \
  instantiate_qmm_wm2_wn2(name, type, mode, group_size, bits, has_global_scale, 0, 0, bm, bn)         \
  instantiate_qmm_wm2_wn2(name, type, mode, group_size, bits, has_global_scale, 0, 1, bm, bn)         \
  instantiate_qmm_wm2_wn2(name, type, mode, group_size, bits, has_global_scale, 1, 0, bm, bn)         \
  instantiate_qmm_wm2_wn2(name, type, mode, group_size, bits, has_global_scale, 1, 1, bm, bn)

#define instantiate_qmm_all(type, mode, group_size, bits, has_global_scale)                          \
  instantiate_qmm_aligned_batched(qmm_t_nax, type, mode, group_size, bits, has_global_scale, 64, 64) \
  instantiate_qmm_aligned_batched(qmm_t_nax, type, mode, group_size, bits, has_global_scale, 32, 64) \
  instantiate_qmm_aligned(gather_qmm_t_nax, type, mode, group_size, bits, has_global_scale, 64, 64)

#define instantiate_qmm_all_missing_global_scale(type, mode, group_size, bits)              \
  instantiate_qmm_wm2_wn2(qmm_t_nax_splitk, type, mode, group_size, bits, 0, 1, 0, 32, 64)  \
  instantiate_qmm_wm2_wn2(qmm_t_nax_splitk, type, mode, group_size, bits, 0, 1, 0, 64, 64)  \

// Specializations for staged qmm_t_nax and qmm_t_nax_splitk.
#define instantiate_qmm_staged_all_missing_global_scale(type, mode, group_size, bits) \
  instantiate_qmm(qmm_t_nax, type, mode, group_size, bits, 0, 1, 0, 64, 64, 4, 1)     \
  instantiate_qmm(qmm_t_nax, type, mode, group_size, bits, 0, 1, 1, 64, 64, 4, 1)     \
  instantiate_qmm(qmm_t_nax_splitk, type, mode, group_size, bits, 0, 1, 0, 64, 64, 4, 1)

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
  instantiate_grouped_qmm_all(type, mode, group_size, bits, has_global_scale)

#define instantiate_quantized_all_missing_global_scale(type, mode, group_size, bits) \
  instantiate_qmm_all_missing_global_scale(type, mode, group_size, bits)             \
  instantiate_qmv_nax_all_missing_global_scale(type, mode, group_size, bits)

#define instantiate_quantized_all_types(mode, group_size, bits, has_global_scale) \
  instantiate_quantized_all(float, mode, group_size, bits, has_global_scale)      \
  instantiate_quantized_all(float16_t, mode, group_size, bits, has_global_scale)  \
  instantiate_quantized_all(bfloat16_t, mode, group_size, bits, has_global_scale) \

#define instantiate_quantized_all_types_missing_global_scale(mode, group_size, bits) \
  instantiate_quantized_all_missing_global_scale(float, mode, group_size, bits)      \
  instantiate_quantized_all_missing_global_scale(float16_t, mode, group_size, bits)  \
  instantiate_quantized_all_missing_global_scale(bfloat16_t, mode, group_size, bits) \
  instantiate_qmm_staged_all_missing_global_scale(float16_t, mode, group_size, bits) \
  instantiate_qmm_staged_all_missing_global_scale(bfloat16_t, mode, group_size, bits)

instantiate_quantized_all_types(mxfp4, 32, 4, 0)
instantiate_quantized_all_types(mxfp8, 32, 8, 0)
instantiate_quantized_all_types(nvfp4, 16, 4, 0)
instantiate_quantized_all_types(nvfp4, 16, 4, 1)
instantiate_quantized_all_types_missing_global_scale(mxfp4, 32, 4)
instantiate_quantized_all_types_missing_global_scale(mxfp8, 32, 8)
instantiate_quantized_all_types_missing_global_scale(nvfp4, 16, 4)
// clang-format on
