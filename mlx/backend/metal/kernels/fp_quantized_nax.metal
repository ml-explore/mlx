// Copyright © 2025 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/fp_quantized_nax.h"


#define instantiate_quantized_batched(mode, name, type, bm, bn, bk, wm, wn, batched) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_32_b_4_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn "_batch_" #batched, \
      fp_ ## name,  \
      type,         \
      32,           \
      4,            \
      batched)

#define instantiate_quantized_aligned(mode, name, type, bm, bn, bk, wm, wn, aligned) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_32_b_4_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn "_alN_" #aligned, \
      fp_ ## name, \
      type,        \
      32,          \
      4,           \
      aligned)

#define instantiate_quantized_aligned_batched(mode, name, type, bm, bn, bk, wm, wn, aligned, batched) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_32_b_4_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn "_alN_" #aligned "_batch_" #batched, \
      fp_ ## name,    \
      type,    \
      32,      \
      4,       \
      aligned, \
      batched)

#define instantiate_gather_qmm_rhs(func, name, type, bm, bn, bk, wm, wn, transpose) \
  instantiate_kernel( \
      #name "_" #type "_gs_32_b_4_bm_" #bm "_bn_" #bn "_bk_" #bk "_wm_" #wm "_wn_" #wn, \
      func,    \
      type,    \
      32,      \
      4,       \
      bm,      \
      bn,      \
      bk,      \
      wm,      \
      wn,      \
      transpose)


#define instantiate_quantized_all_aligned(type) \
  instantiate_quantized_aligned(mxfp4, gather_qmm_t_nax, type, 64, 64, 64, 2, 2, true)      \
  instantiate_quantized_aligned(mxfp4, gather_qmm_t_nax, type, 64, 64, 64, 2, 2, false)     \
  instantiate_quantized_aligned_batched(mxfp4, qmm_t_nax, type, 64, 64, 64, 2, 2, true, 1)  \
  instantiate_quantized_aligned_batched(mxfp4, qmm_t_nax, type, 64, 64, 64, 2, 2, true, 0)  \
  instantiate_quantized_aligned_batched(mxfp4, qmm_t_nax, type, 64, 64, 64, 2, 2, false, 1) \
  instantiate_quantized_aligned_batched(mxfp4, qmm_t_nax, type, 64, 64, 64, 2, 2, false, 0)


#define instantiate_quantized_all_rhs(type) \
  instantiate_gather_qmm_rhs(fp_gather_qmm_rhs_nax, mxfp4_gather_qmm_rhs_nax_nt, type, 64, 64, 64, 2, 2, true) \
  instantiate_gather_qmm_rhs(fp_gather_qmm_rhs_nax, mxfp4_gather_qmm_rhs_nax_nn, type, 64, 64, 64, 2, 2, false) 

#define instantiate_quantized_types(type) \
  instantiate_quantized_all_aligned(type) \
  instantiate_quantized_all_rhs(type)

instantiate_quantized_types(float)
instantiate_quantized_types(bfloat16_t)
instantiate_quantized_types(float16_t)
    // clang-format on
