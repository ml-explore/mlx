// Copyright © 2025 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/fp4_quantized.h"

#define instantiate_quantized(name, type) \
  instantiate_kernel( \
      #name "_" #type "_gs_32_b_4", \
      name, \
      type, \
      32,   \
      uint8_t)

#define instantiate_quantized_batched(name, type, batched) \
  instantiate_kernel( \
      #name "_" #type "_gs_32_b_4_batch_" #batched, \
      name,    \
      type,    \
      32,      \
      uint8_t, \
      batched)

#define instantiate_quantized_aligned(name, type, aligned) \
  instantiate_kernel( \
      #name "_" #type "_gs_32_b_4_alN_" #aligned, \
      name,    \
      type,    \
      32,      \
      uint8_t, \
      aligned)

#define instantiate_quantized_aligned_batched(name, type, aligned, batched) \
  instantiate_kernel( \
      #name "_" #type "_gs_32_b_4_alN_" #aligned "_batch_" #batched, \
      name,    \
      type,    \
      32,      \
      uint8_t, \
      aligned, \
      batched)

#define instantiate_quantized_quad(name, type, D, batched) \
  instantiate_kernel( \
      #name "_" #type "_gs_32_b_4_d_" #D "_batch_" #batched, \
      name,    \
      type,    \
      32,      \
      uint8_t, \
      D,       \
      batched)

#define instantiate_quantized_split_k(name, type, split_k) \
  instantiate_kernel( \
      #name "_" #type "_gs_32_b_4_spk_" #split_k, \
      name,    \
      type,    \
      32,      \
      uint8_t, \
      split_k)

#define instantiate_gather_qmm_rhs(func, name, type, bm, bn, bk, wm, wn, transpose) \
  instantiate_kernel( \
      #name "_" #type "_gs_32_b_4_bm_" #bm "_bn_" #bn "_bk_" #bk "_wm_" #wm "_wn_" #wn, \
      func,    \
      type,    \
      32,      \
      uint8_t, \
      bm,      \
      bn,      \
      bk,      \
      wm,      \
      wn,      \
      transpose)

#define instantiate_quantized_batched_wrap(name, type) \
  instantiate_quantized_batched(name, type, 1)         \
  instantiate_quantized_batched(name, type, 0)

#define instantiate_quantized_all_batched(type) \
  instantiate_quantized_batched_wrap(mxfp4_qmv_fast, type) \
  instantiate_quantized_batched_wrap(mxfp4_qmv, type)      \
  instantiate_quantized_batched_wrap(mxfp4_qvm, type)      \
  instantiate_quantized_batched_wrap(mxfp4_qmm_n, type)

#define instantiate_quantized_all_single(type) \
  instantiate_quantized(mxfp4_gather_qmv_fast, type) \
  instantiate_quantized(mxfp4_gather_qmv, type)      \
  instantiate_quantized(mxfp4_gather_qvm, type)      \
  instantiate_quantized(mxfp4_gather_qmm_n, type)

#define instantiate_quantized_all_aligned(type) \
  instantiate_quantized_aligned(mxfp4_gather_qmm_t, type, true)      \
  instantiate_quantized_aligned(mxfp4_gather_qmm_t, type, false)     \
  instantiate_quantized_aligned_batched(mxfp4_qmm_t, type, true, 1)  \
  instantiate_quantized_aligned_batched(mxfp4_qmm_t, type, true, 0)  \
  instantiate_quantized_aligned_batched(mxfp4_qmm_t, type, false, 1) \
  instantiate_quantized_aligned_batched(mxfp4_qmm_t, type, false, 0)

#define instantiate_quantized_all_quad(type) \
  instantiate_quantized_quad(mxfp4_qmv_quad, type, 64, 1)  \
  instantiate_quantized_quad(mxfp4_qmv_quad, type, 64, 0)  \
  instantiate_quantized_quad(mxfp4_qmv_quad, type, 128, 1) \
  instantiate_quantized_quad(mxfp4_qmv_quad, type, 128, 0)

#define instantiate_quantized_all_splitk(type) \
  instantiate_quantized_split_k(mxfp4_qvm_split_k, type, 8) \
  instantiate_quantized_split_k(mxfp4_qvm_split_k, type, 32)

#define instantiate_quantized_all_rhs(type) \
  instantiate_gather_qmm_rhs(mxfp4_gather_qmm_rhs, mxfp4_gather_qmm_rhs_nt, type, 16, 32, 32, 1, 2, true) \
  instantiate_gather_qmm_rhs(mxfp4_gather_qmm_rhs, mxfp4_gather_qmm_rhs_nn, type, 16, 32, 32, 1, 2, false)

#define instantiate_quantized_types(type) \
  instantiate_quantized_all_batched(type) \
  instantiate_quantized_all_quad(type)    \
  instantiate_quantized_all_splitk(type)  \
  instantiate_quantized_all_single(type)  \
  instantiate_quantized_all_aligned(type) \
  instantiate_quantized_all_rhs(type)

instantiate_quantized_types(float)
instantiate_quantized_types(bfloat16_t)
instantiate_quantized_types(float16_t)
    // clang-format on
