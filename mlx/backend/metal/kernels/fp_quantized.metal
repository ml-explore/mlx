// Copyright © 2025 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/fp_quantized.h"

#define instantiate_quantized(mode, name, type) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_32_b_4", \
      fp_ ## name, \
      type, \
      32,   \
      4)

#define instantiate_quantized_batched(mode, name, type, batched) \
  instantiate_kernel( \
      #mode  "_" #name "_" #type "_gs_32_b_4_batch_" #batched, \
      fp_ ## name,    \
      type,    \
      32,      \
      4,       \
      batched)

#define instantiate_quantized_aligned(mode, name, type, aligned) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_32_b_4_alN_" #aligned, \
      fp_ ## name,    \
      type,    \
      32,      \
      4,       \
      aligned)

#define instantiate_quantized_aligned_batched(mode, name, type, aligned, batched) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_32_b_4_alN_" #aligned "_batch_" #batched, \
      fp_ ## name,    \
      type,    \
      32,      \
      4,       \
      aligned, \
      batched)

#define instantiate_quantized_quad(mode, name, type, D, batched) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_32_b_4_d_" #D "_batch_" #batched, \
      fp_ ## name,    \
      type,    \
      32,      \
      4,       \
      D,       \
      batched)

#define instantiate_quantized_split_k(mode, name, type, split_k) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_32_b_4_spk_" #split_k, \
      fp_ ## name,    \
      type,    \
      32,      \
      4,       \
      split_k)

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

#define instantiate_quantized_batched_wrap(mode, name, type) \
  instantiate_quantized_batched(mode, name, type, 1)         \
  instantiate_quantized_batched(mode, name, type, 0)

#define instantiate_quantized_all_batched(type) \
  instantiate_quantized_batched_wrap(mxfp4, qmv_fast, type) \
  instantiate_quantized_batched_wrap(mxfp4, qmv, type)      \
  instantiate_quantized_batched_wrap(mxfp4, qvm, type)      \
  instantiate_quantized_batched_wrap(mxfp4, qmm_n, type)

#define instantiate_quantized_all_single(type) \
  instantiate_quantized(mxfp4, gather_qmv_fast, type) \
  instantiate_quantized(mxfp4, gather_qmv, type)      \
  instantiate_quantized(mxfp4, gather_qvm, type)      \
  instantiate_quantized(mxfp4, gather_qmm_n, type)

#define instantiate_quantized_all_aligned(type) \
  instantiate_quantized_aligned(mxfp4, gather_qmm_t, type, true)      \
  instantiate_quantized_aligned(mxfp4, gather_qmm_t, type, false)     \
  instantiate_quantized_aligned_batched(mxfp4, qmm_t, type, true, 1)  \
  instantiate_quantized_aligned_batched(mxfp4, qmm_t, type, true, 0)  \
  instantiate_quantized_aligned_batched(mxfp4, qmm_t, type, false, 1) \
  instantiate_quantized_aligned_batched(mxfp4, qmm_t, type, false, 0)

#define instantiate_quantized_all_quad(type) \
  instantiate_quantized_quad(mxfp4, qmv_quad, type, 64, 1)  \
  instantiate_quantized_quad(mxfp4, qmv_quad, type, 64, 0)  \
  instantiate_quantized_quad(mxfp4, qmv_quad, type, 128, 1) \
  instantiate_quantized_quad(mxfp4, qmv_quad, type, 128, 0)

#define instantiate_quantized_all_splitk(type) \
  instantiate_quantized_split_k(mxfp4, qvm_split_k, type, 8) \
  instantiate_quantized_split_k(mxfp4, qvm_split_k, type, 32)

#define instantiate_quantized_all_rhs(type) \
  instantiate_gather_qmm_rhs(fp_gather_qmm_rhs, mxfp4_gather_qmm_rhs_nt, type, 16, 32, 32, 1, 2, true) \
  instantiate_gather_qmm_rhs(fp_gather_qmm_rhs, mxfp4_gather_qmm_rhs_nn, type, 16, 32, 32, 1, 2, false)

#define instantiate_quantize_dequantize(type, mode, group_size, bits) \
  instantiate_kernel( \
    #mode "_quantize_" #type "_gs_" #group_size "_b_" #bits, \
    fp_quantize, \
    type, \
    group_size,  \
    bits) \
  instantiate_kernel( \
    #mode "_dequantize_" #type "_gs_" #group_size "_b_" #bits, \
    fp_dequantize, \
    type, \
    group_size,  \
    bits)

#define instantiate_quantize_dequantize_modes(type) \
  instantiate_quantize_dequantize(type, mxfp4, 32, 4) \
  instantiate_quantize_dequantize(type, nvfp4, 16, 4) \
  instantiate_quantize_dequantize(type, mxfp8, 32, 8)

#define instantiate_quantized_types(type) \
  instantiate_quantized_all_batched(type) \
  instantiate_quantized_all_quad(type)    \
  instantiate_quantized_all_splitk(type)  \
  instantiate_quantized_all_single(type)  \
  instantiate_quantized_all_aligned(type) \
  instantiate_quantized_all_rhs(type)     \
  instantiate_quantize_dequantize_modes(type)

instantiate_quantized_types(float)
instantiate_quantized_types(bfloat16_t)
instantiate_quantized_types(float16_t)
    // clang-format on
