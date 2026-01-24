// Copyright © 2025 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/fp_quantized_nax.h"


#define instantiate_quantized_batched(mode, name, type, bm, bn, bk, wm, wn, batched, group_size, bits) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn "_batch_" #batched, \
      fp_ ## name,  \
      type,         \
      group_size,           \
      bits,            \
      batched)

#define instantiate_quantized_aligned(mode, name, type, bm, bn, bk, wm, wn, aligned, group_size, bits) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn "_alN_" #aligned, \
      fp_ ## name, \
      type,        \
      group_size,          \
      bits,           \
      aligned)

#define instantiate_quantized_aligned_batched(mode, name, type, bm, bn, bk, wm, wn, aligned, batched, group_size, bits) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn "_alN_" #aligned "_batch_" #batched, \
      fp_ ## name,    \
      type,    \
      group_size,      \
      bits,       \
      aligned, \
      batched)

#define instantiate_gather_qmm_rhs(func, name, type, bm, bn, bk, wm, wn, transpose, mode, group_size, bits) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_bm_" #bm "_bn_" #bn "_bk_" #bk "_wm_" #wm "_wn_" #wn, \
      func,    \
      type,    \
      group_size,      \
      bits,       \
      bm,      \
      bn,      \
      bk,      \
      wm,      \
      wn,      \
      transpose)


#define instantiate_quantized_all_aligned(type, mode, group_size, bits) \
  instantiate_quantized_aligned(mode, gather_qmm_t_nax, type, 64, 64, 64, 2, 2, true, group_size, bits)      \
  instantiate_quantized_aligned(mode, gather_qmm_t_nax, type, 64, 64, 64, 2, 2, false, group_size, bits)     \
  instantiate_quantized_aligned_batched(mode, qmm_t_nax, type, 64, 64, 64, 2, 2, true, 1, group_size, bits)  \
  instantiate_quantized_aligned_batched(mode, qmm_t_nax, type, 64, 64, 64, 2, 2, true, 0, group_size, bits)  \
  instantiate_quantized_aligned_batched(mode, qmm_t_nax, type, 64, 64, 64, 2, 2, false, 1, group_size, bits) \
  instantiate_quantized_aligned_batched(mode, qmm_t_nax, type, 64, 64, 64, 2, 2, false, 0, group_size, bits)


#define instantiate_quantized_all_rhs(type, mode, group_size, bits) \
  instantiate_gather_qmm_rhs(fp_gather_qmm_rhs_nax, gather_qmm_rhs_nax_nt, type, 64, 64, 64, 2, 2, true, mode, group_size, bits) \
  instantiate_gather_qmm_rhs(fp_gather_qmm_rhs_nax, gather_qmm_rhs_nax_nn, type, 64, 64, 64, 2, 2, false, mode, group_size, bits)

#define instantiate_quantized_modes(type, mode, group_size, bits) \
  instantiate_quantized_all_aligned(type, mode, group_size, bits) \
  instantiate_quantized_all_rhs(type, mode, group_size, bits)

#define instantiate_quantized_types(type) \
  instantiate_quantized_modes(type, nvfp4, 16, 4) \
  instantiate_quantized_modes(type, mxfp8, 32, 8) \
  instantiate_quantized_modes(type, mxfp4, 32, 4)

instantiate_quantized_types(float)
instantiate_quantized_types(bfloat16_t)
instantiate_quantized_types(float16_t)
    // clang-format on
