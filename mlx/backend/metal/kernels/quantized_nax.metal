// Copyright © 2023-2024 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"
#include "mlx/backend/metal/kernels/quantized_nax.h"

///////////////////////////////////////////////////////////////////////////////
// affine_qmm_t_nax, affine_qmm_n_nax, affine_gather_qmm_t_nax,
// affine_qmm_t_nax_splitk
///////////////////////////////////////////////////////////////////////////////

#define instantiate_qmm(name, type, group_size, bits, aligned_n, batch, bm, bn, wm, wn) \
  instantiate_kernel( \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs_0_alN_" #aligned_n "_batch_" #batch "_bm_" #bm "_bn_" #bn "_wm_" #wm "_wn_" #wn, \
      name,       \
      type,       \
      group_size, \
      bits,       \
      false,      \
      aligned_n,  \
      batch,      \
      bm,         \
      64,         \
      bn,         \
      wm,         \
      wn)

#define instantiate_qmm_aligned(name, type, group_size, bits, bm, bn, wm, wn) \
  instantiate_qmm(name, type, group_size, bits, 0, 0, bm, bn, wm, wn)         \
  instantiate_qmm(name, type, group_size, bits, 1, 0, bm, bn, wm, wn)

#define instantiate_qmm_batched(name, type, group_size, bits, bm, bn, wm, wn) \
  instantiate_qmm(name, type, group_size, bits, 0, 0, bm, bn, wm, wn)         \
  instantiate_qmm(name, type, group_size, bits, 0, 1, bm, bn, wm, wn)

#define instantiate_qmm_aligned_batched(name, type, group_size, bits, bm, bn, wm, wn) \
  instantiate_qmm(name, type, group_size, bits, 0, 0, bm, bn, wm, wn)                 \
  instantiate_qmm(name, type, group_size, bits, 0, 1, bm, bn, wm, wn)                 \
  instantiate_qmm(name, type, group_size, bits, 1, 0, bm, bn, wm, wn)                 \
  instantiate_qmm(name, type, group_size, bits, 1, 1, bm, bn, wm, wn)

#define instantiate_qmm_all(type, group_size, bits)                                       \
  instantiate_qmm_aligned_batched(affine_qmm_t_nax, type, group_size, bits, 32, 64, 2, 2) \
  instantiate_qmm_aligned_batched(affine_qmm_t_nax, type, group_size, bits, 64, 64, 2, 2) \
  instantiate_qmm_batched(affine_qmm_n_nax, type, group_size, bits, 64, 64, 2, 2)         \
  instantiate_qmm_aligned(affine_gather_qmm_t_nax, type, group_size, bits, 64, 64, 2, 2)

// Specializations for staged qmm_t_nax.
#define instantiate_qmm_all_staged(type, group_size, bits)                      \
  instantiate_qmm(affine_qmm_t_nax, type, group_size, bits, 1, 0, 64, 64, 4, 1) \
  instantiate_qmm(affine_qmm_t_nax, type, group_size, bits, 1, 1, 64, 64, 4, 1)

// Specializations for qmm_t_nax_splitk.
#define instantiate_qmm_all_splitk(group_size, bits)                                        \
  instantiate_qmm(affine_qmm_t_nax_splitk, float, group_size, bits, 1, 0, 32, 64, 2, 2)     \
  instantiate_qmm(affine_qmm_t_nax_splitk, float, group_size, bits, 1, 0, 64, 64, 2, 2)     \
  instantiate_qmm(affine_qmm_t_nax_splitk, float16_t, group_size, bits, 1, 0, 64, 64, 4, 1) \
  instantiate_qmm(affine_qmm_t_nax_splitk, bfloat16_t, group_size, bits, 1, 0, 64, 64, 4, 1)

///////////////////////////////////////////////////////////////////////////////
// affine_gather_qmm_rhs_nax
///////////////////////////////////////////////////////////////////////////////

#define instantiate_grouped_qmm(name, type, group_size, bits, transpose, bm, bn) \
  instantiate_kernel( \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs_0_t_" #transpose "_bm_" #bm "_bn_" #bn, \
      name,           \
      type,           \
      group_size,     \
      bits,           \
      false,          \
      transpose,      \
      bm,             \
      bn,             \
      64,             \
      2,              \
      2)

#define instantiate_grouped_qmm_all(type, group_size, bits)                             \
  instantiate_grouped_qmm(affine_gather_qmm_rhs_nax, type, group_size, bits, 0, 32, 64) \
  instantiate_grouped_qmm(affine_gather_qmm_rhs_nax, type, group_size, bits, 0, 64, 64) \
  instantiate_grouped_qmm(affine_gather_qmm_rhs_nax, type, group_size, bits, 1, 32, 64) \
  instantiate_grouped_qmm(affine_gather_qmm_rhs_nax, type, group_size, bits, 1, 64, 64)

///////////////////////////////////////////////////////////////////////////////
// Instantiation
///////////////////////////////////////////////////////////////////////////////

#define instantiate_quantized_all(group_size, bits)        \
  instantiate_qmm_all(float, group_size, bits)             \
  instantiate_qmm_all(float16_t, group_size, bits)         \
  instantiate_qmm_all(bfloat16_t, group_size, bits)        \
  instantiate_qmm_all_staged(float16_t, group_size, bits)  \
  instantiate_qmm_all_staged(bfloat16_t, group_size, bits) \
  instantiate_qmm_all_splitk(group_size, bits)             \
  instantiate_grouped_qmm_all(float, group_size, bits)     \
  instantiate_grouped_qmm_all(float16_t, group_size, bits) \
  instantiate_grouped_qmm_all(bfloat16_t, group_size, bits)

#define instantiate_quantized_all_bits(group_size) \
  instantiate_quantized_all(group_size, 1)         \
  instantiate_quantized_all(group_size, 2)         \
  instantiate_quantized_all(group_size, 3)         \
  instantiate_quantized_all(group_size, 4)         \
  instantiate_quantized_all(group_size, 5)         \
  instantiate_quantized_all(group_size, 6)         \
  instantiate_quantized_all(group_size, 8)

instantiate_quantized_all_bits(32)
instantiate_quantized_all_bits(64)
instantiate_quantized_all_bits(128)

// Staged int8 qmm
#define instantiate_qmm_i8_span(type, group_size, bits, span)       \
  instantiate_kernel(                                               \
      "affine_qmm_t_nax_i8_" #type "_gs_" #group_size "_b_" #bits   \
      "_xs_" #span,                                                 \
      affine_qmm_t_nax_i8,                                          \
      type,                                                         \
      group_size,                                                   \
      bits,                                                         \
      span)

#define instantiate_qmm_i8(type, group_size, bits)     \
  instantiate_qmm_i8_span(type, group_size, bits, 0)   \
  instantiate_qmm_i8_span(type, group_size, bits, 1)   \
  instantiate_qmm_i8_span(type, group_size, bits, 4)

#define instantiate_qmm_i8_groups(type, bits)        \
  instantiate_kernel(                                \
      "affine_qmm_i8_row_scales_" #type "_b_" #bits, \
      affine_qmm_i8_row_scales,                      \
      type,                                          \
      bits)                                          \
  instantiate_qmm_i8(type, 128, bits)                \
  instantiate_qmm_i8(type, 64, bits)                 \
  instantiate_qmm_i8(type, 32, bits)

#define instantiate_qmm_i8_types(bits)       \
  instantiate_qmm_i8_groups(float, bits)     \
  instantiate_qmm_i8_groups(float16_t, bits) \
  instantiate_qmm_i8_groups(bfloat16_t, bits)

#define instantiate_qmm_i8_quantize(type) \
  instantiate_kernel(                     \
      "affine_qmm_i8_quantize_x_" #type,  \
      affine_qmm_i8_quantize_x,           \
      type)

instantiate_qmm_i8_types(2)
instantiate_qmm_i8_types(3)
instantiate_qmm_i8_types(4)
instantiate_qmm_i8_types(5)
instantiate_qmm_i8_types(6)
instantiate_qmm_i8_types(8)
instantiate_qmm_i8_quantize(float)
instantiate_qmm_i8_quantize(float16_t)
instantiate_qmm_i8_quantize(bfloat16_t)
// clang-format on
