// Copyright © 2023-2024 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized.h"

#define instantiate_quantized(name, type, group_size, bits)     \
  instantiate_kernel(                                                    \
      #name "_" #type "_gs_" #group_size "_b_" #bits,                    \
      name,                                                              \
      type,                                                              \
      group_size,                                                        \
      bits)

#define instantiate_quantized_batched(name, type, group_size, bits, batched)     \
  instantiate_kernel(                                                    \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_batch_" #batched, \
      name,                                                              \
      type,                                                              \
      group_size,                                                        \
      bits,                                                              \
      batched)

#define instantiate_quantized_aligned(name, type, group_size, bits, aligned)     \
  instantiate_kernel(                                                                     \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned, \
      name,                                                                  \
      type,                                                                  \
      group_size,                                                            \
      bits,                                                                  \
      aligned)

#define instantiate_quantized_aligned_batched(name, type, group_size, bits, aligned, batched)     \
  instantiate_kernel(                                                                     \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned "_batch_" #batched, \
      name,                                                                  \
      type,                                                                  \
      group_size,                                                            \
      bits,                                                                  \
      aligned,                                                               \
      batched)

#define instantiate_quantized_quad(name, type, group_size, bits, D, batched)     \
  instantiate_kernel(                                                            \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_d_" #D "_batch_" #batched, \
      name,                                                         \
      type,                                                         \
      group_size,                                                   \
      bits,                                                         \
      D,                                                            \
      batched)

#define instantiate_quantized_split_k(name, type, group_size, bits, split_k)     \
  instantiate_kernel(                                                            \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_spk_" #split_k, \
      name,                                                         \
      type,                                                         \
      group_size,                                                   \
      bits,                                                         \
      split_k)

#define instantiate_quantized_batched_wrap(name, type, group_size, bits) \
  instantiate_quantized_batched(name, type, group_size, bits, 1)      \
  instantiate_quantized_batched(name, type, group_size, bits, 0)

#define instantiate_quantized_all_batched(type, group_size, bits) \
  instantiate_quantized_batched_wrap(qmv_fast, type, group_size, bits)     \
  instantiate_quantized_batched_wrap(qmv, type, group_size, bits)     \
  instantiate_quantized_batched_wrap(qvm, type, group_size, bits)     \
  instantiate_quantized_batched_wrap(qmm_n, type, group_size, bits)

#define instantiate_quantized_all_single(type, group_size, bits) \
  instantiate_quantized(affine_quantize, type, group_size, bits) \
  instantiate_quantized(affine_dequantize, type, group_size, bits)     \
  instantiate_quantized(bs_qmv_fast, type, group_size, bits)     \
  instantiate_quantized(bs_qmv, type, group_size, bits)     \
  instantiate_quantized(bs_qvm, type, group_size, bits)     \
  instantiate_quantized(bs_qmm_n, type, group_size, bits)

#define instantiate_quantized_all_aligned(type, group_size, bits)   \
  instantiate_quantized_aligned(bs_qmm_t, type, group_size, bits, true) \
  instantiate_quantized_aligned(bs_qmm_t, type, group_size, bits, false) \
  instantiate_quantized_aligned_batched(qmm_t, type, group_size, bits, true, 1) \
  instantiate_quantized_aligned_batched(qmm_t, type, group_size, bits, true, 0) \
  instantiate_quantized_aligned_batched(qmm_t, type, group_size, bits, false, 1) \
  instantiate_quantized_aligned_batched(qmm_t, type, group_size, bits, false, 0)

#define instantiate_quantized_all_quad(type, group_size, bits)   \
  instantiate_quantized_quad(qmv_quad, type, group_size, bits, 64, 1)   \
  instantiate_quantized_quad(qmv_quad, type, group_size, bits, 64, 0)  \
  instantiate_quantized_quad(qmv_quad, type, group_size, bits, 128, 1)  \
  instantiate_quantized_quad(qmv_quad, type, group_size, bits, 128, 0)

#define instantiate_quantized_all_splitk(type, group_size, bits)   \
  instantiate_quantized_split_k(qvm_split_k, type, group_size, bits, 8)   \
  instantiate_quantized_split_k(qvm_split_k, type, group_size, bits, 32)

#define instantiate_quantized_funcs(type, group_size, bits) \
  instantiate_quantized_all_single(type, group_size, bits)  \
  instantiate_quantized_all_batched(type, group_size, bits) \
  instantiate_quantized_all_aligned(type, group_size, bits) \
  instantiate_quantized_all_quad(type, group_size, bits)    \
  instantiate_quantized_all_splitk(type, group_size, bits)

#define instantiate_quantized_types(group_size, bits)       \
  instantiate_quantized_funcs(float, group_size, bits)      \
  instantiate_quantized_funcs(float16_t, group_size, bits)  \
  instantiate_quantized_funcs(bfloat16_t, group_size, bits)

#define instantiate_quantized_groups(bits) \
  instantiate_quantized_types(128, bits)   \
  instantiate_quantized_types(64, bits)    \
  instantiate_quantized_types(32, bits)

#define instantiate_quantized_all() \
  instantiate_quantized_groups(2) \
  instantiate_quantized_groups(3) \
  instantiate_quantized_groups(4) \
  instantiate_quantized_groups(6) \
  instantiate_quantized_groups(8)

instantiate_quantized_all() // clang-format on
