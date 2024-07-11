// Copyright © 2023-2024 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized.h"

#define instantiate_quantized(name, type, group_size, bits) \
  instantiate_kernel(                                       \
      #name "_" #type "_gs_" #group_size "_b_" #bits,       \
      name,                                                 \
      type,                                                 \
      group_size,                                           \
      bits)

#define instantiate_quantized_types(name, group_size, bits) \
  instantiate_quantized(name, float, group_size, bits)      \
  instantiate_quantized(name, float16_t, group_size, bits)  \
  instantiate_quantized(name, bfloat16_t, group_size, bits)

#define instantiate_quantized_groups(name, bits) \
  instantiate_quantized_types(name, 128, bits)   \
  instantiate_quantized_types(name, 64, bits)    \
  instantiate_quantized_types(name, 32, bits)

#define instantiate_quantized_all(name) \
  instantiate_quantized_groups(name, 2) \
  instantiate_quantized_groups(name, 4) \
  instantiate_quantized_groups(name, 8)

instantiate_quantized_all(qmv_fast)
instantiate_quantized_all(qmv)
instantiate_quantized_all(qvm)
instantiate_quantized_all(qmm_n)
instantiate_quantized_all(bs_qmv_fast)
instantiate_quantized_all(bs_qmv)
instantiate_quantized_all(bs_qvm)
instantiate_quantized_all(bs_qmm_n)
instantiate_quantized_all(affine_quantize)
instantiate_quantized_all(affine_dequantize)

#define instantiate_quantized_aligned(name, type, group_size, bits, aligned) \
  instantiate_kernel(                                                        \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned,       \
      name,                                                                  \
      type,                                                                  \
      group_size,                                                            \
      bits,                                                                  \
      aligned)

#define instantiate_quantized_types_aligned(name, group_size, bits)       \
  instantiate_quantized_aligned(name, float, group_size, bits, true)      \
  instantiate_quantized_aligned(name, float16_t, group_size, bits, true)  \
  instantiate_quantized_aligned(name, bfloat16_t, group_size, bits, true) \
  instantiate_quantized_aligned(name, float, group_size, bits, false)     \
  instantiate_quantized_aligned(name, float16_t, group_size, bits, false) \
  instantiate_quantized_aligned(name, bfloat16_t, group_size, bits, false)

#define instantiate_quantized_groups_aligned(name, bits) \
  instantiate_quantized_types_aligned(name, 128, bits)   \
  instantiate_quantized_types_aligned(name, 64, bits)    \
  instantiate_quantized_types_aligned(name, 32, bits)

#define instantiate_quantized_all_aligned(name) \
  instantiate_quantized_groups_aligned(name, 2) \
  instantiate_quantized_groups_aligned(name, 4) \
  instantiate_quantized_groups_aligned(name, 8) \

instantiate_quantized_all_aligned(qmm_t)
instantiate_quantized_all_aligned(bs_qmm_t) // clang-format on
