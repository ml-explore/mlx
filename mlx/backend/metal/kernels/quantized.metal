// Copyright © 2023-2024 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized.h"


#define instantiate_qmv_fast(itype, group_size, bits)       \
  instantiate_kernel(                                       \
      "qmv_" #itype "_gs_" #group_size "_b_" #bits "_fast", \
      qmv_fast,                                             \
      itype,                                                \
      group_size,                                           \
      bits)

#define instantiate_qmv_fast_types(group_size, bits)     \
  instantiate_qmv_fast(float, group_size, bits) \
  instantiate_qmv_fast(float16_t, group_size, bits)  \
  instantiate_qmv_fast(bfloat16_t, group_size, bits)

instantiate_qmv_fast_types(128, 2)
instantiate_qmv_fast_types(128, 4)
instantiate_qmv_fast_types(128, 8)
instantiate_qmv_fast_types( 64, 2)
instantiate_qmv_fast_types( 64, 4)
instantiate_qmv_fast_types( 64, 8)
instantiate_qmv_fast_types( 32, 2)
instantiate_qmv_fast_types( 32, 4)
instantiate_qmv_fast_types( 32, 8)

#define instantiate_qmv(itype, group_size, bits)    \
  instantiate_kernel(                               \
      "qmv_" #itype "_gs_" #group_size "_b_" #bits, \
      qmv,                                          \
      itype,                                        \
      group_size,                                   \
      bits)

#define instantiate_qmv_types(group_size, bits)     \
  instantiate_qmv(float, group_size, bits) \
  instantiate_qmv(float16_t, group_size, bits)  \
  instantiate_qmv(bfloat16_t, group_size, bits)

instantiate_qmv_types(128, 2)
instantiate_qmv_types(128, 4)
instantiate_qmv_types(128, 8)
instantiate_qmv_types( 64, 2)
instantiate_qmv_types( 64, 4)
instantiate_qmv_types( 64, 8)
instantiate_qmv_types( 32, 2)
instantiate_qmv_types( 32, 4)
instantiate_qmv_types( 32, 8)

#define instantiate_qvm(itype, group_size, bits)    \
  instantiate_kernel(                               \
      "qvm_" #itype "_gs_" #group_size "_b_" #bits, \
      qvm,                                          \
      itype,                                        \
      group_size,                                   \
      bits)

#define instantiate_qvm_types(group_size, bits)     \
  instantiate_qvm(float, group_size, bits) \
  instantiate_qvm(float16_t, group_size, bits)  \
  instantiate_qvm(bfloat16_t, group_size, bits)

instantiate_qvm_types(128, 2)
instantiate_qvm_types(128, 4)
instantiate_qvm_types(128, 8)
instantiate_qvm_types( 64, 2)
instantiate_qvm_types( 64, 4)
instantiate_qvm_types( 64, 8)
instantiate_qvm_types( 32, 2)
instantiate_qvm_types( 32, 4)
instantiate_qvm_types( 32, 8)

#define instantiate_qmm_t(itype, group_size, bits, aligned_N)            \
  instantiate_kernel(                                                    \
      "qmm_t_" #itype "_gs_" #group_size "_b_" #bits "_alN_" #aligned_N, \
      qmm_t,                                                             \
      itype,                                                             \
      group_size,                                                        \
      bits,                                                              \
      aligned_N)

#define instantiate_qmm_t_types(group_size, bits)                  \
  instantiate_qmm_t(float, group_size, bits, false)       \
  instantiate_qmm_t(float16_t, group_size, bits, false)        \
  instantiate_qmm_t(bfloat16_t, group_size, bits, false) \
  instantiate_qmm_t(float, group_size, bits, true)        \
  instantiate_qmm_t(float16_t, group_size, bits, true)         \
  instantiate_qmm_t(bfloat16_t, group_size, bits, true)

instantiate_qmm_t_types(128, 2)
instantiate_qmm_t_types(128, 4)
instantiate_qmm_t_types(128, 8)
instantiate_qmm_t_types( 64, 2)
instantiate_qmm_t_types( 64, 4)
instantiate_qmm_t_types( 64, 8)
instantiate_qmm_t_types( 32, 2)
instantiate_qmm_t_types( 32, 4)
instantiate_qmm_t_types( 32, 8)

#define instantiate_qmm_n(itype, group_size, bits)    \
  instantiate_kernel(                                 \
      "qmm_n_" #itype "_gs_" #group_size "_b_" #bits, \
      qmm_n,                                          \
      itype,                                          \
      group_size,                                     \
      bits)

#define instantiate_qmm_n_types(group_size, bits)     \
  instantiate_qmm_n(float, group_size, bits) \
  instantiate_qmm_n(float16_t, group_size, bits)  \
  instantiate_qmm_n(bfloat16_t, group_size, bits)

instantiate_qmm_n_types(128, 2)
instantiate_qmm_n_types(128, 4)
instantiate_qmm_n_types(128, 8)
instantiate_qmm_n_types( 64, 2)
instantiate_qmm_n_types( 64, 4)
instantiate_qmm_n_types( 64, 8)
instantiate_qmm_n_types( 32, 2)
instantiate_qmm_n_types( 32, 4)
instantiate_qmm_n_types( 32, 8)

#define instantiate_bs_qmv_fast(itype, group_size, bits)       \
  instantiate_kernel(                                          \
      "bs_qmv_" #itype "_gs_" #group_size "_b_" #bits "_fast", \
      bs_qmv_fast,                                             \
      itype,                                                   \
      group_size,                                              \
      bits)

#define instantiate_bs_qmv_fast_types(group_size, bits)     \
  instantiate_bs_qmv_fast(float, group_size, bits) \
  instantiate_bs_qmv_fast(float16_t, group_size, bits)  \
  instantiate_bs_qmv_fast(bfloat16_t, group_size, bits)

instantiate_bs_qmv_fast_types(128, 2)
instantiate_bs_qmv_fast_types(128, 4)
instantiate_bs_qmv_fast_types(128, 8)
instantiate_bs_qmv_fast_types( 64, 2)
instantiate_bs_qmv_fast_types( 64, 4)
instantiate_bs_qmv_fast_types( 64, 8)
instantiate_bs_qmv_fast_types( 32, 2)
instantiate_bs_qmv_fast_types( 32, 4)
instantiate_bs_qmv_fast_types( 32, 8)

#define instantiate_bs_qmv(itype, group_size, bits)    \
  instantiate_kernel(                                  \
      "bs_qmv_" #itype "_gs_" #group_size "_b_" #bits, \
      bs_qmv,                                          \
      itype,                                           \
      group_size,                                      \
      bits)

#define instantiate_bs_qmv_types(group_size, bits)     \
  instantiate_bs_qmv(float, group_size, bits) \
  instantiate_bs_qmv(float16_t, group_size, bits)  \
  instantiate_bs_qmv(bfloat16_t, group_size, bits)

instantiate_bs_qmv_types(128, 2)
instantiate_bs_qmv_types(128, 4)
instantiate_bs_qmv_types(128, 8)
instantiate_bs_qmv_types( 64, 2)
instantiate_bs_qmv_types( 64, 4)
instantiate_bs_qmv_types( 64, 8)
instantiate_bs_qmv_types( 32, 2)
instantiate_bs_qmv_types( 32, 4)
instantiate_bs_qmv_types( 32, 8)

#define instantiate_bs_qvm(itype, group_size, bits)    \
  instantiate_kernel(                                  \
      "bs_qvm_" #itype "_gs_" #group_size "_b_" #bits, \
      bs_qvm,                                          \
      itype,                                           \
      group_size,                                      \
      bits)

#define instantiate_bs_qvm_types(group_size, bits)     \
  instantiate_bs_qvm(float, group_size, bits) \
  instantiate_bs_qvm(float16_t, group_size, bits)  \
  instantiate_bs_qvm(bfloat16_t, group_size, bits)

instantiate_bs_qvm_types(128, 2)
instantiate_bs_qvm_types(128, 4)
instantiate_bs_qvm_types(128, 8)
instantiate_bs_qvm_types( 64, 2)
instantiate_bs_qvm_types( 64, 4)
instantiate_bs_qvm_types( 64, 8)
instantiate_bs_qvm_types( 32, 2)
instantiate_bs_qvm_types( 32, 4)
instantiate_bs_qvm_types( 32, 8)

#define instantiate_bs_qmm_t(itype, group_size, bits, aligned_N)            \
  instantiate_kernel(                                                       \
      "bs_qmm_t_" #itype "_gs_" #group_size "_b_" #bits "_alN_" #aligned_N, \
      bs_qmm_t,                                                             \
      itype,                                                                \
      group_size,                                                           \
      bits,                                                                 \
      aligned_N)

#define instantiate_bs_qmm_t_types(group_size, bits)                  \
  instantiate_bs_qmm_t(float, group_size, bits, false)       \
  instantiate_bs_qmm_t(float16_t, group_size, bits, false)        \
  instantiate_bs_qmm_t(bfloat16_t, group_size, bits, false) \
  instantiate_bs_qmm_t(float, group_size, bits, true)        \
  instantiate_bs_qmm_t(float16_t, group_size, bits, true)         \
  instantiate_bs_qmm_t(bfloat16_t, group_size, bits, true)

instantiate_bs_qmm_t_types(128, 2)
instantiate_bs_qmm_t_types(128, 4)
instantiate_bs_qmm_t_types(128, 8)
instantiate_bs_qmm_t_types( 64, 2)
instantiate_bs_qmm_t_types( 64, 4)
instantiate_bs_qmm_t_types( 64, 8)
instantiate_bs_qmm_t_types( 32, 2)
instantiate_bs_qmm_t_types( 32, 4)
instantiate_bs_qmm_t_types( 32, 8)

#define instantiate_bs_qmm_n(itype, group_size, bits)    \
  instantiate_kernel(                                    \
      "bs_qmm_n_" #itype "_gs_" #group_size "_b_" #bits, \
      bs_qmm_n,                                          \
      itype,                                             \
      group_size,                                        \
      bits)

#define instantiate_bs_qmm_n_types(group_size, bits)     \
  instantiate_bs_qmm_n(float, group_size, bits) \
  instantiate_bs_qmm_n(float16_t, group_size, bits)  \
  instantiate_bs_qmm_n(bfloat16_t, group_size, bits)

instantiate_bs_qmm_n_types(128, 2)
instantiate_bs_qmm_n_types(128, 4)
instantiate_bs_qmm_n_types(128, 8)
instantiate_bs_qmm_n_types( 64, 2)
instantiate_bs_qmm_n_types( 64, 4)
instantiate_bs_qmm_n_types( 64, 8)
instantiate_bs_qmm_n_types( 32, 2)
instantiate_bs_qmm_n_types( 32, 4)
instantiate_bs_qmm_n_types( 32, 8) // clang-format on
