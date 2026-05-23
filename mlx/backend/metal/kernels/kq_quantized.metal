// Copyright © 2026 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/kq_quantized.h"

#define instantiate_kquant_batched(func, type, gs, bits, batched, codec)  \
  instantiate_kernel(                                                     \
      "kquant_" #codec "_" #func "_" #type "_gs_" #gs "_b_" #bits         \
          "_batch_" #batched,                                             \
      kq_ ## codec ## _ ## func,                                          \
      type,                                                               \
      gs,                                                                 \
      bits,                                                               \
      batched)

#define instantiate_kquant_qmm_t(type, gs, bits, aligned_N, batched, codec) \
  instantiate_kernel(                                                       \
      "kquant_" #codec "_qmm_t_" #type "_gs_" #gs "_b_" #bits               \
          "_alN_" #aligned_N "_batch_" #batched,                            \
      kq_ ## codec ## _qmm_t,                                               \
      type,                                                                 \
      gs,                                                                   \
      bits,                                                                 \
      aligned_N,                                                            \
      batched)

#define instantiate_kquant_qmm_t_splitk(type, gs, bits, aligned_N, codec) \
  instantiate_kernel(                                                     \
      "kquant_" #codec "_qmm_t_splitk_" #type "_gs_" #gs "_b_" #bits      \
          "_alN_" #aligned_N,                                             \
      kq_ ## codec ## _qmm_t_splitk,                                      \
      type,                                                               \
      gs,                                                                 \
      bits,                                                               \
      aligned_N)

#define instantiate_kquant_qmm_n(type, gs, bits, batched, codec)        \
  instantiate_kernel(                                                   \
      "kquant_" #codec "_qmm_n_" #type "_gs_" #gs "_b_" #bits           \
          "_batch_" #batched,                                           \
      kq_ ## codec ## _qmm_n,                                           \
      type,                                                             \
      gs,                                                               \
      bits,                                                             \
      batched)

#define instantiate_kquant_gather_qmv(func, type, gs, bits, codec)        \
  instantiate_kernel(                                                     \
      "kquant_" #codec "_" #func "_" #type "_gs_" #gs "_b_" #bits,        \
      kq_ ## codec ## _ ## func,                                          \
      type,                                                               \
      gs,                                                                 \
      bits)

#define instantiate_kquant_gather_qmm_t(type, gs, bits, aligned_N, codec) \
  instantiate_kernel(                                                     \
      "kquant_" #codec "_gather_qmm_t_" #type "_gs_" #gs "_b_" #bits      \
          "_alN_" #aligned_N,                                             \
      kq_ ## codec ## _gather_qmm_t,                                      \
      type,                                                               \
      gs,                                                                 \
      bits,                                                               \
      aligned_N)

#define instantiate_kquant_gather_qmm_n(type, gs, bits, codec)            \
  instantiate_kernel(                                                     \
      "kquant_" #codec "_gather_qmm_n_" #type "_gs_" #gs "_b_" #bits,     \
      kq_ ## codec ## _gather_qmm_n,                                      \
      type,                                                               \
      gs,                                                                 \
      bits)

#define instantiate_kquant_dequantize(type, gs, bits, codec)              \
  instantiate_kernel(                                                     \
      "kquant_" #codec "_dequantize_" #type "_gs_" #gs "_b_" #bits,       \
      kq_ ## codec ## _dequantize,                                        \
      type,                                                               \
      gs,                                                                 \
      bits)

#define instantiate_kquant_q8_0_for_type(type)                          \
  instantiate_kquant_batched(qmv_fast, type, 32, 8, 0, q8_0)            \
  instantiate_kquant_batched(qmv_fast, type, 32, 8, 1, q8_0)            \
  instantiate_kquant_batched(qmv,      type, 32, 8, 0, q8_0)            \
  instantiate_kquant_batched(qmv,      type, 32, 8, 1, q8_0)            \
  instantiate_kquant_qmm_t(type, 32, 8, true, 0, q8_0)                  \
  instantiate_kquant_qmm_t(type, 32, 8, true, 1, q8_0)                  \
  instantiate_kquant_qmm_t(type, 32, 8, false, 0, q8_0)                 \
  instantiate_kquant_qmm_t(type, 32, 8, false, 1, q8_0)                 \
  instantiate_kquant_qmm_t_splitk(type, 32, 8, true, q8_0)              \
  instantiate_kquant_qmm_t_splitk(type, 32, 8, false, q8_0)             \
  instantiate_kquant_qmm_n(type, 32, 8, 0, q8_0)                        \
  instantiate_kquant_qmm_n(type, 32, 8, 1, q8_0)                        \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 32, 8, q8_0)     \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 32, 8, q8_0)     \
  instantiate_kquant_gather_qmm_t(type, 32, 8, true, q8_0)              \
  instantiate_kquant_gather_qmm_t(type, 32, 8, false, q8_0)             \
  instantiate_kquant_gather_qmm_n(type, 32, 8, q8_0)                    \
  instantiate_kquant_dequantize(type, 32, 8, q8_0)

instantiate_kquant_q8_0_for_type(float)
instantiate_kquant_q8_0_for_type(bfloat16_t)
instantiate_kquant_q8_0_for_type(float16_t)

#define instantiate_kquant_q5_1_for_type(type)                          \
  instantiate_kquant_batched(qmv_fast, type, 32, 5, 0, q5_1)            \
  instantiate_kquant_batched(qmv_fast, type, 32, 5, 1, q5_1)            \
  instantiate_kquant_batched(qmv,      type, 32, 5, 0, q5_1)            \
  instantiate_kquant_batched(qmv,      type, 32, 5, 1, q5_1)            \
  instantiate_kquant_qmm_t(type, 32, 5, true, 0, q5_1)                  \
  instantiate_kquant_qmm_t(type, 32, 5, true, 1, q5_1)                  \
  instantiate_kquant_qmm_t(type, 32, 5, false, 0, q5_1)                 \
  instantiate_kquant_qmm_t(type, 32, 5, false, 1, q5_1)                 \
  instantiate_kquant_qmm_t_splitk(type, 32, 5, true, q5_1)              \
  instantiate_kquant_qmm_t_splitk(type, 32, 5, false, q5_1)             \
  instantiate_kquant_qmm_n(type, 32, 5, 0, q5_1)                        \
  instantiate_kquant_qmm_n(type, 32, 5, 1, q5_1)                        \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 32, 5, q5_1)     \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 32, 5, q5_1)     \
  instantiate_kquant_gather_qmm_t(type, 32, 5, true, q5_1)              \
  instantiate_kquant_gather_qmm_t(type, 32, 5, false, q5_1)             \
  instantiate_kquant_gather_qmm_n(type, 32, 5, q5_1)                    \
  instantiate_kquant_dequantize(type, 32, 5, q5_1)

instantiate_kquant_q5_1_for_type(float)
instantiate_kquant_q5_1_for_type(bfloat16_t)
instantiate_kquant_q5_1_for_type(float16_t)

#define instantiate_kquant_q4_0_for_type(type)                          \
  instantiate_kquant_batched(qmv_fast, type, 32, 4, 0, q4_0)            \
  instantiate_kquant_batched(qmv_fast, type, 32, 4, 1, q4_0)            \
  instantiate_kquant_batched(qmv,      type, 32, 4, 0, q4_0)            \
  instantiate_kquant_batched(qmv,      type, 32, 4, 1, q4_0)            \
  instantiate_kquant_qmm_t(type, 32, 4, true, 0, q4_0)                  \
  instantiate_kquant_qmm_t(type, 32, 4, true, 1, q4_0)                  \
  instantiate_kquant_qmm_t(type, 32, 4, false, 0, q4_0)                 \
  instantiate_kquant_qmm_t(type, 32, 4, false, 1, q4_0)                 \
  instantiate_kquant_qmm_t_splitk(type, 32, 4, true, q4_0)              \
  instantiate_kquant_qmm_t_splitk(type, 32, 4, false, q4_0)             \
  instantiate_kquant_qmm_n(type, 32, 4, 0, q4_0)                        \
  instantiate_kquant_qmm_n(type, 32, 4, 1, q4_0)                        \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 32, 4, q4_0)     \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 32, 4, q4_0)     \
  instantiate_kquant_gather_qmm_t(type, 32, 4, true, q4_0)              \
  instantiate_kquant_gather_qmm_t(type, 32, 4, false, q4_0)             \
  instantiate_kquant_gather_qmm_n(type, 32, 4, q4_0)                    \
  instantiate_kquant_dequantize(type, 32, 4, q4_0)

instantiate_kquant_q4_0_for_type(float)
instantiate_kquant_q4_0_for_type(bfloat16_t)
instantiate_kquant_q4_0_for_type(float16_t)

#define instantiate_kquant_q4_1_for_type(type)                          \
  instantiate_kquant_batched(qmv_fast, type, 32, 4, 0, q4_1)            \
  instantiate_kquant_batched(qmv_fast, type, 32, 4, 1, q4_1)            \
  instantiate_kquant_batched(qmv,      type, 32, 4, 0, q4_1)            \
  instantiate_kquant_batched(qmv,      type, 32, 4, 1, q4_1)            \
  instantiate_kquant_qmm_t(type, 32, 4, true, 0, q4_1)                  \
  instantiate_kquant_qmm_t(type, 32, 4, true, 1, q4_1)                  \
  instantiate_kquant_qmm_t(type, 32, 4, false, 0, q4_1)                 \
  instantiate_kquant_qmm_t(type, 32, 4, false, 1, q4_1)                 \
  instantiate_kquant_qmm_t_splitk(type, 32, 4, true, q4_1)              \
  instantiate_kquant_qmm_t_splitk(type, 32, 4, false, q4_1)             \
  instantiate_kquant_qmm_n(type, 32, 4, 0, q4_1)                        \
  instantiate_kquant_qmm_n(type, 32, 4, 1, q4_1)                        \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 32, 4, q4_1)     \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 32, 4, q4_1)     \
  instantiate_kquant_gather_qmm_t(type, 32, 4, true, q4_1)              \
  instantiate_kquant_gather_qmm_t(type, 32, 4, false, q4_1)             \
  instantiate_kquant_gather_qmm_n(type, 32, 4, q4_1)                    \
  instantiate_kquant_dequantize(type, 32, 4, q4_1)

instantiate_kquant_q4_1_for_type(float)
instantiate_kquant_q4_1_for_type(bfloat16_t)
instantiate_kquant_q4_1_for_type(float16_t)

#define instantiate_kquant_q5_0_for_type(type)                          \
  instantiate_kquant_batched(qmv_fast, type, 32, 5, 0, q5_0)            \
  instantiate_kquant_batched(qmv_fast, type, 32, 5, 1, q5_0)            \
  instantiate_kquant_batched(qmv,      type, 32, 5, 0, q5_0)            \
  instantiate_kquant_batched(qmv,      type, 32, 5, 1, q5_0)            \
  instantiate_kquant_qmm_t(type, 32, 5, true, 0, q5_0)                  \
  instantiate_kquant_qmm_t(type, 32, 5, true, 1, q5_0)                  \
  instantiate_kquant_qmm_t(type, 32, 5, false, 0, q5_0)                 \
  instantiate_kquant_qmm_t(type, 32, 5, false, 1, q5_0)                 \
  instantiate_kquant_qmm_t_splitk(type, 32, 5, true, q5_0)              \
  instantiate_kquant_qmm_t_splitk(type, 32, 5, false, q5_0)             \
  instantiate_kquant_qmm_n(type, 32, 5, 0, q5_0)                        \
  instantiate_kquant_qmm_n(type, 32, 5, 1, q5_0)                        \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 32, 5, q5_0)     \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 32, 5, q5_0)     \
  instantiate_kquant_gather_qmm_t(type, 32, 5, true, q5_0)              \
  instantiate_kquant_gather_qmm_t(type, 32, 5, false, q5_0)             \
  instantiate_kquant_gather_qmm_n(type, 32, 5, q5_0)                    \
  instantiate_kquant_dequantize(type, 32, 5, q5_0)

instantiate_kquant_q5_0_for_type(float)
instantiate_kquant_q5_0_for_type(bfloat16_t)
instantiate_kquant_q5_0_for_type(float16_t)

#define instantiate_kquant_q4_k_for_type(type)                          \
  instantiate_kquant_batched(qmv_fast, type, 256, 4, 0, q4_k)            \
  instantiate_kquant_batched(qmv_fast, type, 256, 4, 1, q4_k)            \
  instantiate_kquant_batched(qmv,      type, 256, 4, 0, q4_k)            \
  instantiate_kquant_batched(qmv,      type, 256, 4, 1, q4_k)            \
  instantiate_kquant_qmm_t(type, 256, 4, true, 0, q4_k)                  \
  instantiate_kquant_qmm_t(type, 256, 4, true, 1, q4_k)                  \
  instantiate_kquant_qmm_t(type, 256, 4, false, 0, q4_k)                 \
  instantiate_kquant_qmm_t(type, 256, 4, false, 1, q4_k)                 \
  instantiate_kquant_qmm_t_splitk(type, 256, 4, true, q4_k)              \
  instantiate_kquant_qmm_t_splitk(type, 256, 4, false, q4_k)             \
  instantiate_kquant_qmm_n(type, 256, 4, 0, q4_k)                        \
  instantiate_kquant_qmm_n(type, 256, 4, 1, q4_k)                        \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 256, 4, q4_k)     \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 256, 4, q4_k)     \
  instantiate_kquant_gather_qmm_t(type, 256, 4, true, q4_k)              \
  instantiate_kquant_gather_qmm_t(type, 256, 4, false, q4_k)             \
  instantiate_kquant_gather_qmm_n(type, 256, 4, q4_k)                    \
  instantiate_kquant_dequantize(type, 256, 4, q4_k)

instantiate_kquant_q4_k_for_type(float)
instantiate_kquant_q4_k_for_type(bfloat16_t)
instantiate_kquant_q4_k_for_type(float16_t)

#define instantiate_kquant_q5_k_for_type(type)                          \
  instantiate_kquant_batched(qmv_fast, type, 256, 5, 0, q5_k)            \
  instantiate_kquant_batched(qmv_fast, type, 256, 5, 1, q5_k)            \
  instantiate_kquant_batched(qmv,      type, 256, 5, 0, q5_k)            \
  instantiate_kquant_batched(qmv,      type, 256, 5, 1, q5_k)            \
  instantiate_kquant_qmm_t(type, 256, 5, true, 0, q5_k)                  \
  instantiate_kquant_qmm_t(type, 256, 5, true, 1, q5_k)                  \
  instantiate_kquant_qmm_t(type, 256, 5, false, 0, q5_k)                 \
  instantiate_kquant_qmm_t(type, 256, 5, false, 1, q5_k)                 \
  instantiate_kquant_qmm_t_splitk(type, 256, 5, true, q5_k)              \
  instantiate_kquant_qmm_t_splitk(type, 256, 5, false, q5_k)             \
  instantiate_kquant_qmm_n(type, 256, 5, 0, q5_k)                        \
  instantiate_kquant_qmm_n(type, 256, 5, 1, q5_k)                        \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 256, 5, q5_k)     \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 256, 5, q5_k)     \
  instantiate_kquant_gather_qmm_t(type, 256, 5, true, q5_k)              \
  instantiate_kquant_gather_qmm_t(type, 256, 5, false, q5_k)             \
  instantiate_kquant_gather_qmm_n(type, 256, 5, q5_k)                    \
  instantiate_kquant_dequantize(type, 256, 5, q5_k)

instantiate_kquant_q5_k_for_type(float)
instantiate_kquant_q5_k_for_type(bfloat16_t)
instantiate_kquant_q5_k_for_type(float16_t)

#define instantiate_kquant_q6_k_for_type(type)                          \
  instantiate_kquant_batched(qmv_fast, type, 256, 6, 0, q6_k)            \
  instantiate_kquant_batched(qmv_fast, type, 256, 6, 1, q6_k)            \
  instantiate_kquant_batched(qmv,      type, 256, 6, 0, q6_k)            \
  instantiate_kquant_batched(qmv,      type, 256, 6, 1, q6_k)            \
  instantiate_kquant_qmm_t(type, 256, 6, true, 0, q6_k)                  \
  instantiate_kquant_qmm_t(type, 256, 6, true, 1, q6_k)                  \
  instantiate_kquant_qmm_t(type, 256, 6, false, 0, q6_k)                 \
  instantiate_kquant_qmm_t(type, 256, 6, false, 1, q6_k)                 \
  instantiate_kquant_qmm_t_splitk(type, 256, 6, true, q6_k)              \
  instantiate_kquant_qmm_t_splitk(type, 256, 6, false, q6_k)             \
  instantiate_kquant_qmm_n(type, 256, 6, 0, q6_k)                        \
  instantiate_kquant_qmm_n(type, 256, 6, 1, q6_k)                        \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 256, 6, q6_k)     \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 256, 6, q6_k)     \
  instantiate_kquant_gather_qmm_t(type, 256, 6, true, q6_k)              \
  instantiate_kquant_gather_qmm_t(type, 256, 6, false, q6_k)             \
  instantiate_kquant_gather_qmm_n(type, 256, 6, q6_k)                    \
  instantiate_kquant_dequantize(type, 256, 6, q6_k)

instantiate_kquant_q6_k_for_type(float)
instantiate_kquant_q6_k_for_type(bfloat16_t)
instantiate_kquant_q6_k_for_type(float16_t)

#define instantiate_kquant_q3_k_for_type(type)                          \
  instantiate_kquant_batched(qmv_fast, type, 256, 3, 0, q3_k)            \
  instantiate_kquant_batched(qmv_fast, type, 256, 3, 1, q3_k)            \
  instantiate_kquant_batched(qmv,      type, 256, 3, 0, q3_k)            \
  instantiate_kquant_batched(qmv,      type, 256, 3, 1, q3_k)            \
  instantiate_kquant_qmm_t(type, 256, 3, true, 0, q3_k)                  \
  instantiate_kquant_qmm_t(type, 256, 3, true, 1, q3_k)                  \
  instantiate_kquant_qmm_t(type, 256, 3, false, 0, q3_k)                 \
  instantiate_kquant_qmm_t(type, 256, 3, false, 1, q3_k)                 \
  instantiate_kquant_qmm_t_splitk(type, 256, 3, true, q3_k)              \
  instantiate_kquant_qmm_t_splitk(type, 256, 3, false, q3_k)             \
  instantiate_kquant_qmm_n(type, 256, 3, 0, q3_k)                        \
  instantiate_kquant_qmm_n(type, 256, 3, 1, q3_k)                        \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 256, 3, q3_k)     \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 256, 3, q3_k)     \
  instantiate_kquant_gather_qmm_t(type, 256, 3, true, q3_k)              \
  instantiate_kquant_gather_qmm_t(type, 256, 3, false, q3_k)             \
  instantiate_kquant_gather_qmm_n(type, 256, 3, q3_k)                    \
  instantiate_kquant_dequantize(type, 256, 3, q3_k)

instantiate_kquant_q3_k_for_type(float)
instantiate_kquant_q3_k_for_type(bfloat16_t)
instantiate_kquant_q3_k_for_type(float16_t)

#define instantiate_kquant_q2_k_for_type(type)                          \
  instantiate_kquant_batched(qmv_fast, type, 256, 2, 0, q2_k)            \
  instantiate_kquant_batched(qmv_fast, type, 256, 2, 1, q2_k)            \
  instantiate_kquant_batched(qmv,      type, 256, 2, 0, q2_k)            \
  instantiate_kquant_batched(qmv,      type, 256, 2, 1, q2_k)            \
  instantiate_kquant_qmm_t(type, 256, 2, true, 0, q2_k)                  \
  instantiate_kquant_qmm_t(type, 256, 2, true, 1, q2_k)                  \
  instantiate_kquant_qmm_t(type, 256, 2, false, 0, q2_k)                 \
  instantiate_kquant_qmm_t(type, 256, 2, false, 1, q2_k)                 \
  instantiate_kquant_qmm_t_splitk(type, 256, 2, true, q2_k)              \
  instantiate_kquant_qmm_t_splitk(type, 256, 2, false, q2_k)             \
  instantiate_kquant_qmm_n(type, 256, 2, 0, q2_k)                        \
  instantiate_kquant_qmm_n(type, 256, 2, 1, q2_k)                        \
  instantiate_kquant_gather_qmv(gather_qmv_fast, type, 256, 2, q2_k)     \
  instantiate_kquant_gather_qmv(gather_qmv,      type, 256, 2, q2_k)     \
  instantiate_kquant_gather_qmm_t(type, 256, 2, true, q2_k)              \
  instantiate_kquant_gather_qmm_t(type, 256, 2, false, q2_k)             \
  instantiate_kquant_gather_qmm_n(type, 256, 2, q2_k)                    \
  instantiate_kquant_dequantize(type, 256, 2, q2_k)

instantiate_kquant_q2_k_for_type(float)
instantiate_kquant_q2_k_for_type(bfloat16_t)
instantiate_kquant_q2_k_for_type(float16_t)
    // clang-format on
