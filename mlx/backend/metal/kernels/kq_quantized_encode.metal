// Copyright © 2026 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/kq_quantized_encode.h"

#define instantiate_kquant_quantize(type, gs, bits, codec)                \
  instantiate_kernel(                                                     \
      "kquant_" #codec "_quantize_" #type "_gs_" #gs "_b_" #bits,         \
      kq_ ## codec ## _quantize,                                          \
      type,                                                               \
      gs,                                                                 \
      bits)

#define instantiate_kquant_q8_0_quantize_for_type(type) \
  instantiate_kquant_quantize(type, 32, 8, q8_0)

instantiate_kquant_q8_0_quantize_for_type(float)
instantiate_kquant_q8_0_quantize_for_type(float16_t)
instantiate_kquant_q8_0_quantize_for_type(bfloat16_t)

#define instantiate_kquant_q4_k_quantize_for_type(type) \
  instantiate_kquant_quantize(type, 256, 4, q4_k)

instantiate_kquant_q4_k_quantize_for_type(float)
instantiate_kquant_q4_k_quantize_for_type(float16_t)
instantiate_kquant_q4_k_quantize_for_type(bfloat16_t)

#define instantiate_kquant_q6_k_quantize_for_type(type) \
  instantiate_kquant_quantize(type, 256, 6, q6_k)

instantiate_kquant_q6_k_quantize_for_type(float)
instantiate_kquant_q6_k_quantize_for_type(float16_t)
instantiate_kquant_q6_k_quantize_for_type(bfloat16_t)

#define instantiate_kquant_q5_k_quantize_for_type(type) \
  instantiate_kquant_quantize(type, 256, 5, q5_k)

instantiate_kquant_q5_k_quantize_for_type(float)
instantiate_kquant_q5_k_quantize_for_type(float16_t)
instantiate_kquant_q5_k_quantize_for_type(bfloat16_t)

#define instantiate_kquant_q3_k_quantize_for_type(type) \
  instantiate_kquant_quantize(type, 256, 3, q3_k)

instantiate_kquant_q3_k_quantize_for_type(float)
instantiate_kquant_q3_k_quantize_for_type(float16_t)
instantiate_kquant_q3_k_quantize_for_type(bfloat16_t)

#define instantiate_kquant_q2_k_quantize_for_type(type) \
  instantiate_kquant_quantize(type, 256, 2, q2_k)

instantiate_kquant_q2_k_quantize_for_type(float)
instantiate_kquant_q2_k_quantize_for_type(float16_t)
instantiate_kquant_q2_k_quantize_for_type(bfloat16_t)

#define instantiate_kquant_q4_0_quantize_for_type(type) \
  instantiate_kquant_quantize(type, 32, 4, q4_0)

instantiate_kquant_q4_0_quantize_for_type(float)
instantiate_kquant_q4_0_quantize_for_type(float16_t)
instantiate_kquant_q4_0_quantize_for_type(bfloat16_t)

#define instantiate_kquant_q4_1_quantize_for_type(type) \
  instantiate_kquant_quantize(type, 32, 4, q4_1)

instantiate_kquant_q4_1_quantize_for_type(float)
instantiate_kquant_q4_1_quantize_for_type(float16_t)
instantiate_kquant_q4_1_quantize_for_type(bfloat16_t)

#define instantiate_kquant_q5_0_quantize_for_type(type) \
  instantiate_kquant_quantize(type, 32, 5, q5_0)

instantiate_kquant_q5_0_quantize_for_type(float)
instantiate_kquant_q5_0_quantize_for_type(float16_t)
instantiate_kquant_q5_0_quantize_for_type(bfloat16_t)

#define instantiate_kquant_q5_1_quantize_for_type(type) \
  instantiate_kquant_quantize(type, 32, 5, q5_1)

instantiate_kquant_q5_1_quantize_for_type(float)
instantiate_kquant_q5_1_quantize_for_type(float16_t)
instantiate_kquant_q5_1_quantize_for_type(bfloat16_t)
    // clang-format on
