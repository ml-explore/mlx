// Copyright © 2023-2024 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/ternary_quantized.h"

#define instantiate_ternary(type, group_size) \
  instantiate_kernel(                         \
      "ternary_quantize_" #type "_gs_" #group_size "_b_2", ternary_quantize, type, group_size, 2) \
  instantiate_kernel(                         \
      "ternary_dequantize_" #type "_gs_" #group_size "_b_2", ternary_dequantize, type, group_size, 2)

#define instantiate_ternary_types(group_size) \
  instantiate_ternary(float, group_size)      \
  instantiate_ternary(float16_t, group_size)  \
  instantiate_ternary(bfloat16_t, group_size)

instantiate_ternary_types(32)
instantiate_ternary_types(64)
instantiate_ternary_types(128)
// clang-format on
