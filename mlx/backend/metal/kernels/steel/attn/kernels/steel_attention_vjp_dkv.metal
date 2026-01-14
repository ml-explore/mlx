// Copyright © 2024-25 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/attn/attn.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_vjp_dkv.h"

// Instantiate VJP dKV kernels for all supported types and head dimensions
// tname: string name for kernel lookup (matches type_to_name output)
// dtype: actual C++ type for the kernel template

instantiate_attention_vjp_dkv_all(float32, float);
instantiate_attention_vjp_dkv_all(float16, half);
instantiate_attention_vjp_dkv_all(bfloat16, bfloat16_t);
// clang-format on
