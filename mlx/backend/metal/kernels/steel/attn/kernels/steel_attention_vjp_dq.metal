// Copyright © 2024-25 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/attn.h"

#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_vjp_dq.h"

instantiate_attention_vjp_dq_all(float16, half);
instantiate_attention_vjp_dq_all(bfloat16, bfloat16_t);
// clang-format on
