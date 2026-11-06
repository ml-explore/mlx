// Copyright © 2026 Apple Inc.

#include <metal_common>
#include <metal_simdgroup>

using namespace metal;

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/cross_entropy.h"

#define instantiate_cross_entropy(name, itype)                            \
  instantiate_kernel("cross_entropy_" #name, cross_entropy, itype)        \
  instantiate_kernel("cross_entropy_vjp_" #name, cross_entropy_vjp, itype)

instantiate_cross_entropy(float32, float)
instantiate_cross_entropy(float16, half)
instantiate_cross_entropy(bfloat16, bfloat16_t) // clang-format on
