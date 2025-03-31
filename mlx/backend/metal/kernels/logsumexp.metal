// Copyright © 2023-2024 Apple Inc.

#include <metal_common>
#include <metal_simdgroup>

using namespace metal;

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/logsumexp.h"

#define instantiate_logsumexp(name, itype)                               \
  instantiate_kernel("block_logsumexp_" #name, logsumexp, itype)         \
  instantiate_kernel("looped_logsumexp_" #name, logsumexp_looped, itype) \

instantiate_logsumexp(float32, float)
instantiate_logsumexp(float16, half)
instantiate_logsumexp(bfloat16, bfloat16_t) // clang-format on
