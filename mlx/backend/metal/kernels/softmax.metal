// Copyright © 2023-2024 Apple Inc.

#include <metal_common>
#include <metal_simdgroup>

using namespace metal;

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/softmax.h"

#define instantiate_softmax(name, itype)                                \
  instantiate_kernel("block_softmax_" #name, softmax_single_row, itype) \
  instantiate_kernel("looped_softmax_" #name, softmax_looped, itype)

#define instantiate_softmax_precise(name, itype)                                       \
  instantiate_kernel("block_softmax_precise_" #name, softmax_single_row, itype, float) \
  instantiate_kernel("looped_softmax_precise_" #name, softmax_looped, itype, float)

instantiate_softmax(float32, float)
instantiate_softmax(float16, half)
instantiate_softmax(bfloat16, bfloat16_t)
instantiate_softmax_precise(float16, half)
instantiate_softmax_precise(bfloat16, bfloat16_t) // clang-format on
