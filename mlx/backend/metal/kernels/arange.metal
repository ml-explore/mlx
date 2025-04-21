// Copyright © 2023-2024 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/arange.h"

#define instantiate_arange(tname, type)                                 \
  instantiate_kernel("arange" #tname, arange, type)

instantiate_arange(uint8, uint8_t)
instantiate_arange(uint16, uint16_t)
instantiate_arange(uint32, uint32_t)
instantiate_arange(uint64, uint64_t)
instantiate_arange(int8, int8_t)
instantiate_arange(int16, int16_t)
instantiate_arange(int32, int32_t)
instantiate_arange(int64, int64_t)
instantiate_arange(float16, half)
instantiate_arange(float32, float)
instantiate_arange(bfloat16, bfloat16_t) // clang-format on
