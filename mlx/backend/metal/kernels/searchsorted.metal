// Copyright © 2026 Apple Inc.

#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/sort.h"
#include "mlx/backend/metal/kernels/searchsorted.h"

#define instantiate_searchsorted(tname, type)         \
  instantiate_kernel("searchsorted_" #tname "_left",  \
                     searchsorted, type, false)       \
  instantiate_kernel("searchsorted_" #tname "_right", \
                     searchsorted, type, true)

instantiate_searchsorted(bool_, bool)
instantiate_searchsorted(uint8, uint8_t)
instantiate_searchsorted(uint16, uint16_t)
instantiate_searchsorted(uint32, uint32_t)
instantiate_searchsorted(uint64, uint64_t)
instantiate_searchsorted(int8, int8_t)
instantiate_searchsorted(int16, int16_t)
instantiate_searchsorted(int32, int32_t)
instantiate_searchsorted(int64, int64_t)
instantiate_searchsorted(float16, half)
instantiate_searchsorted(float32, float)
instantiate_searchsorted(bfloat16, bfloat16_t)
instantiate_searchsorted(complex64, complex64_t) // clang-format on
