// Copyright © 2026 Apple Inc.

#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/sort.h"
#include "mlx/backend/metal/kernels/searchsorted.h"

#define instantiate_searchsorted_side(tname, type, sname, side)      \
  instantiate_kernel("searchsorted_v_" #tname "_" #sname,            \
                     searchsorted_v, type, side)                     \
  instantiate_kernel("searchsorted_g_" #tname "_" #sname,            \
                     searchsorted_g, type, side)

#define instantiate_searchsorted(tname, type)             \
  instantiate_searchsorted_side(tname, type, left, false) \
  instantiate_searchsorted_side(tname, type, right, true)

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
