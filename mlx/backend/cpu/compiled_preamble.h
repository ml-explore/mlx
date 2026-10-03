// Copyright © 2023-2026 Apple Inc.

#pragma once

// clang-format off
#include "mlx/types/half_types.h"
#include "mlx/types/complex.h"
#include "mlx/backend/cpu/unary_ops.h"
#include "mlx/backend/cpu/binary_ops.h"
// clang-format on

const char* get_prebuilt_preamble();

#ifdef MLX_HAVE_JIT_PREAMBLE_HWY_AVX2
const char* get_prebuilt_preamble_avx2();
#endif
#ifdef MLX_HAVE_JIT_PREAMBLE_HWY_AVX3
const char* get_prebuilt_preamble_avx3();
#endif
