#pragma once

#include "mlx/backend/cpu/simd/base_simd.h"

#ifdef MLX_USE_ACCELERATE
#if defined(__x86_64__)
// the accelerate_simd implementation require neon -- use base implementation
#else
#include "mlx/backend/cpu/simd/accelerate_simd.h"
#endif
#endif

// x86 SIMD: MLX_ENABLE_AVX2 uses Highway as the implementation substrate.
#if !defined(MLX_USE_ACCELERATE)
#if defined(MLX_USE_HIGHWAY)
#include "mlx/backend/cpu/simd/highway_simd.h"
#endif
#endif
