#pragma once

#include "mlx/backend/cpu/simd/base_simd.h"

#ifdef MLX_USE_ACCELERATE
#if defined(__x86_64__)
// the accelerate_simd implementation require neon -- use base implementation
#else
#include "mlx/backend/cpu/simd/accelerate_simd.h"
#endif
#endif

// x86 SIMD: AVX2 implementation
#if !defined(MLX_USE_ACCELERATE)
#if defined(__AVX2__)
#include "mlx/backend/cpu/simd/avx_simd.h"
#endif
#endif
