#pragma once

#include "mlx/backend/cpu/simd/base_simd.h"

#ifdef MLX_USE_ACCELERATE
#if defined(__x86_64__)
// the accelerate_simd implementation require neon -- use base implementation
#else
#include "mlx/backend/cpu/simd/accelerate_simd.h"
#endif
#endif

// x86 SIMD implementations
#if defined(__SSE4_2__) && !defined(MLX_USE_ACCELERATE)
#include "mlx/backend/cpu/simd/sse_simd.h"
#endif
