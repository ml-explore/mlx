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
// Use the most advanced SIMD available (only one active at a time)
#if !defined(MLX_USE_ACCELERATE)
  #if defined(__AVX2__)
    #include "mlx/backend/cpu/simd/avx_simd.h"
  #elif defined(__SSE4_2__)
    #include "mlx/backend/cpu/simd/sse_simd.h"
  #endif
#endif
