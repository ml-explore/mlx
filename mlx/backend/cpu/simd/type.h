#pragma once

#include "mlx/backend/cpu/simd/base_simd.h"

#if defined(MLX_USE_HIGHWAY_KERNELS) || defined(MLX_USE_HIGHWAY_JIT_SIMD)
#include "mlx/backend/cpu/simd/highway_simd.h"
#endif

#ifdef MLX_USE_ACCELERATE
#if defined(__x86_64__)
// the accelerate_simd implementation require neon -- use base implementation
#else
#include "mlx/backend/cpu/simd/accelerate_simd.h"
#endif
#endif
