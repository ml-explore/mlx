// Copyright © 2025 Apple Inc.

#pragma once

#define MLX_UNROLL _Pragma("unroll")

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define MLX_CUDA_SM_80_ENABLED
#endif
