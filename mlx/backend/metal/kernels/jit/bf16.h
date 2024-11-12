// Copyright © 2024 Apple Inc.

// clang-format off
#define jit_if #if
#define jit_else #else
#define jit_endif #endif

jit_if (__METAL_VERSION__ >= 310)

#include "mlx/backend/metal/kernels/metal_3_1/bf16.h"

jit_else

#include "mlx/backend/metal/kernels/metal_3_0/bf16.h"

jit_endif // clang-format on
