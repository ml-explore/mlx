// Copyright © 2024 Apple Inc.

#pragma once

#include <metal_atomic>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/steel/utils.h"
#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/reduction/ops.h"

static constant constexpr const uint8_t simd_size = 32;