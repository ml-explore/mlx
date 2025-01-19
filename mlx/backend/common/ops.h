// Copyright © 2023-2024 Apple Inc.

#pragma once

#ifdef MLX_BUILD_ACCELERATE
#include "mlx/backend/accelerate/unary_ops.h"
#else
#include "mlx/backend/common/unary_ops.h"
#endif

#include "mlx/backend/common/binary_ops.h"
