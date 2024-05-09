// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/metal/kernels/atomic.h"
#include "mlx/backend/metal/kernels/binary.h"
#include "mlx/backend/metal/kernels/indexing.h"
#include "mlx/backend/metal/kernels/reduction/ops.h"
#include "mlx/backend/metal/kernels/ternary.h"
#include "mlx/backend/metal/kernels/unary.h"

typedef half float16_t;
