// Copyright © 2026 Apple Inc.
#pragma once

#include "mlx/backend/cuda/device.h"

#include <cusolverDn.h>

namespace mlx::core {

void check_cusolver_error(const char* name, cusolverStatus_t err);

#define CHECK_CUSOLVER_ERROR(cmd) check_cusolver_error(#cmd, (cmd))

void init_cusolver_handles_cache();

cusolverDnHandle_t get_cusolver_handle(cu::Device& device);

} // namespace mlx::core
