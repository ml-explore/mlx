// Copyright © 2023-2024 Apple Inc.

#pragma once

#include "mlx/backend/metal/device.h"

namespace mlx::core::metal {

MTL::ResidencySet* setup_residency_set(MTL::Device* d);

} // namespace mlx::core::metal
