// Copyright © 2024 Apple Inc.
#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "mlx/array.h"
#include "mlx/utils.h"

namespace mlx::core {

std::pair<std::vector<std::vector<int>>, std::string> einsum_path(
    const std::string& subscripts,
    const std::vector<array>& operands);

array einsum(
    const std::string& subscripts,
    const std::vector<array>& operands,
    StreamOrDevice s = {});

} // namespace mlx::core
