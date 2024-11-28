// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/array.h"

namespace mlx::core {

/**
 * Export a function to a file.
 */
void export_function(
    std::string path,
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<array>& inputs,
    bool shapeless = false);

/**
 * Import a function from a file.
 */
std::function<std::vector<array>(const std::vector<array>&)> import_function(
    std::string path);

} // namespace mlx::core
