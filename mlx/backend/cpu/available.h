// Copyright © 2025 Apple Inc.

#pragma once

#include <string>
#include <unordered_map>
#include <variant>

namespace mlx::core::cpu {

bool is_available();

/**
 * Get the number of available CPU devices.
 *
 * For CPU, always returns 1.
 */
int device_count();

/**
 * Get CPU device information.
 *
 * Returns a map with basic CPU device properties.
 */
const std::unordered_map<std::string, std::variant<std::string, size_t>>& device_info(int device_index = 0);

} // namespace mlx::core::cpu
