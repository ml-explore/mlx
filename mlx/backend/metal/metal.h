// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <string>
#include <unordered_map>
#include <variant>

namespace mlx::core::metal {

/* Check if the Metal backend is available. */
bool is_available();

/** Capture a GPU trace, saving it to an absolute file `path` */
void start_capture(std::string path = "");
void stop_capture();

/** Get information about the GPU and system settings. */
const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info();

} // namespace mlx::core::metal
