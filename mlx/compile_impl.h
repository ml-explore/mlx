// Copyright © 2023-2024 Apple Inc.

#pragma once

#include "mlx/device.h"

namespace mlx::core::detail {

// This is not part of the general C++ API as calling with a bad id is a bad
// idea.
std::function<std::vector<array>(const std::vector<array>&)> compile(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    std::uintptr_t fun_id,
    bool shapeless = false,
    std::vector<uint64_t> constants = {});

// Erase cached compile functions
void compile_erase(std::uintptr_t fun_id);

// Clear the compiler cache causing a recompilation of all compiled functions
// when called again.
void compile_clear_cache();

bool compile_available_for_device(const Device& device);
} // namespace mlx::core::detail
