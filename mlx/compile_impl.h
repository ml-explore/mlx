// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <unordered_map>

#include "mlx/array.h"

namespace mlx::core::detail {

// This is not part of the general C++ API as calling with a bad id is a bad
// idea.
std::function<std::vector<array>(const std::vector<array>&)> compile(
    std::function<std::vector<array>(const std::vector<array>&)> fun,
    std::uintptr_t fun_id,
    bool shapeless = false,
    std::vector<uint64_t> constants = {});

// Erase cached compile functions
void compile_erase(std::uintptr_t fun_id);

// Clear the compiler cache causing a recompilation of all compiled functions
// when called again.
void compile_clear_cache();

bool compile_available_for_device(const Device& device);

std::pair<std::vector<array>, std::vector<array>> compile_trace(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<array>& inputs,
    bool shapeless);

using ParentsMap =
    std::unordered_map<std::uintptr_t, std::vector<std::pair<array, int>>>;

// Traverses the graph to build a tape and a map of array ids to their parents
std::pair<std::vector<array>, ParentsMap> compile_dfs(
    const std::vector<array>& inputs,
    const std::vector<array>& outputs,
    const std::vector<array>& original_inputs);

// Simplify the tape.
void compile_simplify(
    std::vector<array>& tape,
    ParentsMap& parents_map,
    std::vector<array>& outputs,
    int passes);

std::vector<array> compile_replace(
    const std::vector<array>& tape,
    const std::vector<array>& trace_inputs,
    const std::vector<array>& trace_outputs,
    const std::vector<array>& inputs,
    bool shapeless);

void compile_validate_shapeless(const std::vector<array>& tape);

} // namespace mlx::core::detail
