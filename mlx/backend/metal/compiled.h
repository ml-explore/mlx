// Copyright © 2026 Apple Inc.

#pragma once

#include <functional>
#include <string>
#include <unordered_set>

#include "mlx/array.h"
#include "mlx/graph_utils.h"

namespace mlx::core {

// Write the computation of the tape into tmp_* variables.
void build_tape(
    std::string& os,
    const std::vector<array>& tape,
    NodeNamer& namer);

// Evaluate a Compiled primitive whose tape ends with a Reduce.
void compiled_reduce(
    const Stream& s,
    const std::string& lib_name,
    const std::vector<array>& trace_inputs,
    const std::vector<array>& tape,
    const std::unordered_set<uintptr_t>& constant_ids,
    const std::function<bool(size_t)>& is_constant,
    const std::vector<array>& inputs,
    array& out);

} // namespace mlx::core
