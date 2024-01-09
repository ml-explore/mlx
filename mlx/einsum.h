// Copyright © 2023 Apple Inc.
#pragma once

#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "mlx/array.h"

namespace mlx::core {

struct EinsumPath {
  std::vector<int> args;
  std::set<char> removing;
  std::string einsum_str;
  bool can_dot;
};

std::vector<EinsumPath> einsum_path(
    const std::string& equation,
    const std::vector<array>& operands);

std::pair<std::vector<std::string>, std::string> einsum_parse(
    const std::string& equation);

} // namespace mlx::core