// Copyright © 2023 Apple Inc.

#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "mlx/array.h"

namespace mlx::core {
std::vector<std::tuple<
    std::vector<int>,
    std::set<char>,
    std::string,
    std::vector<std::string>,
    bool>>
einsum_path(const std::string& equation, const std::vector<array>& operands);

std::pair<std::vector<std::string>, std::string> einsum_parse(
    const std::string& equation);

std::map<char, int> str_idx_map(const std::string inp);
} // namespace mlx::core