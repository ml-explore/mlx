#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "array.h"

namespace mlx::core {
std::vector<std::tuple<
    std::vector<int>,
    std::set<char>,
    std::string,
    std::vector<std::string>,
    bool>>
einsum_path(const std::string& equation, const std::vector<array>& operands);
}