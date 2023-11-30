// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/array.h"

namespace mlx::core {

void print_graph(std::ostream& os, const std::vector<array>& outputs);

template <typename... Arrays>
void print_graph(std::ostream& os, Arrays... outputs) {
  print_graph(os, std::vector<array>{std::forward<Arrays>(outputs)...});
}

void export_to_dot(std::ostream& os, const std::vector<array>& outputs);

template <typename... Arrays>
void export_to_dot(std::ostream& os, Arrays... outputs) {
  export_to_dot(os, std::vector<array>{std::forward<Arrays>(outputs)...});
}

} // namespace mlx::core
