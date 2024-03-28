// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/array.h"

namespace mlx::core {

struct NodeNamer {
  std::unordered_map<std::uintptr_t, std::string> names;

  const std::string& get_name(const array& x);
};

void print_graph(std::ostream& os, const std::vector<array>& outputs);

template <typename... Arrays, typename = enable_for_arrays_t<Arrays...>>
void print_graph(std::ostream& os, Arrays&&... outputs) {
  print_graph(os, std::vector<array>{std::forward<Arrays>(outputs)...});
}

void export_to_dot(std::ostream& os, const std::vector<array>& outputs);

template <typename... Arrays, typename = enable_for_arrays_t<Arrays...>>
void export_to_dot(std::ostream& os, Arrays&&... outputs) {
  export_to_dot(os, std::vector<array>{std::forward<Arrays>(outputs)...});
}

} // namespace mlx::core
