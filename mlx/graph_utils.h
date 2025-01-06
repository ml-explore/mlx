// Copyright © 2023 Apple Inc.

#pragma once

#include <unordered_map>

#include "mlx/array.h"

namespace mlx::core {

struct NodeNamer {
  std::unordered_map<std::uintptr_t, std::string> names;

  const std::string& get_name(const array& x);
  void set_name(const array& x, std::string n);
};

void print_graph(
    std::ostream& os,
    NodeNamer namer,
    const std::vector<array>& outputs);

inline void print_graph(std::ostream& os, const std::vector<array>& outputs) {
  print_graph(os, NodeNamer{}, outputs);
}

template <typename... Arrays, typename = enable_for_arrays_t<Arrays...>>
inline void print_graph(std::ostream& os, Arrays&&... outputs) {
  print_graph(
      os, NodeNamer{}, std::vector<array>{std::forward<Arrays>(outputs)...});
}

template <typename... Arrays, typename = enable_for_arrays_t<Arrays...>>
inline void
print_graph(std::ostream& os, NodeNamer namer, Arrays&&... outputs) {
  print_graph(
      os,
      std::move(namer),
      std::vector<array>{std::forward<Arrays>(outputs)...});
}

void export_to_dot(
    std::ostream& os,
    NodeNamer namer,
    const std::vector<array>& outputs);

inline void export_to_dot(std::ostream& os, const std::vector<array>& outputs) {
  export_to_dot(os, NodeNamer{}, outputs);
}

template <typename... Arrays, typename = enable_for_arrays_t<Arrays...>>
inline void export_to_dot(std::ostream& os, Arrays&&... outputs) {
  export_to_dot(
      os, NodeNamer{}, std::vector<array>{std::forward<Arrays>(outputs)...});
}

template <typename... Arrays, typename = enable_for_arrays_t<Arrays...>>
inline void
export_to_dot(std::ostream& os, NodeNamer namer, Arrays&&... outputs) {
  export_to_dot(
      os,
      std::move(namer),
      std::vector<array>{std::forward<Arrays>(outputs)...});
}

} // namespace mlx::core
