// Copyright © 2023 Apple Inc.

#include <functional>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "mlx/graph_utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

using OptionalArrayRef = std::optional<std::reference_wrapper<const array>>;

struct ArrayNames {
  std::unordered_map<std::uintptr_t, std::string> names;

  std::string get_name(const array& x) {
    auto it = names.find(x.id());
    if (it == names.end()) {
      // Get the next name in the sequence
      // [A, B, ..., Z, AA, AB, ...]
      std::vector<char> letters;
      auto var_num = names.size() + 1;
      while (var_num > 0) {
        letters.push_back('A' + (var_num - 1) % 26);
        var_num = (var_num - 1) / 26;
      }
      std::string name(letters.rbegin(), letters.rend());
      names.insert({x.id(), name});
      return name;
    }
    return it->second;
  }
};

void depth_first_traversal(
    std::function<void(GraphNode)> callback,
    const std::vector<array>& outputs) {
  std::function<void(const GraphNode&)> recurse;
  std::unordered_set<std::uintptr_t> cache;
  recurse = [&](const GraphNode& x) {
    auto id = x.id();
    if (cache.find(id) != cache.end()) {
      return;
    }
    cache.insert(id);
    for (auto& in : x.inputs()) {
      recurse(in.graph_node());
    }
    callback(x);
  };

  for (auto& o : outputs) {
    recurse(o.graph_node());
  }
}

void print_graph(std::ostream& os, const std::vector<array>& outputs) {
  std::vector<GraphNode> tape;
  std::vector<array> inputs;

  depth_first_traversal(
      [&](const GraphNode& x) {
        if (x.has_primitive()) {
          tape.push_back(x);
        } else {
          inputs.insert(inputs.end(), x.inputs().begin(), x.inputs().end());
        }
      },
      outputs);

  ArrayNames namer;
  auto print_arrs = [&namer, &os](const std::vector<array>& arrs) {
    for (auto& arr : arrs) {
      os << namer.get_name(arr);
      os << " [" << arr.shape() << ", " << arr.dtype() << "]";
      if (&arr != &arrs.back()) {
        os << ", ";
      }
    }
  };

  os << "Inputs: ";
  print_arrs(inputs);
  os << "\nOutputs: ";
  print_arrs(outputs);
  os << "\n";

  for (auto& arr : tape) {
    arr.primitive().print(os);
    os << " ";
    print_arrs(arr.inputs());
    os << " -> ";
    print_arrs(arr.outputs());
    os << "\n";
  }
}

void export_to_dot(std::ostream& os, const std::vector<array>& outputs) {
  os << "digraph {" << std::endl;

  std::unordered_set<std::uintptr_t> output_set;
  for (auto& o : outputs) {
    output_set.insert(o.id());
  }
  std::unordered_set<std::uintptr_t> input_set;
  ArrayNames namer;
  depth_first_traversal(
      [&](const GraphNode& x) {
        for (auto& a : x.inputs()) {
          // Record inputs
          if (!a.has_primitive() && input_set.find(a.id()) != input_set.end()) {
            input_set.insert(a.id());
            os << "{ rank=source; " << namer.get_name(a) << "; }" << std::endl;
          }
        }
        for (auto& a : x.outputs()) {
          os << "{ ";
          if (output_set.find(a.id()) != output_set.end()) {
            os << "rank=sink; ";
          }
          os << namer.get_name(a);
          os << " [label =\"";
          x.primitive().print(os);
          os << "\"]";
          os << "; }" << std::endl;
          for (auto c : x.inputs()) {
            os << namer.get_name(c) << " -> " << namer.get_name(a) << std::endl;
          }
        }
      },
      outputs);

  os << "}";
}

} // namespace mlx::core
