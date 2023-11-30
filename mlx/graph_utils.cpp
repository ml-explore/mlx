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
    std::function<void(OptionalArrayRef, const array&, int)> callback,
    const std::vector<array>& outputs) {
  std::function<void(OptionalArrayRef, const array&, int)> recurse;
  std::unordered_set<std::uintptr_t> cache;
  recurse = [&](OptionalArrayRef parent, const array& x, int input_index) {
    auto id = x.id();
    if (cache.find(id) != cache.end()) {
      return;
    }
    cache.insert(id);
    for (int i = 0; i < x.inputs().size(); i++) {
      recurse(x, x.inputs()[i], i);
    }
    callback(parent, x, input_index);
  };

  for (auto x : outputs) {
    recurse(std::nullopt, x, 0);
  }
}

void depth_first_traversal(
    std::function<void(const array&)> callback,
    const std::vector<array>& outputs) {
  depth_first_traversal(
      [&callback](OptionalArrayRef p, const array& x, int input_index) {
        callback(x);
      },
      outputs);
}

void print_graph(std::ostream& os, const std::vector<array>& outputs) {
  std::vector<array> tape;
  std::vector<array> inputs;

  depth_first_traversal(
      [&](const array& x) {
        if (x.has_primitive()) {
          tape.push_back(x);
        } else {
          inputs.push_back(x);
        }
      },
      outputs);

  ArrayNames namer;
  auto print_arr = [&namer, &os](const array& a) {
    os << namer.get_name(a);
    os << " [" << a.shape() << ", " << a.dtype() << "]";
  };

  auto print_arrs = [&](const std::vector<array>& arrs) {
    for (auto& arr : arrs) {
      print_arr(arr);
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
    print_arr(arr);
    os << "\n";
  }
}

void export_to_dot(std::ostream& os, const std::vector<array>& outputs) {
  os << "digraph {" << std::endl;

  ArrayNames namer;
  depth_first_traversal(
      [&namer, &os](auto parent, const array& x, int input_index) {
        os << "{ ";
        if (!x.has_primitive()) {
          os << "rank=source; ";
        }
        if (!parent) {
          os << "rank=sink; ";
        }
        os << namer.get_name(x);
        if (x.has_primitive()) {
          os << " [label =\"";
          x.primitive().print(os);
          os << "\"]";
        }
        os << "; }" << std::endl;

        for (auto c : x.inputs()) {
          os << namer.get_name(c) << " -> " << namer.get_name(x) << std::endl;
        }
      },
      outputs);

  os << "}";
}

} // namespace mlx::core
