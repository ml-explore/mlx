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

const std::string& NodeNamer::get_name(const array& x) {
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
    names.emplace(x.id(), std::string(letters.rbegin(), letters.rend()));

    return get_name(x);
  }
  return it->second;
}

void NodeNamer::set_name(const array& x, std::string n) {
  names[x.id()] = std::move(n);
}

void depth_first_traversal(
    std::function<void(array)> callback,
    const std::vector<array>& outputs) {
  std::function<void(const array&)> recurse;
  std::unordered_set<std::uintptr_t> cache;
  recurse = [&](const array& x) {
    auto id = x.id();
    if (cache.find(id) != cache.end()) {
      return;
    }
    cache.insert(id);
    for (auto& s : x.siblings()) {
      cache.insert(s.id());
    }
    for (auto& in : x.inputs()) {
      recurse(in);
    }
    callback(x);
  };

  for (auto& o : outputs) {
    recurse(o);
  }
}

void print_graph(
    std::ostream& os,
    NodeNamer namer,
    const std::vector<array>& outputs) {
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

  auto print_arrs = [&namer, &os](std::vector<array> arrs) {
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

void export_to_dot(
    std::ostream& os,
    NodeNamer namer,
    const std::vector<array>& nodes) {
  // Perform one DFS to mark arrays as intermediate if they are used as inputs
  // to other arrays.
  std::unordered_set<std::uintptr_t> intermediate_set;
  depth_first_traversal(
      [&](const array& x) {
        // No primitive so it is an input
        if (!x.has_primitive()) {
          return;
        }

        for (auto& a : x.inputs()) {
          intermediate_set.insert(a.id());
        }
      },
      nodes);

  // Now we got everything we need to make the graph. Arrays can be one of 3
  // things:
  //  1. Inputs, when they have no primitive ie are evaluated
  //  2. Intermediates, when they are the intermediate set
  //  3. Outputs, if they are not inputs and not intermediates

  os << "digraph {" << std::endl;

  depth_first_traversal(
      [&](const array& x) {
        if (!x.has_primitive()) {
          os << "{ rank=source; \"" << namer.get_name(x) << "\"; }"
             << std::endl;
          return;
        }

        // Node for primitive
        if (x.has_primitive()) {
          os << "{ ";
          os << x.primitive_id();
          os << " [label =\"";
          x.primitive().print(os);
          os << "\", shape=rectangle]";
          os << "; }" << std::endl;
          // Arrows to primitive's inputs
          for (auto& a : x.inputs()) {
            os << '"' << namer.get_name(a) << "\" -> " << x.primitive_id()
               << std::endl;
          }
        }

        // Point outputs to their primitive
        for (auto& a : x.outputs()) {
          os << "{ ";
          if (intermediate_set.find(a.id()) == intermediate_set.end()) {
            os << "rank=sink; ";
          }
          os << '"' << namer.get_name(a);
          os << "\"; }" << std::endl;
          if (x.has_primitive()) {
            os << x.primitive_id() << " -> \"" << namer.get_name(a) << '"'
               << std::endl;
          }
        }
      },
      nodes);

  os << "}";
}

} // namespace mlx::core
