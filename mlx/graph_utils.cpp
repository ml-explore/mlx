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
    std::string name(letters.rbegin(), letters.rend());
    names.insert({x.id(), name});

    return get_name(x);
  }
  return it->second;
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

  NodeNamer namer;
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

void export_to_dot(std::ostream& os, const std::vector<array>& outputs) {
  os << "digraph {" << std::endl;

  std::unordered_set<std::uintptr_t> output_set;
  for (auto& o : outputs) {
    output_set.insert(o.id());
  }
  std::unordered_set<std::uintptr_t> input_set;
  NodeNamer namer;
  depth_first_traversal(
      [&](const array& x) {
        if (!x.has_primitive()) {
          input_set.insert(x.id());
          os << "{ rank=source; " << namer.get_name(x) << "; }" << std::endl;
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
            os << namer.get_name(a) << " -> " << x.primitive_id() << std::endl;
          }
        }

        // Point outputs to their primitive
        for (auto& a : x.outputs()) {
          os << "{ ";
          if (output_set.find(a.id()) != output_set.end()) {
            os << "rank=sink; ";
          }
          os << namer.get_name(a);
          os << "; }" << std::endl;
          if (x.has_primitive()) {
            os << x.primitive_id() << " -> " << namer.get_name(a) << std::endl;
          }
        }
      },
      outputs);

  os << "}";
}

void generate_kernel(std::ostream& os, const std::vector<array>& outputs) {
  auto get_type_string = [](Dtype d) {
    if (d == float32) {
      return "float";
    } else {
      throw std::runtime_error("Unsupported type");
    }
  };

  std::unordered_set<std::uintptr_t> input_set;
  std::vector<array> inputs;
  NodeNamer namer;

  // Collect inputs
  depth_first_traversal(
      [&](const array& x) {
        if (!x.has_primitive()) {
          if (auto [__, inserted] = input_set.insert(x.id()); inserted) {
            inputs.push_back(x);
          }
        }
      },
      outputs);

  // Start the kernel
  os << "[[kernel]] void some_generated_kernel_name(" << std::endl;
  for (auto& x : inputs) {
    os << "    device const " << get_type_string(x.dtype()) << "* "
       << namer.get_name(x) << "," << std::endl;
  }
  for (auto& x : outputs) {
    os << "    device " << get_type_string(x.dtype()) << "* "
       << namer.get_name(x) << "," << std::endl;
  }
  os << "    uint index [[thread_position_in_grid]]) {" << std::endl;

  // Read the inputs in tmps
  for (auto& x : inputs) {
    os << "  " << get_type_string(x.dtype()) << " tmp_" << namer.get_name(x)
       << " = " << namer.get_name(x) << "[index];" << std::endl;
  }

  // Actually write the computation
  depth_first_traversal(
      [&](const array& x) {
        if (!x.has_primitive()) {
          return;
        }

        os << "  " << get_type_string(x.dtype()) << " tmp_" << namer.get_name(x)
           << " = ";
        x.primitive().print(os);
        os << "()(";
        for (int i = 0; i < x.inputs().size() - 1; i++) {
          os << "tmp_" << namer.get_name(x.inputs()[i]) << ", ";
        }
        os << "tmp_" << namer.get_name(x.inputs().back()) << ");" << std::endl;
      },
      outputs);

  // Write the outputs from tmps
  for (auto& x : outputs) {
    os << "  " << namer.get_name(x) << "[index] = tmp_" << namer.get_name(x)
       << ";" << std::endl;
  }

  // Finish the kernel
  os << "}" << std::endl;
}

} // namespace mlx::core
