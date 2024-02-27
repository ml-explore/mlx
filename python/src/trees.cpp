// Copyright © 2023-2024 Apple Inc.

#include "python/src/trees.h"

void tree_visit(py::object tree, std::function<void(py::handle)> visitor) {
  std::function<void(py::handle)> recurse;
  recurse = [&](py::handle subtree) {
    if (py::isinstance<py::list>(subtree) ||
        py::isinstance<py::tuple>(subtree)) {
      for (auto item : subtree) {
        recurse(item);
      }
    } else if (py::isinstance<py::dict>(subtree)) {
      for (auto item : py::cast<py::dict>(subtree)) {
        recurse(item.second);
      }
    } else {
      visitor(subtree);
    }
  };

  recurse(tree);
}

template <typename T, typename U, typename V>
void validate_subtrees(const std::vector<py::object>& subtrees) {
  int len = py::cast<T>(subtrees[0]).size();
  for (auto& subtree : subtrees) {
    if ((py::isinstance<T>(subtree) && py::cast<T>(subtree).size() != len) ||
        py::isinstance<U>(subtree) || py::isinstance<V>(subtree)) {
      throw std::invalid_argument(
          "[tree_map] Additional input tree is not a valid prefix of the first tree.");
    }
  }
}

py::object tree_map(
    const std::vector<py::object>& trees,
    std::function<py::object(const std::vector<py::object>&)> transform) {
  std::function<py::object(const std::vector<py::object>&)> recurse;

  recurse = [&](const std::vector<py::object>& subtrees) {
    if (py::isinstance<py::list>(subtrees[0])) {
      py::list l;
      std::vector<py::object> items(subtrees.size());
      validate_subtrees<py::list, py::tuple, py::dict>(subtrees);
      for (int i = 0; i < py::cast<py::list>(subtrees[0]).size(); ++i) {
        for (int j = 0; j < subtrees.size(); ++j) {
          if (py::isinstance<py::list>(subtrees[j])) {
            items[j] = py::cast<py::list>(subtrees[j])[i];
          } else {
            items[j] = subtrees[j];
          }
        }
        l.append(recurse(items));
      }
      return py::cast<py::object>(l);
    } else if (py::isinstance<py::tuple>(subtrees[0])) {
      //  Check the rest of the subtrees
      std::vector<py::object> items(subtrees.size());
      int len = py::cast<py::tuple>(subtrees[0]).size();
      py::tuple l(len);
      validate_subtrees<py::tuple, py::list, py::dict>(subtrees);
      for (int i = 0; i < len; ++i) {
        for (int j = 0; j < subtrees.size(); ++j) {
          if (py::isinstance<py::tuple>(subtrees[j])) {
            items[j] = py::cast<py::tuple>(subtrees[j])[i];
          } else {
            items[j] = subtrees[j];
          }
        }
        l[i] = recurse(items);
      }
      return py::cast<py::object>(l);
    } else if (py::isinstance<py::dict>(subtrees[0])) {
      std::vector<py::object> items(subtrees.size());
      validate_subtrees<py::dict, py::list, py::tuple>(subtrees);
      py::dict d;
      for (auto item : py::cast<py::dict>(subtrees[0])) {
        for (int j = 0; j < subtrees.size(); ++j) {
          if (py::isinstance<py::dict>(subtrees[j])) {
            auto subdict = py::cast<py::dict>(subtrees[j]);
            if (!subdict.contains(item.first)) {
              throw std::invalid_argument(
                  "[tree_map] Tree is not a valid prefix tree of the first tree.");
            }
            items[j] = subdict[item.first];
          } else {
            items[j] = subtrees[j];
          }
        }
        d[item.first] = recurse(items);
      }
      return py::cast<py::object>(d);
    } else {
      return transform(subtrees);
    }
  };
  return recurse(trees);
}

py::object tree_map(
    py::object tree,
    std::function<py::object(py::handle)> transform) {
  return tree_map({tree}, [&](std::vector<py::object> inputs) {
    return transform(inputs[0]);
  });
}

void tree_visit_update(
    py::object tree,
    std::function<py::object(py::handle)> visitor) {
  std::function<py::object(py::handle)> recurse;
  recurse = [&](py::handle subtree) {
    if (py::isinstance<py::list>(subtree)) {
      auto l = py::cast<py::list>(subtree);
      for (int i = 0; i < l.size(); ++i) {
        l[i] = recurse(l[i]);
      }
      return py::cast<py::object>(l);
    } else if (py::isinstance<py::tuple>(subtree)) {
      for (auto item : subtree) {
        recurse(item);
      }
      return py::cast<py::object>(subtree);
    } else if (py::isinstance<py::dict>(subtree)) {
      auto d = py::cast<py::dict>(subtree);
      for (auto item : d) {
        d[item.first] = recurse(item.second);
      }
      return py::cast<py::object>(d);
    } else if (py::isinstance<array>(subtree)) {
      return visitor(subtree);
    } else {
      return py::cast<py::object>(subtree);
    }
  };
  recurse(tree);
}

// Fill a pytree (recursive dict or list of dict or list)
// in place with the given arrays
// Non dict or list nodes are ignored
void tree_fill(py::object& tree, const std::vector<array>& values) {
  size_t index = 0;
  tree_visit_update(
      tree, [&](py::handle node) { return py::cast(values[index++]); });
}

// Replace all the arrays from the src values with the dst values in the tree
void tree_replace(
    py::object& tree,
    const std::vector<array>& src,
    const std::vector<array>& dst) {
  std::unordered_map<uintptr_t, array> src_to_dst;
  for (int i = 0; i < src.size(); ++i) {
    src_to_dst.insert({src[i].id(), dst[i]});
  }
  tree_visit_update(tree, [&](py::handle node) {
    auto arr = py::cast<array>(node);
    if (auto it = src_to_dst.find(arr.id()); it != src_to_dst.end()) {
      return py::cast(it->second);
    }
    return py::cast(arr);
  });
}

std::vector<array> tree_flatten(py::object tree, bool strict /* = true */) {
  std::vector<array> flat_tree;

  tree_visit(tree, [&](py::handle obj) {
    if (py::isinstance<array>(obj)) {
      flat_tree.push_back(py::cast<array>(obj));
    } else if (strict) {
      throw std::invalid_argument(
          "[tree_flatten] The argument should contain only arrays");
    }
  });

  return flat_tree;
}

py::object tree_unflatten(
    py::object tree,
    const std::vector<array>& values,
    int index /* = 0 */) {
  return tree_map(tree, [&](py::handle obj) {
    if (py::isinstance<array>(obj)) {
      return py::cast(values[index++]);
    } else {
      return py::cast<py::object>(obj);
    }
  });
}

py::object structure_sentinel() {
  static py::object sentinel;

  if (sentinel.ptr() == nullptr) {
    sentinel = py::capsule(&sentinel);
    // probably not needed but this should make certain that we won't ever
    // delete the sentinel
    sentinel.inc_ref();
  }

  return sentinel;
}

std::pair<std::vector<array>, py::object> tree_flatten_with_structure(
    py::object tree,
    bool strict /* = true */) {
  auto sentinel = structure_sentinel();
  std::vector<array> flat_tree;
  auto structure = tree_map(
      tree,
      [&flat_tree, sentinel = std::move(sentinel), strict](py::handle obj) {
        if (py::isinstance<array>(obj)) {
          flat_tree.push_back(py::cast<array>(obj));
          return sentinel;
        } else if (!strict) {
          return py::cast<py::object>(obj);
        } else {
          throw std::invalid_argument(
              "[tree_flatten] The argument should contain only arrays");
        }
      });

  return {flat_tree, structure};
}

py::object tree_unflatten_from_structure(
    py::object structure,
    const std::vector<array>& values,
    int index /* = 0 */) {
  auto sentinel = structure_sentinel();
  return tree_map(structure, [&](py::handle obj) {
    if (obj.is(sentinel)) {
      return py::cast(values[index++]);
    } else {
      return py::cast<py::object>(obj);
    }
  });
}
