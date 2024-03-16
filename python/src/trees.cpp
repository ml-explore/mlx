// Copyright © 2023-2024 Apple Inc.

#include "python/src/trees.h"

void tree_visit(nb::object tree, std::function<void(nb::handle)> visitor) {
  std::function<void(nb::handle)> recurse;
  recurse = [&](nb::handle subtree) {
    if (nb::isinstance<nb::list>(subtree) ||
        nb::isinstance<nb::tuple>(subtree)) {
      for (auto item : subtree) {
        recurse(item);
      }
    } else if (nb::isinstance<nb::dict>(subtree)) {
      for (auto item : nb::cast<nb::dict>(subtree)) {
        recurse(item.second);
      }
    } else {
      visitor(subtree);
    }
  };

  recurse(tree);
}

template <typename T, typename U, typename V>
void validate_subtrees(const std::vector<nb::object>& subtrees) {
  int len = nb::cast<T>(subtrees[0]).size();
  for (auto& subtree : subtrees) {
    if ((nb::isinstance<T>(subtree) && nb::cast<T>(subtree).size() != len) ||
        nb::isinstance<U>(subtree) || nb::isinstance<V>(subtree)) {
      throw std::invalid_argument(
          "[tree_map] Additional input tree is not a valid prefix of the first tree.");
    }
  }
}

nb::object tree_map(
    const std::vector<nb::object>& trees,
    std::function<nb::object(const std::vector<nb::object>&)> transform) {
  std::function<nb::object(const std::vector<nb::object>&)> recurse;

  recurse = [&](const std::vector<nb::object>& subtrees) {
    if (nb::isinstance<nb::list>(subtrees[0])) {
      nb::list l;
      std::vector<nb::object> items(subtrees.size());
      validate_subtrees<nb::list, nb::tuple, nb::dict>(subtrees);
      for (int i = 0; i < nb::cast<nb::list>(subtrees[0]).size(); ++i) {
        for (int j = 0; j < subtrees.size(); ++j) {
          if (nb::isinstance<nb::list>(subtrees[j])) {
            items[j] = nb::cast<nb::list>(subtrees[j])[i];
          } else {
            items[j] = subtrees[j];
          }
        }
        l.append(recurse(items));
      }
      return nb::cast<nb::object>(l);
    } else if (nb::isinstance<nb::tuple>(subtrees[0])) {
      //  Check the rest of the subtrees
      std::vector<nb::object> items(subtrees.size());
      int len = nb::cast<nb::tuple>(subtrees[0]).size();
      nb::list l;
      validate_subtrees<nb::tuple, nb::list, nb::dict>(subtrees);
      for (int i = 0; i < len; ++i) {
        for (int j = 0; j < subtrees.size(); ++j) {
          if (nb::isinstance<nb::tuple>(subtrees[j])) {
            items[j] = nb::cast<nb::tuple>(subtrees[j])[i];
          } else {
            items[j] = subtrees[j];
          }
        }
        l.append(recurse(items));
      }
      return nb::cast<nb::object>(nb::tuple(l));
    } else if (nb::isinstance<nb::dict>(subtrees[0])) {
      std::vector<nb::object> items(subtrees.size());
      validate_subtrees<nb::dict, nb::list, nb::tuple>(subtrees);
      nb::dict d;
      for (auto item : nb::cast<nb::dict>(subtrees[0])) {
        for (int j = 0; j < subtrees.size(); ++j) {
          if (nb::isinstance<nb::dict>(subtrees[j])) {
            auto subdict = nb::cast<nb::dict>(subtrees[j]);
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
      return nb::cast<nb::object>(d);
    } else {
      return transform(subtrees);
    }
  };
  return recurse(trees);
}

nb::object tree_map(
    nb::object tree,
    std::function<nb::object(nb::handle)> transform) {
  return tree_map({tree}, [&](std::vector<nb::object> inputs) {
    return transform(inputs[0]);
  });
}

void tree_visit_update(
    nb::object tree,
    std::function<nb::object(nb::handle)> visitor) {
  std::function<nb::object(nb::handle)> recurse;
  recurse = [&](nb::handle subtree) {
    if (nb::isinstance<nb::list>(subtree)) {
      auto l = nb::cast<nb::list>(subtree);
      for (int i = 0; i < l.size(); ++i) {
        l[i] = recurse(l[i]);
      }
      return nb::cast<nb::object>(l);
    } else if (nb::isinstance<nb::tuple>(subtree)) {
      for (auto item : subtree) {
        recurse(item);
      }
      return nb::cast<nb::object>(subtree);
    } else if (nb::isinstance<nb::dict>(subtree)) {
      auto d = nb::cast<nb::dict>(subtree);
      for (auto item : d) {
        d[item.first] = recurse(item.second);
      }
      return nb::cast<nb::object>(d);
    } else if (nb::isinstance<array>(subtree)) {
      return visitor(subtree);
    } else {
      return nb::cast<nb::object>(subtree);
    }
  };
  recurse(tree);
}

// Fill a pytree (recursive dict or list of dict or list)
// in place with the given arrays
// Non dict or list nodes are ignored
void tree_fill(nb::object& tree, const std::vector<array>& values) {
  size_t index = 0;
  tree_visit_update(
      tree, [&](nb::handle node) { return nb::cast(values[index++]); });
}

// Replace all the arrays from the src values with the dst values in the tree
void tree_replace(
    nb::object& tree,
    const std::vector<array>& src,
    const std::vector<array>& dst) {
  std::unordered_map<uintptr_t, array> src_to_dst;
  for (int i = 0; i < src.size(); ++i) {
    src_to_dst.insert({src[i].id(), dst[i]});
  }
  tree_visit_update(tree, [&](nb::handle node) {
    auto arr = nb::cast<array>(node);
    if (auto it = src_to_dst.find(arr.id()); it != src_to_dst.end()) {
      return nb::cast(it->second);
    }
    return nb::cast(arr);
  });
}

std::vector<array> tree_flatten(nb::object tree, bool strict /* = true */) {
  std::vector<array> flat_tree;

  tree_visit(tree, [&](nb::handle obj) {
    if (nb::isinstance<array>(obj)) {
      flat_tree.push_back(nb::cast<array>(obj));
    } else if (strict) {
      throw std::invalid_argument(
          "[tree_flatten] The argument should contain only arrays");
    }
  });

  return flat_tree;
}

nb::object tree_unflatten(
    nb::object tree,
    const std::vector<array>& values,
    int index /* = 0 */) {
  return tree_map(tree, [&](nb::handle obj) {
    if (nb::isinstance<array>(obj)) {
      return nb::cast(values[index++]);
    } else {
      return nb::cast<nb::object>(obj);
    }
  });
}

nb::object structure_sentinel() {
  static nb::object sentinel;

  if (sentinel.ptr() == nullptr) {
    sentinel = nb::capsule(&sentinel);
    // probably not needed but this should make certain that we won't ever
    // delete the sentinel
    sentinel.inc_ref();
  }

  return sentinel;
}

std::pair<std::vector<array>, nb::object> tree_flatten_with_structure(
    nb::object tree,
    bool strict /* = true */) {
  auto sentinel = structure_sentinel();
  std::vector<array> flat_tree;
  auto structure = tree_map(
      tree,
      [&flat_tree, sentinel = std::move(sentinel), strict](nb::handle obj) {
        if (nb::isinstance<array>(obj)) {
          flat_tree.push_back(nb::cast<array>(obj));
          return sentinel;
        } else if (!strict) {
          return nb::cast<nb::object>(obj);
        } else {
          throw std::invalid_argument(
              "[tree_flatten] The argument should contain only arrays");
        }
      });

  return {flat_tree, structure};
}

nb::object tree_unflatten_from_structure(
    nb::object structure,
    const std::vector<array>& values,
    int index /* = 0 */) {
  auto sentinel = structure_sentinel();
  return tree_map(structure, [&](nb::handle obj) {
    if (obj.is(sentinel)) {
      return nb::cast(values[index++]);
    } else {
      return nb::cast<nb::object>(obj);
    }
  });
}
