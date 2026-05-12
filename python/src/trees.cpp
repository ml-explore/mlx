// Copyright © 2023-2024 Apple Inc.

#include <unordered_map>

#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include "python/src/trees.h"

namespace {

struct PytreeNodeDef {
  nb::callable flatten_fn;
  nb::callable unflatten_fn;
};

// The map is intentionally heap-allocated and never freed.  Holding
// nb::callable references in a function-local static triggers a
// use-after-finalize when the C++ runtime tears down the static during
// interpreter shutdown — the Python state is already gone, so decrefing the
// stored callables segfaults.  This is the same lifetime trick used by
// structure_sentinel() below.
std::unordered_map<PyObject*, PytreeNodeDef>& registry() {
  static auto* r = new std::unordered_map<PyObject*, PytreeNodeDef>();
  return *r;
}

// Calls the registered flatten_fn for obj and returns (children, aux).
std::pair<std::vector<nb::object>, nb::object> flatten_registered(
    nb::handle obj) {
  auto& def = registry().at(reinterpret_cast<PyObject*>(Py_TYPE(obj.ptr())));
  auto seq = nb::cast<nb::sequence>(def.flatten_fn(obj));
  if (nb::len(seq) != 2) {
    throw std::invalid_argument(
        "[register_pytree_node] flatten_fn must return a (children, aux_data) "
        "pair.");
  }
  auto children = nb::cast<std::vector<nb::object>>(seq[0]);
  return {std::move(children), nb::cast<nb::object>(seq[1])};
}

// Recreates the original object from aux + children for the type of `like`.
nb::object unflatten_registered(
    nb::handle like,
    nb::object aux,
    const std::vector<nb::object>& children) {
  auto& def = registry().at(reinterpret_cast<PyObject*>(Py_TYPE(like.ptr())));
  nb::list children_list;
  for (const auto& c : children) {
    children_list.append(c);
  }
  return def.unflatten_fn(aux, children_list);
}

} // namespace

void register_pytree_node(
    nb::type_object cls,
    nb::callable flatten_fn,
    nb::callable unflatten_fn) {
  registry()[cls.ptr()] = PytreeNodeDef{flatten_fn, unflatten_fn};
}

bool is_registered_pytree(nb::handle obj) {
  if (!obj.ptr()) {
    return false;
  }
  return registry().find(reinterpret_cast<PyObject*>(Py_TYPE(obj.ptr()))) !=
      registry().end();
}

std::vector<nb::object> pytree_children(nb::handle obj) {
  return flatten_registered(obj).first;
}

uint64_t registered_pytree_fingerprint(nb::handle obj) {
  PyObject* type = reinterpret_cast<PyObject*>(Py_TYPE(obj.ptr()));
  uint64_t fp = reinterpret_cast<uintptr_t>(type);

  // Mix in hash(aux_data) so structurally distinct registered nodes don't
  // collide.  We re-call flatten_fn purely to retrieve aux; this is the same
  // cost as the structural recurse below and keeps the fingerprint in sync
  // with how the node will be expanded.
  auto it = registry().find(type);
  if (it != registry().end()) {
    try {
      auto seq = nb::cast<nb::sequence>(it->second.flatten_fn(obj));
      if (nb::len(seq) == 2) {
        nb::object aux = nb::cast<nb::object>(seq[1]);
        if (!aux.is_none()) {
          try {
            uint64_t aux_hash = static_cast<uint64_t>(nb::hash(aux));
            fp ^= aux_hash + 0x9e3779b97f4a7c15ULL + (fp << 6) + (fp >> 2);
          } catch (...) {
            // Unhashable aux — fall back to type-only fingerprint.
          }
        }
      }
    } catch (...) {
      // flatten_fn failed — fall back to type-only fingerprint.
    }
  }
  return fp;
}

void init_trees(nb::module_& m) {
  m.def(
      "register_pytree_node",
      &register_pytree_node,
      nb::arg("cls"),
      nb::arg("flatten_fn"),
      nb::arg("unflatten_fn"),
      R"pbdoc(
        Register a custom class as a pytree node.

        Once registered, instances of ``cls`` are treated as interior nodes
        (not leaves) by :func:`mlx.core.compile`, :func:`mlx.utils.tree_map`,
        :func:`mlx.utils.tree_flatten`, and friends.

        Args:
            cls (type): The class to register.
            flatten_fn (callable): ``flatten_fn(obj) -> (children, aux_data)``
                where *children* is a list or tuple of sub-trees (may contain
                :class:`array` or further nested structures) and *aux_data* is
                any hashable metadata needed to reconstruct the object.  When
                a registered object appears as a :func:`compile` argument the
                hash of *aux_data* participates in the compile cache key, so
                two instances with different aux trigger a retrace.
            unflatten_fn (callable): ``unflatten_fn(aux_data, children) -> obj``
                that recreates the original object from *aux_data* and the
                (possibly updated) *children* list.

        Example:

            >>> import mlx.core as mx
            >>>
            >>> class Pair:
            ...     def __init__(self, a, b):
            ...         self.a = a
            ...         self.b = b
            ...
            >>> mx.register_pytree_node(
            ...     Pair,
            ...     lambda p: ([p.a, p.b], None),
            ...     lambda _, children: Pair(*children),
            ... )
            >>>
            >>> @mx.compile
            ... def add_pair(p):
            ...     return p.a + p.b
            ...
            >>> add_pair(Pair(mx.array(1), mx.array(2)))
            array(3, dtype=int32)
      )pbdoc");
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
      auto type = subtrees[0].type();
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
      if (PyTuple_CheckExact(subtrees[0].ptr())) {
        return nb::cast<nb::object>(nb::tuple(l));
      }
      return nb::hasattr(type, "_fields") ? type(*l) : type(l);
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
    } else if (is_registered_pytree(subtrees[0])) {
      auto [children, aux] = flatten_registered(subtrees[0]);
      PyTypeObject* type = Py_TYPE(subtrees[0].ptr());

      // Pre-flatten every other subtree so we can index parallel children.
      std::vector<std::vector<nb::object>> other_children(subtrees.size());
      other_children[0] = std::move(children);
      for (size_t j = 1; j < subtrees.size(); ++j) {
        if (is_registered_pytree(subtrees[j]) &&
            Py_TYPE(subtrees[j].ptr()) == type) {
          other_children[j] = flatten_registered(subtrees[j]).first;
          if (other_children[j].size() != other_children[0].size()) {
            throw std::invalid_argument(
                "[tree_map] Additional input tree is not a valid prefix of "
                "the first tree.");
          }
        }
      }

      std::vector<nb::object> new_children;
      new_children.reserve(other_children[0].size());
      for (size_t i = 0; i < other_children[0].size(); ++i) {
        std::vector<nb::object> items(subtrees.size());
        for (size_t j = 0; j < subtrees.size(); ++j) {
          if (!other_children[j].empty()) {
            items[j] = other_children[j][i];
          } else {
            items[j] = subtrees[j];
          }
        }
        new_children.push_back(recurse(items));
      }
      return unflatten_registered(subtrees[0], aux, new_children);
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

void tree_visit(
    const std::vector<nb::object>& trees,
    std::function<void(const std::vector<nb::object>&)> visitor) {
  std::function<void(const std::vector<nb::object>&)> recurse;

  recurse = [&](const std::vector<nb::object>& subtrees) {
    if (nb::isinstance<nb::list>(subtrees[0])) {
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
        recurse(items);
      }
    } else if (nb::isinstance<nb::tuple>(subtrees[0])) {
      //  Check the rest of the subtrees
      std::vector<nb::object> items(subtrees.size());
      int len = nb::cast<nb::tuple>(subtrees[0]).size();
      validate_subtrees<nb::tuple, nb::list, nb::dict>(subtrees);
      for (int i = 0; i < len; ++i) {
        for (int j = 0; j < subtrees.size(); ++j) {
          if (nb::isinstance<nb::tuple>(subtrees[j])) {
            items[j] = nb::cast<nb::tuple>(subtrees[j])[i];
          } else {
            items[j] = subtrees[j];
          }
        }
        recurse(items);
      }
    } else if (nb::isinstance<nb::dict>(subtrees[0])) {
      std::vector<nb::object> items(subtrees.size());
      validate_subtrees<nb::dict, nb::list, nb::tuple>(subtrees);
      for (auto item : nb::cast<nb::dict>(subtrees[0])) {
        for (int j = 0; j < subtrees.size(); ++j) {
          if (nb::isinstance<nb::dict>(subtrees[j])) {
            auto subdict = nb::cast<nb::dict>(subtrees[j]);
            if (!subdict.contains(item.first)) {
              throw std::invalid_argument(
                  "[tree_visit] Tree is not a valid prefix tree of the first tree.");
            }
            items[j] = subdict[item.first];
          } else {
            items[j] = subtrees[j];
          }
        }
        recurse(items);
      }
    } else if (is_registered_pytree(subtrees[0])) {
      PyTypeObject* type = Py_TYPE(subtrees[0].ptr());
      std::vector<std::vector<nb::object>> other_children(subtrees.size());
      other_children[0] = flatten_registered(subtrees[0]).first;
      for (size_t j = 1; j < subtrees.size(); ++j) {
        if (is_registered_pytree(subtrees[j]) &&
            Py_TYPE(subtrees[j].ptr()) == type) {
          other_children[j] = flatten_registered(subtrees[j]).first;
          if (other_children[j].size() != other_children[0].size()) {
            throw std::invalid_argument(
                "[tree_visit] Additional input tree is not a valid prefix of "
                "the first tree.");
          }
        }
      }
      for (size_t i = 0; i < other_children[0].size(); ++i) {
        std::vector<nb::object> items(subtrees.size());
        for (size_t j = 0; j < subtrees.size(); ++j) {
          if (!other_children[j].empty()) {
            items[j] = other_children[j][i];
          } else {
            items[j] = subtrees[j];
          }
        }
        recurse(items);
      }
    } else {
      visitor(subtrees);
    }
  };
  return recurse(trees);
}

void tree_visit(nb::handle tree, std::function<void(nb::handle)> visitor) {
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
    } else if (is_registered_pytree(subtree)) {
      auto [children, _] = flatten_registered(subtree);
      for (const auto& child : children) {
        recurse(child);
      }
    } else {
      visitor(subtree);
    }
  };

  recurse(tree);
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
      auto type = subtree.type();
      nb::list l(subtree);
      for (int i = 0; i < l.size(); ++i) {
        l[i] = recurse(l[i]);
      }
      if (PyTuple_CheckExact(subtree.ptr())) {
        return nb::cast<nb::object>(nb::tuple(l));
      }
      return nb::hasattr(type, "_fields") ? type(*l) : type(l);
    } else if (nb::isinstance<nb::dict>(subtree)) {
      auto d = nb::cast<nb::dict>(subtree);
      for (auto item : d) {
        d[item.first] = recurse(item.second);
      }
      return nb::cast<nb::object>(d);
    } else if (is_registered_pytree(subtree)) {
      auto [children, aux] = flatten_registered(subtree);
      std::vector<nb::object> new_children;
      new_children.reserve(children.size());
      for (auto& c : children) {
        new_children.push_back(recurse(c));
      }
      return unflatten_registered(subtree, aux, new_children);
    } else if (nb::isinstance<mx::array>(subtree)) {
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
void tree_fill(nb::object& tree, const std::vector<mx::array>& values) {
  size_t index = 0;
  tree_visit_update(
      tree, [&](nb::handle node) { return nb::cast(values[index++]); });
}

// Replace all the arrays from the src values with the dst values in the tree
void tree_replace(
    nb::object& tree,
    const std::vector<mx::array>& src,
    const std::vector<mx::array>& dst) {
  std::unordered_map<uintptr_t, mx::array> src_to_dst;
  for (int i = 0; i < src.size(); ++i) {
    src_to_dst.insert({src[i].id(), dst[i]});
  }
  tree_visit_update(tree, [&](nb::handle node) {
    auto arr = nb::cast<mx::array>(node);
    if (auto it = src_to_dst.find(arr.id()); it != src_to_dst.end()) {
      return nb::cast(it->second);
    }
    return nb::cast(arr);
  });
}

std::vector<mx::array> tree_flatten(nb::handle tree, bool strict /* = true */) {
  std::vector<mx::array> flat_tree;

  tree_visit(tree, [&](nb::handle obj) {
    if (nb::isinstance<mx::array>(obj)) {
      flat_tree.push_back(nb::cast<mx::array>(obj));
    } else if (strict) {
      throw std::invalid_argument(
          "[tree_flatten] The argument should contain only arrays");
    }
  });

  return flat_tree;
}

nb::object tree_unflatten(
    nb::object tree,
    const std::vector<mx::array>& values,
    int index /* = 0 */) {
  return tree_map(tree, [&](nb::handle obj) {
    if (nb::isinstance<mx::array>(obj)) {
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

std::pair<std::vector<mx::array>, nb::object> tree_flatten_with_structure(
    nb::object tree,
    bool strict /* = true */) {
  auto sentinel = structure_sentinel();
  std::vector<mx::array> flat_tree;
  auto structure = tree_map(
      tree,
      [&flat_tree, sentinel = std::move(sentinel), strict](nb::handle obj) {
        if (nb::isinstance<mx::array>(obj)) {
          flat_tree.push_back(nb::cast<mx::array>(obj));
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
    const std::vector<mx::array>& values,
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
