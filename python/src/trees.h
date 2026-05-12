// Copyright © 2023-2024 Apple Inc.
#pragma once
#include <nanobind/nanobind.h>
#include <vector>

#include "mlx/array.h"

namespace mx = mlx::core;
namespace nb = nanobind;

// --------------------------------------------------------------------------
// Pytree node registry
//
// Allows third-party Python classes to participate in MLX tree utilities
// and in mx.compile argument flattening.  Mirrors the API of
// jax.tree_util.register_pytree_node:
//
//   flatten_fn(obj)            -> (children: Sequence, aux_data: Any)
//   unflatten_fn(aux, children) -> obj
// --------------------------------------------------------------------------

void register_pytree_node(
    nb::type_object cls,
    nb::callable flatten_fn,
    nb::callable unflatten_fn);

// True if type(obj) has been registered as a pytree node.
bool is_registered_pytree(nb::handle obj);

// Returns the children list for a registered pytree node.  Caller must ensure
// is_registered_pytree(obj) is true.
std::vector<nb::object> pytree_children(nb::handle obj);

// Compile cache fingerprint for a registered pytree's type+aux pair.
// Combines id(type) and hash(aux) so that compile retraces if either changes.
uint64_t registered_pytree_fingerprint(nb::handle obj);

void tree_visit(
    const std::vector<nb::object>& trees,
    std::function<void(const std::vector<nb::object>&)> visitor);
void tree_visit(nb::handle tree, std::function<void(nb::handle)> visitor);

nb::object tree_map(
    const std::vector<nb::object>& trees,
    std::function<nb::object(const std::vector<nb::object>&)> transform);

nb::object tree_map(
    nb::object tree,
    std::function<nb::object(nb::handle)> transform);

void tree_visit_update(
    nb::object tree,
    std::function<nb::object(nb::handle)> visitor);

/**
 * Fill a pytree (recursive dict or list of dict or list) in place with the
 * given arrays. */
void tree_fill(nb::object& tree, const std::vector<mx::array>& values);

/**
 * Replace all the arrays from the src values with the dst values in the
 * tree.
 */
void tree_replace(
    nb::object& tree,
    const std::vector<mx::array>& src,
    const std::vector<mx::array>& dst);

/**
 * Flatten a tree into a vector of arrays. If strict is true, then the
 * function will throw if the tree contains a leaf which is not an array.
 */
std::vector<mx::array> tree_flatten(nb::handle tree, bool strict = true);

/**
 * Unflatten a tree from a vector of arrays.
 */
nb::object tree_unflatten(
    nb::object tree,
    const std::vector<mx::array>& values,
    int index = 0);

std::pair<std::vector<mx::array>, nb::object> tree_flatten_with_structure(
    nb::object tree,
    bool strict = true);

nb::object tree_unflatten_from_structure(
    nb::object structure,
    const std::vector<mx::array>& values,
    int index = 0);
