// Copyright © 2023-2024 Apple Inc.
#pragma once
#include <nanobind/nanobind.h>

#include "mlx/array.h"

namespace nb = nanobind;
using namespace mlx::core;

void tree_visit(nb::object tree, std::function<void(nb::handle)> visitor);

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
void tree_fill(nb::object& tree, const std::vector<array>& values);

/**
 * Replace all the arrays from the src values with the dst values in the
 * tree.
 */
void tree_replace(
    nb::object& tree,
    const std::vector<array>& src,
    const std::vector<array>& dst);

/**
 * Flatten a tree into a vector of arrays. If strict is true, then the
 * function will throw if the tree contains a leaf which is not an array.
 */
std::vector<array> tree_flatten(nb::object tree, bool strict = true);

/**
 * Unflatten a tree from a vector of arrays.
 */
nb::object tree_unflatten(
    nb::object tree,
    const std::vector<array>& values,
    int index = 0);

std::pair<std::vector<array>, nb::object> tree_flatten_with_structure(
    nb::object tree,
    bool strict = true);

nb::object tree_unflatten_from_structure(
    nb::object structure,
    const std::vector<array>& values,
    int index = 0);
