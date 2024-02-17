// Copyright © 2023-2024 Apple Inc.
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlx/array.h"

namespace py = pybind11;
using namespace mlx::core;

void tree_visit(py::object tree, std::function<void(py::handle)> visitor);

py::object tree_map(
    const std::vector<py::object>& trees,
    std::function<py::object(const std::vector<py::object>&)> transform);

py::object tree_map(
    py::object tree,
    std::function<py::object(py::handle)> transform);

void tree_visit_update(
    py::object tree,
    std::function<py::object(py::handle)> visitor);

/**
 * Fill a pytree (recursive dict or list of dict or list) in place with the
 * given arrays. */
void tree_fill(py::object& tree, const std::vector<array>& values);

/**
 * Replace all the arrays from the src values with the dst values in the
 * tree.
 */
void tree_replace(
    py::object& tree,
    const std::vector<array>& src,
    const std::vector<array>& dst);

/**
 * Flatten a tree into a vector of arrays. If strict is true, then the
 * function will throw if the tree contains a leaf which is not an array.
 */
std::vector<array> tree_flatten(py::object tree, bool strict = true);

/**
 * Unflatten a tree from a vector of arrays.
 */
py::object tree_unflatten(
    py::object tree,
    const std::vector<array>& values,
    int index = 0);

std::pair<std::vector<array>, py::object> tree_flatten_with_structure(
    py::object tree,
    bool strict = true);

py::object tree_unflatten_from_structure(
    py::object structure,
    const std::vector<array>& values,
    int index = 0);
