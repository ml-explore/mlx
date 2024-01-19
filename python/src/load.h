// Copyright © 2023 Apple Inc.

#pragma once

#include <pybind11/pybind11.h>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include "mlx/io.h"

namespace py = pybind11;
using namespace mlx::core;

using LoadOutputTypes = std::variant<
    array,
    std::unordered_map<std::string, array>,
    std::pair<
        std::unordered_map<std::string, array>,
        std::unordered_map<std::string, MetaData>>>;

std::unordered_map<std::string, array> mlx_load_safetensor_helper(
    py::object file,
    StreamOrDevice s);
void mlx_save_safetensor_helper(py::object file, py::dict d);

std::pair<
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, MetaData>>
mlx_load_gguf_helper(py::object file, StreamOrDevice s);
void mlx_save_gguf_helper(
    py::object file,
    py::dict d,
    std::optional<py::dict> m);

LoadOutputTypes mlx_load_helper(
    py::object file,
    std::optional<std::string> format,
    bool return_metadata,
    StreamOrDevice s);
void mlx_save_helper(py::object file, array a);
void mlx_savez_helper(
    py::object file,
    py::args args,
    const py::kwargs& kwargs,
    bool compressed = false);
