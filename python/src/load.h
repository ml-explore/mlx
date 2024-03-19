// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include "mlx/io.h"

namespace nb = nanobind;
using namespace mlx::core;

using LoadOutputTypes = std::variant<
    array,
    std::unordered_map<std::string, array>,
    SafetensorsLoad,
    GGUFLoad>;

SafetensorsLoad mlx_load_safetensor_helper(nb::object file, StreamOrDevice s);
void mlx_save_safetensor_helper(
    nb::object file,
    nb::dict d,
    std::optional<nb::dict> m);

GGUFLoad mlx_load_gguf_helper(nb::object file, StreamOrDevice s);

void mlx_save_gguf_helper(
    nb::object file,
    nb::dict d,
    std::optional<nb::dict> m);

LoadOutputTypes mlx_load_helper(
    nb::object file,
    std::optional<std::string> format,
    bool return_metadata,
    StreamOrDevice s);
void mlx_save_helper(nb::object file, array a);
void mlx_savez_helper(
    nb::object file,
    nb::args args,
    const nb::kwargs& kwargs,
    bool compressed = false);
