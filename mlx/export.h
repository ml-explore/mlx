// Copyright © 2024 Apple Inc.

#pragma once

#include <optional>
#include <set>
#include <unordered_map>
#include <variant>
#include "mlx/array.h"
#include "mlx/mlx_export.h"

namespace mlx::core {

using Args = std::vector<array>;
using Kwargs = std::unordered_map<std::string, array>;

// Possible types for a Primitive's state
using StateT = std::variant<
    bool,
    int,
    size_t,
    float,
    double,
    Dtype,
    Shape,
    Strides,
    std::vector<int>,
    std::vector<size_t>,
    std::vector<std::tuple<bool, bool, bool>>,
    std::vector<std::variant<bool, int, float>>,
    std::optional<float>,
    std::string>;

using ExportCallbackInput = std::unordered_map<
    std::string,
    std::variant<
        std::vector<std::tuple<std::string, Shape, Dtype>>,
        std::vector<std::pair<std::string, array>>,
        std::vector<std::pair<std::string, std::string>>,
        std::vector<StateT>,
        std::string>>;
using ExportCallback = std::function<void(const ExportCallbackInput&)>;

struct FunctionExporter;

/**
 * Make an exporter to save multiple traces of a given function to
 * the same file.
 */
MLX_API FunctionExporter exporter(
    const std::string& file,
    const std::function<std::vector<array>(const Args&)>& fun,
    bool shapeless = false);

MLX_API FunctionExporter exporter(
    const std::string& file,
    const std::function<std::vector<array>(const Kwargs&)>& fun,
    bool shapeless = false);

MLX_API FunctionExporter exporter(
    const std::string& path,
    const std::function<std::vector<array>(const Args&, const Kwargs&)>& fun,
    bool shapeless = false);

/**
 * Export a function to a file.
 */
MLX_API void export_function(
    const std::string& file,
    const std::function<std::vector<array>(const Args&)>& fun,
    const Args& args,
    bool shapeless = false);

MLX_API void export_function(
    const std::string& file,
    const std::function<std::vector<array>(const Kwargs&)>& fun,
    const Kwargs& kwargs,
    bool shapeless = false);

MLX_API void export_function(
    const std::string& file,
    const std::function<std::vector<array>(const Args&, const Kwargs&)>& fun,
    const Args& args,
    const Kwargs& kwargs,
    bool shapeless = false);

struct ImportedFunction;

/**
 * Import a function from a file.
 */
MLX_API ImportedFunction import_function(const std::string& file);

/**
 * Make an exporter to export multiple traces of a given function with the same
 * callback.
 */
MLX_API FunctionExporter exporter(
    const ExportCallback& callback,
    const std::function<std::vector<array>(const Args&)>& fun,
    bool shapeless = false);

MLX_API FunctionExporter exporter(
    const ExportCallback& callback,
    const std::function<std::vector<array>(const Kwargs&)>& fun,
    bool shapeless = false);

MLX_API FunctionExporter exporter(
    const ExportCallback& callback,
    const std::function<std::vector<array>(const Args&, const Kwargs&)>& fun,
    bool shapeless = false);

/**
 * Export a function with a callback.
 */
MLX_API void export_function(
    const ExportCallback& callback,
    const std::function<std::vector<array>(const Args&)>& fun,
    const Args& args,
    bool shapeless = false);

MLX_API void export_function(
    const ExportCallback& callback,
    const std::function<std::vector<array>(const Kwargs&)>& fun,
    const Kwargs& kwargs,
    bool shapeless = false);

MLX_API void export_function(
    const ExportCallback& callback,
    const std::function<std::vector<array>(const Args&, const Kwargs&)>& fun,
    const Args& args,
    const Kwargs& kwargs,
    bool shapeless = false);

} // namespace mlx::core

#include "mlx/export_impl.h"
