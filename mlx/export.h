// Copyright © 2024 Apple Inc.

#pragma once

#include <map>
#include <set>
#include "mlx/array.h"

namespace mlx::core {

using Args = std::vector<array>;
using Kwargs = std::map<std::string, array>;

struct FunctionExporter;

/**
 * Make an exporter to save multiple traces of a given function to
 * the same file.
 */
FunctionExporter exporter(
    const std::string& file,
    const std::function<std::vector<array>(const Args&)>& fun,
    bool shapeless = false);

FunctionExporter exporter(
    const std::string& file,
    const std::function<std::vector<array>(const Kwargs&)>& fun,
    bool shapeless = false);

FunctionExporter exporter(
    const std::string& path,
    const std::function<std::vector<array>(const Args&, const Kwargs&)>& fun,
    bool shapeless = false);

/**
 * Export a function to a file.
 */
void export_function(
    const std::string& file,
    const std::function<std::vector<array>(const Args&)>& fun,
    const Args& args,
    bool shapeless = false);

void export_function(
    const std::string& file,
    const std::function<std::vector<array>(const Kwargs&)>& fun,
    const Kwargs& kwargs,
    bool shapeless = false);

void export_function(
    const std::string& file,
    const std::function<std::vector<array>(const Args&, const Kwargs&)>& fun,
    const Args& args,
    const Kwargs& kwargs,
    bool shapeless = false);

struct ImportedFunction;

/**
 * Import a function from a file.
 */
ImportedFunction import_function(const std::string& file);

} // namespace mlx::core

#include "mlx/export_impl.h"
