// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <variant>

#include "mlx/array.h"
#include "mlx/io/load.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

namespace mlx::core {

/** Save array to out stream in .npy format */
void save(std::shared_ptr<io::Writer> out_stream, array a);

/** Save array to file in .npy format */
void save(const std::string& file, array a);

/** Load array from reader in .npy format */
array load(std::shared_ptr<io::Reader> in_stream, StreamOrDevice s = {});

/** Load array from file in .npy format */
array load(const std::string& file, StreamOrDevice s = {});

/** Load array map from .safetensors file format */
std::unordered_map<std::string, array> load_safetensors(
    std::shared_ptr<io::Reader> in_stream,
    StreamOrDevice s = {});
std::unordered_map<std::string, array> load_safetensors(
    const std::string& file,
    StreamOrDevice s = {});

void save_safetensors(
    std::shared_ptr<io::Writer> in_stream,
    std::unordered_map<std::string, array>);
void save_safetensors(
    const std::string& file,
    std::unordered_map<std::string, array>);

using MetaData =
    std::variant<std::monostate, array, std::string, std::vector<std::string>>;

/** Load array map and metadata from .gguf file format */
std::pair<
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, MetaData>>
load_gguf(const std::string& file, StreamOrDevice s = {});

void save_gguf(
    std::string file,
    std::unordered_map<std::string, array> array_map,
    std::unordered_map<std::string, MetaData> meta_data = {});

} // namespace mlx::core
