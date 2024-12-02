// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <unordered_map>
#include <variant>

#include "mlx/array.h"
#include "mlx/io/load.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mlx::core {
using GGUFMetaData =
    std::variant<std::monostate, array, std::string, std::vector<std::string>>;
using GGUFLoad = std::pair<
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, GGUFMetaData>>;
using SafetensorsLoad = std::pair<
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, std::string>>;

/** Save array to out stream in .npy format */
void save(std::shared_ptr<io::Writer> out_stream, array a);

/** Save array to file in .npy format */
void save(std::string file, array a);

/** Load array from reader in .npy format */
array load(std::shared_ptr<io::Reader> in_stream, StreamOrDevice s = {});

/** Load array from file in .npy format */
array load(std::string file, StreamOrDevice s = {});

/** Load array map from .safetensors file format */
SafetensorsLoad load_safetensors(
    std::shared_ptr<io::Reader> in_stream,
    StreamOrDevice s = {});
SafetensorsLoad load_safetensors(
    const std::string& file,
    StreamOrDevice s = {});

void save_safetensors(
    std::shared_ptr<io::Writer> in_stream,
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, std::string> metadata = {});
void save_safetensors(
    std::string file,
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, std::string> metadata = {});

/** Load array map and metadata from .gguf file format */

GGUFLoad load_gguf(const std::string& file, StreamOrDevice s = {});

void save_gguf(
    std::string file,
    std::unordered_map<std::string, array> array_map,
    std::unordered_map<std::string, GGUFMetaData> meta_data = {});

} // namespace mlx::core
