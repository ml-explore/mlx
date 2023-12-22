#include "mlx/io/safetensor.h"

#include <stack>

namespace mlx::core {

std::string dtype_to_safetensor_str(Dtype t) {
  if (t == float32) {
    return ST_F32;
  } else if (t == bfloat16) {
    return ST_BF16;
  } else if (t == float16) {
    return ST_F16;
  } else if (t == int64) {
    return ST_I64;
  } else if (t == int32) {
    return ST_I32;
  } else if (t == int16) {
    return ST_I16;
  } else if (t == int8) {
    return ST_I8;
  } else if (t == uint64) {
    return ST_U64;
  } else if (t == uint32) {
    return ST_U32;
  } else if (t == uint16) {
    return ST_U16;
  } else if (t == uint8) {
    return ST_U8;
  } else if (t == bool_) {
    return ST_BOOL;
  } else if (t == complex64) {
    return ST_C64;
  } else {
    throw std::runtime_error("[safetensor] unsupported dtype");
  }
}

Dtype dtype_from_safetensor_str(std::string str) {
  if (str == ST_F32) {
    return float32;
  } else if (str == ST_F16) {
    return float16;
  } else if (str == ST_BF16) {
    return bfloat16;
  } else if (str == ST_I64) {
    return int64;
  } else if (str == ST_I32) {
    return int32;
  } else if (str == ST_I16) {
    return int16;
  } else if (str == ST_I8) {
    return int8;
  } else if (str == ST_U64) {
    return uint64;
  } else if (str == ST_U32) {
    return uint32;
  } else if (str == ST_U16) {
    return uint16;
  } else if (str == ST_U8) {
    return uint8;
  } else if (str == ST_BOOL) {
    return bool_;
  } else if (str == ST_C64) {
    return complex64;
  } else {
    throw std::runtime_error("[safetensor] unsupported dtype " + str);
  }
}

/** Load array from reader in safetensor format */
std::unordered_map<std::string, array> load_safetensor(
    std::shared_ptr<io::Reader> in_stream,
    StreamOrDevice s) {
  ////////////////////////////////////////////////////////
  // Open and check file
  if (!in_stream->good() || !in_stream->is_open()) {
    throw std::runtime_error(
        "[load_safetensor] Failed to open " + in_stream->label());
  }

  uint64_t jsonHeaderLength = 0;
  in_stream->read(reinterpret_cast<char*>(&jsonHeaderLength), 8);
  if (jsonHeaderLength <= 0) {
    throw std::runtime_error(
        "[load_safetensor] Invalid json header length " + in_stream->label());
  }
  // Load the json metadata
  char rawJson[jsonHeaderLength];
  in_stream->read(rawJson, jsonHeaderLength);
  auto metadata = json::parse(rawJson, rawJson + jsonHeaderLength);
  // Should always be an object on the top-level
  if (!metadata.is_object()) {
    throw std::runtime_error(
        "[load_safetensor] Invalid json metadata " + in_stream->label());
  }
  size_t offset = jsonHeaderLength + 8;
  // Load the arrays using metadata
  std::unordered_map<std::string, array> res;
  for (const auto& item : metadata.items()) {
    if (item.key() == "__metadata__") {
      // ignore metadata for now
      continue;
    }
    std::string dtype = item.value().at("dtype");
    std::vector<int> shape = item.value().at("shape");
    std::vector<size_t> data_offsets = item.value().at("data_offsets");
    Dtype type = dtype_from_safetensor_str(dtype);
    auto loaded_array = array(
        shape,
        type,
        std::make_unique<Load>(
            to_stream(s), in_stream, offset + data_offsets.at(0), false),
        std::vector<array>{});
    res.insert({item.key(), loaded_array});
  }
  return res;
}

std::unordered_map<std::string, array> load_safetensor(
    const std::string& file,
    StreamOrDevice s) {
  return load_safetensor(std::make_shared<io::FileReader>(file), s);
}

/** Save array to out stream in .npy format */
void save_safetensor(
    std::shared_ptr<io::Writer> out_stream,
    std::unordered_map<std::string, array> a,
    std::optional<bool> retain_graph_) {
  ////////////////////////////////////////////////////////
  // Check file
  if (!out_stream->good() || !out_stream->is_open()) {
    throw std::runtime_error(
        "[save_safetensor] Failed to open " + out_stream->label());
  }

  ////////////////////////////////////////////////////////
  // Check array map
  json parent;
  parent["__metadata__"] = json::object({
      {"format", "mlx"},
  });
  size_t offset = 0;
  for (auto& [key, arr] : a) {
    arr.eval(retain_graph_.value_or(arr.is_tracer()));
    if (arr.nbytes() == 0) {
      throw std::invalid_argument(
          "[save_safetensor] cannot serialize an empty array key: " + key);
    }

    if (!arr.flags().contiguous) {
      throw std::invalid_argument(
          "[save_safetensor] cannot serialize a non-contiguous array key: " +
          key);
    }
    json child;
    child["dtype"] = dtype_to_safetensor_str(arr.dtype());
    child["shape"] = arr.shape();
    child["data_offsets"] = std::vector<size_t>{offset, offset + arr.nbytes()};
    parent[key] = child;
    offset += arr.nbytes();
  }

  auto header = parent.dump();
  uint64_t header_len = header.length();
  out_stream->write(reinterpret_cast<char*>(&header_len), 8);
  out_stream->write(header.c_str(), header_len);
  for (auto& [key, arr] : a) {
    out_stream->write(arr.data<char>(), arr.nbytes());
  }
}

void save_safetensor(
    const std::string& file_,
    std::unordered_map<std::string, array> a,
    std::optional<bool> retain_graph) {
  // Open and check file
  std::string file = file_;

  // Add .safetensors to file name if it is not there
  if (file.length() < 12 ||
      file.substr(file.length() - 12, 12) != ".safetensors")
    file += ".safetensors";

  // Serialize array
  save_safetensor(std::make_shared<io::FileWriter>(file), a, retain_graph);
}

} // namespace mlx::core