// Copyright © 2023 Apple Inc.
//
#include <json.hpp>
#include <memory>
#include <stack>

#include "mlx/fast.h"
#include "mlx/io.h"
#include "mlx/io/load.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"

using json = nlohmann::json;

#define ST_F16 "F16"
#define ST_BF16 "BF16"
#define ST_F32 "F32"

#define ST_BOOL "BOOL"
#define ST_I8 "I8"
#define ST_I16 "I16"
#define ST_I32 "I32"
#define ST_I64 "I64"
#define ST_U8 "U8"
#define ST_U16 "U16"
#define ST_U32 "U32"
#define ST_U64 "U64"
#define ST_F8_E4M3 "F8_E4M3"

// Note: Complex numbers aren't in the spec yet so this could change -
// https://github.com/huggingface/safetensors/issues/389
#define ST_C64 "C64"

namespace mlx::core {

std::string dtype_to_safetensor_str(Dtype t) {
  switch (t) {
    case float32:
      return ST_F32;
    case bfloat16:
      return ST_BF16;
    case float16:
      return ST_F16;
    case int64:
      return ST_I64;
    case int32:
      return ST_I32;
    case int16:
      return ST_I16;
    case int8:
      return ST_I8;
    case uint64:
      return ST_U64;
    case uint32:
      return ST_U32;
    case uint16:
      return ST_U16;
    case uint8:
      return ST_U8;
    case bool_:
      return ST_BOOL;
    case complex64:
      return ST_C64;
    default:
      throw std::runtime_error("[save_safetensors] received invalid dtype.");
  }
}

Dtype dtype_from_safetensor_str(std::string_view str) {
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
  } else if (str == ST_F8_E4M3) {
    // We convert this manually later
    return uint8;
  } else {
    throw std::runtime_error(
        "[safetensor] unsupported dtype " + std::string(str));
  }
}

array f8_e4m3_to_float(array x, Dtype dtype, StreamOrDevice s) {
  if (to_stream(s).device == Device::gpu) {
    // From PyTorch:
    // https://github.com/pytorch/pytorch/blob/e3643e1e0e923f0fc063dfab6f45c956d568919d/c10/util/Float8_e4m3fn.h#L46
    std::string source = R"(
      uint elem = thread_position_in_grid.x;
      uint8_t val = x[elem];

      const uint32_t w = (uint32_t)val << 24;
      const uint32_t sign = w & 0x80000000;
      const uint32_t nonsign = w & 0x7FFFFFFF;

      uint32_t renorm_shift = metal::clz(nonsign);
      renorm_shift = renorm_shift > 4 ? renorm_shift - 4 : 0;

      const int32_t inf_nan_mask =
          ((int32_t)(nonsign + 0x01000000) >> 8) & 0x7F800000;
      const int32_t zero_mask = (int32_t)(nonsign - 1) >> 31;
      uint32_t result = sign |
          ((((nonsign << renorm_shift >> 4) + ((0x78 - renorm_shift) << 23)) |
              inf_nan_mask) &
              ~zero_mask);

      float out = *(reinterpret_cast<thread float*>(&result));
      y[elem] = static_cast<T>(out);
    )";
    auto kernel = fast::metal_kernel("f8_e4m3", {"x"}, {"y"}, source);
    auto outputs = kernel(
        {x},
        {x.shape()},
        {dtype},
        {x.size(), 1, 1},
        {256, 1, 1},
        {{"T", dtype}},
        std::nullopt,
        false,
        s);
    return outputs[0];
  } else {
    auto w = left_shift(astype(x, uint32, s), array({24}, uint32), s);
    auto sign = bitwise_and(w, array({0x80000000}, uint32), s);
    auto nonsign = bitwise_and(w, array({0x7FFFFFFF}, uint32), s);

    // Emulate a clz op with a lookup table
    auto clz_table =
        array({28, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}, uint32);
    auto renorm_shift = take(clz_table, bitwise_and(x, array({0xf}), s), s);
    renorm_shift = where(
        greater(
            bitwise_and(x, array({0x70}, uint32), s), array({0}, uint32), s),
        array({0}, uint32),
        renorm_shift,
        s);
    auto inf_nan_mask = bitwise_and(
        right_shift(
            astype(add(nonsign, array(0x01000000, int32), s), int32, s),
            array({8}, int32),
            s),
        array({0x7F800000}, int32),
        s);
    auto zero_mask = right_shift(
        astype(subtract(nonsign, array({1}, uint32), s), int32, s),
        array({31}, int32),
        s);
    zero_mask = astype(zero_mask, uint32, s);
    inf_nan_mask = astype(inf_nan_mask, uint32, s);
    auto result =
        add(right_shift(
                left_shift(nonsign, renorm_shift, s), array({4}, uint32), s),
            left_shift(
                subtract(array({0x78}, uint32), renorm_shift, s),
                array({23}, uint32),
                s),
            s);
    result = bitwise_or(
        sign,
        bitwise_and(
            bitwise_or(result, inf_nan_mask, s),
            bitwise_invert(zero_mask, s),
            s),
        s);
    result = astype(view(result, float32, s), dtype, s);
    return result;
  }
}

/** Load array from reader in safetensor format */
SafetensorsLoad load_safetensors(
    std::shared_ptr<io::Reader> in_stream,
    StreamOrDevice s) {
  ////////////////////////////////////////////////////////
  // Open and check file
  if (!in_stream->good() || !in_stream->is_open()) {
    throw std::runtime_error(
        "[load_safetensors] Failed to open " + in_stream->label());
  }

  uint64_t jsonHeaderLength = 0;
  // This is the same limit as in the original Rust Safetensors code.
  constexpr uint64_t kMaxJsonHeaderLength = 100000000;
  in_stream->read(reinterpret_cast<char*>(&jsonHeaderLength), 8);
  if (jsonHeaderLength <= 0 || jsonHeaderLength >= kMaxJsonHeaderLength) {
    throw std::runtime_error(
        "[load_safetensors] Invalid json header length " + in_stream->label());
  }
  // Load the json metadata
  auto rawJson = std::make_unique<char[]>(jsonHeaderLength);
  in_stream->read(rawJson.get(), jsonHeaderLength);
  auto metadata = json::parse(rawJson.get(), rawJson.get() + jsonHeaderLength);
  // Should always be an object on the top-level
  if (!metadata.is_object()) {
    throw std::runtime_error(
        "[load_safetensors] Invalid json metadata " + in_stream->label());
  }
  size_t offset = jsonHeaderLength + 8;
  // Load the arrays using metadata
  std::unordered_map<std::string, array> res;
  std::unordered_map<std::string, std::string> metadata_map;
  for (const auto& item : metadata.items()) {
    if (item.key() == "__metadata__") {
      for (const auto& meta_item : item.value().items()) {
        metadata_map.insert({meta_item.key(), meta_item.value()});
      }
      continue;
    }
    const std::string& dtype = item.value().at("dtype");
    const Shape& shape = item.value().at("shape");
    const std::vector<size_t>& data_offsets = item.value().at("data_offsets");
    Dtype type = dtype_from_safetensor_str(dtype);
    auto loaded_array = array(
        shape,
        type,
        std::make_shared<Load>(
            to_stream(s), in_stream, offset + data_offsets.at(0), false),
        std::vector<array>{});
    if (dtype == ST_F8_E4M3) {
      loaded_array = f8_e4m3_to_float(loaded_array, bfloat16, s);
    }
    res.insert({item.key(), loaded_array});
  }
  return {res, metadata_map};
}

SafetensorsLoad load_safetensors(const std::string& file, StreamOrDevice s) {
  return load_safetensors(std::make_shared<io::ParallelFileReader>(file), s);
}

void save_safetensors(
    std::shared_ptr<io::Writer> out_stream,
    std::unordered_map<std::string, array> a,
    std::unordered_map<std::string, std::string> metadata /* = {} */) {
  ////////////////////////////////////////////////////////
  // Check file
  if (!out_stream->good() || !out_stream->is_open()) {
    throw std::runtime_error(
        "[save_safetensors] Failed to open " + out_stream->label());
  }

  ////////////////////////////////////////////////////////
  // Check array map
  json parent;
  json _metadata;
  for (auto& [key, value] : metadata) {
    _metadata[key] = value;
  }
  parent["__metadata__"] = _metadata;

  {
    std::vector<array> to_eval;
    to_eval.reserve(a.size());
    for (auto& p : a) {
      p.second = contiguous(p.second);
      to_eval.push_back(p.second);
    }
    eval(std::move(to_eval));
  }

  size_t offset = 0;
  for (auto& [key, arr] : a) {
    if (arr.nbytes() == 0) {
      throw std::invalid_argument(
          "[save_safetensors] cannot serialize an empty array key: " + key);
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

void save_safetensors(
    std::string file,
    std::unordered_map<std::string, array> a,
    std::unordered_map<std::string, std::string> metadata /* = {} */) {
  // Add .safetensors to file name if it is not there
  if (file.length() < 12 ||
      file.substr(file.length() - 12, 12) != ".safetensors")
    file += ".safetensors";

  // Serialize array
  save_safetensors(
      std::make_shared<io::FileWriter>(std::move(file)), a, metadata);
}

} // namespace mlx::core
