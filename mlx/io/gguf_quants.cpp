// Copyright © 2023-2024 Apple Inc.

#include <cstdint>
#include <cstring>
#include <numeric>

#include "mlx/io/gguf.h"

namespace mlx::core {

const KQuantCodec* gguf_type_to_kquant_codec(uint32_t gguf_type) {
  switch (gguf_type) {
    case GGUF_TYPE_Q4_0:
      return kquant_codec_by_name("q4_0");
    case GGUF_TYPE_Q4_1:
      return kquant_codec_by_name("q4_1");
    case GGUF_TYPE_Q5_0:
      return kquant_codec_by_name("q5_0");
    case GGUF_TYPE_Q8_0:
      return kquant_codec_by_name("q8_0");
    case GGUF_TYPE_Q5_1:
      return kquant_codec_by_name("q5_1");
    case GGUF_TYPE_Q2_K:
      return kquant_codec_by_name("q2_k");
    case GGUF_TYPE_Q3_K:
      return kquant_codec_by_name("q3_k");
    case GGUF_TYPE_Q4_K:
      return kquant_codec_by_name("q4_k");
    case GGUF_TYPE_Q5_K:
      return kquant_codec_by_name("q5_k");
    case GGUF_TYPE_Q6_K:
      return kquant_codec_by_name("q6_k");
    default:
      return nullptr;
  }
}

void gguf_load_kquant(
    std::unordered_map<std::string, array>& a,
    const gguf_tensor& tensor,
    const KQuantCodec& codec,
    std::vector<std::string>& kquant_entries) {
  std::string name(tensor.name, tensor.namelen);

  auto logical_shape = get_shape(tensor);
  if (logical_shape.empty()) {
    std::ostringstream msg;
    msg << "[load_gguf] kquant tensor " << name << " has no dimensions";
    throw std::runtime_error(msg.str());
  }
  auto last_dim = logical_shape.back();
  if (last_dim % codec.weights_per_block != 0) {
    std::ostringstream msg;
    msg << "[load_gguf] kquant tensor " << name << " last dim " << last_dim
        << " is not divisible by weights_per_block " << codec.weights_per_block
        << " for codec " << codec.name;
    throw std::runtime_error(msg.str());
  }
  auto bytes_per_row =
      (last_dim / codec.weights_per_block) * codec.bytes_per_block;
  auto packed_shape = logical_shape;
  packed_shape.back() = bytes_per_row;

  size_t total_bytes = std::accumulate(
      packed_shape.begin(),
      packed_shape.end(),
      static_cast<size_t>(1),
      std::multiplies<size_t>());
  if (total_bytes != tensor.bsize) {
    std::ostringstream msg;
    msg << "[load_gguf] kquant tensor " << name << " (" << codec.name
        << ") computed byte size " << total_bytes
        << " does not match tensor.bsize " << tensor.bsize;
    throw std::runtime_error(msg.str());
  }

  auto buf = allocator::malloc(total_bytes);
  std::memcpy(buf.raw_ptr(), tensor.weights_data, total_bytes);
  array weight(buf, std::move(packed_shape), uint8);

  constexpr std::string_view weight_suffix = ".weight";
  std::string name_prefix;
  if (name.size() > weight_suffix.size() &&
      name.compare(
          name.size() - weight_suffix.size(),
          weight_suffix.size(),
          weight_suffix) == 0) {
    name_prefix = name.substr(0, name.size() - weight_suffix.size());
  } else {
    name_prefix = name;
  }

  auto sb = allocator::malloc(uint8.size());
  *static_cast<uint8_t*>(sb.raw_ptr()) = 0;
  array scales_ph(sb, Shape{1}, uint8);

  auto check_insert = [](const auto& inserted) {
    if (!inserted.second) {
      std::ostringstream msg;
      msg << "[load_gguf] Duplicate parameter name " << inserted.first->second
          << " this can happen when loading quantized tensors.";
      throw std::runtime_error(msg.str());
    }
  };

  check_insert(a.emplace(name, std::move(weight)));
  check_insert(a.emplace(name_prefix + ".scales", std::move(scales_ph)));

  kquant_entries.push_back(name + ":" + std::string(codec.name));
}

} // namespace mlx::core
