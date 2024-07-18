// Copyright © 2023-2024 Apple Inc.

#include <cstdint>
#include <cstring>
#include <numeric>

#include "mlx/io/gguf.h"

namespace mlx::core {

void unpack_32_4(uint8_t* data, int8_t* dst) {
  std::fill_n(dst, 16, 0);
  for (int j = 0; j < 16; ++j) {
    uint8_t x = (data[j + 2] & 0x0F); // j+2 to skip scale bytes.
    if (j % 2 != 0) {
      x <<= 4;
    }
    dst[j / 2] += x;
  }
  // Last 16 weights are in the higher bits
  for (int j = 0; j < 16; ++j) {
    uint8_t x = (data[j + 2] >> 4);
    if (j % 2 != 0) {
      x <<= 4;
    }
    dst[8 + j / 2] += x;
  }
}

// Extracts (weight, scales, biases) from Q4_0 tensors.
// Data layout is: |16 bit scale|32 x 4bit weights|.
void extract_q4_0_data(
    const gguf_tensor& tensor,
    array& weights_arr,
    array& scales_arr,
    array& biases_arr) {
  const uint64_t bytes_per_block = 18; // 2 bytes scale, 32x0.5 byte weights
  auto data = static_cast<uint8_t*>(tensor.weights_data);
  auto weights = weights_arr.data<int8_t>();
  auto scales = scales_arr.data<float16_t>();
  auto biases = biases_arr.data<float16_t>();
  for (int64_t i = 0; i < scales_arr.size(); i++) {
    scales[i] = *((float16_t*)data);
    biases[i] = -8 * scales[i];
    unpack_32_4(data, weights);
    weights += 16;
    data += bytes_per_block;
  }
}

// Extracts (weight, scales, biases) from Q4_1 tensors.
// Data layout is: |16 bit scale|16 bit bias|32 x 4bit weights|.
void extract_q4_1_data(
    const gguf_tensor& tensor,
    array& weights_arr,
    array& scales_arr,
    array& biases_arr) {
  const uint64_t bytes_per_block =
      20; // 2 bytes scale, 2 bytes bias, 32x0.5 byte weights
  auto data = static_cast<uint8_t*>(tensor.weights_data);
  auto weights = weights_arr.data<int8_t>();
  auto scales = scales_arr.data<float16_t>();
  auto biases = biases_arr.data<float16_t>();
  for (int64_t i = 0; i < scales_arr.size(); i++) {
    scales[i] = *((float16_t*)data);
    biases[i] = *((float16_t*)(data) + 1);
    unpack_32_4(data, weights);
    weights += 16;
    data += bytes_per_block;
  }
}

// Extracts (weight, scales, biases) from Q8_0 tensors.
// Data layout is: |16 bit scale|32 x 8bit weights|.
void extract_q8_0_data(
    const gguf_tensor& tensor,
    array& weights_arr,
    array& scales_arr,
    array& biases_arr) {
  const uint64_t weights_per_block = 32;
  const uint64_t bytes_per_block = 34; // 2 bytes scale, 32x1 byte weights
  auto data = static_cast<uint8_t*>(tensor.weights_data);
  auto weights = weights_arr.data<int8_t>();
  auto scales = scales_arr.data<float16_t>();
  auto biases = biases_arr.data<float16_t>();
  for (int64_t i = 0; i < scales_arr.size(); i++) {
    uint8_t* block_data = data + i * bytes_per_block;
    scales[i] = *((float16_t*)block_data);
    biases[i] = -128 * scales[i];
    for (int64_t j = 0; j < weights_per_block; ++j) {
      uint8_t x = block_data[j + 2]; // j+2 to skip the scale bytes.
      // Original data is in int8_t, so we add a bias of -128 and invert the
      // first bit.
      x ^= 1 << 7;
      weights[i * weights_per_block + j] = x;
    }
  }
}

void gguf_load_quantized(
    std::unordered_map<std::string, array>& a,
    const gguf_tensor& tensor) {
  uint64_t weights_per_byte;
  if (tensor.type == GGUF_TYPE_Q4_0 || tensor.type == GGUF_TYPE_Q4_1) {
    weights_per_byte = 2;
  } else { // tensor.type == GGUF_TYPE_Q8_0
    weights_per_byte = 1;
  }

  std::string name(tensor.name, tensor.namelen);

  std::vector<int> shape = get_shape(tensor);
  const uint64_t weights_per_block = 32;
  if (shape[shape.size() - 1] % weights_per_block != 0) {
    std::ostringstream msg;
    msg << "[load_gguf] tensor " << name
        << "has incompatible last dim shape: " << shape[shape.size() - 1];
    throw std::runtime_error(msg.str());
  }

  std::vector<int> weights_shape = shape;
  weights_shape.back() /= (weights_per_byte * 4);
  auto w_nbytes = uint32.size *
      std::accumulate(weights_shape.begin(),
                      weights_shape.end(),
                      1,
                      std::multiplies<size_t>());

  array weights(allocator::malloc(w_nbytes), std::move(weights_shape), uint32);

  // For scales and bias
  shape[shape.size() - 1] = shape[shape.size() - 1] / weights_per_block;
  auto sb_nbytes = float16.size *
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

  array scales(allocator::malloc(sb_nbytes), shape, float16);
  array biases(allocator::malloc(sb_nbytes), std::move(shape), float16);
  if (tensor.type == GGUF_TYPE_Q4_0) {
    extract_q4_0_data(tensor, weights, scales, biases);
  } else if (tensor.type == GGUF_TYPE_Q4_1) {
    extract_q4_1_data(tensor, weights, scales, biases);
  } else if (tensor.type == GGUF_TYPE_Q8_0) {
    extract_q8_0_data(tensor, weights, scales, biases);
  }

  a.emplace(name, std::move(weights));

  auto check_insert = [](const auto& inserted) {
    if (!inserted.second) {
      std::ostringstream msg;
      msg << "[load_gguf] Duplicate parameter name " << inserted.first->second
          << " this can happend when loading quantized tensors.";
      throw std::runtime_error(msg.str());
    }
  };

  constexpr std::string_view weight_suffix = ".weight";
  const std::string name_prefix =
      name.substr(0, name.length() - weight_suffix.length());
  check_insert(a.emplace(name_prefix + ".scales", std::move(scales)));
  check_insert(a.emplace(name_prefix + ".biases", std::move(biases)));
}

} // namespace mlx::core
