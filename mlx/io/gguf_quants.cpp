// Copyright © 2023-2024 Apple Inc.

#include <cstdint>
#include <cstring>

#include <mlx/io/gguf.h>

namespace mlx::core {

// Extracts (weight, scales, biases) from Q4_0 tensors.
// Data layout is: |16 bit scale|32 x 4bit weights|.
void extract_q4_0_data(
    std::unordered_map<std::string, array>& a,
    const gguf_tensor& tensor) {
  std::string name = std::string(tensor.name, tensor.namelen);
  const std::vector<int> shape = get_shape(tensor);
  const uint64_t weights_per_byte = 2;
  const uint64_t weights_per_block = 32;
  if (shape[shape.size() - 1] % weights_per_block != 0) {
    std::ostringstream msg;
    msg << "[load_gguf] tensor " << name
        << "has incompatible last dim shape: " << shape[shape.size() - 1];
    throw std::runtime_error(msg.str());
  }
  const uint64_t bytes_per_block = 18; // 2 bytes scale, 32x0.5 byte weights
  const uint64_t num_blocks = tensor.num_weights / weights_per_block;
  allocator::Buffer weights_buffer =
      allocator::malloc(tensor.num_weights / weights_per_byte);
  allocator::Buffer scales_buffer = allocator::malloc(num_blocks * 2);
  allocator::Buffer biases_buffer = allocator::malloc(num_blocks * 2);
  auto data = (uint8_t*)tensor.weights_data;
  auto weigths = (uint8_t*)weights_buffer.raw_ptr();
  auto scales = (float16_t*)scales_buffer.raw_ptr();
  auto biases = (float16_t*)biases_buffer.raw_ptr();
  for (int64_t i = 0; i < num_blocks; i++) {
    uint8_t* block_data = data + i * bytes_per_block;
    scales[i] = *((float16_t*)block_data);
    biases[i] = -8 * scales[i];
    // First 16 weights are in the lower bits
    for (int64_t j = 0; j < 16; ++j) {
      uint8_t x = (block_data[j + 2] & 0x0F); // j+2 to skip scale bytes.
      if (j % 2 != 0) {
        x <<= 4;
      }
      weigths[i * 16 + j / 2] += x;
    }
    // Last 16 weights are in the higher bits
    for (int64_t j = 0; j < 16; ++j) {
      uint8_t x = (block_data[j + 2] >> 4);
      if (j % 2 != 0) {
        x <<= 4;
      }
      weigths[i * 16 + 8 + j / 2] += x;
    }
  }
  std::vector<int> weights_shape = shape;
  weights_shape[shape.size() - 1] =
      weights_shape[shape.size() - 1] / weights_per_byte / 4;
  a.insert({name, array(weights_buffer, weights_shape, uint32)});

  const std::string weight_suffix = ".weight";
  const std::string name_prefix =
      name.substr(0, name.length() - weight_suffix.length());
  std::vector<int> scale_bias_shape = shape;
  scale_bias_shape[shape.size() - 1] =
      scale_bias_shape[shape.size() - 1] / weights_per_block;

  const std::string scales_name =
      (std::ostringstream() << name_prefix << ".scales").str();
  a.insert({scales_name, array(scales_buffer, scale_bias_shape, float16)});

  const std::string biases_name =
      (std::ostringstream() << name_prefix << ".biases").str();
  a.insert({biases_name, array(biases_buffer, scale_bias_shape, float16)});
}

// Extracts (weight, scales, biases) from Q4_1 tensors.
// Data layout is: |16 bit scale|32 x 4bit weights|.
void extract_q4_1_data(
    std::unordered_map<std::string, array>& a,
    const gguf_tensor& tensor) {
  std::string name = std::string(tensor.name, tensor.namelen);
  const std::vector<int> shape = get_shape(tensor);
  const uint64_t weights_per_byte = 2;
  const uint64_t weights_per_block = 32;
  if (shape[shape.size() - 1] % weights_per_block != 0) {
    std::ostringstream msg;
    msg << "[load_gguf] tensor " << name
        << "has incompatible last dim shape: " << shape[shape.size() - 1];
    throw std::runtime_error(msg.str());
  }
  const uint64_t bytes_per_block =
      20; // 2 bytes scale, 2 bytes bias, 32x0.5 byte weights
  const uint64_t num_blocks = tensor.num_weights / weights_per_block;
  allocator::Buffer weights_buffer =
      allocator::malloc(tensor.num_weights / weights_per_byte);
  allocator::Buffer scales_buffer = allocator::malloc(num_blocks * 2);
  allocator::Buffer biases_buffer = allocator::malloc(num_blocks * 2);
  auto data = (uint8_t*)tensor.weights_data;
  auto weigths = (uint8_t*)weights_buffer.raw_ptr();
  auto scales = (float16_t*)scales_buffer.raw_ptr();
  auto biases = (float16_t*)biases_buffer.raw_ptr();
  for (int64_t i = 0; i < num_blocks; i++) {
    uint8_t* block_data = data + i * bytes_per_block;
    scales[i] = *((float16_t*)block_data);
    biases[i] = *((float16_t*)block_data + 1);
    // First 16 weights are in the lower bits
    for (int64_t j = 0; j < 16; ++j) {
      uint8_t x =
          (block_data[j + 4] & 0x0F); // j+4 to skip scale and biases bytes.
      if (j % 2 != 0) {
        x <<= 4;
      }
      weigths[i * 16 + j / 2] += x;
    }
    // Last 16 weights are in the higher bits
    for (int64_t j = 0; j < 16; ++j) {
      uint8_t x = (block_data[j + 4] >> 4);
      if (j % 2 != 0) {
        x <<= 4;
      }
      weigths[i * 16 + 8 + j / 2] += x;
    }
  }
  std::vector<int> weights_shape = shape;
  weights_shape[shape.size() - 1] =
      weights_shape[shape.size() - 1] / weights_per_byte / 4;
  a.insert({name, array(weights_buffer, weights_shape, uint32)});

  const std::string weight_suffix = ".weight";
  const std::string name_prefix =
      name.substr(0, name.length() - weight_suffix.length());
  std::vector<int> scale_bias_shape = shape;
  scale_bias_shape[shape.size() - 1] =
      scale_bias_shape[shape.size() - 1] / weights_per_block;

  const std::string scales_name =
      (std::ostringstream() << name_prefix << ".scales").str();
  a.insert({scales_name, array(scales_buffer, scale_bias_shape, float16)});

  const std::string biases_name =
      (std::ostringstream() << name_prefix << ".biases").str();
  a.insert({biases_name, array(biases_buffer, scale_bias_shape, float16)});
}

// Extracts (weight, scales, biases) from Q8_0 tensors.
// Data layout is: |16 bit scale|32 x 8bit weights|.
void extract_q8_0_data(
    std::unordered_map<std::string, array>& a,
    const gguf_tensor& tensor) {
  std::string name = std::string(tensor.name, tensor.namelen);
  const std::vector<int> shape = get_shape(tensor);
  const uint64_t weights_per_byte = 1;
  const uint64_t weights_per_block = 32;
  if (shape[shape.size() - 1] % weights_per_block != 0) {
    std::ostringstream msg;
    msg << "[load_gguf] tensor " << name
        << "has incompatible last dim shape: " << shape[shape.size() - 1];
    throw std::runtime_error(msg.str());
  }
  const uint64_t bytes_per_block = 34; // 2 bytes scale, 32x1 byte weights
  const uint64_t num_blocks = tensor.num_weights / weights_per_block;
  allocator::Buffer weights_buffer =
      allocator::malloc(tensor.num_weights / weights_per_byte);
  allocator::Buffer scales_buffer = allocator::malloc(num_blocks * 2);
  allocator::Buffer biases_buffer = allocator::malloc(num_blocks * 2);
  auto data = (uint8_t*)tensor.weights_data;
  auto weigths = (int8_t*)weights_buffer.raw_ptr();
  auto scales = (float16_t*)scales_buffer.raw_ptr();
  auto biases = (float16_t*)biases_buffer.raw_ptr();
  for (int64_t i = 0; i < num_blocks; i++) {
    uint8_t* block_data = data + i * bytes_per_block;
    scales[i] = *((float16_t*)block_data);
    biases[i] = -128 * scales[i];
    for (int64_t j = 0; j < weights_per_block; ++j) {
      uint8_t x = block_data[j + 2]; // j+2 to skip the scale bytes.
      // Original data is in int8_t, so we add a bias of -128 and invert the
      // first bit.
      x ^= 1 << 7;
      weigths[i * weights_per_block + j] = x;
    }
  }
  std::vector<int> weights_shape = shape;
  weights_shape[shape.size() - 1] =
      weights_shape[shape.size() - 1] / weights_per_byte / 4;
  a.insert({name, array(weights_buffer, weights_shape, uint32)});

  const std::string weight_suffix = ".weight";
  const std::string name_prefix =
      name.substr(0, name.length() - weight_suffix.length());
  std::vector<int> scale_bias_shape = shape;
  scale_bias_shape[shape.size() - 1] =
      scale_bias_shape[shape.size() - 1] / weights_per_block;

  const std::string scales_name =
      (std::ostringstream() << name_prefix << ".scales").str();
  a.insert({scales_name, array(scales_buffer, scale_bias_shape, float16)});

  const std::string biases_name =
      (std::ostringstream() << name_prefix << ".biases").str();
  a.insert({biases_name, array(biases_buffer, scale_bias_shape, float16)});
}

void gguf_load_quantized(
    std::unordered_map<std::string, array>& a,
    const gguf_tensor& tensor) {
  if (tensor.type == GGUF_TYPE_Q4_0) {
    extract_q4_0_data(a, tensor);
  } else if (tensor.type == GGUF_TYPE_Q4_1) {
    extract_q4_1_data(a, tensor);
  } else if (tensor.type == GGUF_TYPE_Q8_0) {
    extract_q8_0_data(a, tensor);
  }
}

} // namespace mlx::core
