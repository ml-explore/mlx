// Copyright © 2023 Apple Inc.

#include <cstdint>
#include <cstring>

#include "mlx/io.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

extern "C" {
#include <gguflib.h>
}

namespace mlx::core {

std::optional<uint32_t> dtype_to_gguf_tensor_type(const Dtype& dtype) {
  switch (dtype) {
    case float32:
      return GGUF_TYPE_F32;
    case float16:
      return GGUF_TYPE_F16;
    case int8:
      return GGUF_TYPE_I8;
    case int16:
      return GGUF_TYPE_I16;
    case int32:
      return GGUF_TYPE_I32;
    default:
      return {};
  }
}

std::optional<Dtype> gguf_type_to_dtype(const uint32_t& gguf_type) {
  switch (gguf_type) {
    case GGUF_TYPE_F32:
      return float32;
    case GGUF_TYPE_F16:
      return float16;
    case GGUF_TYPE_I8:
      return int8;
    case GGUF_TYPE_I16:
      return int16;
    case GGUF_TYPE_I32:
      return int32;
    default:
      return {};
  }
}

std::pair<allocator::Buffer, Dtype> extract_tensor_data(gguf_tensor* tensor) {
  std::optional<Dtype> equivalent_dtype = gguf_type_to_dtype(tensor->type);
  // If there's an equivalent type, we can simply copy.
  if (equivalent_dtype.has_value()) {
    allocator::Buffer buffer = allocator::malloc(tensor->bsize);
    memcpy(
        buffer.raw_ptr(),
        tensor->weights_data,
        tensor->num_weights * equivalent_dtype.value().size);
    return {buffer, equivalent_dtype.value()};
  }
  // Otherwise, we convert to float16.
  // TODO: Add other dequantization options.
  int16_t* data = gguf_tensor_to_f16(tensor);
  if (data == NULL) {
    throw std::runtime_error("[load_gguf] gguf_tensor_to_f16 failed");
  }
  const size_t new_size = tensor->num_weights * sizeof(int16_t);
  allocator::Buffer buffer = allocator::malloc(new_size);
  memcpy(buffer.raw_ptr(), data, new_size);
  free(data);
  return {buffer, float16};
}

void metadata_value_callback(
    void* privdata,
    uint32_t type,
    union gguf_value* val,
    uint64_t in_array,
    uint64_t array_len) {
  auto value = static_cast<MetaData*>(privdata);
  // TODO: Support all other types.
  switch (type) {
    case GGUF_VALUE_TYPE_ARRAY_START:
      break;
    case GGUF_VALUE_TYPE_ARRAY_END:
      break;
    case GGUF_VALUE_TYPE_UINT8:
      break;
    case GGUF_VALUE_TYPE_INT8:
      break;
    case GGUF_VALUE_TYPE_UINT16:
      break;
    case GGUF_VALUE_TYPE_INT16:
      break;
    case GGUF_VALUE_TYPE_UINT32:
      break;
    case GGUF_VALUE_TYPE_INT32:
      break;
    case GGUF_VALUE_TYPE_FLOAT32:
      break;
    case GGUF_VALUE_TYPE_BOOL:
      break;
    case GGUF_VALUE_TYPE_STRING:
      *value =
          std::string(val->string.string, static_cast<int>(val->string.len));
      break;
    case GGUF_VALUE_TYPE_UINT64:
      break;
    case GGUF_VALUE_TYPE_INT64:
      break;
    case GGUF_VALUE_TYPE_FLOAT64:
      break;
    default:
      throw std::runtime_error("[load_gguf] unknown value type");
      break;
  }
}

std::unordered_map<std::string, MetaData> load_metadata(gguf_ctx* ctx) {
  std::unordered_map<std::string, MetaData> metadata;
  gguf_key key;
  while (gguf_get_key(ctx, &key)) {
    std::string key_name = std::string(key.name, key.namelen);
    MetaData value;
    gguf_do_with_value(
        ctx, key.type, key.val, &value, 0, 0, metadata_value_callback);
    metadata.insert({key_name, value});
  }
  return metadata;
}

std::unordered_map<std::string, array> load_arrays(gguf_ctx* ctx) {
  std::unordered_map<std::string, array> array_map;
  gguf_tensor tensor;
  while (gguf_get_tensor(ctx, &tensor)) {
    std::vector<int> shape;
    // The dimension order in GGML is the reverse of the order used in MLX.
    for (int i = tensor.ndim - 1; i >= 0; i--) {
      shape.push_back(tensor.dim[i]);
    }
    const auto& [data, dtype] = extract_tensor_data(&tensor);
    array loaded_array = array(data, shape, dtype);
    std::string name = std::string(tensor.name, tensor.namelen);
    array_map.insert({name, loaded_array});
  }
  return array_map;
}

std::pair<
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, MetaData>>
load_gguf(const std::string& file, StreamOrDevice s) {
  gguf_ctx* ctx = gguf_open(file.c_str());
  if (!ctx) {
    throw std::runtime_error("[load_gguf] gguf_init failed");
  }
  auto metadata = load_metadata(ctx);
  auto arrays = load_arrays(ctx);
  gguf_close(ctx);
  return {arrays, metadata};
}

void save_gguf(
    std::string file,
    std::unordered_map<std::string, array> array_map,
    std::unordered_map<std::string, MetaData> metadata /* = {} */) {
  // Add .gguf to file name if it is not there
  if (file.length() < 5 || file.substr(file.length() - 5, 5) != ".gguf") {
    file += ".gguf";
  }

  gguf_ctx* ctx = gguf_create(file.c_str(), GGUF_OVERWRITE);
  if (!ctx) {
    throw std::runtime_error("[save_gguf] gguf_create failed");
  }

  // Save any meta data
  for (const auto& [key, value] : metadata) {
    if (auto pv = std::get_if<std::string>(&value); pv) {
      const std::string& str = *pv;
      size_t size = sizeof(gguf_string) + str.length();
      std::vector<char> val_vec(size + 1);
      gguf_string* val = reinterpret_cast<gguf_string*>(val_vec.data());
      val->len = str.length();
      memcpy(val->string, str.c_str(), str.length());
      val->string[str.length()] = '\0';
      gguf_append_kv(
          ctx,
          key.c_str(),
          key.length(),
          GGUF_VALUE_TYPE_STRING,
          static_cast<void*>(val),
          size);
    }
    // TODO: serialize other types
  }

  // Tensor offsets are relative to data section, so we start at offset 0.
  uint64_t tensor_offset = 0;

  // First, append the tensor info
  for (auto& [key, arr] : array_map) {
    arr.eval();

    // Try to make it row contiguous
    if (!arr.flags().row_contiguous) {
      arr = reshape(flatten(arr), arr.shape());
      arr.eval();
    }

    // Has to be row-major now but, check one more time in case
    // any of the above change in the future
    if (!arr.flags().row_contiguous) {
      throw std::invalid_argument(
          "[save_gguf] can only serialize row-major arrays");
    }

    tensor_offset += gguf_get_alignment_padding(ctx->alignment, tensor_offset);
    const std::optional<uint32_t> gguf_type =
        dtype_to_gguf_tensor_type(arr.dtype());
    if (!gguf_type.has_value()) {
      std::ostringstream msg;
      msg << "[save_gguf] dtype " << arr.dtype() << " is not supported";
      throw std::runtime_error(msg.str());
    }
    const char* tensorname = key.c_str();
    const uint64_t namelen = key.length();
    const uint32_t num_dim = arr.ndim();
    uint64_t dim[num_dim];
    for (int i = 0; i < num_dim; i++) {
      dim[i] = arr.shape()[num_dim - 1 - i];
    }
    if (!gguf_append_tensor_info(
            ctx,
            tensorname,
            namelen,
            num_dim,
            dim,
            gguf_type.value(),
            tensor_offset)) {
      throw std::runtime_error("[save_gguf] gguf_append_tensor_info failed");
    }
    tensor_offset += arr.nbytes();
  }

  // Then, append the tensor weights
  for (const auto& [key, arr] : array_map) {
    if (!gguf_append_tensor_data(ctx, (void*)arr.data<void>(), arr.nbytes())) {
      throw std::runtime_error("[save_gguf] gguf_append_tensor_data failed");
    }
  }
  gguf_close(ctx);
}

} // namespace mlx::core
