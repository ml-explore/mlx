// Copyright © 2023-2024 Apple Inc.

#include <cstdint>
#include <cstring>
#include <numeric>

#include <mlx/io/gguf.h>

namespace mlx::core {

// https://github.com/antirez/gguf-tools/blob/af7d88d808a7608a33723fba067036202910acb3/gguflib.h#L102-L108
constexpr int gguf_array_header_size = 12;

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

std::vector<int> get_shape(const gguf_tensor& tensor) {
  std::vector<int> shape;
  // The dimension order in GGML is the reverse of the order used in MLX.
  for (int i = tensor.ndim - 1; i >= 0; i--) {
    shape.push_back(tensor.dim[i]);
  }
  return shape;
}

std::tuple<allocator::Buffer, Dtype> extract_tensor_data(gguf_tensor* tensor) {
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

void set_mx_value_from_gguf(
    gguf_ctx* ctx,
    uint32_t type,
    gguf_value* val,
    MetaData& value) {
  switch (type) {
    case GGUF_VALUE_TYPE_UINT8:
      value = array(val->uint8, uint8);
      break;
    case GGUF_VALUE_TYPE_INT8:
      value = array(val->int8, int8);
      break;
    case GGUF_VALUE_TYPE_UINT16:
      value = array(val->uint16, uint16);
      break;
    case GGUF_VALUE_TYPE_INT16:
      value = array(val->int16, int16);
      break;
    case GGUF_VALUE_TYPE_UINT32:
      value = array(val->uint32, uint32);
      break;
    case GGUF_VALUE_TYPE_INT32:
      value = array(val->int32, int32);
      break;
    case GGUF_VALUE_TYPE_UINT64:
      value = array(val->uint64, uint64);
      break;
    case GGUF_VALUE_TYPE_INT64:
      value = array(val->int64, int64);
      break;
    case GGUF_VALUE_TYPE_FLOAT32:
      value = array(val->float32, float32);
      break;
    case GGUF_VALUE_TYPE_BOOL:
      value = array(val->boolval, bool_);
      break;
    case GGUF_VALUE_TYPE_STRING:
      value =
          std::string(val->string.string, static_cast<int>(val->string.len));
      break;
    case GGUF_VALUE_TYPE_FLOAT64:
      value = array(val->float64, float32);
      break;
    case GGUF_VALUE_TYPE_ARRAY: {
      ctx->off += gguf_array_header_size; // Skip header
      char* data = reinterpret_cast<char*>(val) + gguf_array_header_size;
      auto size = static_cast<int>(val->array.len);
      if (val->array.type == GGUF_VALUE_TYPE_ARRAY) {
        throw std::invalid_argument(
            "[load_gguf] Only supports loading 1-layer of nested arrays.");
      }
      switch (val->array.type) {
        case GGUF_VALUE_TYPE_UINT8:
          value = array(reinterpret_cast<uint8_t*>(data), {size}, uint8);
          break;
        case GGUF_VALUE_TYPE_INT8:
          value = array(reinterpret_cast<int8_t*>(data), {size}, int8);
          break;
        case GGUF_VALUE_TYPE_UINT16:
          value = array(reinterpret_cast<uint16_t*>(data), {size}, uint16);
          break;
        case GGUF_VALUE_TYPE_INT16:
          value = array(reinterpret_cast<int16_t*>(data), {size}, int16);
          break;
        case GGUF_VALUE_TYPE_UINT32:
          value = array(reinterpret_cast<uint32_t*>(data), {size}, uint32);
          break;
        case GGUF_VALUE_TYPE_INT32:
          value = array(reinterpret_cast<int32_t*>(data), {size}, int32);
          break;
        case GGUF_VALUE_TYPE_UINT64:
          value = array(reinterpret_cast<uint64_t*>(data), {size}, uint64);
          break;
        case GGUF_VALUE_TYPE_INT64:
          value = array(reinterpret_cast<uint64_t*>(data), {size}, int64);
          break;
        case GGUF_VALUE_TYPE_FLOAT32:
          value = array(reinterpret_cast<float*>(data), {size}, float32);
          break;
        case GGUF_VALUE_TYPE_BOOL:
          value = array(reinterpret_cast<bool*>(data), {size}, bool_);
          break;
        case GGUF_VALUE_TYPE_STRING: {
          std::vector<std::string> strs(size);
          for (auto& str : strs) {
            auto str_val = reinterpret_cast<gguf_string*>(data);
            data += (str_val->len + sizeof(gguf_string));
            str = std::string(str_val->string, static_cast<int>(str_val->len));
            ctx->off += (str_val->len + sizeof(gguf_string));
          }
          value = std::move(strs);
          break;
        }
        case GGUF_VALUE_TYPE_FLOAT64:
          value = array(reinterpret_cast<double*>(data), {size}, float32);
          break;
        default:
          throw std::runtime_error(
              "[load_gguf] Multiple levels of nested arrays are not supported.");
      }
      break;
    }
    default:
      throw std::runtime_error("[load_gguf] Received unexpected type.");
      break;
  }
  if (type == GGUF_VALUE_TYPE_STRING) {
    ctx->off += (sizeof(gguf_string) + std::get<std::string>(value).size());
  } else if (auto pv = std::get_if<array>(&value); pv) {
    ctx->off += pv->nbytes();
  }
}

std::unordered_map<std::string, MetaData> load_metadata(gguf_ctx* ctx) {
  std::unordered_map<std::string, MetaData> metadata;
  gguf_key key;
  while (gguf_get_key(ctx, &key)) {
    std::string key_name = std::string(key.name, key.namelen);
    auto& val = metadata.insert({key_name, MetaData{}}).first->second;
    set_mx_value_from_gguf(ctx, key.type, key.val, val);
  }
  return metadata;
}

std::unordered_map<std::string, array> load_arrays(gguf_ctx* ctx) {
  std::unordered_map<std::string, array> array_map;
  gguf_tensor tensor;

  auto check_insert = [](auto inserted) {
    if (!inserted.second) {
      std::ostringstream msg;
      msg << "[load_gguf] Duplicate parameter name " << inserted.first->second
          << " this can happend when loading quantized tensors.";
      throw std::runtime_error(msg.str());
    }
  };

  while (gguf_get_tensor(ctx, &tensor)) {
    if (tensor.type == GGUF_TYPE_Q4_0 || tensor.type == GGUF_TYPE_Q4_1 ||
        tensor.type == GGUF_TYPE_Q8_0) {
      gguf_load_quantized(array_map, tensor);
    } else {
      std::string name = std::string(tensor.name, tensor.namelen);

      const auto& [data, dtype] = extract_tensor_data(&tensor);
      array loaded_array = array(data, get_shape(tensor), dtype);
      array_map.insert({name, loaded_array});
    }
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

void append_kv_array(
    gguf_ctx* ctx,
    const std::string& key,
    array& val,
    uint32_t gguf_type) {
  if (val.ndim() == 1) {
    size_t gguf_size = val.nbytes() + gguf_array_header_size;
    std::vector<char> val_vec(gguf_size);
    gguf_value* gguf_val = reinterpret_cast<gguf_value*>(val_vec.data());
    gguf_val->array.type = gguf_type;
    gguf_val->array.len = val.size();
    memcpy(
        val_vec.data() + gguf_array_header_size,
        val.data<char>(),
        val.nbytes());
    gguf_append_kv(
        ctx,
        key.c_str(),
        key.length(),
        GGUF_VALUE_TYPE_ARRAY,
        reinterpret_cast<void*>(val_vec.data()),
        gguf_size);
  } else {
    gguf_append_kv(
        ctx,
        key.c_str(),
        key.length(),
        gguf_type,
        reinterpret_cast<void*>(val.data<char>()),
        val.nbytes());
  }
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

  auto string_to_gguf = [](char* dst, const std::string& src) {
    gguf_string* val = reinterpret_cast<gguf_string*>(dst);
    val->len = src.length();
    memcpy(val->string, src.c_str(), src.length());
  };

  // Save any meta data
  for (auto& [key, value] : metadata) {
    if (auto pv = std::get_if<std::string>(&value); pv) {
      const std::string& str = *pv;
      size_t size = sizeof(gguf_string) + str.length();
      std::vector<char> val_vec(size);
      string_to_gguf(val_vec.data(), str);
      gguf_append_kv(
          ctx,
          key.c_str(),
          key.length(),
          GGUF_VALUE_TYPE_STRING,
          static_cast<void*>(val_vec.data()),
          size);
    } else if (auto pv = std::get_if<std::vector<std::string>>(&value); pv) {
      const auto& str_vec = *pv;
      auto mem_size = std::accumulate(
          str_vec.begin(), str_vec.end(), 0, [](size_t accum, const auto& s) {
            return accum + s.size();
          });
      mem_size += str_vec.size() * sizeof(gguf_string) + gguf_array_header_size;
      std::vector<char> val_vec(mem_size);
      gguf_value* val = reinterpret_cast<gguf_value*>(val_vec.data());
      val->array.type = GGUF_VALUE_TYPE_STRING;
      val->array.len = str_vec.size();
      auto str_ptr = val_vec.data() + gguf_array_header_size;
      for (auto& str : str_vec) {
        string_to_gguf(str_ptr, str);
        str_ptr += str.length() + sizeof(gguf_string);
      }
      gguf_append_kv(
          ctx,
          key.c_str(),
          key.length(),
          GGUF_VALUE_TYPE_ARRAY,
          static_cast<void*>(val),
          mem_size);
    } else if (auto pv = std::get_if<array>(&value); pv) {
      array v = *pv;
      if (v.ndim() > 1) {
        throw std::runtime_error(
            "[save_gguf] Cannot save arrays with more than one dimension.");
      }
      if (v.size() == 0) {
        throw std::runtime_error("[save_gguf] Cannot save empty arrays.");
      }

      eval(v);
      if (!v.flags().row_contiguous) {
        v = reshape(flatten(v), v.shape());
      }
      if (!v.flags().row_contiguous) {
        throw std::runtime_error(
            "[save_gguf] Cannot save non contiguous arrays.");
      }
      switch (v.dtype()) {
        case float32:
          append_kv_array(ctx, key, v, GGUF_VALUE_TYPE_FLOAT32);
          break;
        case int64:
          append_kv_array(ctx, key, v, GGUF_VALUE_TYPE_INT64);
          break;
        case int32:
          append_kv_array(ctx, key, v, GGUF_VALUE_TYPE_INT32);
          break;
        case int16:
          append_kv_array(ctx, key, v, GGUF_VALUE_TYPE_INT16);
          break;
        case int8:
          append_kv_array(ctx, key, v, GGUF_VALUE_TYPE_INT8);
          break;
        case uint64:
          append_kv_array(ctx, key, v, GGUF_VALUE_TYPE_UINT64);
          break;
        case uint32:
          append_kv_array(ctx, key, v, GGUF_VALUE_TYPE_UINT32);
          break;
        case uint16:
          append_kv_array(ctx, key, v, GGUF_VALUE_TYPE_UINT16);
          break;
        case uint8:
          append_kv_array(ctx, key, v, GGUF_VALUE_TYPE_UINT8);
          break;
        case bool_:
          append_kv_array(ctx, key, v, GGUF_VALUE_TYPE_BOOL);
          break;
        default:
          std::ostringstream msg;
          msg << "[save_gguf] array type " << v.dtype()
              << " not support for metadata.";
          throw std::invalid_argument(msg.str());
      }
    } else {
      throw std::runtime_error(
          "[save_gguf] Received unexpected type in metadata");
    }
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
