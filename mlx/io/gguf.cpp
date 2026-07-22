// Copyright © 2023-2024 Apple Inc.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <numeric>

#include "mlx/io/gguf.h"
#include "mlx/ops.h"

namespace mlx::core {

// https://github.com/antirez/gguf-tools/blob/af7d88d808a7608a33723fba067036202910acb3/gguflib.h#L102-L108
constexpr int gguf_array_header_size = 12;

namespace {

// GGUF metadata is byte-packed, so values need not satisfy host alignment.
template <typename T>
T read_unaligned(const void* src) {
  T value;
  std::memcpy(&value, src, sizeof(value));
  return value;
}

template <typename T>
void write_unaligned(void* dst, T value) {
  std::memcpy(dst, &value, sizeof(value));
}

array array_from_bytes(const void* src, int size, Dtype dtype) {
  auto nbytes = static_cast<size_t>(size) * size_of(dtype);
  auto buffer = allocator::malloc(nbytes);
  std::memcpy(buffer.raw_ptr(), src, nbytes);
  return array(buffer, {size}, dtype);
}

} // namespace

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

Shape get_shape(const gguf_tensor& tensor) {
  Shape shape;
  // The dimension order in GGML is the reverse of the order used in MLX.
  for (int i = tensor.ndim - 1; i >= 0; i--) {
    shape.push_back(tensor.dim[i]);
  }
  return shape;
}

std::tuple<allocator::Buffer, Dtype> extract_tensor_data(gguf_tensor* tensor) {
  if (tensor == nullptr) {
    throw std::invalid_argument(
        "[extract_tensor_data] Input tensor pointer is null.");
  }
  std::optional<Dtype> equivalent_dtype = gguf_type_to_dtype(tensor->type);
  // If there's an equivalent type, we can simply copy.
  if (equivalent_dtype.has_value()) {
    if (tensor->weights_data == nullptr) {
      throw std::runtime_error("[load_gguf] NULL tensor data pointer");
    }
    allocator::Buffer buffer = allocator::malloc(tensor->bsize);
    memcpy(
        buffer.raw_ptr(),
        tensor->weights_data,
        tensor->num_weights * equivalent_dtype.value().size());
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
    GGUFMetaData& value) {
  switch (type) {
    case GGUF_VALUE_TYPE_UINT8:
      value = array(read_unaligned<uint8_t>(val), uint8);
      break;
    case GGUF_VALUE_TYPE_INT8:
      value = array(read_unaligned<int8_t>(val), int8);
      break;
    case GGUF_VALUE_TYPE_UINT16:
      value = array(read_unaligned<uint16_t>(val), uint16);
      break;
    case GGUF_VALUE_TYPE_INT16:
      value = array(read_unaligned<int16_t>(val), int16);
      break;
    case GGUF_VALUE_TYPE_UINT32:
      value = array(read_unaligned<uint32_t>(val), uint32);
      break;
    case GGUF_VALUE_TYPE_INT32:
      value = array(read_unaligned<int32_t>(val), int32);
      break;
    case GGUF_VALUE_TYPE_UINT64:
      value = array(read_unaligned<uint64_t>(val), uint64);
      break;
    case GGUF_VALUE_TYPE_INT64:
      value = array(read_unaligned<int64_t>(val), int64);
      break;
    case GGUF_VALUE_TYPE_FLOAT32:
      value = array(read_unaligned<float>(val), float32);
      break;
    case GGUF_VALUE_TYPE_BOOL:
      value = array(read_unaligned<uint8_t>(val), bool_);
      break;
    case GGUF_VALUE_TYPE_STRING: {
      auto length = read_unaligned<uint64_t>(val);
      auto data = reinterpret_cast<const char*>(val) + sizeof(length);
      value = std::string(data, static_cast<size_t>(length));
      break;
    }
    case GGUF_VALUE_TYPE_FLOAT64:
      value = array(read_unaligned<double>(val), float32);
      break;
    case GGUF_VALUE_TYPE_ARRAY: {
      ctx->off += gguf_array_header_size; // Skip header
      auto data = reinterpret_cast<const char*>(val) + gguf_array_header_size;
      auto array_type = read_unaligned<uint32_t>(val);
      auto size = static_cast<int>(read_unaligned<uint64_t>(
          reinterpret_cast<const char*>(val) + sizeof(array_type)));
      if (array_type == GGUF_VALUE_TYPE_ARRAY) {
        throw std::invalid_argument(
            "[load_gguf] Only supports loading 1-layer of nested arrays.");
      }
      switch (array_type) {
        case GGUF_VALUE_TYPE_UINT8:
          value = array_from_bytes(data, size, uint8);
          break;
        case GGUF_VALUE_TYPE_INT8:
          value = array_from_bytes(data, size, int8);
          break;
        case GGUF_VALUE_TYPE_UINT16:
          value = array_from_bytes(data, size, uint16);
          break;
        case GGUF_VALUE_TYPE_INT16:
          value = array_from_bytes(data, size, int16);
          break;
        case GGUF_VALUE_TYPE_UINT32:
          value = array_from_bytes(data, size, uint32);
          break;
        case GGUF_VALUE_TYPE_INT32:
          value = array_from_bytes(data, size, int32);
          break;
        case GGUF_VALUE_TYPE_UINT64:
          value = array_from_bytes(data, size, uint64);
          break;
        case GGUF_VALUE_TYPE_INT64:
          value = array_from_bytes(data, size, int64);
          break;
        case GGUF_VALUE_TYPE_FLOAT32:
          value = array_from_bytes(data, size, float32);
          break;
        case GGUF_VALUE_TYPE_BOOL:
          value = array_from_bytes(data, size, bool_);
          break;
        case GGUF_VALUE_TYPE_STRING: {
          std::vector<std::string> strs(size);
          for (auto& str : strs) {
            auto length = read_unaligned<uint64_t>(data);
            data += sizeof(length);
            str = std::string(data, static_cast<size_t>(length));
            data += length;
            ctx->off += (length + sizeof(gguf_string));
          }
          value = std::move(strs);
          break;
        }
        case GGUF_VALUE_TYPE_FLOAT64: {
          std::vector<double> values(size);
          std::memcpy(values.data(), data, values.size() * sizeof(double));
          value = array(values.begin(), {size}, float32);
          break;
        }
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

std::unordered_map<std::string, GGUFMetaData> load_metadata(gguf_ctx* ctx) {
  std::unordered_map<std::string, GGUFMetaData> metadata;
  gguf_key key;
  while (gguf_get_key(ctx, &key)) {
    std::string key_name = std::string(key.name, key.namelen);
    auto& val = metadata.insert({key_name, GGUFMetaData{}}).first->second;
    set_mx_value_from_gguf(ctx, key.type, key.val, val);
  }
  return metadata;
}

std::unordered_map<std::string, array> load_arrays(gguf_ctx* ctx) {
  std::unordered_map<std::string, array> array_map;
  gguf_tensor tensor;

  auto check_insert = [](const auto& inserted) {
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
      std::string name(tensor.name, tensor.namelen);
      const auto& [data, dtype] = extract_tensor_data(&tensor);
      array loaded_array = array(data, get_shape(tensor), dtype);
      check_insert(array_map.insert({name, loaded_array}));
    }
  }
  return array_map;
}

GGUFLoad load_gguf(const std::string& file, StreamOrDevice s) {
  bool exists;
  {
    std::ifstream f(file.c_str());
    exists = f.good();
  }
  if (!exists) {
    throw std::invalid_argument("[load_gguf] Failed to open " + file);
  }

  std::unique_ptr<gguf_ctx, decltype(&gguf_close)> ctx(
      gguf_open(file.data()), gguf_close);
  if (!ctx) {
    throw std::runtime_error("[load_gguf] gguf_init failed");
  }
  auto metadata = load_metadata(ctx.get());
  auto arrays = load_arrays(ctx.get());
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
    write_unaligned(val_vec.data(), gguf_type);
    write_unaligned(
        val_vec.data() + sizeof(gguf_type), static_cast<uint64_t>(val.size()));
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
    std::unordered_map<std::string, GGUFMetaData> metadata /* = {} */) {
  // Add .gguf to file name if it is not there
  if (file.length() < 5 || file.substr(file.length() - 5, 5) != ".gguf") {
    file += ".gguf";
  }

  std::unique_ptr<gguf_ctx, decltype(&gguf_close)> ctx(
      gguf_create(file.c_str(), GGUF_OVERWRITE), gguf_close);
  if (!ctx) {
    throw std::runtime_error("[save_gguf] gguf_create failed");
  }

  auto string_to_gguf = [](char* dst, const std::string& src) {
    write_unaligned(dst, static_cast<uint64_t>(src.length()));
    memcpy(dst + sizeof(uint64_t), src.c_str(), src.length());
  };

  // Save any meta data
  for (auto& [key, value] : metadata) {
    if (auto pv = std::get_if<std::string>(&value); pv) {
      const std::string& str = *pv;
      size_t size = sizeof(gguf_string) + str.length();
      std::vector<char> val_vec(size);
      string_to_gguf(val_vec.data(), str);
      gguf_append_kv(
          ctx.get(),
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
      write_unaligned(
          val_vec.data(), static_cast<uint32_t>(GGUF_VALUE_TYPE_STRING));
      write_unaligned(
          val_vec.data() + sizeof(uint32_t),
          static_cast<uint64_t>(str_vec.size()));
      auto str_ptr = val_vec.data() + gguf_array_header_size;
      for (auto& str : str_vec) {
        string_to_gguf(str_ptr, str);
        str_ptr += str.length() + sizeof(gguf_string);
      }
      gguf_append_kv(
          ctx.get(),
          key.c_str(),
          key.length(),
          GGUF_VALUE_TYPE_ARRAY,
          static_cast<void*>(val_vec.data()),
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
          append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_FLOAT32);
          break;
        case int64:
          append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_INT64);
          break;
        case int32:
          append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_INT32);
          break;
        case int16:
          append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_INT16);
          break;
        case int8:
          append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_INT8);
          break;
        case uint64:
          append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_UINT64);
          break;
        case uint32:
          append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_UINT32);
          break;
        case uint16:
          append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_UINT16);
          break;
        case uint8:
          append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_UINT8);
          break;
        case bool_:
          append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_BOOL);
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
    std::vector<uint64_t> dim(num_dim);
    for (int i = 0; i < num_dim; i++) {
      dim[i] = arr.shape()[num_dim - 1 - i];
    }
    if (!gguf_append_tensor_info(
            ctx.get(),
            tensorname,
            namelen,
            num_dim,
            dim.data(),
            gguf_type.value(),
            tensor_offset)) {
      throw std::runtime_error("[save_gguf] gguf_append_tensor_info failed");
    }
    tensor_offset += arr.nbytes();
  }

  // Then, append the tensor weights
  for (const auto& [key, arr] : array_map) {
    if (!gguf_append_tensor_data(
            ctx.get(), (void*)arr.data<void>(), arr.nbytes())) {
      throw std::runtime_error("[save_gguf] gguf_append_tensor_data failed");
    }
  }
}

} // namespace mlx::core
