#include "mlx/io/gguf.h"

namespace mlx::core {

std::optional<uint32_t> dtype_to_gguf_type(const Dtype& dtype) {
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
  // Otherwise, we need to convert to float32.
  const float* data = gguf_tensor_to_float(tensor);
  if (data == NULL) {
    throw std::runtime_error("[load_gguf] gguf_tensor_to_float failed");
  }
  allocator::Buffer buffer = allocator::malloc(tensor->bsize);
  std::copy(data, data + tensor->num_weights, (float*)buffer.raw_ptr());
  return {buffer, float32};
}

std::unordered_map<std::string, array> load_gguf(
    const std::string& file,
    StreamOrDevice s) {
  std::unordered_map<std::string, array> result;
  gguf_ctx* ctx = gguf_init(file.c_str());
  if (!ctx) {
    throw std::runtime_error("[load_gguf] gguf_init failed");
  }
  gguf_skip_key_values_section(ctx);
  gguf_tensor tensor;
  uint64_t params = 0;
  while (gguf_get_tensor(ctx, &tensor)) {
    params += tensor.num_weights;
    std::vector<int> shape;
    for (int i = 0; i < tensor.ndim; i++) {
      shape.push_back(tensor.dim[i]);
    }
    const auto& [data, dtype] = extract_tensor_data(&tensor);
    array loaded_array = array(data, shape, dtype);
    std::string name = std::string(tensor.name, tensor.namelen);
    result.insert({name, loaded_array});
  }
  gguf_end(ctx);
  return result;
}

void save_gguf(
    const std::string& file_,
    std::unordered_map<std::string, array> a,
    std::optional<bool> retain_graph) {
  std::string file = file_;

  // Add .gguf to file name if it is not there
  if (file.length() < 5 || file.substr(file.length() - 5, 5) != ".gguf")
    file += ".gguf";

  gguf_ctx* ctx = gguf_create(file.c_str());
  if (!ctx) {
    throw std::runtime_error("[save_gguf] gguf_create failed");
  }

  // Tensor offsets are relative to data section, so we start at offset 0.
  uint64_t tensor_offset = 0;

  // First, append the tensor info
  for (auto& [key, arr] : a) {
    arr.eval(retain_graph.value_or(arr.is_tracer()));

    tensor_offset += gguf_get_alignment_padding(ctx->alignment, tensor_offset);
    const std::optional<uint32_t> gguf_type = dtype_to_gguf_type(arr.dtype());

    if (!gguf_type.has_value()) {
      throw std::runtime_error("[save_gguf] dtype is not supported");
    }
    const char* tensorname = key.c_str();
    const uint64_t namelen = key.length();
    const uint32_t num_dim = arr.shape().size();
    uint64_t dim[num_dim];
    for (int i = 0; i < num_dim; i++) {
      dim[i] = arr.shape()[i];
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
  for (const auto& [key, arr] : a) {
    if (!gguf_append_tensor_data(ctx, (void*)arr.data<void>(), arr.nbytes())) {
      throw std::runtime_error("[save_gguf] gguf_append_tensor_data failed");
    }
  }
}

} // namespace mlx::core
