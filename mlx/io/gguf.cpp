#include "mlx/io/gguf.h"

namespace mlx::core {

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
    const float* data = gguf_tensor_to_float(&tensor);
    if (data == NULL) {
      throw std::runtime_error("[load_gguf] gguf_tensor_to_float failed");
    }
    allocator::Buffer buffer = allocator::malloc(tensor.bsize);
    std::copy(data, data + tensor.num_weights, (float*)buffer.raw_ptr());
    array loaded_array = array(data, shape, float32);

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

  // Add .safetensors to file name if it is not there
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
    if (arr.dtype() != float32) {
      throw std::runtime_error("[save_gguf] only float32 supported");
    }
    const uint32_t type = GGUF_TYPE_F32;
    const char* tensorname = key.c_str();
    const uint64_t namelen = key.length();
    const uint32_t num_dim = arr.shape().size();
    uint64_t dim[num_dim];
    for (int i = 0; i < num_dim; i++) {
      dim[i] = arr.shape()[i];
    }
    if (!gguf_append_tensor_info(
            ctx, tensorname, namelen, num_dim, dim, type, tensor_offset)) {
      throw std::runtime_error("[save_gguf] gguf_append_tensor_info failed");
    }
    tensor_offset += arr.nbytes();
  }

  // Then, append the tensor weights
  for (const auto& [key, arr] : a) {
    if (!gguf_append_tensor_data(ctx, (void*)arr.data<float>(), arr.nbytes())) {
      throw std::runtime_error("[save_gguf] gguf_append_tensor_data failed");
    }
  }
}

} // namespace mlx::core
