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
  while (gguf_get_tensor(ctx,&tensor)) {
    printf("\n[load_gguf] %s tensor %.*s @%llu, %llu weights, %llu bytes\n",
        gguf_get_tensor_type_name(tensor.type),
        (int)tensor.namelen,
        tensor.name,
        tensor.offset,
        tensor.num_weights,
        tensor.bsize);
    params += tensor.num_weights;

    std::vector<int> shape;
    for (int i = 0; i < tensor.ndim; i++) {
      shape.push_back(tensor.dim[i]);
    }
    const float* data = gguf_tensor_to_float(&tensor);
    if (data == NULL) {
      throw std::runtime_error("[load_gguf] gguf_tensor_to_float failed");
    }
    for (int i = 0; i < tensor.num_weights; i++) {
      printf("data %d: %f ", i, data[i]);
    }
    array loaded_array = ones(shape, float32, s);
    // loaded_array.set_data((float*)data);
    
    std::string name = std::string(tensor.name, tensor.namelen);
    result.insert({name, loaded_array});
  }
  printf("[load_gguf] parameters: %.02fB\n",
    (double)params/1000000000);
  gguf_end(ctx);
  return result;
}

void save_gguf(
    const std::string& file_,
    std::unordered_map<std::string, array> a,
    std::optional<bool> retain_graph) {
  // Open and check file
  std::string file = file_;

  // Add .safetensors to file name if it is not there
  if (file.length() < 5 ||
      file.substr(file.length() - 5, 5) != ".gguf")
    file += ".gguf";

  gguf_ctx* ctx = gguf_create(file.c_str());
  if (!ctx) {
    throw std::runtime_error("[save_gguf] gguf_create failed");
  }

  // Tensor offsets are relative to data section, so we start at offset 0.
  uint64_t tensor_offset = 0;
  
  // First, append the tensor info
  for (auto& [key, arr] : a) {
    tensor_offset += gguf_get_alignment_padding(ctx->alignment, tensor_offset);
    if (arr.dtype() != float32) {
      throw std::runtime_error("[save_gguf] only float32 supported");
    }
    const uint32_t type = GGUF_TYPE_F32;
    const char *tensorname = key.c_str();
    const uint64_t namelen = key.length();
    const uint32_t num_dim = arr.shape().size();
    uint64_t dim[num_dim];
    for (int i = 0; i < num_dim; i++) {
      dim[i] = arr.shape()[i];
    }
    printf("info: %s; nbytes: %zu; offset: %llu\n", tensorname, arr.nbytes(), tensor_offset);
    if (!gguf_append_tensor_info(ctx, tensorname, namelen, num_dim, dim, type, tensor_offset)) {
      throw std::runtime_error("[save_gguf] gguf_append_tensor_info failed");
    }
    tensor_offset += arr.nbytes();
  }

  // Then, append the tensors weights
  for (auto& [key, arr] : a) {
    printf("weights: %s, nbytes: %zu\n", key.c_str(), arr.nbytes());
    if (!gguf_append_tensor_data(ctx, arr.data<float>(), arr.nbytes())) {
      throw std::runtime_error("[save_gguf] gguf_append_tensor_data failed");
    }
  }
}

} // namespace mlx::core
