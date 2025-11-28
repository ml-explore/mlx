// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device/config.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/dtype_utils.h"

#include <cudnn_frontend.h>
#include <fmt/format.h>

namespace mlx::core {

namespace cu {
class CommandEncoder;
}

namespace fe = cudnn_frontend;

#define CHECK_CUDNN_FE_ERROR(cmd)                                    \
  do {                                                               \
    auto error = cmd;                                                \
    if (!error.is_good()) {                                          \
      throw std::runtime_error(                                      \
          fmt::format("{} failed: {}.", #cmd, error.get_message())); \
    }                                                                \
  } while (0)

// Return pointer alignment of |x|'s data.
inline uint8_t get_alignment(const array& x) {
  uint8_t alignment = 1;
  uintptr_t address = reinterpret_cast<uintptr_t>(gpu_ptr<void>(x));
  for (; alignment < 32; alignment *= 2) {
    if (address % (alignment * 2)) {
      return alignment;
    }
  }
  return alignment;
}

// Convert the type of elements in |vec| to |T|.
template <typename T, typename Vec>
inline std::vector<T> convert_vector(const Vec& vec) {
  return std::vector<T>(vec.begin(), vec.end());
}

// Map dtype to cudnn data type.
inline fe::DataType_t dtype_to_cudnn_type(Dtype dtype) {
  switch (dtype) {
    case int8:
      return fe::DataType_t::INT8;
    case int32:
      return fe::DataType_t::INT32;
    case uint8:
      return fe::DataType_t::UINT8;
    case float16:
      return fe::DataType_t::HALF;
    case bfloat16:
      return fe::DataType_t::BFLOAT16;
    case float32:
      return fe::DataType_t::FLOAT;
    case float64:
      return fe::DataType_t::DOUBLE;
    default:
      throw std::runtime_error(fmt::format(
          "Unsupported dtype in cuDNN: {}.", dtype_to_string(dtype)));
  }
}

// Return an array that can be used as map key for |vec| with size <= MAX_NDIM.
//
// There are 2 differences from the const_param util from kernel_utils.cuh:
// 1. The rest of array is filled with 0.
// 2. This util can be used in .cpp files.
template <int NDIM = MAX_NDIM, typename T, template <typename U> class Vec>
inline std::array<T, NDIM> vector_key(const Vec<T>& vec) {
  if (vec.size() > NDIM) {
    throw std::runtime_error(
        fmt::format("ndim can not be larger than {}.", NDIM));
  }
  std::array<T, NDIM> result = {};
  std::copy_n(vec.begin(), vec.size(), result.begin());
  return result;
}

// Extends cuDNN graph with helpers.
class DnnGraph : public fe::graph::Graph {
 public:
  DnnGraph(cudnnHandle_t handle, Dtype io_dtype, Dtype compute_dtype = float32)
      : handle_(handle) {
    set_io_data_type(dtype_to_cudnn_type(io_dtype));
    set_intermediate_data_type(dtype_to_cudnn_type(compute_dtype));
    set_compute_data_type(dtype_to_cudnn_type(compute_dtype));
  }

  // Create a cuDNN tensor description from MLX array |x|.
  auto& tensor(
      std::shared_ptr<fe::graph::Tensor_attributes>& attrs,
      int64_t uid,
      const array& x) {
    set_tensor_attrs(attrs, uid, x);
    return attrs;
  }
  auto tensor(const char* name, int64_t uid, const array& x) {
    auto attrs = Graph::tensor(fe::graph::Tensor_attributes().set_name(name));
    tensor(attrs, uid, x);
    return attrs;
  }

  // Create a cuDNN tensor description from MLX array |x|, and transpose it from
  // NHWC layout to NCHW.
  auto& tensor_nchw(
      std::shared_ptr<fe::graph::Tensor_attributes>& attrs,
      int64_t uid,
      const array& x) {
    set_tensor_attrs_nchw(attrs, uid, x);
    return attrs;
  }
  auto tensor_nchw(const char* name, int64_t uid, const array& x) {
    auto attrs = Graph::tensor(fe::graph::Tensor_attributes().set_name(name));
    tensor_nchw(attrs, uid, x);
    return attrs;
  }

  // Create a cuDNN tensor for scalar.
  auto scalar(const char* name, int64_t uid, Dtype dtype) {
    return Graph::tensor(fe::graph::Tensor_attributes()
                             .set_name(name)
                             .set_uid(uid)
                             .set_dim({1, 1, 1, 1})
                             .set_stride({1, 1, 1, 1})
                             .set_is_pass_by_value(true)
                             .set_data_type(dtype_to_cudnn_type(dtype)));
  }

  // Call this before setting notes.
  fe::error_t prepare();
  // Call this after setting notes.
  fe::error_t build();

  // Add cuDNN graph to CUDA graph, using native CUDA graph API.
  fe::error_t encode_graph(
      cu::CommandEncoder& encoder,
      std::unordered_map<int64_t, void*> variant_pack);
  // Add cuDNN graph to CUDA graph, using stream capture.
  fe::error_t encode_capturing(
      cu::CommandEncoder& encoder,
      std::unordered_map<int64_t, void*> variant_pack);

 private:
  void* prepare_workspace(cu::CommandEncoder& encoder);

  void set_tensor_attrs(
      std::shared_ptr<fe::graph::Tensor_attributes>& tensor,
      int64_t uid,
      const array& x,
      const std::vector<int64_t>& shape,
      const std::vector<int64_t>& strides);
  void set_tensor_attrs(
      std::shared_ptr<fe::graph::Tensor_attributes>& tensor,
      int64_t uid,
      const array& x);
  void set_tensor_attrs_nchw(
      std::shared_ptr<fe::graph::Tensor_attributes>& tensor,
      int64_t uid,
      const array& x);

  cudnnHandle_t handle_;
};

} // namespace mlx::core
