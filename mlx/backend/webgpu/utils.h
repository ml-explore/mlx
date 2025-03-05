// Copyright © 2025 Apple Inc.

#pragma once

#include <betann/betann.h>
#include <fmt/format.h>

#include "mlx/array.h"
#include "mlx/backend/webgpu/allocator.h"
#include "mlx/utils.h"

namespace mlx::core {

inline void throw_unsupported_dtype_error(Dtype dtype) {
  std::ostringstream os;
  os << dtype;
  throw std::runtime_error(
      fmt::format("Unsupported dtype in WGSL: {}", os.str()));
}

inline betann::DataType dtype_to_webgpu(Dtype dtype) {
  switch (dtype) {
    case bool_:
      return betann::DataType::Bool;
    case int8:
    case int16:
    case int32:
      return betann::DataType::I32;
    case uint8:
    case uint16:
    case uint32:
      return betann::DataType::U32;
    case float32:
      return betann::DataType::F32;
    case float16:
      return betann::DataType::F16;
    default:
      throw_unsupported_dtype_error(dtype);
      return betann::DataType::U32; // suppress warning
  }
}

inline size_t gpu_size_factor(Dtype dtype) {
  switch (dtype) {
    case int32:
    case uint32:
    case float16:
    case float32:
      return 1;
    case bool_:
    case int8:
    case uint8:
      return 4;
    case int16:
    case uint16:
      return 2;
    default:
      throw_unsupported_dtype_error(dtype);
      return 1; // suppress warning
  }
}

template <typename T>
inline std::vector<uint32_t> to_u32_vector(const std::vector<T>& vec) {
  return std::vector<uint32_t>(vec.begin(), vec.end());
}

inline betann::Buffer get_gpu_buffer(const array& arr) {
  size_t size_factor = gpu_size_factor(arr.dtype());
  betann::Buffer buffer =
      static_cast<const webgpu::DoubleBuffer*>(arr.buffer().ptr())->gpu_data();
  buffer.size = arr.data_size() * arr.itemsize() * size_factor;
  buffer.offset = arr.offset() * arr.itemsize() * size_factor;
  return buffer;
}

} // namespace mlx::core
