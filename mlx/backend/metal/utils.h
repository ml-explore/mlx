// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <type_traits>

#include "mlx/array.h"
#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"

namespace mlx::core {

std::string type_to_name(const Dtype& t);
std::string type_to_name(const array& a);

// Compute the grid and block dimensions, check backend/common/utils.h for docs.
MTL::Size get_block_dims(int dim0, int dim1, int dim2, int pow2 = 10);
MTL::Size get_2d_grid_dims(const Shape& shape, const Strides& strides);
MTL::Size
get_2d_grid_dims(const Shape& shape, const Strides& strides, size_t divisor);

inline NS::String* make_string(std::ostringstream& os) {
  std::string string = os.str();
  return NS::String::string(string.c_str(), NS::UTF8StringEncoding);
}

inline void debug_set_stream_queue_label(MTL::CommandQueue* queue, int index) {
#ifdef MLX_METAL_DEBUG
  std::ostringstream label;
  label << "Stream " << index;
  queue->setLabel(make_string(label));
#endif
}

inline void debug_set_primitive_buffer_label(
    MTL::CommandBuffer* command_buffer,
    Primitive& primitive) {
#ifdef MLX_METAL_DEBUG
  std::ostringstream label;
  if (auto cbuf_label = command_buffer->label(); cbuf_label) {
    label << cbuf_label->utf8String();
  }
  label << primitive.name();
  command_buffer->setLabel(make_string(label));
#endif
}

template <typename T>
constexpr bool is_numeric_except_char = std::is_arithmetic_v<T> &&
    !std::is_same_v<T, char> && !std::is_same_v<T, signed char> &&
    !std::is_same_v<T, unsigned char> && !std::is_same_v<T, wchar_t>;

template <typename T>
void concatenate(std::string& acc, T first) {
  if constexpr (is_numeric_except_char<T>) {
    acc += std::to_string(first);
  } else {
    acc += first;
  }
}

template <typename T, typename... Args>
void concatenate(std::string& acc, T first, Args... args) {
  if constexpr (is_numeric_except_char<T>) {
    acc += std::to_string(first);
  } else {
    acc += first;
  }
  concatenate(acc, args...);
}

inline int get_work_per_thread(Dtype dtype) {
  return std::max(1, 8 / dtype.size());
}
inline int get_work_per_thread(Dtype dtype, size_t size) {
  constexpr size_t wpt_threshold = 1 << 16;
  return size < wpt_threshold ? 1 : std::max(1, 8 / dtype.size());
}

inline size_t ceildiv(size_t n, size_t m) {
  return (n + m - 1) / m;
}

} // namespace mlx::core
