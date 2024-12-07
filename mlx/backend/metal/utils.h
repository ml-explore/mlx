// Copyright © 2023-2024 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"

namespace mlx::core {

std::string type_to_name(const Dtype& t);
std::string type_to_name(const array& a);

// Compute the thread block dimensions which fit the given
// input dimensions.
// - The thread block dimensions will be powers of two
// - The thread block size will be less than 2^pow2
MTL::Size get_block_dims(int dim0, int dim1, int dim2, int pow2 = 10);

// Computes a 2D grid where each element is < UINT_MAX
// Assumes:
// - overall size (product of non-broadcasted dimensions) is < UINT_MAX^2
// - shape and strides correspond to a contiguous (no holes) but
//   possibly broadcasted array
MTL::Size get_2d_grid_dims(const Shape& shape, const Strides& strides);

// Same as above but we do an implicit division with divisor.
// Basically, equivalent to factorizing
//    Prod(s \forall s in shape if strides[s] > 0) / divisor.
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
  primitive.print(label);
  command_buffer->setLabel(make_string(label));
#endif
}

std::string get_primitive_string(Primitive* primitive);

template <typename T>
void concatenate(std::string& acc, T first) {
  acc += first;
}

template <typename T, typename... Args>
void concatenate(std::string& acc, T first, Args... args) {
  acc += first;
  concatenate(acc, args...);
}

} // namespace mlx::core
