// Copyright © 2023-2024 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"

namespace mlx::core {

using metal::CommandEncoder;

template <typename T>
inline void set_vector_bytes(
    CommandEncoder& enc,
    const std::vector<T>& vec,
    size_t nelems,
    int idx) {
  enc->setBytes(vec.data(), nelems * sizeof(T), idx);
}

template <typename T>
inline void
set_vector_bytes(CommandEncoder& enc, const std::vector<T>& vec, int idx) {
  return set_vector_bytes(enc, vec, vec.size(), idx);
}

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
MTL::Size get_2d_grid_dims(
    const std::vector<int>& shape,
    const std::vector<size_t>& strides);

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

} // namespace mlx::core
