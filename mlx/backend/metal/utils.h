// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

inline void
set_array_buffer(MTL::ComputeCommandEncoder* enc, const array& a, int idx) {
  auto a_buf = static_cast<const MTL::Buffer*>(a.buffer().ptr());
  auto offset = a.data<char>() -
      static_cast<char*>(const_cast<MTL::Buffer*>(a_buf)->contents());
  enc->setBuffer(a_buf, offset, idx);
}

inline void set_array_buffer(
    MTL::ComputeCommandEncoder* enc,
    const array& a,
    int64_t offset,
    int idx) {
  auto a_buf = static_cast<const MTL::Buffer*>(a.buffer().ptr());
  auto base_offset = a.data<char>() -
      static_cast<char*>(const_cast<MTL::Buffer*>(a_buf)->contents());
  base_offset += offset;
  enc->setBuffer(a_buf, base_offset, idx);
}

template <typename T>
inline void set_vector_bytes(
    MTL::ComputeCommandEncoder* enc,
    const std::vector<T>& vec,
    size_t nelems,
    int idx) {
  enc->setBytes(vec.data(), nelems * sizeof(T), idx);
}

template <typename T>
inline void set_vector_bytes(
    MTL::ComputeCommandEncoder* enc,
    const std::vector<T>& vec,
    int idx) {
  return set_vector_bytes(enc, vec, vec.size(), idx);
}

std::string type_to_name(const array& a) {
  std::string tname;
  switch (a.dtype()) {
    case bool_:
      tname = "bool_";
      break;
    case uint8:
      tname = "uint8";
      break;
    case uint16:
      tname = "uint16";
      break;
    case uint32:
      tname = "uint32";
      break;
    case uint64:
      tname = "uint64";
      break;
    case int8:
      tname = "int8";
      break;
    case int16:
      tname = "int16";
      break;
    case int32:
      tname = "int32";
      break;
    case int64:
      tname = "int64";
      break;
    case float16:
      tname = "float16";
      break;
    case float32:
      tname = "float32";
      break;
    case bfloat16:
      tname = "bfloat16";
      break;
    case complex64:
      tname = "complex64";
      break;
  }
  return tname;
}

MTL::Size get_block_dims(int dim0, int dim1, int dim2) {
  int pows[3] = {0, 0, 0};
  int sum = 0;
  while (true) {
    int presum = sum;
    // Check all the pows
    if (dim0 >= (1 << (pows[0] + 1))) {
      pows[0]++;
      sum++;
    }
    if (sum == 10) {
      break;
    }
    if (dim1 >= (1 << (pows[1] + 1))) {
      pows[1]++;
      sum++;
    }
    if (sum == 10) {
      break;
    }
    if (dim2 >= (1 << (pows[2] + 1))) {
      pows[2]++;
      sum++;
    }
    if (sum == presum || sum == 10) {
      break;
    }
  }
  return MTL::Size{1ul << pows[0], 1ul << pows[1], 1ul << pows[2]};
}

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
  primitive.print(label);
  command_buffer->setLabel(make_string(label));
#endif
}

} // namespace

} // namespace mlx::core
