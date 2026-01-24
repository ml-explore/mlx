// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/rocm/copy/copy.hpp"

namespace mlx::core {

void copy_gpu_inplace(
    const array& in,
    array& out,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out,
    int64_t offset_in,
    int64_t offset_out,
    CopyType ctype,
    const Stream& s,
    std::optional<array> dynamic_offset_in,
    std::optional<array> dynamic_offset_out) {
  if (out.size() == 0) {
    return;
  }

  auto& encoder = rocm::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  if (ctype == CopyType::Scalar || ctype == CopyType::Vector) {
    copy_contiguous(encoder, ctype, in, out, offset_in, offset_out);
    return;
  }

  // For General and GeneralGeneral copy types, we need more complex handling
  // For now, fall back to a simpler implementation
  if (ctype == CopyType::General || ctype == CopyType::GeneralGeneral) {
    // TODO: Implement general copy with strided access
    throw std::runtime_error("General copy not yet fully implemented for ROCm.");
  }
}

void fill_gpu(const array& in, array& out, const Stream& s) {
  if (out.size() == 0) {
    return;
  }
  out.set_data(allocator::malloc(out.nbytes()));
  auto& encoder = rocm::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  copy_contiguous(encoder, CopyType::Scalar, in, out, 0, 0);
}

} // namespace mlx::core
