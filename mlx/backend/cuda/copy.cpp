// Copyright © 2025 Apple Inc.

#include "mlx/backend/gpu/copy.h"

namespace mlx::core {

void copy_gpu_inplace(
    const array& in,
    array& out,
    const Shape& data_shape,
    const Strides& strides_in_pre,
    const Strides& strides_out_pre,
    int64_t inp_offset,
    int64_t out_offset,
    CopyType ctype,
    const Stream& s,
    const std::optional<array>& dynamic_i_offset /* = std::nullopt */,
    const std::optional<array>& dynamic_o_offset /* = std::nullopt */) {
  throw std::runtime_error("copy_gpu_inplace not implemented in CUDA backend.");
}

void fill_gpu(const array& val, array& out, const Stream& s) {
  throw std::runtime_error("fill_gpu not implemented in CUDA backend.");
}

} // namespace mlx::core
