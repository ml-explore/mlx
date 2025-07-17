// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/primitives.h"

#include <fmt/format.h>
#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

#define CHECK_CUDNNS_ERROR(cmd) check_cudnn_error(#cmd, (cmd))

void check_cudnn_error(const char* name, cudnnStatus_t err) {
  if (err != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error(
        fmt::format("{} failed: {}.", name, cudnnGetErrorString(err)));
  }
}

void Convolution::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Convolution::eval_gpu");
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  assert(inputs.size() == 2);
  const auto& in = inputs[0];
  const auto& wt = inputs[1];
  out.set_data(allocator::malloc(out.nbytes()));

  throw std::runtime_error("NYI");
}

} // namespace mlx::core
