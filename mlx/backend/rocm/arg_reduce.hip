// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>

namespace mlx::core {

void ArgReduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  // For now, use a simple implementation
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);
  
  const array& in = inputs[0];
  out.set_data(allocator::malloc(out.nbytes()));
  
  // TODO: Implement proper arg reduce using rocPrim
  throw std::runtime_error("ArgReduce not yet fully implemented for ROCm");
}

} // namespace mlx::core
