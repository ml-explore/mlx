// Copyright © 2026 Apple Inc.

namespace mlx::core::cu {

__global__ void graph_root_kernel() {}

void* graph_root_kernel_ptr() {
  return reinterpret_cast<void*>(&graph_root_kernel);
}

} // namespace mlx::core::cu
