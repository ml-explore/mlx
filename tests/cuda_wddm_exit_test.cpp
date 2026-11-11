// Copyright © 2026 Apple Inc.

#include "mlx/mlx.h"

#include <cuda_runtime.h>
#include <windows.h>

#include <iostream>
#include <limits>

int main() {
  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess ||
      !properties.memoryPoolsSupported || properties.integrated ||
      properties.tccDriver || properties.luidDeviceNodeMask == 0) {
    return 77;
  }
  try {
    using namespace mlx::core;
    set_default_device(Device::gpu);
    set_memory_limit(std::numeric_limits<size_t>::max());
    if (get_memory_limit() == std::numeric_limits<size_t>::max()) {
      std::cerr << "WDDM memory budget was not initialized.\n";
      return 1;
    }
    {
      auto result = sum(ones({256}, float32));
      if (result.item<float>() != 256.0f) {
        return 1;
      }
    }
    clear_streams();
    std::cout << "GPU work completed; testing process exit." << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
