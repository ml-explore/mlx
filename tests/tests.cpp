// Copyright © 2023 Apple Inc.

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"

#include <cstdlib>

#include "mlx/mlx.h"

using namespace mlx::core;

int main(int argc, char** argv) {
  doctest::Context context;

  const char* device = std::getenv("DEVICE");
  if (device != nullptr && std::string(device) == "cpu") {
    set_default_device(Device::cpu);
  } else if (is_available(Device::gpu)) {
    // Use generic GPU availability check (works for Metal on macOS, or CUDA on
    // Linux/Windows)
    set_default_device(Device::gpu);
  }

  context.applyCommandLine(argc, argv);
  return context.run();
}
