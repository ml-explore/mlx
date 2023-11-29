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
  } else if (metal::is_available()) {
    set_default_device(Device::gpu);
  }

  context.applyCommandLine(argc, argv);
  return context.run();
}
