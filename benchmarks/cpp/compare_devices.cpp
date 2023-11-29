#include <iostream>
#include "mlx/mlx.h"
#include "time_utils.h"

using namespace mlx::core;

void time_add_op() {
  std::vector<int> sizes(1, 1);
  for (int i = 0; i < 9; ++i) {
    sizes.push_back(10 * sizes.back());
  }
  set_default_device(Device::cpu);
  for (auto size : sizes) {
    auto a = random::uniform({size});
    auto b = random::uniform({size});
    eval(a, b);
    std::cout << "Size " << size << std::endl;
    TIMEM("cpu", add, a, b, Device::cpu);
    TIMEM("gpu", add, a, b, Device::gpu);
  }
}

int main() {
  time_add_op();
}
