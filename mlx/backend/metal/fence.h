// Copyright © 2024 Apple Inc.

#include "mlx/backend/metal/device.h"

namespace mlx::core {

class Fence {
 public:
  Fence(const Stream& stream);

  void update_gpu(const array& x);
  void wait_gpu(array& x);

  void wait();
  void update();

 private:
  Stream stream_;
  std::shared_ptr<void> fence_;
  uint32_t cpu_count_{0};
  uint32_t gpu_count_{0};
  bool use_fast_;
  std::atomic_uint* cpu_value();
};

} // namespace mlx::core
