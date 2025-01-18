// Copyright © 2024 Apple Inc.

#include "mlx/backend/metal/device.h"

namespace mlx::core {

/* A fence to be used for synchronizing work between the CPU and GPU
 *
 * Calls to `update_gpu` should be paired with calls to `wait`. This ensures
 * that the array passed to `update_gpu` is computed and visible to the CPU
 * after the call to `wait` returns.
 *
 * Calls to `update` should be paired with calls to `wait_gpu`. This ensures
 * that the array passed to `wait_gpu` will not be read by the GPU until the CPU
 * has called `update`.
 *
 * The fence supports slow (default) and fast mode. Fast mode requires setting
 * the environment variable `MLX_METAL_FAST_SYNCH=1`. Fast mode also requires
 * Metal 3.2+ (macOS 15+, iOS 18+).
 */
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
