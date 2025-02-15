// Copyright © 2024 Apple Inc.

#include <vector>

#include "mlx/array.h"

namespace mlx::core {

/* A fence to be used for synchronizing work between streams.
 *
 * Calls to `wait` wait in the given stream until all previous calls to update
 * are complete on their given stream.
 *
 * The arrays passed to `update` are computed and visible after the call to
 * `wait` returns. The array passed to `wait` will not be read until all
 * previous calls to `update` have completed.
 *
 * The fence supports slow (default) and fast mode. Fast mode requires setting
 * the environment variable `MLX_METAL_FAST_SYNCH=1`. Fast mode also requires
 * Metal 3.2+ (macOS 15+, iOS 18+).
 */
class Fence {
 public:
  Fence();

  void update(Stream stream, const std::vector<array>& arrays);
  void wait(Stream stream, const array& array);

 private:
  std::shared_ptr<void> fence_;
  uint32_t count_{0};
  bool use_fast_;
  std::atomic_uint* cpu_value();
};

} // namespace mlx::core
