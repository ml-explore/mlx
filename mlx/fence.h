// Copyright © 2024 Apple Inc.

#include <vector>

#include "mlx/array.h"

namespace mlx::core {

/* A fence to be used for synchronizing work between streams.
 *
 * Calls to `wait` wait in the given stream until all previous calls to update
 * are complete on their given stream.
 *
 * The array passed to `update` is computed and visible after the call to
 * `wait` returns. The array passed to `wait` will not be read until all
 * previous calls to `update` have completed.
 *
 * Note, calls to `update` should always from the same thread or explicitly
 * synchronized so that they occur in sequence. Calls to `wait` can be on any
 * thread.
 *
 * For the Metal back-end the fence supports slow (default) and fast mode.
 * Fast mode requires setting the environment variable
 * `MLX_METAL_FAST_SYNCH=1`. Fast mode also requires Metal 3.2+ (macOS 15+,
 * iOS 18+).
 */
class Fence {
 public:
  Fence() {};
  explicit Fence(Stream stream);

  void update(Stream stream, const array& x);
  void wait(Stream stream, const array& x);

 private:
  std::shared_ptr<void> fence_{nullptr};
};

} // namespace mlx::core
