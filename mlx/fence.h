// Copyright © 2024 Apple Inc.

#include <cstdint>

#include "mlx/array.h"

namespace mlx::core {

/* A fence to be used for synchronizing work between streams.
 *
 * `update` returns a value that marks when its array is computed and visible.
 * `wait` orders work in the consumer stream after that value is signaled.
 * Later updates do not extend an earlier array's wait.
 *
 * Note, calls to `update` should always be from the same thread or explicitly
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

  uint32_t update(Stream stream, const array& x, bool cross_device);
  void wait(Stream stream, const array& x, uint32_t value);

  template <typename T>
  auto& cast() const {
    return *static_cast<T*>(fence_.get());
  }

 private:
  std::shared_ptr<void> fence_;
};

} // namespace mlx::core
