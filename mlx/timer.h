// Copyright © 2026 Apple Inc.

#pragma once

#include <memory>
#include <vector>

#include "mlx/api.h"
#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx::core {

class MLX_API Timer {
 public:
  explicit Timer(Stream stream);

  void start(
      const std::vector<array>& inputs,
      const std::vector<array>& outputs);
  void stop(
      const std::vector<array>& inputs,
      const std::vector<array>& outputs);
  double elapsed_time();

  const Stream& stream() const {
    return stream_;
  }

 private:
  Stream stream_;
  std::shared_ptr<void> impl_;
};

MLX_API std::vector<array> timer_start(
    const std::vector<array>& inputs,
    const Timer& timer);

MLX_API std::vector<array> timer_stop(
    const std::vector<array>& inputs,
    const Timer& timer);

} // namespace mlx::core
