// Copyright © 2026 Apple Inc.

#pragma once

namespace mlx::core {

enum class MathMode {
  Safe = 0,
  Relaxed = 1,
  Fast = 2,
};

struct CompileOptions {
  MathMode math_mode = MathMode::Safe;

  CompileOptions() = default;
  bool operator==(const CompileOptions&) const = default;

  // A simple way to make export work, needs more work when adding new options.
  using Data = int;
  CompileOptions(Data data) : math_mode(static_cast<MathMode>(data)) {}
  Data serialize() const {
    return static_cast<int>(math_mode);
  }
};

} // namespace mlx::core
