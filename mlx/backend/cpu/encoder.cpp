// Copyright © 2025 Apple Inc.

#include "mlx/backend/cpu/encoder.h"

#include <fmt/format.h>

namespace mlx::core::cpu {

std::unordered_map<int, CommandEncoder>& get_command_encoders() {
  static thread_local std::unordered_map<int, CommandEncoder> encoders;
  return encoders;
}

CommandEncoder& get_command_encoder(Stream stream) {
  auto& encoders = get_command_encoders();
  auto it = encoders.find(stream.index);
  if (it == encoders.end()) {
    throw std::runtime_error(
        fmt::format(
            "There is no Stream(cpu, {}) in current thread.", stream.index));
  }
  return it->second;
}

} // namespace mlx::core::cpu
