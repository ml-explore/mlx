// Copyright © 2025 Apple Inc.

#include "mlx/backend/cpu/encoder.h"

namespace mlx::core::cpu {

CommandEncoder& get_command_encoder(Stream stream) {
  static std::unordered_map<int, CommandEncoder> encoder_map;
  auto it = encoder_map.find(stream.index);
  if (it == encoder_map.end()) {
    it = encoder_map.emplace(stream.index, stream).first;
  }
  return it->second;
}

} // namespace mlx::core::cpu
