// Copyright © 2023 Apple Inc.

#include "mlx/backend/cpu/threefry.h"

namespace mlx::core::random {

std::pair<uint32_t, uint32_t> threefry2x32_hash(
    const std::pair<uint32_t, uint32_t>& key,
    std::pair<uint32_t, uint32_t> count) {
  constexpr static uint32_t rotations[2][4] = {
      {13, 15, 26, 6}, {17, 29, 16, 24}};

  uint32_t ks[3] = {key.first, key.second, key.first ^ key.second ^ 0x1BD11BDA};

  count.first += ks[0];
  count.second += ks[1];

  for (int i = 0; i < 5; ++i) {
    for (auto r : rotations[i % 2]) {
      count.first += count.second;
      count.second = (count.second << r) | (count.second >> (32 - r));
      count.second ^= count.first;
    }
    count.first += ks[(i + 1) % 3];
    count.second += ks[(i + 2) % 3] + i + 1;
  }

  return count;
}

} // namespace mlx::core::random
