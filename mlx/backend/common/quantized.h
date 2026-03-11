// Copyright © 2026 Apple Inc.

namespace mlx::core {

inline constexpr short get_pack_factor(int bits, int wsize = 8) {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

inline constexpr short get_bytes_per_pack(int bits, int wsize = 8) {
  bool power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

} // namespace mlx::core
