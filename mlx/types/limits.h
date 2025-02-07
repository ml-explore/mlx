// Copyright © 2024 Apple Inc.
#pragma once

#include <limits>
#include "mlx/types/half_types.h"

namespace mlx::core {

template <typename T>
struct numeric_limits;

template <>
struct numeric_limits<float> : public std::numeric_limits<float> {};

template <>
struct numeric_limits<double> : public std::numeric_limits<double> {};

template <>
struct numeric_limits<float16_t> {
 private:
  union half_or_bits {
    uint16_t bits;
    float16_t value;
  };
  constexpr static float16_t bits_to_half(uint16_t v) {
    return half_or_bits{v}.value;
  }

 public:
  constexpr static float16_t lowest() {
    return bits_to_half(0xFBFF);
  }
  static constexpr float16_t max() {
    return bits_to_half(0x7BFF);
  }
  static constexpr float16_t infinity() {
    return bits_to_half(0x7C00);
  }
};

template <>
struct numeric_limits<bfloat16_t> {
 private:
  union bfloat_or_bits {
    uint16_t bits;
    bfloat16_t value;
  };
  constexpr static bfloat16_t bits_to_bfloat(uint16_t v) {
    return bfloat_or_bits{v}.value;
  }

 public:
  constexpr static bfloat16_t lowest() {
    return bits_to_bfloat(0xFF7F);
  }
  static constexpr bfloat16_t max() {
    return bits_to_bfloat(0x7F7F);
  }
  static constexpr bfloat16_t infinity() {
    return bits_to_bfloat(0x7F80);
  }
};

} // namespace mlx::core
