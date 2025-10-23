#pragma once

struct __nv_fp4_e2m1 {
  __device__ __nv_fp4_e2m1(float x) {
    if (std::isnan(x)) {
      __x = 0x7;
      return;
    }

    const uint8_t sign_bit = (std::signbit(x)) ? 0x8 : 0x0;
    x = std::abs(x);

    if (x > 5.0f) {
      __x = 0x7;
    } else if (x >= 3.5f) {
      __x = 0x6;
    } else if (x > 2.5f) {
      __x = 0x5;
    } else if (x >= 1.75f) {
      __x = 0x4;
    } else if (x > 1.25f) {
      __x = 0x3;
    } else if (x >= 0.75f) {
      __x = 0x2;
    } else if (x > 0.25f) {
      __x = 0x1;
    } else {
      __x = 0x0;
    }
    __x |= sign_bit;
  }

  __device__ operator float() {
    static const float LUT[16] = {
        0.0f,
        0.5f,
        1.0f,
        1.5f,
        2.0f,
        3.0f,
        4.0f,
        6.0f,
        -0.0f,
        -0.5f,
        -1.0f,
        -1.5f,
        -2.0f,
        -3.0f,
        -4.0f,
        -6.0f};

    return LUT[__x];
  }
  uint8_t __x{0};
};
