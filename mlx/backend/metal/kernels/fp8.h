#pragma once

inline float fp32_from_bits(uint32_t bits) {
  return *(reinterpret_cast<thread float*>(&bits));
}
inline float fp32_to_bits(float x) {
  return *(reinterpret_cast<thread uint32_t*>(&x));
}

struct fp8_e4m3 {
  template <typename T>
  fp8_e4m3(T f) {
    // From PyTorch
    // https://github.com/pytorch/pytorch/blob/e3643e1e0e923f0fc063dfab6f45c956d568919d/c10/util/Float8_e4m3fn.h#L148
    uint32_t fp8_max = 543 << 21;
    uint32_t denorm_mask = 141 << 23;
    uint32_t f_bits = fp32_to_bits(static_cast<float>(f));
    uint32_t sign = f_bits & 0x80000000;
    f_bits ^= sign;
    if (f_bits >= fp8_max) {
      // Default behavior saturates to min/max
      bits = 0x7E;
    } else {
      if (f_bits < (121 << 23)) {
        f_bits =
            fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
        bits = static_cast<uint8_t>(f_bits - denorm_mask);
      } else {
        // resulting mantissa is odd
        uint8_t mant_odd = (f_bits >> 20) & 1;
        f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
        f_bits += mant_odd;
        bits = static_cast<uint8_t>(f_bits >> 20);
      }
    }
    bits |= static_cast<uint8_t>(sign >> 24);
  }

  operator float() {
    // From PyTorch:
    // https://github.com/pytorch/pytorch/blob/e3643e1e0e923f0fc063dfab6f45c956d568919d/c10/util/Float8_e4m3fn.h#L46
    uint32_t w = static_cast<uint32_t>(bits) << 24;
    uint32_t sign = w & 0x80000000;
    uint32_t nonsign = w & 0x7FFFFFFF;

    uint32_t renorm_shift = metal::clz(nonsign);
    renorm_shift = renorm_shift > 4 ? renorm_shift - 4 : 0;

    int32_t inf_nan_mask =
        (static_cast<int32_t>(nonsign + 0x01000000) >> 8) & 0x7F800000;
    int32_t zero_mask = static_cast<int32_t>(nonsign - 1) >> 31;
    uint32_t result = sign |
        ((((nonsign << renorm_shift >> 4) + ((0x78 - renorm_shift) << 23)) |
          inf_nan_mask) &
         ~zero_mask);
    return fp32_from_bits(result);
  }

  uint8_t bits;
};

struct fp8_e8m0 {
  fp8_e8m0(float x) {
    if (!metal::isfinite(x)) {
      bits = 0xFF;
      return;
    }
    if (x < 0.0f) {
      bits = 0x00;
      return;
    }
    float le = metal::log2(x);
    int n = int(metal::round(le));

    n = n < -127 ? -127 : n;
    n = n > 127 ? 127 : n;
    bits = static_cast<uint8_t>(n + 127);
  }

  operator float() {
    if (bits == 0xFF) {
      return metal::numeric_limits<float>::quiet_NaN();
    }
    return metal::ldexp(1.0f, static_cast<int>(bits) - 127);
  }

  uint8_t bits;
};
