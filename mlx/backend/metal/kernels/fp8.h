#pragma once

struct fp8_e4m3 {
  template <typename T>
  fp8_e4m3(T f) {
    // From PyTorch
    // https://github.com/pytorch/pytorch/blob/e3643e1e0e923f0fc063dfab6f45c956d568919d/c10/util/Float8_e4m3fn.h#L148
    uint32_t fp8_max = 543 << 21;
    uint32_t denorm_mask = 141 << 23;
    uint32_t f_bits = as_type<uint32_t>(static_cast<float>(f));
    uint32_t sign = f_bits & 0x80000000;
    f_bits ^= sign;
    if (f_bits >= fp8_max) {
      // Default behavior saturates to min/max
      bits = 0x7E;
    } else {
      if (f_bits < (121 << 23)) {
        f_bits = as_type<uint32_t>(
            as_type<float>(f_bits) + as_type<float>(denorm_mask));
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

  operator float16_t() {
    uint16_t v = (bits & 127) << 7;
    half converted = as_type<half>(v);
    converted *= 256.0;
    auto sign = bits & 128;
    return (sign ? -converted : converted);
  }

  operator bfloat16_t() {
    return static_cast<bfloat16_t>(this->operator float16_t());
  }

  operator float() {
    return static_cast<float>(this->operator float16_t());
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

  operator bfloat16_t() {
    uint16_t out = (bits == 0 ? 0x40 : (static_cast<uint16_t>(bits) << 7));
    return as_type<bfloat16_t>(out);
  }

  operator float() {
    uint32_t out = (bits == 0 ? 0x400000 : (static_cast<uint16_t>(bits) << 23));
    return as_type<float>(out);
  }

  uint8_t bits;
};
