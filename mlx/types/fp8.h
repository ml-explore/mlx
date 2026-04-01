// Copyright © 2023 Apple Inc.

#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

// Float8 E4M3FN: 1 sign, 4 exponent (bias 7), 3 mantissa bits
// No infinities; NaN is exp=1111, mantissa=111 only
#define __MLX_FP8_NAN__ 0x7F
#define __MLX_FP8_ONE__ 0x38

namespace mlx::core {

namespace {
union float_bits_fp8 {
  float f;
  uint32_t u;
};
} // namespace

struct _MLX_Float8 {
  uint8_t bits_;

  // Default constructor
  _MLX_Float8() = default;

  // Default copy constructor
  _MLX_Float8(_MLX_Float8 const&) = default;

  // Appease std::vector<bool> for being special
  _MLX_Float8& operator=(std::vector<bool>::reference x) {
    bits_ = (x) ? __MLX_FP8_ONE__ : 0;
    return (*this);
  }

  _MLX_Float8& operator=(const float& x) {
    return (*this = _MLX_Float8(x));
  }

  // From float32
  _MLX_Float8(const float& x) : bits_(0) {
    if (std::isnan(x)) {
      bits_ = __MLX_FP8_NAN__;
    } else {
      float_bits_fp8 in;
      in.f = x;

      // Extract sign
      uint8_t sign = (in.u >> 24) & 0x80;

      // Handle infinity and values beyond E4M3 max (448.0)
      float abs_x = std::abs(x);
      if (abs_x > 448.0f) {
        // Clamp to max representable: exp=1111, mantissa=110 = 0x7E
        bits_ = sign | 0x7E;
        return;
      }

      // Extract float32 exponent and re-bias for E4M3 (bias 7)
      int32_t exp = (int32_t)((in.u >> 23) & 0xFF) - 127 + 7;

      // Extract top 3 mantissa bits
      uint32_t mantissa = (in.u >> 20) & 0x07;

      // Round to nearest even
      uint32_t round_bit = (in.u >> 19) & 1;
      uint32_t sticky = (in.u & 0x0007FFFF) != 0;
      if (round_bit && (sticky || (mantissa & 1))) {
        mantissa++;
        if (mantissa > 7) {
          mantissa = 0;
          exp++;
        }
      }

      // Check for overflow after rounding (max normal is exp=15, mantissa=110)
      if (exp >= 15 && (exp > 15 || mantissa > 6)) {
        // Clamp to max (not NaN, which is exp=15 mantissa=7)
        bits_ = sign | 0x7E;
      } else if (exp <= 0) {
        if (exp >= -3) {
          // Denormalized: shift mantissa with implicit 1
          mantissa = (mantissa | 0x08) >> (1 - exp);
          bits_ = sign | (uint8_t)(mantissa & 0x07);
        } else {
          // Underflow to zero
          bits_ = sign;
        }
      } else {
        bits_ = sign | ((uint8_t)exp << 3) | (uint8_t)mantissa;
      }
    }
  }

  // To float32
  operator float() const {
    float_bits_fp8 out;

    uint32_t sign = (bits_ >> 7) & 1;
    uint32_t exp = (bits_ >> 3) & 0x0F;
    uint32_t mantissa = bits_ & 0x07;

    // NaN check: exp=1111, mantissa=111
    if (exp == 0x0F && mantissa == 0x07) {
      out.u = (sign << 31) | 0x7F800000 | (1 << 22); // quiet NaN
    } else if (exp == 0) {
      if (mantissa == 0) {
        // Zero (signed)
        out.u = sign << 31;
      } else {
        // Denormalized: normalize
        while ((mantissa & 0x08) == 0) {
          mantissa <<= 1;
          exp--;
        }
        mantissa &= 0x07; // remove implicit 1
        uint32_t f_exp = (uint32_t)((int32_t)exp + 127 - 7 + 1);
        out.u = (sign << 31) | (f_exp << 23) | (mantissa << 20);
      }
    } else {
      // Normalized (includes exp=15 mantissa=0..6 which are valid values)
      uint32_t f_exp = exp - 7 + 127;
      out.u = (sign << 31) | (f_exp << 23) | (mantissa << 20);
    }

    return out.f;
  }
};

#define fp8_binop_base(__op__, __operator__, otype, atype, btype, ctype) \
  inline otype __operator__(atype lhs, btype rhs) {                      \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);       \
  }

#define fp8_binop_helper(__op__, __operator__, otype, itype, ctype) \
  inline otype __operator__(_MLX_Float8 lhs, itype rhs) {           \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);  \
  }                                                                  \
  inline otype __operator__(itype lhs, _MLX_Float8 rhs) {           \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);  \
  }

// Operators
#define fp8_binop(__op__, __operator__)                                       \
  fp8_binop_base(                                                             \
      __op__, __operator__, _MLX_Float8, _MLX_Float8, _MLX_Float8, float);   \
  fp8_binop_helper(__op__, __operator__, float, float, float);               \
  fp8_binop_helper(__op__, __operator__, double, double, double);            \
  fp8_binop_helper(__op__, __operator__, _MLX_Float8, bool, float);          \
  fp8_binop_helper(__op__, __operator__, _MLX_Float8, int32_t, float);       \
  fp8_binop_helper(__op__, __operator__, _MLX_Float8, uint32_t, float);      \
  fp8_binop_helper(__op__, __operator__, _MLX_Float8, int64_t, float);       \
  fp8_binop_helper(__op__, __operator__, _MLX_Float8, uint64_t, float);

fp8_binop(+, operator+);
fp8_binop(-, operator-);
fp8_binop(*, operator*);
fp8_binop(/, operator/);

#undef fp8_binop

// Comparison ops
#define fp8_compop(__op__, __operator__)                                \
  fp8_binop_base(                                                       \
      __op__, __operator__, bool, _MLX_Float8, _MLX_Float8, float);    \
  fp8_binop_helper(__op__, __operator__, bool, float, float);           \
  fp8_binop_helper(__op__, __operator__, bool, double, double);         \
  fp8_binop_helper(__op__, __operator__, bool, int32_t, float);         \
  fp8_binop_helper(__op__, __operator__, bool, uint32_t, float);        \
  fp8_binop_helper(__op__, __operator__, bool, int64_t, float);         \
  fp8_binop_helper(__op__, __operator__, bool, uint64_t, float);

fp8_compop(>, operator>);
fp8_compop(<, operator<);
fp8_compop(>=, operator>=);
fp8_compop(<=, operator<=);
fp8_compop(==, operator==);
fp8_compop(!=, operator!=);

#undef fp8_compop

// Negative
inline _MLX_Float8 operator-(_MLX_Float8 lhs) {
  return -static_cast<float>(lhs);
}

// Inplace ops
#define fp8_inplace_op(__op__, __operator__)                              \
  inline _MLX_Float8& __operator__(_MLX_Float8& lhs, const float& rhs) { \
    lhs = lhs __op__ rhs;                                                \
    return lhs;                                                          \
  }                                                                      \
  inline float& __operator__(float& lhs, _MLX_Float8 rhs) {             \
    lhs = lhs __op__ rhs;                                                \
    return lhs;                                                          \
  }

fp8_inplace_op(+, operator+=);
fp8_inplace_op(-, operator-=);
fp8_inplace_op(*, operator*=);
fp8_inplace_op(/, operator/=);

#undef fp8_inplace_op

// Bitwise ops

#define fp8_bitop(__op__, __operator__)                                  \
  inline _MLX_Float8 __operator__(_MLX_Float8 lhs, _MLX_Float8 rhs) {   \
    _MLX_Float8 out;                                                     \
    out.bits_ = lhs.bits_ __op__ rhs.bits_;                              \
    return out;                                                          \
  }                                                                      \
  inline _MLX_Float8 __operator__(_MLX_Float8 lhs, uint8_t rhs) {       \
    _MLX_Float8 out;                                                     \
    out.bits_ = lhs.bits_ __op__ rhs;                                    \
    return out;                                                          \
  }                                                                      \
  inline _MLX_Float8 __operator__(uint8_t lhs, _MLX_Float8 rhs) {       \
    _MLX_Float8 out;                                                     \
    out.bits_ = lhs __op__ rhs.bits_;                                    \
    return out;                                                          \
  }

fp8_bitop(|, operator|);
fp8_bitop(&, operator&);
fp8_bitop(^, operator^);

#undef fp8_bitop

#define fp8_inplace_bitop(__op__, __operator__)                            \
  inline _MLX_Float8& __operator__(_MLX_Float8& lhs, _MLX_Float8 rhs) {  \
    lhs.bits_ = lhs.bits_ __op__ rhs.bits_;                               \
    return lhs;                                                            \
  }                                                                        \
  inline _MLX_Float8& __operator__(_MLX_Float8& lhs, uint8_t rhs) {      \
    lhs.bits_ = lhs.bits_ __op__ rhs;                                     \
    return lhs;                                                            \
  }

fp8_inplace_bitop(|, operator|=);
fp8_inplace_bitop(&, operator&=);
fp8_inplace_bitop(^, operator^=);

#undef fp8_inplace_bitop

} // namespace mlx::core
