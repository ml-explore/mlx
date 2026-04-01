// Copyright © 2023 Apple Inc.

#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

// BFloat8 (E5M2): 1 sign, 5 exponent (bias 15), 2 mantissa bits
#define __MLX_BF8_NAN__ 0x7F
#define __MLX_BF8_ONE__ 0x3C

namespace mlx::core {

namespace {
union float_bits_bf8 {
  float f;
  uint32_t u;
};
} // namespace

struct _MLX_BFloat8 {
  uint8_t bits_;

  // Default constructor
  _MLX_BFloat8() = default;

  // Default copy constructor
  _MLX_BFloat8(_MLX_BFloat8 const&) = default;

  // Appease std::vector<bool> for being special
  _MLX_BFloat8& operator=(std::vector<bool>::reference x) {
    bits_ = (x) ? __MLX_BF8_ONE__ : 0;
    return (*this);
  }

  _MLX_BFloat8& operator=(const float& x) {
    return (*this = _MLX_BFloat8(x));
  }

  // From float32
  _MLX_BFloat8(const float& x) : bits_(0) {
    if (std::isnan(x)) {
      bits_ = __MLX_BF8_NAN__;
    } else if (std::isinf(x)) {
      // Infinity: exp=11111, mantissa=00, preserve sign
      bits_ = (x > 0) ? 0x7C : 0xFC;
    } else {
      float_bits_bf8 in;
      in.f = x;

      // Extract sign
      uint8_t sign = (in.u >> 24) & 0x80;

      // Extract float32 exponent and re-bias for E5M2
      int32_t exp = (int32_t)((in.u >> 23) & 0xFF) - 127 + 15;

      // Extract top 2 mantissa bits
      uint32_t mantissa = (in.u >> 21) & 0x03;

      // Round to nearest even
      uint32_t round_bit = (in.u >> 20) & 1;
      uint32_t sticky = (in.u & 0x000FFFFF) != 0;
      if (round_bit && (sticky || (mantissa & 1))) {
        mantissa++;
        if (mantissa > 3) {
          mantissa = 0;
          exp++;
        }
      }

      if (exp >= 31) {
        // Overflow to infinity
        bits_ = sign | 0x7C;
      } else if (exp <= 0) {
        if (exp >= -2) {
          // Denormalized: shift mantissa with implicit 1
          mantissa = (mantissa | 0x04) >> (1 - exp);
          bits_ = sign | (uint8_t)(mantissa & 0x03);
        } else {
          // Underflow to zero
          bits_ = sign;
        }
      } else {
        bits_ = sign | ((uint8_t)exp << 2) | (uint8_t)mantissa;
      }
    }
  }

  // To float32
  operator float() const {
    float_bits_bf8 out;

    uint32_t sign = (bits_ >> 7) & 1;
    uint32_t exp = (bits_ >> 2) & 0x1F;
    uint32_t mantissa = bits_ & 0x03;

    if (exp == 0x1F) {
      // Inf (mantissa==0) or NaN (mantissa!=0)
      out.u = (sign << 31) | 0x7F800000 | (mantissa << 21);
    } else if (exp == 0) {
      if (mantissa == 0) {
        // Zero (signed)
        out.u = sign << 31;
      } else {
        // Denormalized: normalize
        while ((mantissa & 0x04) == 0) {
          mantissa <<= 1;
          exp--;
        }
        mantissa &= 0x03; // remove implicit 1
        uint32_t f_exp = (uint32_t)((int32_t)exp + 127 - 15 + 1);
        out.u = (sign << 31) | (f_exp << 23) | (mantissa << 21);
      }
    } else {
      // Normalized
      uint32_t f_exp = exp - 15 + 127;
      out.u = (sign << 31) | (f_exp << 23) | (mantissa << 21);
    }

    return out.f;
  }
};

#define bf8_binop_base(__op__, __operator__, otype, atype, btype, ctype) \
  inline otype __operator__(atype lhs, btype rhs) {                      \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);       \
  }

#define bf8_binop_helper(__op__, __operator__, otype, itype, ctype) \
  inline otype __operator__(_MLX_BFloat8 lhs, itype rhs) {          \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);  \
  }                                                                  \
  inline otype __operator__(itype lhs, _MLX_BFloat8 rhs) {          \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);  \
  }

// Operators
#define bf8_binop(__op__, __operator__)                                       \
  bf8_binop_base(                                                             \
      __op__, __operator__, _MLX_BFloat8, _MLX_BFloat8, _MLX_BFloat8, float); \
  bf8_binop_helper(__op__, __operator__, float, float, float);                \
  bf8_binop_helper(__op__, __operator__, double, double, double);             \
  bf8_binop_helper(__op__, __operator__, _MLX_BFloat8, bool, float);          \
  bf8_binop_helper(__op__, __operator__, _MLX_BFloat8, int32_t, float);       \
  bf8_binop_helper(__op__, __operator__, _MLX_BFloat8, uint32_t, float);      \
  bf8_binop_helper(__op__, __operator__, _MLX_BFloat8, int64_t, float);       \
  bf8_binop_helper(__op__, __operator__, _MLX_BFloat8, uint64_t, float);

bf8_binop(+, operator+);
bf8_binop(-, operator-);
bf8_binop(*, operator*);
bf8_binop(/, operator/);

#undef bf8_binop

// Comparison ops
#define bf8_compop(__op__, __operator__)                                \
  bf8_binop_base(                                                       \
      __op__, __operator__, bool, _MLX_BFloat8, _MLX_BFloat8, float);  \
  bf8_binop_helper(__op__, __operator__, bool, float, float);           \
  bf8_binop_helper(__op__, __operator__, bool, double, double);         \
  bf8_binop_helper(__op__, __operator__, bool, int32_t, float);         \
  bf8_binop_helper(__op__, __operator__, bool, uint32_t, float);        \
  bf8_binop_helper(__op__, __operator__, bool, int64_t, float);         \
  bf8_binop_helper(__op__, __operator__, bool, uint64_t, float);

bf8_compop(>, operator>);
bf8_compop(<, operator<);
bf8_compop(>=, operator>=);
bf8_compop(<=, operator<=);
bf8_compop(==, operator==);
bf8_compop(!=, operator!=);

#undef bf8_compop

// Negative
inline _MLX_BFloat8 operator-(_MLX_BFloat8 lhs) {
  return -static_cast<float>(lhs);
}

// Inplace ops
#define bf8_inplace_op(__op__, __operator__)                              \
  inline _MLX_BFloat8& __operator__(_MLX_BFloat8& lhs, const float& rhs) { \
    lhs = lhs __op__ rhs;                                                \
    return lhs;                                                          \
  }                                                                      \
  inline float& __operator__(float& lhs, _MLX_BFloat8 rhs) {            \
    lhs = lhs __op__ rhs;                                                \
    return lhs;                                                          \
  }

bf8_inplace_op(+, operator+=);
bf8_inplace_op(-, operator-=);
bf8_inplace_op(*, operator*=);
bf8_inplace_op(/, operator/=);

#undef bf8_inplace_op

// Bitwise ops

#define bf8_bitop(__op__, __operator__)                                  \
  inline _MLX_BFloat8 __operator__(_MLX_BFloat8 lhs, _MLX_BFloat8 rhs) { \
    _MLX_BFloat8 out;                                                    \
    out.bits_ = lhs.bits_ __op__ rhs.bits_;                              \
    return out;                                                          \
  }                                                                      \
  inline _MLX_BFloat8 __operator__(_MLX_BFloat8 lhs, uint8_t rhs) {     \
    _MLX_BFloat8 out;                                                    \
    out.bits_ = lhs.bits_ __op__ rhs;                                    \
    return out;                                                          \
  }                                                                      \
  inline _MLX_BFloat8 __operator__(uint8_t lhs, _MLX_BFloat8 rhs) {     \
    _MLX_BFloat8 out;                                                    \
    out.bits_ = lhs __op__ rhs.bits_;                                    \
    return out;                                                          \
  }

bf8_bitop(|, operator|);
bf8_bitop(&, operator&);
bf8_bitop(^, operator^);

#undef bf8_bitop

#define bf8_inplace_bitop(__op__, __operator__)                            \
  inline _MLX_BFloat8& __operator__(_MLX_BFloat8& lhs, _MLX_BFloat8 rhs) { \
    lhs.bits_ = lhs.bits_ __op__ rhs.bits_;                               \
    return lhs;                                                            \
  }                                                                        \
  inline _MLX_BFloat8& __operator__(_MLX_BFloat8& lhs, uint8_t rhs) {     \
    lhs.bits_ = lhs.bits_ __op__ rhs;                                     \
    return lhs;                                                            \
  }

bf8_inplace_bitop(|, operator|=);
bf8_inplace_bitop(&, operator&=);
bf8_inplace_bitop(^, operator^=);

#undef bf8_inplace_bitop

} // namespace mlx::core
