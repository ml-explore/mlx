// Copyright © 2023-2024 Apple Inc.

const std::string preamble = R"(
#include <cmath>
#include <complex>
#include <stdint.h>

#ifdef __ARM_FEATURE_FP16_SCALAR_ARITHMETIC

#include <arm_fp16.h>
typedef __fp16 float16_t;

#else

#define ADD_HALF_BINOPS
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#define __MLX_HALF_NAN__ 0x7D00


namespace {
union float_bits_fp16 {
  float f;
  uint32_t u;
};
} // namespace

struct _MLX_Float16 {
  uint16_t bits_;

  // Default constructor
  _MLX_Float16() = default;

  // Default copy constructor
  _MLX_Float16(_MLX_Float16 const&) = default;

  // Appease std::vector<bool> for being special
  _MLX_Float16& operator=(std::vector<bool>::reference x) {
    bits_ = x;
    return *this;
  }

  _MLX_Float16& operator=(const float& x) {
    return (*this = _MLX_Float16(x));
  }

  // From float32
  _MLX_Float16(const float& x) : bits_(0) {
    // Conversion following
    // https://github.com/Maratyszcza/FP16/blob/master/include/fp16/fp16.h

    // Union
    float_bits_fp16 in;

    // Take fp32 bits
    in.f = x;

    // Find and take sign bit
    uint32_t x_sign_32 = in.u & uint32_t(0x80000000);
    uint16_t x_sign_16 = (x_sign_32 >> 16);

    if (std::isnan(x)) {
      bits_ = x_sign_16 | uint16_t(__MLX_HALF_NAN__);
    } else {
      // Union
      float_bits_fp16 inf_scale, zero_scale, magic_bits;

      // Find exponent bits and take the max supported by half
      uint32_t x_expo_32 = in.u & uint32_t(0x7f800000);
      uint32_t max_expo_32 = uint32_t(0x38800000);
      x_expo_32 = x_expo_32 < max_expo_32 ? max_expo_32 : x_expo_32;
      x_expo_32 += uint32_t(15) << 23;

      // Handle scaling to inf as needed
      inf_scale.u = uint32_t(0x77800000);
      zero_scale.u = uint32_t(0x08800000);

      // Combine with magic and let addition do rounding
      magic_bits.u = x_expo_32;
      magic_bits.f += (std::abs(x) * inf_scale.f) * zero_scale.f;

      // Take the lower 5 bits of the exponent
      uint32_t x_expo_16 = ((magic_bits.u >> 13) & uint32_t(0x7c00));

      // Collect the lower 12 bits which have the mantissa
      uint32_t x_mant_16 = magic_bits.u & uint32_t(0x0fff);

      // Combine sign, exp and mantissa
      bits_ = (x_sign_16 | uint16_t(x_expo_16 + x_mant_16));
    }
  }

  // To float32
  operator float() const {
    // Conversion following
    // https://github.com/Maratyszcza/FP16/blob/master/include/fp16/fp16.h

    // Union
    float_bits_fp16 out;

    uint32_t x_sign_32 = (bits_ << 16) & uint32_t(0x80000000);
    uint32_t base = (bits_ << 16);
    uint32_t two_base = base + base;

    uint32_t denorm_max = 1u << 27;
    if (two_base < denorm_max) {
      out.u = uint32_t(126) << 23; // magic mask
      out.u |= (two_base >> 17); // Bits from fp16
      out.f -= 0.5f; // magic bias
    } else {
      out.u = uint32_t(0xE0) << 23; // exponent offset
      out.u += (two_base >> 4); // Bits from fp16
      float out_unscaled = out.f; // Store value
      out.u = uint32_t(0x7800000); // exponent scale
      out.f *= out_unscaled;
    }

    // Add sign
    out.u |= x_sign_32;

    return out.f;
  }
};

#define half_binop_base(__op__, __operator__, otype, atype, btype, ctype) \
  inline otype __operator__(atype lhs, btype rhs) {                       \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);        \
  }

#define half_binop_helper(__op__, __operator__, otype, itype, ctype) \
  inline otype __operator__(_MLX_Float16 lhs, itype rhs) {           \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);   \
  }                                                                  \
  inline otype __operator__(itype lhs, _MLX_Float16 rhs) {           \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);   \
  }

// Operators
#define half_binop(__op__, __operator__)                                      \
  half_binop_base(                                                            \
      __op__, __operator__, _MLX_Float16, _MLX_Float16, _MLX_Float16, float); \
  half_binop_helper(__op__, __operator__, float, float, float);               \
  half_binop_helper(__op__, __operator__, double, double, double);            \
  half_binop_helper(__op__, __operator__, _MLX_Float16, bool, float);         \
  half_binop_helper(__op__, __operator__, _MLX_Float16, int32_t, float);      \
  half_binop_helper(__op__, __operator__, _MLX_Float16, uint32_t, float);     \
  half_binop_helper(__op__, __operator__, _MLX_Float16, int64_t, float);      \
  half_binop_helper(__op__, __operator__, _MLX_Float16, uint64_t, float);

half_binop(+, operator+);
half_binop(-, operator-);
half_binop(*, operator*);
half_binop(/, operator/);

#undef half_binop

// Comparison ops
#define half_compop(__op__, __operator__)                             \
  half_binop_base(                                                    \
      __op__, __operator__, bool, _MLX_Float16, _MLX_Float16, float); \
  half_binop_helper(__op__, __operator__, bool, float, float);        \
  half_binop_helper(__op__, __operator__, bool, double, double);      \
  half_binop_helper(__op__, __operator__, bool, int32_t, float);      \
  half_binop_helper(__op__, __operator__, bool, uint32_t, float);     \
  half_binop_helper(__op__, __operator__, bool, int64_t, float);      \
  half_binop_helper(__op__, __operator__, bool, uint64_t, float);

half_compop(>, operator>);
half_compop(<, operator<);
half_compop(>=, operator>=);
half_compop(<=, operator<=);
half_compop(==, operator==);
half_compop(!=, operator!=);

#undef half_compop

// Negative
inline _MLX_Float16 operator-(_MLX_Float16 lhs) {
  return -static_cast<float>(lhs);
}

// Inplace ops
#define half_inplace_op(__op__, __operator__)                              \
  inline _MLX_Float16& __operator__(_MLX_Float16& lhs, const float& rhs) { \
    lhs = lhs __op__ rhs;                                                  \
    return lhs;                                                            \
  }                                                                        \
  inline float& __operator__(float& lhs, _MLX_Float16 rhs) {               \
    lhs = lhs __op__ rhs;                                                  \
    return lhs;                                                            \
  }

half_inplace_op(+, operator+=);
half_inplace_op(-, operator-=);
half_inplace_op(*, operator*=);
half_inplace_op(/, operator/=);

#undef half_inplace_op

// Bitwise ops

#define half_bitop(__op__, __operator__)                                 \
  inline _MLX_Float16 __operator__(_MLX_Float16 lhs, _MLX_Float16 rhs) { \
    _MLX_Float16 out;                                                    \
    out.bits_ = lhs.bits_ __op__ rhs.bits_;                              \
    return out;                                                          \
  }                                                                      \
  inline _MLX_Float16 __operator__(_MLX_Float16 lhs, uint16_t rhs) {     \
    _MLX_Float16 out;                                                    \
    out.bits_ = lhs.bits_ __op__ rhs;                                    \
    return out;                                                          \
  }                                                                      \
  inline _MLX_Float16 __operator__(uint16_t lhs, _MLX_Float16 rhs) {     \
    _MLX_Float16 out;                                                    \
    out.bits_ = lhs __op__ rhs.bits_;                                    \
    return out;                                                          \
  }

half_bitop(|, operator|);
half_bitop(&, operator&);
half_bitop(^, operator^);

#undef half_bitop

#define half_inplace_bitop(__op__, __operator__)                           \
  inline _MLX_Float16& __operator__(_MLX_Float16& lhs, _MLX_Float16 rhs) { \
    lhs.bits_ = lhs.bits_ __op__ rhs.bits_;                                \
    return lhs;                                                            \
  }                                                                        \
  inline _MLX_Float16& __operator__(_MLX_Float16& lhs, uint16_t rhs) {     \
    lhs.bits_ = lhs.bits_ __op__ rhs;                                      \
    return lhs;                                                            \
  }

half_inplace_bitop(|, operator|=);
half_inplace_bitop(&, operator&=);
half_inplace_bitop(^, operator^=);

#undef half_inplace_bitop

typedef struct _MLX_Float16 float16_t;

#endif // __ARM_FEATURE_FP16_SCALAR_ARITHMETIC
#ifdef __ARM_FEATURE_BF16

#include <arm_bf16.h>
typedef __bf16 bfloat16_t;

#else

#define ADD_HALF_BINOPS
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#define __MLX_BFLOAT_NAN__ 0x7FC0


namespace {
union float_bits_bf16 {
  float f;
  uint32_t u;
};
} // namespace

struct _MLX_BFloat16 {
  uint16_t bits_;

  // Default constructor
  _MLX_BFloat16() = default;

  // Default copy constructor
  _MLX_BFloat16(_MLX_BFloat16 const&) = default;

  // Appease std::vector<bool> for being special
  _MLX_BFloat16& operator=(std::vector<bool>::reference x) {
    bits_ = x;
    return *this;
  }

  _MLX_BFloat16& operator=(const float& x) {
    return (*this = _MLX_BFloat16(x));
  }

  // From float32
  _MLX_BFloat16(const float& x) {
    if (std::isnan(x)) {
      bits_ = __MLX_BFLOAT_NAN__;
    } else {
      // Union
      float_bits_bf16 in;

      // Take bits
      in.f = x;

      // Round to nearest even
      in.u += (in.u >> 16 & 1) + uint32_t(0x7FFF);

      // Take upper 16 bits
      bits_ = in.u >> 16;
    }
  }

  // To float32
  operator float() const {
    // Union
    float_bits_bf16 out;

    // Upper 16 bits are the data and lower 16 bits are 0s
    out.u = ((uint32_t)bits_) << 16;

    return out.f;
  }
};

#define bfloat_binop_base(__op__, __operator__, otype, atype, btype, ctype) \
  inline otype __operator__(atype lhs, btype rhs) {                         \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);          \
  }

#define bfloat_binop_helper(__op__, __operator__, otype, itype, ctype) \
  inline otype __operator__(_MLX_BFloat16 lhs, itype rhs) {            \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);     \
  }                                                                    \
  inline otype __operator__(itype lhs, _MLX_BFloat16 rhs) {            \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);     \
  }

// Operators
#define bfloat_binop(_op_, _operator_)                                       \
  bfloat_binop_base(                                                         \
      _op_, _operator_, _MLX_BFloat16, _MLX_BFloat16, _MLX_BFloat16, float); \
  bfloat_binop_helper(_op_, _operator_, float, float, float);                \
  bfloat_binop_helper(_op_, _operator_, double, double, double);             \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, bool, float);         \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int32_t, float);      \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint32_t, float);     \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int64_t, float);      \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint64_t, float);

bfloat_binop(+, operator+);
bfloat_binop(-, operator-);
bfloat_binop(*, operator*);
bfloat_binop(/, operator/);

#undef bfloat_binop

// Comparison ops
#define bfloat_compop(__op__, __operator__)                             \
  bfloat_binop_base(                                                    \
      __op__, __operator__, bool, _MLX_BFloat16, _MLX_BFloat16, float); \
  bfloat_binop_helper(__op__, __operator__, bool, float, float);        \
  bfloat_binop_helper(__op__, __operator__, bool, double, double);      \
  bfloat_binop_helper(__op__, __operator__, bool, int32_t, float);      \
  bfloat_binop_helper(__op__, __operator__, bool, uint32_t, float);     \
  bfloat_binop_helper(__op__, __operator__, bool, int64_t, float);      \
  bfloat_binop_helper(__op__, __operator__, bool, uint64_t, float);

bfloat_compop(>, operator>);
bfloat_compop(<, operator<);
bfloat_compop(>=, operator>=);
bfloat_compop(<=, operator<=);
bfloat_compop(==, operator==);
bfloat_compop(!=, operator!=);

#undef bfloat_compop

// Negative
inline _MLX_BFloat16 operator-(_MLX_BFloat16 lhs) {
  return -static_cast<float>(lhs);
}

// Inplace ops
#define bfloat_inplace_op(__op__, __operator__)                              \
  inline _MLX_BFloat16& __operator__(_MLX_BFloat16& lhs, const float& rhs) { \
    lhs = lhs __op__ rhs;                                                    \
    return lhs;                                                              \
  }                                                                          \
  inline float& __operator__(float& lhs, _MLX_BFloat16 rhs) {                \
    lhs = lhs __op__ rhs;                                                    \
    return lhs;                                                              \
  }

bfloat_inplace_op(+, operator+=);
bfloat_inplace_op(-, operator-=);
bfloat_inplace_op(*, operator*=);
bfloat_inplace_op(/, operator/=);

#undef bfloat_inplace_op

// Bitwise ops

#define bfloat_bitop(__op__, __operator__)                                  \
  inline _MLX_BFloat16 __operator__(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { \
    _MLX_BFloat16 out;                                                      \
    out.bits_ = lhs.bits_ __op__ rhs.bits_;                                 \
    return out;                                                             \
  }                                                                         \
  inline _MLX_BFloat16 __operator__(_MLX_BFloat16 lhs, uint16_t rhs) {      \
    _MLX_BFloat16 out;                                                      \
    out.bits_ = lhs.bits_ __op__ rhs;                                       \
    return out;                                                             \
  }                                                                         \
  inline _MLX_BFloat16 __operator__(uint16_t lhs, _MLX_BFloat16 rhs) {      \
    _MLX_BFloat16 out;                                                      \
    out.bits_ = lhs __op__ rhs.bits_;                                       \
    return out;                                                             \
  }

bfloat_bitop(|, operator|);
bfloat_bitop(&, operator&);
bfloat_bitop(^, operator^);

#undef bfloat_bitop

#define bfloat_inplace_bitop(__op__, __operator__)                            \
  inline _MLX_BFloat16& __operator__(_MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { \
    lhs.bits_ = lhs.bits_ __op__ rhs.bits_;                                   \
    return lhs;                                                               \
  }                                                                           \
  inline _MLX_BFloat16& __operator__(_MLX_BFloat16& lhs, uint16_t rhs) {      \
    lhs.bits_ = lhs.bits_ __op__ rhs;                                         \
    return lhs;                                                               \
  }

bfloat_inplace_bitop(|, operator|=);
bfloat_inplace_bitop(&, operator&=);
bfloat_inplace_bitop(^, operator^=);

#undef bfloat_inplace_bitop

typedef struct _MLX_BFloat16 bfloat16_t;

#endif // __ARM_FEATURE_BF16

#ifdef ADD_HALF_BINOPS

// clang-format off
#define fp16_bf16_binop_helper(__op__, __operator__)               \
  inline float __operator__(float16_t lhs, bfloat16_t rhs) {       \
    return static_cast<float>(lhs) __op__ static_cast<float>(rhs); \
  }                                                                \
  inline float __operator__(bfloat16_t lhs, float16_t rhs) {       \
    return static_cast<float>(lhs) __op__ static_cast<float>(rhs); \
  }

fp16_bf16_binop_helper(+, operator+)
fp16_bf16_binop_helper(-, operator-)
fp16_bf16_binop_helper(*, operator*)
fp16_bf16_binop_helper(/, operator/)
// clang-format on

#endif


struct complex64_t;

template <typename T>
static constexpr bool can_convert_to_complex64 =
    !std::is_same_v<T, complex64_t> && std::is_convertible_v<T, float>;

struct complex64_t : public std::complex<float> {
  complex64_t(float v, float u) : std::complex<float>(v, u){};
  complex64_t(std::complex<float> v) : std::complex<float>(v){};

  template <
      typename T,
      typename = typename std::enable_if<can_convert_to_complex64<T>>::type>
  complex64_t(T x) : std::complex<float>(x){};

  operator float() const {
    return real();
  };
};

inline bool operator>=(const complex64_t& a, const complex64_t& b) {
  return (a.real() > b.real()) ||
      (a.real() == b.real() && a.imag() >= b.imag());
}

inline bool operator>(const complex64_t& a, const complex64_t& b) {
  return (a.real() > b.real()) || (a.real() == b.real() && a.imag() > b.imag());
}

inline complex64_t operator%(complex64_t a, complex64_t b) {
  auto real = a.real() - (b.real() * static_cast<int64_t>(a.real() / b.real()));
  auto imag = a.imag() - (b.imag() * static_cast<int64_t>(a.imag() / b.imag()));
  if (real != 0 && (real < 0 != b.real() < 0))
    real += b.real();
  if (imag != 0 && (imag < 0 != b.imag() < 0))
    imag += b.imag();
  return {real, imag};
}

inline bool operator<=(const complex64_t& a, const complex64_t& b) {
  return operator>=(b, a);
}

inline bool operator<(const complex64_t& a, const complex64_t& b) {
  return operator>(b, a);
}

inline complex64_t operator-(const complex64_t& v) {
  return -static_cast<std::complex<float>>(v);
}

// clang-format off
#define complex_binop_helper(_op_, _operator_, itype)            \
  inline complex64_t _operator_(itype x, const complex64_t& y) { \
    return static_cast<complex64_t>(x) _op_ y;           \
  }                                                              \
  inline complex64_t _operator_(const complex64_t& x, itype y) { \
    return x _op_ static_cast<complex64_t>(y);           \
  }

#define complex_binop(_op_, _operator_)                                               \
  inline complex64_t _operator_(const std::complex<float>& x, const complex64_t& y) { \
    return x _op_ static_cast<std::complex<float>>(y);                                \
  }                                                                                   \
  inline complex64_t _operator_(const complex64_t& x, const std::complex<float>& y) { \
    return static_cast<std::complex<float>>(x) _op_ y;                                \
  }                                                                                   \
  inline complex64_t _operator_(const complex64_t& x, const complex64_t& y) {         \
    return static_cast<std::complex<float>>(x)                                        \
        _op_ static_cast<std::complex<float>>(y);                                     \
  }                                                                                   \
  complex_binop_helper(_op_, _operator_, bool)                                        \
  complex_binop_helper(_op_, _operator_, uint32_t)                                    \
  complex_binop_helper(_op_, _operator_, uint64_t)                                    \
  complex_binop_helper(_op_, _operator_, int32_t)                                     \
  complex_binop_helper(_op_, _operator_, int64_t)                                     \
  complex_binop_helper(_op_, _operator_, float16_t)                                   \
  complex_binop_helper(_op_, _operator_, bfloat16_t)                                  \
  complex_binop_helper(_op_, _operator_, float)
// clang-format on

complex_binop(+, operator+)

typedef union {
  int i;
  float f;
} IntOrFloat;

inline float fast_exp(float x) {
  x *= 1.442695; // multiply with log_2(e)
  float ipart, fpart;
  IntOrFloat epart;
  x = std::max(-80.f, std::min(x, 80.f));
  ipart = std::floor(x + 0.5);
  fpart = x - ipart;

  x = 1.535336188319500e-4f;
  x = x * fpart + 1.339887440266574e-3f;
  x = x * fpart + 9.618437357674640e-3f;
  x = x * fpart + 5.550332471162809e-2f;
  x = x * fpart + 2.402264791363012e-1f;
  x = x * fpart + 6.931472028550421e-1f;
  x = x * fpart + 1.000000000000000f;

  // generate 2**ipart in the floating point representation using integer
  // bitshifting
  epart.i = (int(ipart) + 127) << 23;

  return epart.f * x;
}

float fast_erf(float a) {
  float r, s, t, u;
  t = std::abs(a);
  s = a * a;
  if (t > 0.927734375f) {
    // maximum error 0.99527 ulp
    r = std::fma(
        -1.72853470e-5f, t, 3.83197126e-4f); // -0x1.220000p-16,0x1.91cfb2p-12
    u = std::fma(
        -3.88396438e-3f, t, 2.42546219e-2f); // -0x1.fd1438p-9, 0x1.8d6342p-6
    r = std::fma(r, s, u);
    r = std::fma(r, t, -1.06777877e-1f); // -0x1.b55cb8p-4
    r = std::fma(r, t, -6.34846687e-1f); // -0x1.450aa0p-1
    r = std::fma(r, t, -1.28717512e-1f); // -0x1.079d0cp-3
    r = std::fma(r, t, -t);
    // TODO, replace with expm1 when implemented
    r = 1.0f - std::exp(r);
    r = std::copysign(r, a);
  } else {
    // maximum error 0.98929 ulp
    r = -5.96761703e-4f; // -0x1.38e000p-11
    r = std::fma(r, s, 4.99119423e-3f); //  0x1.471a58p-8
    r = std::fma(r, s, -2.67681349e-2f); // -0x1.b691b2p-6
    r = std::fma(r, s, 1.12819925e-1f); //  0x1.ce1c44p-4
    r = std::fma(r, s, -3.76125336e-1f); // -0x1.812700p-2
    r = std::fma(r, s, 1.28379166e-1f); //  0x1.06eba8p-3
    r = std::fma(r, a, a);
  }
  return r;
}

float fast_erfinv(float a) {
  auto t = std::fma(a, 0.0f - a, 1.0f);
  t = std::log(t);
  float p;
  if (std::abs(t) > 6.125f) { // maximum ulp error = 2.35793
    p = 3.03697567e-10f; //  0x1.4deb44p-32
    p = std::fma(p, t, 2.93243101e-8f); //  0x1.f7c9aep-26
    p = std::fma(p, t, 1.22150334e-6f); //  0x1.47e512p-20
    p = std::fma(p, t, 2.84108955e-5f); //  0x1.dca7dep-16
    p = std::fma(p, t, 3.93552968e-4f); //  0x1.9cab92p-12
    p = std::fma(p, t, 3.02698812e-3f); //  0x1.8cc0dep-9
    p = std::fma(p, t, 4.83185798e-3f); //  0x1.3ca920p-8
    p = std::fma(p, t, -2.64646143e-1f); // -0x1.0eff66p-2
    p = std::fma(p, t, 8.40016484e-1f); //  0x1.ae16a4p-1
  } else { // maximum ulp error = 2.35002
    p = 5.43877832e-9f; //  0x1.75c000p-28
    p = std::fma(p, t, 1.43285448e-7f); //  0x1.33b402p-23
    p = std::fma(p, t, 1.22774793e-6f); //  0x1.499232p-20
    p = std::fma(p, t, 1.12963626e-7f); //  0x1.e52cd2p-24
    p = std::fma(p, t, -5.61530760e-5f); // -0x1.d70bd0p-15
    p = std::fma(p, t, -1.47697632e-4f); // -0x1.35be90p-13
    p = std::fma(p, t, 2.31468678e-3f); //  0x1.2f6400p-9
    p = std::fma(p, t, 1.15392581e-2f); //  0x1.7a1e50p-7
    p = std::fma(p, t, -2.32015476e-1f); // -0x1.db2aeep-3
    p = std::fma(p, t, 8.86226892e-1f); //  0x1.c5bf88p-1
  }
  return a * p;
}

struct Abs {
  template <typename T>
  T operator()(T x) {
    return std::abs(x);
  };
  uint8_t operator()(uint8_t x) {
    return x;
  };
  uint16_t operator()(uint16_t x) {
    return x;
  };
  uint32_t operator()(uint32_t x) {
    return x;
  };
  uint64_t operator()(uint64_t x) {
    return x;
  };
  bool operator()(bool x) {
    return x;
  };
};

struct ArcCos {
  template <typename T>
  T operator()(T x) {
    return std::acos(x);
  };
};

struct ArcCosh {
  template <typename T>
  T operator()(T x) {
    return std::acosh(x);
  };
};

struct ArcSin {
  template <typename T>
  T operator()(T x) {
    return std::asin(x);
  };
};

struct ArcSinh {
  template <typename T>
  T operator()(T x) {
    return std::asinh(x);
  };
};

struct ArcTan {
  template <typename T>
  T operator()(T x) {
    return std::atan(x);
  };
};

struct ArcTanh {
  template <typename T>
  T operator()(T x) {
    return std::atanh(x);
  };
};

struct Ceil {
  template <typename T>
  T operator()(T x) {
    return std::ceil(x);
  };
  int8_t operator()(int8_t x) {
    return x;
  };
  int16_t operator()(int16_t x) {
    return x;
  };
  int32_t operator()(int32_t x) {
    return x;
  };
  int64_t operator()(int64_t x) {
    return x;
  };
  uint8_t operator()(uint8_t x) {
    return x;
  };
  uint16_t operator()(uint16_t x) {
    return x;
  };
  uint32_t operator()(uint32_t x) {
    return x;
  };
  uint64_t operator()(uint64_t x) {
    return x;
  };
  bool operator()(bool x) {
    return x;
  };
};

struct Cos {
  template <typename T>
  T operator()(T x) {
    return std::cos(x);
  };
};

struct Cosh {
  template <typename T>
  T operator()(T x) {
    return std::cosh(x);
  };
};

struct Erf {
  template <typename T>
  T operator()(T x) {
    return static_cast<T>(fast_erf(static_cast<float>(x)));
  };
};

struct ErfInv {
  template <typename T>
  T operator()(T x) {
    return static_cast<T>(fast_erfinv(static_cast<float>(x)));
  };
};

struct Exp {
  template <typename T>
  T operator()(T x) {
    return fast_exp(x);
  };
};

struct Floor {
  template <typename T>
  T operator()(T x) {
    return std::floor(x);
  };
  int8_t operator()(int8_t x) {
    return x;
  };
  int16_t operator()(int16_t x) {
    return x;
  };
  int32_t operator()(int32_t x) {
    return x;
  };
  int64_t operator()(int64_t x) {
    return x;
  };
  uint8_t operator()(uint8_t x) {
    return x;
  };
  uint16_t operator()(uint16_t x) {
    return x;
  };
  uint32_t operator()(uint32_t x) {
    return x;
  };
  uint64_t operator()(uint64_t x) {
    return x;
  };
  bool operator()(bool x) {
    return x;
  };
};

struct Log {
  template <typename T>
  T operator()(T x) {
    return std::log(x);
  };
};

struct Log2 {
  template <typename T>
  T operator()(T x) {
    return std::log2(x);
  };
};

struct Log10 {
  template <typename T>
  T operator()(T x) {
    return std::log10(x);
  };
};

struct Log1p {
  template <typename T>
  T operator()(T x) {
    return log1p(x);
  };
};

struct LogicalNot {
  template <typename T>
  T operator()(T x) {
    return !x;
  };
};

struct Negative {
  template <typename T>
  T operator()(T x) {
    return -x;
  };
};

struct Round {
  template <typename T>
  T operator()(T x) {
    return std::rint(x);
  }

  std::complex<float> operator()(std::complex<float> x) {
    return {std::rint(x.real()), std::rint(x.imag())};
  }
};

struct Sigmoid {
  template <typename T>
  T operator()(T x) {
    auto one = static_cast<decltype(x)>(1.0);
    return one / (one + fast_exp(-x));
  }
};

struct Sign {
  template <typename T>
  T operator()(T x) {
    return (x > T(0)) - (x < T(0));
  }
  uint8_t operator()(uint8_t x) {
    return x != 0;
  }
  uint16_t operator()(uint16_t x) {
    return x != 0;
  }
  uint32_t operator()(uint32_t x) {
    return x != 0;
  }
  uint64_t operator()(uint64_t x) {
    return x != 0;
  }
};

struct Sin {
  template <typename T>
  T operator()(T x) {
    return std::sin(x);
  };
};

struct Sinh {
  template <typename T>
  T operator()(T x) {
    return std::sinh(x);
  };
};

struct Square {
  template <typename T>
  T operator()(T x) {
    return x * x;
  };
};

struct Sqrt {
  template <typename T>
  T operator()(T x) {
    return std::sqrt(x);
  };
};

struct Rsqrt {
  template <typename T>
  T operator()(T x) {
    return static_cast<decltype(x)>(1.0) / std::sqrt(x);
  };
};

struct Tan {
  template <typename T>
  T operator()(T x) {
    return std::tan(x);
  };
};

struct Tanh {
  template <typename T>
  T operator()(T x) {
    return std::tanh(x);
  };
};

struct Add {
  template <typename T>
  T operator()(T x, T y) {
    return x + y;
  }
};

struct Divide {
  template <typename T>
  T operator()(T x, T y) {
    return x / y;
  }
};

struct Remainder {
  template <typename T>
  std::enable_if_t<std::is_integral_v<T> & !std::is_signed_v<T>, T> operator()(
      T numerator,
      T denominator) {
    return numerator % denominator;
  }

  template <typename T>
  std::enable_if_t<std::is_integral_v<T> & std::is_signed_v<T>, T> operator()(
      T numerator,
      T denominator) {
    auto r = numerator % denominator;
    if (r != 0 && (r < 0 != denominator < 0))
      r += denominator;
    return r;
  }

  template <typename T>
  std::enable_if_t<!std::is_integral_v<T>, T> operator()(
      T numerator,
      T denominator) {
    auto r = std::fmod(numerator, denominator);
    if (r != 0 && (r < 0 != denominator < 0)) {
      r += denominator;
    }
    return r;
  }

  std::complex<float> operator()(
      std::complex<float> a, std::complex<float> b) {
      auto real = a.real() - (b.real() * static_cast<int64_t>(a.real() / b.real()));
      auto imag = a.imag() - (b.imag() * static_cast<int64_t>(a.imag() / b.imag()));
      if (real != 0 && ((real < 0) != (b.real() < 0)))
        real += b.real();
      if (imag != 0 && ((imag < 0) != (b.imag() < 0)))
        imag += b.imag();
      return {real, imag};
  }
};

struct Equal {
  template <typename T>
  bool operator()(T x, T y) {
    return x == y;
  }
};

struct NaNEqual {
  template <typename T>
  bool operator()(T x, T y) {
    return x == y || (std::isnan(x) && std::isnan(y));
  }
};

struct Greater {
  template <typename T>
  bool operator()(T x, T y) {
    return x > y;
  }
};

struct GreaterEqual {
  template <typename T>
  bool operator()(T x, T y) {
    return x >= y;
  }
};

struct Less {
  template <typename T>
  bool operator()(T x, T y) {
    return x < y;
  }
};

struct LessEqual {
  template <typename T>
  bool operator()(T x, T y) {
    return x <= y;
  }
};

struct LogAddExp {
  template <typename T>
  T operator()(T x, T y) {
    constexpr float inf = std::numeric_limits<float>::infinity();
    auto maxval = (x > y) ? x : y;
    auto minval = (x > y) ? y : x;
    return (minval == -inf || maxval == inf)
        ? maxval
        : static_cast<decltype(x)>(
              maxval + std::log1p(fast_exp(minval - maxval)));
  };
};

struct Maximum {
  template <typename T>
  std::enable_if_t<std::is_integral_v<T>, T> operator()(T x, T y) {
    return (x > y) ? x : y;
  }

  template <typename T>
  std::enable_if_t<!std::is_integral_v<T>, T> operator()(T x, T y) {
    if (std::isnan(x)) {
      return x;
    }
    return (x > y) ? x : y;
  }
};

struct Minimum {
  template <typename T>
  std::enable_if_t<std::is_integral_v<T>, T> operator()(T x, T y) {
    return x < y ? x : y;
  }

  template <typename T>
  std::enable_if_t<!std::is_integral_v<T>, T> operator()(T x, T y) {
    if (std::isnan(x)) {
      return x;
    }
    return x < y ? x : y;
  }
};

struct Multiply {
  template <typename T>
  T operator()(T x, T y) {
    return x * y;
  }
};

struct NotEqual {
  template <typename T>
  bool operator()(T x, T y) {
    return x != y;
  }
};

struct Power {
  template <typename T>
  std::enable_if_t<!std::is_integral_v<T>, T> operator()(T base, T exp) {
    return std::pow(base, exp);
  }

  template <typename T>
  std::enable_if_t<std::is_integral_v<T>, T> operator()(T base, T exp) {
    T res = 1;
    while (exp) {
      if (exp & 1) {
        res *= base;
      }
      exp >>= 1;
      base *= base;
    }
    return res;
  }
};

struct Subtract {
  template <typename T>
  T operator()(T x, T y) {
    return x - y;
  }
};

struct LogicalAnd {
  template <typename T>
  T operator()(T x, T y) {
    return x && y;
  };
};

struct LogicalOr {
  template <typename T>
  T operator()(T x, T y) {
    return x || y;
  };
};
)";
