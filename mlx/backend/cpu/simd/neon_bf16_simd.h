#pragma once

#include <arm_neon.h>

#include "mlx/backend/cpu/simd/base_simd.h"

namespace mlx::core::simd {

constexpr int BF16_N = 8;

template <>
inline constexpr int max_size<bfloat16_t> = BF16_N;

inline float32x4_t bf16_bits_to_f32(uint16x4_t v) {
  return vreinterpretq_f32_u32(vshll_n_u16(v, 16));
}

inline uint16x4_t f32_to_bf16_bits(float32x4_t f) {
  uint32x4_t bits = vreinterpretq_u32_f32(f);
  uint32x4_t lsb = vandq_u32(vshrq_n_u32(bits, 16), vdupq_n_u32(1));
  uint32x4_t rounded = vaddq_u32(bits, vaddq_u32(vdupq_n_u32(0x7FFF), lsb));
  uint16x4_t out = vshrn_n_u32(rounded, 16);
  uint32x4_t is_nan = vmvnq_u32(vceqq_f32(f, f));
  return vbsl_u16(vmovn_u32(is_nan), vdup_n_u16(0x7FC0), out);
}

template <>
struct Simd<bfloat16_t, BF16_N> {
  static constexpr int size = BF16_N;
  using scalar_t = bfloat16_t;

  Simd<bfloat16_t, BF16_N>() {}

  Simd<bfloat16_t, BF16_N>(uint16x8_t v) : value(v) {}

  template <typename U>
  Simd<bfloat16_t, BF16_N>(U v) {
    bfloat16_t t = static_cast<bfloat16_t>(static_cast<float>(v));
    value = vdupq_n_u16(*reinterpret_cast<uint16_t*>(&t));
  }

  Simd<bfloat16_t, BF16_N>(Simd<float, BF16_N> other) {
    auto f32x4_a = *(float32x4_t*)(&other);
    auto f32x4_b = *((float32x4_t*)(&other) + 1);
    value = vcombine_u16(f32_to_bf16_bits(f32x4_a), f32_to_bf16_bits(f32x4_b));
  }

  template <typename U>
  Simd<bfloat16_t, BF16_N>(Simd<U, BF16_N> other)
      : Simd(Simd<float, BF16_N>(other)) {}

  operator Simd<float, BF16_N>() const {
    float32x4x2_t v;
    v.val[0] = bf16_bits_to_f32(vget_low_u16(value));
    v.val[1] = bf16_bits_to_f32(vget_high_u16(value));
    return load<float, BF16_N>((float*)&v);
  }

  template <typename U>
  operator Simd<U, BF16_N>() const {
    return Simd<U, BF16_N>(Simd<float, BF16_N>(*this));
  }

  bfloat16_t operator[](int idx) const {
    return reinterpret_cast<const bfloat16_t*>(&value)[idx];
  }

  bfloat16_t& operator[](int idx) {
    return reinterpret_cast<bfloat16_t*>(&value)[idx];
  }

  uint16x8_t value;
};

#define DEFINE_BF16_UNARY_OP(name)                                       \
  inline Simd<bfloat16_t, BF16_N> name(Simd<bfloat16_t, BF16_N> a) {     \
    return Simd<bfloat16_t, BF16_N>(name(Simd<float, BF16_N>(a)));       \
  }

DEFINE_BF16_UNARY_OP(abs)
DEFINE_BF16_UNARY_OP(ceil)
DEFINE_BF16_UNARY_OP(floor)
DEFINE_BF16_UNARY_OP(sqrt)
DEFINE_BF16_UNARY_OP(rsqrt)
DEFINE_BF16_UNARY_OP(recip)
DEFINE_BF16_UNARY_OP(rint)
DEFINE_BF16_UNARY_OP(acos)
DEFINE_BF16_UNARY_OP(acosh)
DEFINE_BF16_UNARY_OP(asin)
DEFINE_BF16_UNARY_OP(asinh)
DEFINE_BF16_UNARY_OP(atan)
DEFINE_BF16_UNARY_OP(atanh)
DEFINE_BF16_UNARY_OP(cosh)
DEFINE_BF16_UNARY_OP(expm1)
DEFINE_BF16_UNARY_OP(log)
DEFINE_BF16_UNARY_OP(log2)
DEFINE_BF16_UNARY_OP(log10)
DEFINE_BF16_UNARY_OP(log1p)
DEFINE_BF16_UNARY_OP(sinh)
DEFINE_BF16_UNARY_OP(tan)
DEFINE_BF16_UNARY_OP(tanh)

inline Simd<bfloat16_t, BF16_N> operator-(Simd<bfloat16_t, BF16_N> v) {
  return Simd<bfloat16_t, BF16_N>(veorq_u16(v.value, vdupq_n_u16(0x8000)));
}

inline Simd<bool, BF16_N> operator!(Simd<bfloat16_t, BF16_N> v) {
  return !Simd<float, BF16_N>(v);
}

#define DEFINE_BF16_BINARY_OP(name)                                          \
  inline Simd<bfloat16_t, BF16_N> name(                                     \
      Simd<bfloat16_t, BF16_N> a, Simd<bfloat16_t, BF16_N> b) {             \
    return Simd<bfloat16_t, BF16_N>(                                        \
        name(Simd<float, BF16_N>(a), Simd<float, BF16_N>(b)));              \
  }                                                                          \
  template <typename T>                                                      \
  Simd<bfloat16_t, BF16_N> name(Simd<bfloat16_t, BF16_N> a, T b) {          \
    return name(a, Simd<bfloat16_t, BF16_N>(b));                            \
  }                                                                          \
  template <typename T>                                                      \
  Simd<bfloat16_t, BF16_N> name(T a, Simd<bfloat16_t, BF16_N> b) {          \
    return name(Simd<bfloat16_t, BF16_N>(a), b);                            \
  }

DEFINE_BF16_BINARY_OP(operator+)
DEFINE_BF16_BINARY_OP(operator-)
DEFINE_BF16_BINARY_OP(operator*)
DEFINE_BF16_BINARY_OP(operator/)
DEFINE_BF16_BINARY_OP(maximum)
DEFINE_BF16_BINARY_OP(minimum)
DEFINE_BF16_BINARY_OP(atan2)
DEFINE_BF16_BINARY_OP(remainder)
DEFINE_BF16_BINARY_OP(pow)

#define DEFINE_BF16_COMPARISON(Op)                                           \
  inline Simd<bool, BF16_N> operator Op(                                     \
      Simd<bfloat16_t, BF16_N> a, Simd<bfloat16_t, BF16_N> b) {              \
    return Simd<float, BF16_N>(a) Op Simd<float, BF16_N>(b);                 \
  }                                                                          \
  template <typename T>                                                      \
  Simd<bool, BF16_N> operator Op(Simd<bfloat16_t, BF16_N> a, T b) {          \
    return a Op Simd<bfloat16_t, BF16_N>(b);                                 \
  }                                                                          \
  template <typename T>                                                      \
  Simd<bool, BF16_N> operator Op(T a, Simd<bfloat16_t, BF16_N> b) {          \
    return Simd<bfloat16_t, BF16_N>(a) Op b;                                 \
  }

DEFINE_BF16_COMPARISON(==)
DEFINE_BF16_COMPARISON(!=)
DEFINE_BF16_COMPARISON(>)
DEFINE_BF16_COMPARISON(<)
DEFINE_BF16_COMPARISON(>=)
DEFINE_BF16_COMPARISON(<=)

inline Simd<bfloat16_t, BF16_N> operator&&(
    Simd<bfloat16_t, BF16_N> a,
    Simd<bfloat16_t, BF16_N> b) {
  return Simd<bfloat16_t, BF16_N>(
      Simd<float, BF16_N>(a) && Simd<float, BF16_N>(b));
}

inline Simd<bfloat16_t, BF16_N> operator||(
    Simd<bfloat16_t, BF16_N> a,
    Simd<bfloat16_t, BF16_N> b) {
  return Simd<bfloat16_t, BF16_N>(
      Simd<float, BF16_N>(a) || Simd<float, BF16_N>(b));
}

template <>
inline Simd<bool, BF16_N> isnan(Simd<bfloat16_t, BF16_N> v) {
  return isnan(Simd<float, BF16_N>(v));
}

template <>
inline Simd<bfloat16_t, BF16_N> clamp(
    Simd<bfloat16_t, BF16_N> v,
    Simd<bfloat16_t, BF16_N> min,
    Simd<bfloat16_t, BF16_N> max) {
  return minimum(maximum(v, min), max);
}

template <typename T>
Simd<bfloat16_t, BF16_N>
fma(Simd<bfloat16_t, BF16_N> x, Simd<bfloat16_t, BF16_N> y, T z) {
  return Simd<bfloat16_t, BF16_N>(fma(
      Simd<float, BF16_N>(x),
      Simd<float, BF16_N>(y),
      Simd<float, BF16_N>(Simd<bfloat16_t, BF16_N>(z))));
}

template <typename MaskT>
Simd<bfloat16_t, BF16_N> select(
    Simd<MaskT, BF16_N> mask,
    Simd<bfloat16_t, BF16_N> x,
    Simd<bfloat16_t, BF16_N> y) {
  auto m = Simd<uint16_t, BF16_N>(mask);
  return Simd<bfloat16_t, BF16_N>(
      vbslq_u16(*(uint16x8_t*)(&m.value), x.value, y.value));
}

inline bfloat16_t max(Simd<bfloat16_t, BF16_N> x) {
  return bfloat16_t(max(Simd<float, BF16_N>(x)));
}
inline bfloat16_t min(Simd<bfloat16_t, BF16_N> x) {
  return bfloat16_t(min(Simd<float, BF16_N>(x)));
}
inline bfloat16_t sum(Simd<bfloat16_t, BF16_N> x) {
  return bfloat16_t(sum(Simd<float, BF16_N>(x)));
}
inline bfloat16_t prod(Simd<bfloat16_t, BF16_N> x) {
  return bfloat16_t(prod(Simd<float, BF16_N>(x)));
}

} // namespace mlx::core::simd
