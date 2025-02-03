#pragma once

#include <arm_neon.h>

#include "mlx/backend/cpu/simd/base_simd.h"

namespace mlx::core::simd {

constexpr int N = 8;

template <>
struct Simd<float16_t, N> {
  static constexpr int size = N;
  using scalar_t = float16_t;

  Simd<float16_t, N>() {}

  template <typename U>
  Simd<float16_t, N>(U v) : value(vdupq_n_f16(v)){};

  Simd<float16_t, N>(float16x8_t v) : value(v){};

  Simd<float16_t, N>(Simd<float, N> other) {
    auto f32x4_a = *(float32x4_t*)(&other);
    auto f32x4_b = *((float32x4_t*)(&other) + 1);
    value = vcvt_high_f16_f32(vcvt_f16_f32(f32x4_a), f32x4_b);
  };

  Simd<float16_t, N>(Simd<uint16_t, N> other) {
    value = vcvtq_f16_u16(*(uint16x8_t*)(&other.value));
  };

  operator Simd<int16_t, N>() {
    auto v = vcvtq_s16_f16(value);
    return load<int16_t, N>((int16_t*)&v);
  };

  operator Simd<float, N>() {
    float32x4x2_t v;
    v.val[0] = vcvt_f32_f16(*(float16x4_t*)(&value));
    v.val[1] = vcvt_high_f32_f16(value);
    return load<float, N>((float*)&v);
  }
  float16_t operator[](int idx) const {
    return reinterpret_cast<const float16_t*>(&value)[idx];
  }

  float16_t& operator[](int idx) {
    return reinterpret_cast<float16_t*>(&value)[idx];
  }

  float16x8_t value;
};

#define DEFINE_NEON_UNARY_OP(name, op)                   \
  inline Simd<float16_t, N> name(Simd<float16_t, N> a) { \
    return Simd<float16_t, N>{op(a.value)};              \
  }

DEFINE_NEON_UNARY_OP(abs, vabsq_f16)
DEFINE_NEON_UNARY_OP(ceil, vrndpq_f16)
DEFINE_NEON_UNARY_OP(floor, vrndmq_f16)
DEFINE_NEON_UNARY_OP(sqrt, vsqrtq_f16)
DEFINE_NEON_UNARY_OP(rsqrt, vrsqrteq_f16)
DEFINE_NEON_UNARY_OP(recip, vrecpeq_f16)
DEFINE_NEON_UNARY_OP(rint, vrndnq_f16)

#define DEFINE_NEON_BINARY_OP(name, op)                                        \
  inline Simd<float16_t, N> name(Simd<float16_t, N> a, Simd<float16_t, N> b) { \
    return op(a.value, b.value);                                               \
  }                                                                            \
  template <typename T>                                                        \
  Simd<float16_t, N> name(Simd<float16_t, N> a, T b) {                         \
    return op(a.value, Simd<float16_t, N>(b).value);                           \
  }                                                                            \
  template <typename T>                                                        \
  Simd<float16_t, N> name(T a, Simd<float16_t, N> b) {                         \
    return op(Simd<float16_t, N>(a).value, b.value);                           \
  }

inline Simd<float16_t, N> operator!(Simd<float16_t, N> v) {
  auto out = vceqzq_f16(v.value);
  return Simd<uint16_t, N>(*(uint16_t*)&out);
}

inline Simd<float16_t, N> operator-(Simd<float16_t, N> v) {
  return vnegq_f16(v.value);
}

DEFINE_NEON_BINARY_OP(maximum, vmaxq_f16)
DEFINE_NEON_BINARY_OP(minimum, vminq_f16)
DEFINE_NEON_BINARY_OP(operator+, vaddq_f16)
DEFINE_NEON_BINARY_OP(operator-, vsubq_f16)
DEFINE_NEON_BINARY_OP(operator*, vmulq_f16)
DEFINE_NEON_BINARY_OP(operator/, vdivq_f16)

#define DEFINE_NEON_COMPARISON(Op, op)                   \
  template <typename T>                                  \
  Simd<bool, N> operator Op(Simd<float16_t, N> a, T b) { \
    auto out = op(a.value, Simd<float16_t, N>(b).value); \
    return Simd<uint16_t, N>(*(uint16_t*)(&out));        \
  }                                                      \
  template <typename T>                                  \
  Simd<bool, N> operator Op(T a, Simd<float16_t, N> b) { \
    auto out = op(Simd<float16_t, N>(a).value, b.value); \
    return Simd<uint16_t, N>(*(uint16_t*)(&out));        \
  }                                                      \
  inline Simd<bool, N> operator Op(                      \
      Simd<float16_t, N> a, Simd<float16_t, N> b) {      \
    auto out = op(a.value, b.value);                     \
    return Simd<uint16_t, N>(*(uint16_t*)(&out));        \
  }

DEFINE_NEON_COMPARISON(==, vceqq_f16)
DEFINE_NEON_COMPARISON(>=, vcgeq_f16)
DEFINE_NEON_COMPARISON(<=, vcleq_f16)
DEFINE_NEON_COMPARISON(>, vcgtq_f16)
DEFINE_NEON_COMPARISON(<, vcltq_f16)

template <typename T>
Simd<bool, N> operator!=(Simd<float16_t, N> a, T b) {
  return !(a == b);
}
template <typename T>
Simd<bool, N> operator!=(T a, Simd<float16_t, N> b) {
  return !(a == b);
}
inline Simd<bool, N> operator!=(Simd<float16_t, N> a, Simd<float16_t, N> b) {
  return !(a == b);
}

inline Simd<float16_t, N> operator||(
    Simd<float16_t, N> a,
    Simd<float16_t, N> b) {
  return Simd<uint16_t, N>((a != 0) || (b != 0));
}
template <typename T>
Simd<float16_t, N> operator||(Simd<float16_t, N> a, T b) {
  return Simd<uint16_t, N>((a != 0) || (b != 0));
}
template <typename T>
Simd<float16_t, N> operator||(T a, Simd<float16_t, N> b) {
  return Simd<uint16_t, N>((a != 0) || (b != 0));
}
inline Simd<float16_t, N> operator&&(
    Simd<float16_t, N> a,
    Simd<float16_t, N> b) {
  return Simd<uint16_t, N>((a != 0) && (b != 0));
}
template <typename T>
Simd<float16_t, N> operator&&(Simd<float16_t, N> a, T b) {
  return Simd<uint16_t, N>((a != 0) && (b != 0));
}
template <typename T>
Simd<float16_t, N> operator&&(T a, Simd<float16_t, N> b) {
  return Simd<uint16_t, N>((a != 0) && (b != 0));
}

template <>
inline Simd<bool, N> isnan(Simd<float16_t, N> v) {
  return v != v;
}

template <>
inline Simd<float16_t, N>
clamp(Simd<float16_t, N> v, Simd<float16_t, N> min, Simd<float16_t, N> max) {
  return minimum(maximum(v, min), max);
}

template <typename T>
Simd<float16_t, N> fma(Simd<float16_t, N> x, Simd<float16_t, N> y, T z) {
  return vfmaq_f16(x.value, y.value, Simd<float16_t, N>(z).value);
}

template <typename MaskT>
Simd<float16_t, N>
select(Simd<MaskT, N> mask, Simd<float16_t, N> x, Simd<float16_t, N> y) {
  return vbslq_f16(Simd<uint16_t, N>(mask).value, x.value, y.value);
}

// Reductions
inline float16_t max(Simd<float16_t, N> x) {
  float16x4_t y;
  y = vpmax_f16(vget_low_f16(x.value), vget_high_f16(x.value));
  y = vpmax_f16(y, y);
  y = vpmax_f16(y, y);
  return vget_lane_f16(y, 0);
}
inline float16_t min(Simd<float16_t, N> x) {
  float16x4_t y;
  y = vpmin_f16(vget_low_f16(x.value), vget_high_f16(x.value));
  y = vpmin_f16(y, y);
  y = vpmin_f16(y, y);
  return vget_lane_f16(y, 0);
}
inline float16_t sum(Simd<float16_t, N> x) {
  float16x4_t y;
  y = vpadd_f16(vget_low_f16(x.value), vget_high_f16(x.value));
  y = vpadd_f16(y, y);
  y = vpadd_f16(y, y);
  return vget_lane_f16(y, 0);
}
inline float16_t prod(Simd<float16_t, N> x) {
  auto hx = vmul_f16(vget_low_f16(x.value), vget_high_f16(x.value));
  auto out = hx[0];
  hx[0] *= hx[1];
  hx[0] *= hx[2];
  hx[0] *= hx[3];
  return hx[0];
}

} // namespace mlx::core::simd
