// Copyright © 2026 Apple Inc.

#pragma once

#include "mlx/backend/cpu/simd/base_simd.h"
#include "mlx/types/half_types.h"

#include "hwy/highway.h"

#include <cstring>

HWY_BEFORE_NAMESPACE();
namespace mlx::core::simd {
namespace hn = hwy::HWY_NAMESPACE;

// Highway's AVX3-family x86 targets use 512-bit vectors.
#if HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_DL ||       \
    HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX3_SPR || \
    HWY_TARGET == HWY_AVX10_2
inline constexpr int highway_max_f32_lanes = 16;
inline constexpr int highway_max_f64_lanes = 8;
#elif HWY_TARGET == HWY_AVX2
inline constexpr int highway_max_f32_lanes = 8;
inline constexpr int highway_max_f64_lanes = 4;
#else
inline constexpr int highway_max_f32_lanes = 4;
inline constexpr int highway_max_f64_lanes = 2;
#endif

template <>
inline constexpr int max_size<float> = highway_max_f32_lanes;
template <>
inline constexpr int max_size<double> = highway_max_f64_lanes;
template <>
inline constexpr int max_size<int32_t> = highway_max_f32_lanes;
template <>
inline constexpr int max_size<uint32_t> = highway_max_f32_lanes;
template <>
inline constexpr int max_size<uint8_t> = 8;
template <>
inline constexpr int max_size<float16_t> = highway_max_f32_lanes;
template <>
inline constexpr int max_size<bfloat16_t> = highway_max_f32_lanes;

namespace highway_detail {

template <typename T>
struct LaneType {
  using type = T;
};

template <>
struct LaneType<float16_t> {
  using type = hwy::float16_t;
};

template <>
struct LaneType<bfloat16_t> {
  using type = hwy::bfloat16_t;
};

template <typename T>
using LaneTypeT = typename LaneType<T>::type;

template <typename T, int N>
using D = hn::FixedTag<LaneTypeT<T>, N>;

template <typename T>
struct IsSimd : std::false_type {};

template <typename T, int N>
struct IsSimd<Simd<T, N>> : std::true_type {};

template <int N, typename T = int>
using EnableIfVector = std::enable_if_t<(N > 1), T>;

template <int N, typename T, typename R = int>
using EnableIfVectorNonBool =
    std::enable_if_t<(N > 1) && !std::is_same_v<T, bool>, R>;

template <typename T>
constexpr bool is_half_v =
    std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>;

template <typename T>
uint16_t half_bits(T value) {
  uint16_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

template <typename T, typename U>
LaneTypeT<T> to_lane(U value) {
  if constexpr (std::is_same_v<T, float16_t>) {
    return hwy::float16_t::FromBits(
        half_bits(static_cast<float16_t>(static_cast<float>(value))));
  } else if constexpr (std::is_same_v<T, bfloat16_t>) {
    return hwy::bfloat16_t::FromBits(
        half_bits(static_cast<bfloat16_t>(static_cast<float>(value))));
  } else {
    return static_cast<T>(value);
  }
}

template <typename T, typename Lane>
T from_lane(Lane value) {
  if constexpr (std::is_same_v<T, float16_t>) {
    uint16_t bits = hwy::BitCastScalar<uint16_t>(value);
    float16_t out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
  } else if constexpr (std::is_same_v<T, bfloat16_t>) {
    uint16_t bits = hwy::BitCastScalar<uint16_t>(value);
    bfloat16_t out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
  } else {
    return static_cast<T>(value);
  }
}

template <typename T>
const LaneTypeT<T>* lane_ptr(const T* ptr) {
  return reinterpret_cast<const LaneTypeT<T>*>(ptr);
}

template <typename T>
LaneTypeT<T>* lane_ptr(T* ptr) {
  return reinterpret_cast<LaneTypeT<T>*>(ptr);
}

template <typename T, int N, typename Fn>
Simd<T, N> map_unary(Simd<T, N> x, Fn fn);

template <typename T, int N, typename Fn>
Simd<T, N> map_binary(Simd<T, N> x, Simd<T, N> y, Fn fn);

template <int N>
struct BoolSimdBase {
  static constexpr int size = N;
  uint8_t bits[(N + 7) / 8] = {};

  BoolSimdBase() = default;

  BoolSimdBase(bool value) {
    if (value) {
      for (int i = 0; i < static_cast<int>(sizeof(bits)); ++i) {
        bits[i] = 0xFF;
      }
      bits[sizeof(bits) - 1] &=
          static_cast<uint8_t>((1u << (((N - 1) % 8) + 1)) - 1u);
    }
  }

  template <typename T>
  explicit BoolSimdBase(Simd<T, N> value) {
    using D = highway_detail::D<T, N>;
    const D d;
    hn::StoreMaskBits(d, hn::MaskFromVec(value.value), bits);
  }

  template <class D>
  hn::Mask<D> to_mask(D d) const {
    return hn::LoadMaskBits(d, bits);
  }

  static BoolSimdBase load(const bool* ptr) {
    BoolSimdBase out;
    for (int i = 0; i < N; ++i) {
      if (ptr[i]) {
        out.bits[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
      }
    }
    return out;
  }

  void store(bool* ptr) const {
    for (int i = 0; i < N; ++i) {
      ptr[i] = (bits[i / 8] & static_cast<uint8_t>(1u << (i % 8))) != 0;
    }
  }

  bool operator[](int idx) const {
    return (bits[idx / 8] & static_cast<uint8_t>(1u << (idx % 8))) != 0;
  }
};

} // namespace highway_detail

#define MLX_HIGHWAY_BOOL_SIMD(N)                           \
  template <>                                              \
  struct Simd<bool, N> : highway_detail::BoolSimdBase<N> { \
    using Base = highway_detail::BoolSimdBase<N>;          \
    using Base::Base;                                      \
                                                           \
    static Simd load(const bool* ptr) {                    \
      Simd out;                                            \
      static_cast<Base&>(out) = Base::load(ptr);           \
      return out;                                          \
    }                                                      \
                                                           \
    template <class D>                                     \
    static Simd from_mask(D d, hn::Mask<D> mask) {         \
      Simd out;                                            \
      hn::StoreMaskBits(d, mask, out.bits);                \
      return out;                                          \
    }                                                      \
  };

MLX_HIGHWAY_BOOL_SIMD(2)
MLX_HIGHWAY_BOOL_SIMD(4)
MLX_HIGHWAY_BOOL_SIMD(8)
MLX_HIGHWAY_BOOL_SIMD(16)
#undef MLX_HIGHWAY_BOOL_SIMD

template <typename T, int N>
struct Simd {
  static constexpr int size = N;
  using D = highway_detail::D<T, N>;
  using V = hn::Vec<D>;

  V value;

  Simd() : value(hn::Zero(D())) {}

  template <
      typename U,
      typename =
          std::enable_if_t<!highway_detail::IsSimd<std::decay_t<U>>::value>>
  Simd(U v) : value(hn::Set(D(), highway_detail::to_lane<T>(v))) {}

  template <typename U>
  Simd(Simd<U, N> other) {
    if constexpr (std::is_same_v<T, U>) {
      value = other.value;
    } else if constexpr (
        std::is_same_v<T, float> && highway_detail::is_half_v<U>) {
      value = hn::PromoteTo(D(), other.value);
    } else if constexpr (
        highway_detail::is_half_v<T> && std::is_same_v<U, float>) {
      value = hn::DemoteTo(D(), other.value);
    } else if constexpr (
        !highway_detail::is_half_v<T> && !highway_detail::is_half_v<U> &&
        !std::is_same_v<T, bool> && !std::is_same_v<U, bool> &&
        std::is_integral_v<T> && std::is_integral_v<U> &&
        sizeof(T) == sizeof(U)) {
      value = hn::BitCast(D(), other.value);
    } else if constexpr (
        !highway_detail::is_half_v<T> && !highway_detail::is_half_v<U> &&
        !std::is_same_v<T, bool> && !std::is_same_v<U, bool> &&
        sizeof(T) == sizeof(U)) {
      value = hn::ConvertTo(D(), other.value);
    } else if constexpr (std::is_same_v<U, bool>) {
      const D d;
      value = hn::IfThenElse(
          other.to_mask(d),
          hn::Set(d, highway_detail::to_lane<T>(1)),
          hn::Zero(d));
    } else {
      alignas(64) U tmp_in[N];
      alignas(64) T tmp_out[N];
      other.store(tmp_in);
      for (int i = 0; i < N; ++i) {
        tmp_out[i] = static_cast<T>(tmp_in[i]);
      }
      value = load(tmp_out).value;
    }
  }

  static Simd load(const T* ptr) {
    Simd out;
    out.value = hn::LoadU(D(), highway_detail::lane_ptr(ptr));
    return out;
  }

  void store(T* ptr) const {
    hn::StoreU(value, D(), highway_detail::lane_ptr(ptr));
  }

  T operator[](int idx) const {
    alignas(64) T tmp[N];
    store(tmp);
    return tmp[idx];
  }
};

template <>
struct Simd<int64_t, 4> {
  static constexpr int size = 4;
  Simd<int64_t, 2> lo;
  Simd<int64_t, 2> hi;

  Simd() = default;
  Simd(int64_t v) : lo(v), hi(v) {}

  static Simd load(const int64_t* ptr) {
    Simd out;
    out.lo = simd::load<int64_t, 2>(ptr);
    out.hi = simd::load<int64_t, 2>(ptr + 2);
    return out;
  }

  void store(int64_t* ptr) const {
    simd::store(ptr, lo);
    simd::store(ptr + 2, hi);
  }

  int64_t operator[](int idx) const {
    return idx < 2 ? lo[idx] : hi[idx - 2];
  }
};

template <>
struct Simd<int64_t, 8> {
  static constexpr int size = 8;
  Simd<int64_t, 4> lo;
  Simd<int64_t, 4> hi;

  Simd() = default;
  Simd(int64_t v) : lo(v), hi(v) {}

  static Simd load(const int64_t* ptr) {
    Simd out;
    out.lo = simd::load<int64_t, 4>(ptr);
    out.hi = simd::load<int64_t, 4>(ptr + 4);
    return out;
  }

  void store(int64_t* ptr) const {
    simd::store(ptr, lo);
    simd::store(ptr + 4, hi);
  }

  int64_t operator[](int idx) const {
    return idx < 4 ? lo[idx] : hi[idx - 4];
  }
};

template <>
struct Simd<uint64_t, 4> {
  static constexpr int size = 4;
  Simd<uint64_t, 2> lo;
  Simd<uint64_t, 2> hi;

  Simd() = default;
  Simd(uint64_t v) : lo(v), hi(v) {}

  static Simd load(const uint64_t* ptr) {
    Simd out;
    out.lo = simd::load<uint64_t, 2>(ptr);
    out.hi = simd::load<uint64_t, 2>(ptr + 2);
    return out;
  }

  void store(uint64_t* ptr) const {
    simd::store(ptr, lo);
    simd::store(ptr + 2, hi);
  }

  uint64_t operator[](int idx) const {
    return idx < 2 ? lo[idx] : hi[idx - 2];
  }
};

template <>
struct Simd<uint64_t, 8> {
  static constexpr int size = 8;
  Simd<uint64_t, 4> lo;
  Simd<uint64_t, 4> hi;

  Simd() = default;
  Simd(uint64_t v) : lo(v), hi(v) {}

  static Simd load(const uint64_t* ptr) {
    Simd out;
    out.lo = simd::load<uint64_t, 4>(ptr);
    out.hi = simd::load<uint64_t, 4>(ptr + 4);
    return out;
  }

  void store(uint64_t* ptr) const {
    simd::store(ptr, lo);
    simd::store(ptr + 4, hi);
  }

  uint64_t operator[](int idx) const {
    return idx < 4 ? lo[idx] : hi[idx - 4];
  }
};

namespace highway_detail {

template <typename T, int N, typename Fn>
Simd<T, N> map_unary(Simd<T, N> x, Fn fn) {
  alignas(64) T in[N];
  alignas(64) T out[N];
  x.store(in);
  for (int i = 0; i < N; ++i) {
    out[i] = static_cast<T>(fn(static_cast<float>(in[i])));
  }
  return Simd<T, N>::load(out);
}

template <typename T, int N, typename Fn>
Simd<T, N> map_binary(Simd<T, N> x, Simd<T, N> y, Fn fn) {
  alignas(64) T in_x[N];
  alignas(64) T in_y[N];
  alignas(64) T out[N];
  x.store(in_x);
  y.store(in_y);
  for (int i = 0; i < N; ++i) {
    out[i] = static_cast<T>(
        fn(static_cast<float>(in_x[i]), static_cast<float>(in_y[i])));
  }
  return Simd<T, N>::load(out);
}

template <typename T, int N, typename Fn>
Simd<T, N> map_integer_binary(Simd<T, N> x, Simd<T, N> y, Fn fn) {
  alignas(64) T in_x[N];
  alignas(64) T in_y[N];
  alignas(64) T out[N];
  x.store(in_x);
  y.store(in_y);
  for (int i = 0; i < N; ++i) {
    out[i] = static_cast<T>(fn(in_x[i], in_y[i]));
  }
  return Simd<T, N>::load(out);
}

} // namespace highway_detail

template <int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<bool, N> operator!(Simd<bool, N> x) {
  Simd<bool, N> out;
  for (int i = 0; i < static_cast<int>(sizeof(out.bits)); ++i) {
    out.bits[i] = static_cast<uint8_t>(~x.bits[i]);
  }
  out.bits[sizeof(out.bits) - 1] &=
      static_cast<uint8_t>((1u << (((N - 1) % 8) + 1)) - 1u);
  return out;
}

#define MLX_HIGHWAY_BOOL_OP(name, op)                              \
  template <int N, highway_detail::EnableIfVector<N> = 0>          \
  inline Simd<bool, N> name(Simd<bool, N> a, Simd<bool, N> b) {    \
    Simd<bool, N> out;                                             \
    for (int i = 0; i < static_cast<int>(sizeof(out.bits)); ++i) { \
      out.bits[i] = static_cast<uint8_t>(a.bits[i] op b.bits[i]);  \
    }                                                              \
    return out;                                                    \
  }

MLX_HIGHWAY_BOOL_OP(operator&&, &)
MLX_HIGHWAY_BOOL_OP(operator&, &)
MLX_HIGHWAY_BOOL_OP(operator||, |)
MLX_HIGHWAY_BOOL_OP(operator|, |)
MLX_HIGHWAY_BOOL_OP(operator^, ^)
#undef MLX_HIGHWAY_BOOL_OP

template <int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<bool, N> operator==(Simd<bool, N> a, Simd<bool, N> b) {
  return !(a ^ b);
}

template <int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<bool, N> operator!=(Simd<bool, N> a, Simd<bool, N> b) {
  return a ^ b;
}

#define MLX_HIGHWAY_ARITHMETIC(name, hwy_op)                          \
  template <typename T, int N, highway_detail::EnableIfVector<N> = 0> \
  inline Simd<T, N> name(Simd<T, N> a, Simd<T, N> b) {                \
    if constexpr (highway_detail::is_half_v<T>) {                     \
      return Simd<T, N>(name(Simd<float, N>(a), Simd<float, N>(b)));  \
    } else {                                                          \
      Simd<T, N> out;                                                 \
      out.value = hn::hwy_op(a.value, b.value);                       \
      return out;                                                     \
    }                                                                 \
  }                                                                   \
  template <                                                          \
      typename T,                                                     \
      int N,                                                          \
      typename U,                                                     \
      highway_detail::EnableIfVector<N> = 0>                          \
  inline Simd<T, N> name(Simd<T, N> a, U b) {                         \
    return name(a, Simd<T, N>(b));                                    \
  }                                                                   \
  template <                                                          \
      typename T,                                                     \
      int N,                                                          \
      typename U,                                                     \
      highway_detail::EnableIfVector<N> = 0>                          \
  inline Simd<T, N> name(U a, Simd<T, N> b) {                         \
    return name(Simd<T, N>(a), b);                                    \
  }

MLX_HIGHWAY_ARITHMETIC(operator+, Add)
MLX_HIGHWAY_ARITHMETIC(operator-, Sub)
MLX_HIGHWAY_ARITHMETIC(operator*, Mul)
#undef MLX_HIGHWAY_ARITHMETIC

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<T, N> operator/(Simd<T, N> a, Simd<T, N> b) {
  if constexpr (std::is_integral_v<T>) {
    return highway_detail::map_integer_binary(
        a, b, [](T x, T y) { return static_cast<T>(x / y); });
  } else if constexpr (highway_detail::is_half_v<T>) {
    return Simd<T, N>(Simd<float, N>(a) / Simd<float, N>(b));
  } else {
    Simd<T, N> out;
    out.value = hn::Div(a.value, b.value);
    return out;
  }
}

template <typename T, int N, typename U, highway_detail::EnableIfVector<N> = 0>
inline Simd<T, N> operator/(Simd<T, N> a, U b) {
  return a / Simd<T, N>(b);
}

template <typename T, int N, typename U, highway_detail::EnableIfVector<N> = 0>
inline Simd<T, N> operator/(U a, Simd<T, N> b) {
  return Simd<T, N>(a) / b;
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<T, N> operator-(Simd<T, N> a) {
  if constexpr (highway_detail::is_half_v<T>) {
    return Simd<T, N>(-Simd<float, N>(a));
  } else if constexpr (std::is_unsigned_v<T>) {
    Simd<T, N> out;
    out.value = hn::Sub(hn::Zero(typename Simd<T, N>::D()), a.value);
    return out;
  } else {
    Simd<T, N> out;
    out.value = hn::Neg(a.value);
    return out;
  }
}

#define MLX_HIGHWAY_BITWISE(name, hwy_op)                             \
  template <typename T, int N, highway_detail::EnableIfVector<N> = 0> \
  inline Simd<T, N> name(Simd<T, N> a, Simd<T, N> b) {                \
    Simd<T, N> out;                                                   \
    out.value = hn::hwy_op(a.value, b.value);                         \
    return out;                                                       \
  }                                                                   \
  template <                                                          \
      typename T,                                                     \
      int N,                                                          \
      typename U,                                                     \
      highway_detail::EnableIfVector<N> = 0>                          \
  inline Simd<T, N> name(Simd<T, N> a, U b) {                         \
    return name(a, Simd<T, N>(b));                                    \
  }                                                                   \
  template <                                                          \
      typename T,                                                     \
      int N,                                                          \
      typename U,                                                     \
      highway_detail::EnableIfVector<N> = 0>                          \
  inline Simd<T, N> name(U a, Simd<T, N> b) {                         \
    return name(Simd<T, N>(a), b);                                    \
  }

MLX_HIGHWAY_BITWISE(operator&, And)
MLX_HIGHWAY_BITWISE(operator|, Or)
MLX_HIGHWAY_BITWISE(operator^, Xor)
#undef MLX_HIGHWAY_BITWISE

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<T, N> operator~(Simd<T, N> a) {
  Simd<T, N> out;
  out.value = hn::Not(a.value);
  return out;
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<T, N> operator<<(Simd<T, N> a, int bits) {
  Simd<T, N> out;
  out.value = hn::ShiftLeftSame(a.value, bits);
  return out;
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<T, N> operator>>(Simd<T, N> a, int bits) {
  Simd<T, N> out;
  out.value = hn::ShiftRightSame(a.value, bits);
  return out;
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<T, N> operator<<(Simd<T, N> a, Simd<T, N> b) {
  Simd<T, N> out;
  out.value = a.value << b.value;
  return out;
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<T, N> operator>>(Simd<T, N> a, Simd<T, N> b) {
  Simd<T, N> out;
  out.value = a.value >> b.value;
  return out;
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<bool, N> operator!(Simd<T, N> a) {
  if constexpr (highway_detail::is_half_v<T>) {
    return !Simd<float, N>(a);
  } else {
    const typename Simd<T, N>::D d;
    return Simd<bool, N>::from_mask(d, hn::Eq(a.value, hn::Zero(d)));
  }
}

#define MLX_HIGHWAY_COMPARE(name, hwy_op)                               \
  template <typename T, int N, highway_detail::EnableIfVector<N> = 0>   \
  inline Simd<bool, N> name(Simd<T, N> a, Simd<T, N> b) {               \
    if constexpr (highway_detail::is_half_v<T>) {                       \
      return name(Simd<float, N>(a), Simd<float, N>(b));                \
    } else {                                                            \
      const typename Simd<T, N>::D d;                                   \
      return Simd<bool, N>::from_mask(d, hn::hwy_op(a.value, b.value)); \
    }                                                                   \
  }                                                                     \
  template <                                                            \
      typename T,                                                       \
      int N,                                                            \
      typename U,                                                       \
      highway_detail::EnableIfVector<N> = 0>                            \
  inline Simd<bool, N> name(Simd<T, N> a, U b) {                        \
    return name(a, Simd<T, N>(b));                                      \
  }                                                                     \
  template <                                                            \
      typename T,                                                       \
      int N,                                                            \
      typename U,                                                       \
      highway_detail::EnableIfVector<N> = 0>                            \
  inline Simd<bool, N> name(U a, Simd<T, N> b) {                        \
    return name(Simd<T, N>(a), b);                                      \
  }

MLX_HIGHWAY_COMPARE(operator==, Eq)
MLX_HIGHWAY_COMPARE(operator!=, Ne)
MLX_HIGHWAY_COMPARE(operator<, Lt)
MLX_HIGHWAY_COMPARE(operator>, Gt)
MLX_HIGHWAY_COMPARE(operator<=, Le)
MLX_HIGHWAY_COMPARE(operator>=, Ge)
#undef MLX_HIGHWAY_COMPARE

template <typename T, int N, highway_detail::EnableIfVectorNonBool<N, T> = 0>
inline Simd<T, N> operator&&(Simd<T, N> a, Simd<T, N> b) {
  return Simd<T, N>((a != Simd<T, N>(0)) && (b != Simd<T, N>(0)));
}

template <typename T, int N, highway_detail::EnableIfVectorNonBool<N, T> = 0>
inline Simd<T, N> operator||(Simd<T, N> a, Simd<T, N> b) {
  return Simd<T, N>((a != Simd<T, N>(0)) || (b != Simd<T, N>(0)));
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<T, N> select(Simd<bool, N> mask, Simd<T, N> x, Simd<T, N> y) {
  const typename Simd<T, N>::D d;
  Simd<T, N> out;
  out.value = hn::IfThenElse(mask.to_mask(d), x.value, y.value);
  return out;
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<T, N> abs(Simd<T, N> a) {
  if constexpr (std::is_unsigned_v<T>) {
    return a;
  } else if constexpr (highway_detail::is_half_v<T>) {
    return Simd<T, N>(abs(Simd<float, N>(a)));
  } else {
    Simd<T, N> out;
    out.value = hn::Abs(a.value);
    return out;
  }
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<bool, N> isnan(Simd<T, N> a) {
  if constexpr (std::is_floating_point_v<T>) {
    const typename Simd<T, N>::D d;
    return Simd<bool, N>::from_mask(d, hn::IsNaN(a.value));
  } else if constexpr (highway_detail::is_half_v<T>) {
    return isnan(Simd<float, N>(a));
  } else {
    return Simd<bool, N>(false);
  }
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<T, N> sqrt(Simd<T, N> a) {
  if constexpr (highway_detail::is_half_v<T>) {
    return Simd<T, N>(sqrt(Simd<float, N>(a)));
  } else {
    Simd<T, N> out;
    out.value = hn::Sqrt(a.value);
    return out;
  }
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<T, N> floor(Simd<T, N> a) {
  if constexpr (highway_detail::is_half_v<T>) {
    return Simd<T, N>(floor(Simd<float, N>(a)));
  } else if constexpr (std::is_floating_point_v<T>) {
    Simd<T, N> out;
    out.value = hn::Floor(a.value);
    return out;
  } else {
    return a;
  }
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<T, N> ceil(Simd<T, N> a) {
  if constexpr (highway_detail::is_half_v<T>) {
    return Simd<T, N>(ceil(Simd<float, N>(a)));
  } else if constexpr (std::is_floating_point_v<T>) {
    Simd<T, N> out;
    out.value = hn::Ceil(a.value);
    return out;
  } else {
    return a;
  }
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<T, N> rint(Simd<T, N> a) {
  if constexpr (highway_detail::is_half_v<T>) {
    return Simd<T, N>(rint(Simd<float, N>(a)));
  } else if constexpr (std::is_floating_point_v<T>) {
    Simd<T, N> out;
    out.value = hn::Round(a.value);
    return out;
  } else {
    return a;
  }
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<T, N> recip(Simd<T, N> a) {
  return Simd<T, N>(1) / a;
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<T, N> rsqrt(Simd<T, N> a) {
  return Simd<T, N>(1) / sqrt(a);
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<T, N> maximum(Simd<T, N> a, Simd<T, N> b) {
  if constexpr (highway_detail::is_half_v<T>) {
    return Simd<T, N>(maximum(Simd<float, N>(a), Simd<float, N>(b)));
  } else {
    Simd<T, N> out;
    out.value = hn::Max(a.value, b.value);
    if constexpr (std::is_floating_point_v<T>) {
      out = select(isnan(b), b, select(isnan(a), a, out));
    }
    return out;
  }
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<T, N> minimum(Simd<T, N> a, Simd<T, N> b) {
  if constexpr (highway_detail::is_half_v<T>) {
    return Simd<T, N>(minimum(Simd<float, N>(a), Simd<float, N>(b)));
  } else {
    Simd<T, N> out;
    out.value = hn::Min(a.value, b.value);
    if constexpr (std::is_floating_point_v<T>) {
      out = select(isnan(b), b, select(isnan(a), a, out));
    }
    return out;
  }
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<T, N> clamp(Simd<T, N> v, Simd<T, N> min_v, Simd<T, N> max_v) {
  return minimum(maximum(v, min_v), max_v);
}

template <typename T, int N, typename U, highway_detail::EnableIfVector<N> = 0>
inline Simd<T, N> fma(Simd<T, N> x, Simd<T, N> y, U z) {
  if constexpr (highway_detail::is_half_v<T>) {
    return Simd<T, N>(
        fma(Simd<float, N>(x), Simd<float, N>(y), Simd<float, N>(z)));
  } else {
    Simd<T, N> out;
    out.value = hn::MulAdd(x.value, y.value, Simd<T, N>(z).value);
    return out;
  }
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<T, N> remainder(Simd<T, N> a, Simd<T, N> b) {
  if constexpr (std::is_integral_v<T>) {
    return highway_detail::map_integer_binary(a, b, [](T x, T y) {
      T r = static_cast<T>(x % y);
      if constexpr (std::is_signed_v<T>) {
        if (r != 0 && ((r < 0) != (y < 0))) {
          r = static_cast<T>(r + y);
        }
      }
      return r;
    });
  } else {
    return highway_detail::map_binary(a, b, [](float x, float y) {
      float r = std::fmod(x, y);
      if (r != 0 && (std::signbit(r) != std::signbit(y))) {
        r += y;
      }
      return r;
    });
  }
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<T, N> pow(Simd<T, N> base, Simd<T, N> exp) {
  if constexpr (std::is_integral_v<T>) {
    return highway_detail::map_integer_binary(base, exp, [](T x, T y) {
      T result = 1;
      while (y) {
        if (y & 1) {
          result = static_cast<T>(result * x);
        }
        y = static_cast<T>(y >> 1);
        x = static_cast<T>(x * x);
      }
      return result;
    });
  } else {
    return highway_detail::map_binary(
        base, exp, [](float x, float y) { return std::pow(x, y); });
  }
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<T, N> atan2(Simd<T, N> a, Simd<T, N> b) {
  return highway_detail::map_binary(
      a, b, [](float x, float y) { return std::atan2(x, y); });
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline Simd<T, N> clz(Simd<T, N> x) {
  Simd<T, N> out;
  out.value = hn::LeadingZeroCount(x.value);
  return out;
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline T sum(Simd<T, N> x) {
  if constexpr (highway_detail::is_half_v<T>) {
    return static_cast<T>(sum(Simd<float, N>(x)));
  } else {
    return hn::ReduceSum(typename Simd<T, N>::D(), x.value);
  }
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline T max(Simd<T, N> x) {
  if constexpr (highway_detail::is_half_v<T>) {
    return static_cast<T>(max(Simd<float, N>(x)));
  } else {
    return hn::ReduceMax(typename Simd<T, N>::D(), x.value);
  }
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline T min(Simd<T, N> x) {
  if constexpr (highway_detail::is_half_v<T>) {
    return static_cast<T>(min(Simd<float, N>(x)));
  } else {
    return hn::ReduceMin(typename Simd<T, N>::D(), x.value);
  }
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline T prod(Simd<T, N> x) {
  alignas(64) T tmp[N];
  x.store(tmp);
  T result = 1;
  for (int i = 0; i < N; ++i) {
    result = static_cast<T>(result * tmp[i]);
  }
  return result;
}

template <int N, highway_detail::EnableIfVector<N> = 0>
inline bool any(Simd<bool, N> x) {
  for (int i = 0; i < static_cast<int>(sizeof(x.bits)); ++i) {
    if (x.bits[i] != 0) {
      return true;
    }
  }
  return false;
}

template <int N, highway_detail::EnableIfVector<N> = 0>
inline bool all(Simd<bool, N> x) {
  for (int i = 0; i < N; ++i) {
    if (!x[i]) {
      return false;
    }
  }
  return true;
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline bool any(Simd<T, N> x) {
  return any(!(!x));
}

template <typename T, int N, highway_detail::EnableIfVector<N> = 0>
inline bool all(Simd<T, N> x) {
  return all(!(!x));
}

#define MLX_HIGHWAY_TRANSCENDENTAL(name, std_name)                        \
  template <typename T, int N, highway_detail::EnableIfVector<N> = 0>     \
  inline Simd<T, N> name(Simd<T, N> x) {                                  \
    return highway_detail::map_unary(                                     \
        x, [](float v) { return static_cast<float>(std::std_name(v)); }); \
  }

MLX_HIGHWAY_TRANSCENDENTAL(acos, acos)
MLX_HIGHWAY_TRANSCENDENTAL(acosh, acosh)
MLX_HIGHWAY_TRANSCENDENTAL(asin, asin)
MLX_HIGHWAY_TRANSCENDENTAL(asinh, asinh)
MLX_HIGHWAY_TRANSCENDENTAL(atan, atan)
MLX_HIGHWAY_TRANSCENDENTAL(atanh, atanh)
MLX_HIGHWAY_TRANSCENDENTAL(cosh, cosh)
MLX_HIGHWAY_TRANSCENDENTAL(expm1, expm1)
MLX_HIGHWAY_TRANSCENDENTAL(log, log)
MLX_HIGHWAY_TRANSCENDENTAL(log1p, log1p)
MLX_HIGHWAY_TRANSCENDENTAL(log2, log2)
MLX_HIGHWAY_TRANSCENDENTAL(log10, log10)
MLX_HIGHWAY_TRANSCENDENTAL(sinh, sinh)
MLX_HIGHWAY_TRANSCENDENTAL(tan, tan)
MLX_HIGHWAY_TRANSCENDENTAL(tanh, tanh)
#undef MLX_HIGHWAY_TRANSCENDENTAL

} // namespace mlx::core::simd
HWY_AFTER_NAMESPACE();
