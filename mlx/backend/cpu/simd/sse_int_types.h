#pragma once

// Shared SSE 128-bit integer type definitions
// Used by both sse_simd.h and avx_simd.h
// AVX doesn't have 256-bit integer ops, so it reuses these SSE integer types
//
// IMPORTANT: This header assumes Simd<bool, 4> is already defined by the includer
// - In SSE builds: sse_simd.h defines Simd<bool, 4> with __m128i (4x32-bit masks)
// - In AVX builds: avx_simd.h defines Simd<bool, 4> with __m256i (4x64-bit masks for doubles)
//   but integer comparisons still return the SSE version with __m128i

#include <smmintrin.h> // SSE4.1
#include <cmath>
#include <stdint.h>

namespace mlx::core::simd {

// SSE integer types: 128-bit registers for int32/uint32
// Note: max_size is NOT defined here - each includer defines it
// SSE: max_size<int32_t> = 4, max_size<uint32_t> = 4
// AVX: max_size<int32_t> = 4, max_size<uint32_t> = 4 (same, AVX doesn't add int support)

// Note: Bool reductions (all/any) are defined in sse_simd.h to avoid duplication

// ============================================================================
// int32x4 and uint32x4 types (128-bit SSE)
// ============================================================================

template <>
struct Simd<int32_t, 4> {
  static constexpr int size = 4;
  __m128i value;

  Simd() : value(_mm_setzero_si128()) {}
  Simd(__m128i v) : value(v) {}
  Simd(int32_t v) : value(_mm_set1_epi32(v)) {}
  template <typename U>
  Simd(Simd<U, 4> v) : value(v.value) {}

  int32_t operator[](int idx) const {
    alignas(16) int32_t tmp[4];
    _mm_store_si128((__m128i*)tmp, value);
    return tmp[idx];
  }
};

template <>
struct Simd<uint32_t, 4> {
  static constexpr int size = 4;
  __m128i value;

  Simd() : value(_mm_setzero_si128()) {}
  Simd(__m128i v) : value(v) {}
  Simd(uint32_t v) : value(_mm_set1_epi32(v)) {}
  template <typename U>
  Simd(Simd<U, 4> v) : value(v.value) {}

  uint32_t operator[](int idx) const {
    alignas(16) uint32_t tmp[4];
    _mm_store_si128((__m128i*)tmp, value);
    return tmp[idx];
  }
};

// Load/Store int32x4
template <>
inline Simd<int32_t, 4> load<int32_t, 4>(const int32_t* ptr) {
  return _mm_loadu_si128((__m128i*)ptr);
}

template <>
inline void store<int32_t, 4>(int32_t* ptr, Simd<int32_t, 4> v) {
  _mm_storeu_si128((__m128i*)ptr, v.value);
}

// Load/Store uint32x4
template <>
inline Simd<uint32_t, 4> load<uint32_t, 4>(const uint32_t* ptr) {
  return _mm_loadu_si128((__m128i*)ptr);
}

template <>
inline void store<uint32_t, 4>(uint32_t* ptr, Simd<uint32_t, 4> v) {
  _mm_storeu_si128((__m128i*)ptr, v.value);
}

// Arithmetic uint32x4
inline Simd<uint32_t, 4> operator+(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return _mm_add_epi32(a.value, b.value);
}
inline Simd<uint32_t, 4> operator-(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return _mm_sub_epi32(a.value, b.value);
}
inline Simd<uint32_t, 4> operator*(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return _mm_mullo_epi32(a.value, b.value);
}
inline Simd<uint32_t, 4> operator-(Simd<uint32_t, 4> a) {
  return _mm_sub_epi32(_mm_setzero_si128(), a.value);
}
inline Simd<uint32_t, 4> operator/(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  alignas(16) uint32_t tmp_a[4], tmp_b[4], tmp_r[4];
  _mm_store_si128((__m128i*)tmp_a, a.value);
  _mm_store_si128((__m128i*)tmp_b, b.value);
  for (int i = 0; i < 4; i++) {
    tmp_r[i] = tmp_a[i] / tmp_b[i];
  }
  return _mm_load_si128((__m128i*)tmp_r);
}

// Bitwise uint32x4
inline Simd<uint32_t, 4> operator&(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return _mm_and_si128(a.value, b.value);
}
inline Simd<uint32_t, 4> operator|(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return _mm_or_si128(a.value, b.value);
}
inline Simd<uint32_t, 4> operator^(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return _mm_xor_si128(a.value, b.value);
}
inline Simd<uint32_t, 4> operator~(Simd<uint32_t, 4> a) {
  return _mm_xor_si128(a.value, _mm_set1_epi32(-1));
}

// Shifts uint32x4
inline Simd<uint32_t, 4> operator<<(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  alignas(16) uint32_t tmp_a[4], tmp_b[4], tmp_r[4];
  _mm_store_si128((__m128i*)tmp_a, a.value);
  _mm_store_si128((__m128i*)tmp_b, b.value);
  for (int i = 0; i < 4; i++) {
    tmp_r[i] = tmp_a[i] << tmp_b[i];
  }
  return _mm_load_si128((__m128i*)tmp_r);
}

inline Simd<uint32_t, 4> operator>>(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  alignas(16) uint32_t tmp_a[4], tmp_b[4], tmp_r[4];
  _mm_store_si128((__m128i*)tmp_a, a.value);
  _mm_store_si128((__m128i*)tmp_b, b.value);
  for (int i = 0; i < 4; i++) {
    tmp_r[i] = tmp_a[i] >> tmp_b[i];
  }
  return _mm_load_si128((__m128i*)tmp_r);
}

inline Simd<uint32_t, 4> maximum(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return _mm_max_epu32(a.value, b.value);
}

inline Simd<uint32_t, 4> minimum(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return _mm_min_epu32(a.value, b.value);
}

inline Simd<uint32_t, 4> clamp(Simd<uint32_t, 4> v, Simd<uint32_t, 4> min_val, Simd<uint32_t, 4> max_val) {
  return minimum(maximum(v, min_val), max_val);
}

inline Simd<bool, 4> operator==(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return Simd<bool, 4>(_mm_cmpeq_epi32(a.value, b.value));
}

inline Simd<bool, 4> operator!=(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return !(a == b);
}

inline Simd<bool, 4> operator!(Simd<uint32_t, 4> a) {
  return a == Simd<uint32_t, 4>(0);
}

inline Simd<bool, 4> isnan(Simd<uint32_t, 4> a) {
  return Simd<bool, 4>(false);
}

inline Simd<uint32_t, 4> operator&&(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  auto mask_a = (a != Simd<uint32_t, 4>(0));
  auto mask_b = (b != Simd<uint32_t, 4>(0));
  return Simd<uint32_t, 4>(_mm_and_si128(mask_a.value, mask_b.value));
}

inline Simd<uint32_t, 4> operator||(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  auto mask_a = (a != Simd<uint32_t, 4>(0));
  auto mask_b = (b != Simd<uint32_t, 4>(0));
  return Simd<uint32_t, 4>(_mm_or_si128(mask_a.value, mask_b.value));
}

inline Simd<bool, 4> operator<(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  auto minn = _mm_min_epu32(a.value, b.value);
  return Simd<bool, 4>(_mm_andnot_si128(_mm_cmpeq_epi32(a.value, b.value), _mm_cmpeq_epi32(minn, a.value)));
}

inline Simd<bool, 4> operator>(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return b < a;
}

inline Simd<bool, 4> operator<=(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return !(a > b);
}

inline Simd<bool, 4> operator>=(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return !(a < b);
}

inline Simd<uint32_t, 4> remainder(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return a - b * (a / b);
}

inline Simd<uint32_t, 4> pow(Simd<uint32_t, 4> base, Simd<uint32_t, 4> exp) {
  alignas(16) uint32_t tmp_base[4], tmp_exp[4], tmp_r[4];
  _mm_store_si128((__m128i*)tmp_base, base.value);
  _mm_store_si128((__m128i*)tmp_exp, exp.value);
  for (int i = 0; i < 4; i++) {
    tmp_r[i] = 1;
    for (uint32_t j = 0; j < tmp_exp[i]; j++) {
      tmp_r[i] *= tmp_base[i];
    }
  }
  return _mm_load_si128((__m128i*)tmp_r);
}

inline Simd<uint32_t, 4> select(
    Simd<bool, 4> mask,
    Simd<uint32_t, 4> x,
    Simd<uint32_t, 4> y) {
  return _mm_blendv_epi8(y.value, x.value, mask.value);
}

inline uint32_t sum(Simd<uint32_t, 4> v) {
  __m128i shuf = _mm_shuffle_epi32(v.value, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i sums = _mm_add_epi32(v.value, shuf);
  shuf = _mm_shuffle_epi32(sums, _MM_SHUFFLE(1, 0, 3, 2));
  sums = _mm_add_epi32(sums, shuf);
  return _mm_cvtsi128_si32(sums);
}

inline uint32_t max(Simd<uint32_t, 4> v) {
  __m128i shuf = _mm_shuffle_epi32(v.value, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i maxs = _mm_max_epu32(v.value, shuf);
  shuf = _mm_shuffle_epi32(maxs, _MM_SHUFFLE(1, 0, 3, 2));
  maxs = _mm_max_epu32(maxs, shuf);
  return _mm_cvtsi128_si32(maxs);
}

inline uint32_t min(Simd<uint32_t, 4> v) {
  __m128i shuf = _mm_shuffle_epi32(v.value, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i mins = _mm_min_epu32(v.value, shuf);
  shuf = _mm_shuffle_epi32(mins, _MM_SHUFFLE(1, 0, 3, 2));
  mins = _mm_min_epu32(mins, shuf);
  return _mm_cvtsi128_si32(mins);
}

inline uint32_t prod(Simd<uint32_t, 4> v) {
  __m128i shuf = _mm_shuffle_epi32(v.value, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i prods = _mm_mullo_epi32(v.value, shuf);
  shuf = _mm_shuffle_epi32(prods, _MM_SHUFFLE(1, 0, 3, 2));
  prods = _mm_mullo_epi32(prods, shuf);
  return _mm_cvtsi128_si32(prods);
}

// Arithmetic int32x4
inline Simd<int32_t, 4> operator+(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return _mm_add_epi32(a.value, b.value);
}
inline Simd<int32_t, 4> operator-(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return _mm_sub_epi32(a.value, b.value);
}
inline Simd<int32_t, 4> operator*(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return _mm_mullo_epi32(a.value, b.value);
}
inline Simd<int32_t, 4> operator/(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  alignas(16) int32_t tmp_a[4], tmp_b[4], tmp_r[4];
  _mm_store_si128((__m128i*)tmp_a, a.value);
  _mm_store_si128((__m128i*)tmp_b, b.value);
  for (int i = 0; i < 4; i++) {
    tmp_r[i] = tmp_a[i] / tmp_b[i];
  }
  return _mm_load_si128((__m128i*)tmp_r);
}
inline Simd<int32_t, 4> operator-(Simd<int32_t, 4> a) {
  return _mm_sub_epi32(_mm_setzero_si128(), a.value);
}

// Bitwise int32x4
inline Simd<int32_t, 4> operator&(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return _mm_and_si128(a.value, b.value);
}
inline Simd<int32_t, 4> operator|(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return _mm_or_si128(a.value, b.value);
}
inline Simd<int32_t, 4> operator^(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return _mm_xor_si128(a.value, b.value);
}
inline Simd<int32_t, 4> operator~(Simd<int32_t, 4> a) {
  return _mm_xor_si128(a.value, _mm_set1_epi32(-1));
}

// Shifts int32x4
inline Simd<int32_t, 4> operator<<(Simd<int32_t, 4> a, int bits) {
  return _mm_slli_epi32(a.value, bits);
}
inline Simd<int32_t, 4> operator>>(Simd<int32_t, 4> a, int bits) {
  return _mm_srai_epi32(a.value, bits);
}

inline Simd<int32_t, 4> operator<<(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  alignas(16) int32_t tmp_a[4], tmp_b[4], tmp_r[4];
  _mm_store_si128((__m128i*)tmp_a, a.value);
  _mm_store_si128((__m128i*)tmp_b, b.value);
  for (int i = 0; i < 4; i++) {
    tmp_r[i] = tmp_a[i] << tmp_b[i];
  }
  return _mm_load_si128((__m128i*)tmp_r);
}

inline Simd<int32_t, 4> operator>>(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  alignas(16) int32_t tmp_a[4], tmp_b[4], tmp_r[4];
  _mm_store_si128((__m128i*)tmp_a, a.value);
  _mm_store_si128((__m128i*)tmp_b, b.value);
  for (int i = 0; i < 4; i++) {
    tmp_r[i] = tmp_a[i] >> tmp_b[i];
  }
  return _mm_load_si128((__m128i*)tmp_r);
}

// Comparisons int32x4
inline Simd<bool, 4> operator<(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return Simd<bool, 4>(_mm_cmplt_epi32(a.value, b.value));
}
inline Simd<bool, 4> operator>(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return Simd<bool, 4>(_mm_cmpgt_epi32(a.value, b.value));
}
inline Simd<bool, 4> operator<=(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return !(a > b);
}
inline Simd<bool, 4> operator>=(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return !(a < b);
}
inline Simd<bool, 4> operator==(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return Simd<bool, 4>(_mm_cmpeq_epi32(a.value, b.value));
}
inline Simd<bool, 4> operator!=(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return !(a == b);
}

inline Simd<bool, 4> operator!(Simd<int32_t, 4> a) {
  return a == Simd<int32_t, 4>(0);
}

inline Simd<bool, 4> isnan(Simd<int32_t, 4> a) {
  return Simd<bool, 4>(false);
}

inline Simd<int32_t, 4> maximum(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return _mm_max_epi32(a.value, b.value);
}

inline Simd<int32_t, 4> minimum(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return _mm_min_epi32(a.value, b.value);
}

inline Simd<int32_t, 4> abs(Simd<int32_t, 4> a) {
  return _mm_abs_epi32(a.value);
}

inline Simd<int32_t, 4> clamp(Simd<int32_t, 4> v, Simd<int32_t, 4> min_val, Simd<int32_t, 4> max_val) {
  return minimum(maximum(v, min_val), max_val);
}

inline Simd<int32_t, 4> operator&&(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  auto mask_a = (a != Simd<int32_t, 4>(0));
  auto mask_b = (b != Simd<int32_t, 4>(0));
  return Simd<int32_t, 4>(_mm_and_si128(mask_a.value, mask_b.value));
}

inline Simd<int32_t, 4> operator||(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  auto mask_a = (a != Simd<int32_t, 4>(0));
  auto mask_b = (b != Simd<int32_t, 4>(0));
  return Simd<int32_t, 4>(_mm_or_si128(mask_a.value, mask_b.value));
}

inline Simd<int32_t, 4> select(
    Simd<bool, 4> mask,
    Simd<int32_t, 4> x,
    Simd<int32_t, 4> y) {
  return _mm_blendv_epi8(y.value, x.value, mask.value);
}

inline Simd<int32_t, 4> remainder(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  auto r = a - b * (a / b);
  auto mask = (r != Simd<int32_t, 4>(0)) && ((r < Simd<int32_t, 4>(0)) != (b < Simd<int32_t, 4>(0)));
  return select(mask, r + b, r);
}

inline Simd<int32_t, 4> pow(Simd<int32_t, 4> base, Simd<int32_t, 4> exp) {
  alignas(16) int32_t tmp_base[4], tmp_exp[4], tmp_r[4];
  _mm_store_si128((__m128i*)tmp_base, base.value);
  _mm_store_si128((__m128i*)tmp_exp, exp.value);
  for (int i = 0; i < 4; i++) {
    if (tmp_exp[i] < 0) {
      tmp_r[i] = 0;
    } else {
      tmp_r[i] = 1;
      for (int32_t j = 0; j < tmp_exp[i]; j++) {
        tmp_r[i] *= tmp_base[i];
      }
    }
  }
  return _mm_load_si128((__m128i*)tmp_r);
}

inline int32_t sum(Simd<int32_t, 4> v) {
  __m128i shuf = _mm_shuffle_epi32(v.value, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i sums = _mm_add_epi32(v.value, shuf);
  shuf = _mm_shuffle_epi32(sums, _MM_SHUFFLE(1, 0, 3, 2));
  sums = _mm_add_epi32(sums, shuf);
  return _mm_cvtsi128_si32(sums);
}

inline int32_t max(Simd<int32_t, 4> v) {
  __m128i shuf = _mm_shuffle_epi32(v.value, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i maxs = _mm_max_epi32(v.value, shuf);
  shuf = _mm_shuffle_epi32(maxs, _MM_SHUFFLE(1, 0, 3, 2));
  maxs = _mm_max_epi32(maxs, shuf);
  return _mm_cvtsi128_si32(maxs);
}

inline int32_t min(Simd<int32_t, 4> v) {
  __m128i shuf = _mm_shuffle_epi32(v.value, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i mins = _mm_min_epi32(v.value, shuf);
  shuf = _mm_shuffle_epi32(mins, _MM_SHUFFLE(1, 0, 3, 2));
  mins = _mm_min_epi32(mins, shuf);
  return _mm_cvtsi128_si32(mins);
}

inline int32_t prod(Simd<int32_t, 4> v) {
  __m128i shuf = _mm_shuffle_epi32(v.value, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i prods = _mm_mullo_epi32(v.value, shuf);
  shuf = _mm_shuffle_epi32(prods, _MM_SHUFFLE(1, 0, 3, 2));
  prods = _mm_mullo_epi32(prods, shuf);
  return _mm_cvtsi128_si32(prods);
}

// clz (count leading zeros)
inline Simd<uint32_t, 4> clz(Simd<uint32_t, 4> x) {
  alignas(16) uint32_t tmp[4], res[4];
  _mm_store_si128((__m128i*)tmp, x.value);
  for (int i = 0; i < 4; i++) {
    res[i] = tmp[i] == 0 ? 32 : __builtin_clz(tmp[i]);
  }
  return _mm_load_si128((__m128i*)res);
}

inline Simd<int32_t, 4> clz(Simd<int32_t, 4> x) {
  return Simd<int32_t, 4>(clz(Simd<uint32_t, 4>(x.value)).value);
}

} // namespace mlx::core::simd
