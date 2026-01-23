#pragma once

#ifdef __AVX512F__

// AVX-512 Foundation: 512-bit SIMD with mask registers
// Provides 16 floats, 8 doubles, 16 int32s per register
// Key features: mask registers, enhanced gather/scatter, conflict detection

#include "mlx/backend/cpu/simd/base_simd.h"
#include "mlx/backend/cpu/simd/x86_simd_macros.h"
#include "mlx/types/half_types.h"

#include <immintrin.h> // AVX-512
#include <stdint.h>
#include <cmath>

// ============================================================================
// AVX-512 Specific Macros
// ============================================================================
// These macros handle AVX-512's unique cast-down patterns for int8/int16 types.
// AVX-512 stores these types in the lower portion of 512-bit registers and
// uses AVX2/SSE intrinsics for actual operations.
// ============================================================================

// Bool operations using AVX-512 mask registers
#define DEFINE_AVX512_BOOL_OPS_MASK(N, mask_type, suffix)             \
  inline Simd<bool, N> operator!(Simd<bool, N> a) {                   \
    return Simd<bool, N>(_knot_##suffix(a.value));                    \
  }                                                                   \
  inline Simd<bool, N> operator||(Simd<bool, N> a, Simd<bool, N> b) { \
    return Simd<bool, N>(_kor_##suffix(a.value, b.value));            \
  }                                                                   \
  inline Simd<bool, N> operator&&(Simd<bool, N> a, Simd<bool, N> b) { \
    return Simd<bool, N>(_kand_##suffix(a.value, b.value));           \
  }                                                                   \
  inline Simd<bool, N> operator&(Simd<bool, N> a, Simd<bool, N> b) {  \
    return Simd<bool, N>(_kand_##suffix(a.value, b.value));           \
  }                                                                   \
  inline Simd<bool, N> operator|(Simd<bool, N> a, Simd<bool, N> b) {  \
    return Simd<bool, N>(_kor_##suffix(a.value, b.value));            \
  }                                                                   \
  inline Simd<bool, N> operator^(Simd<bool, N> a, Simd<bool, N> b) {  \
    return Simd<bool, N>(_kxor_##suffix(a.value, b.value));           \
  }                                                                   \
  inline Simd<bool, N> operator==(Simd<bool, N> a, Simd<bool, N> b) { \
    return Simd<bool, N>(_kxnor_##suffix(a.value, b.value));          \
  }                                                                   \
  inline Simd<bool, N> operator!=(Simd<bool, N> a, Simd<bool, N> b) { \
    return Simd<bool, N>(_kxor_##suffix(a.value, b.value));           \
  }

// Comparison operators for float/double using _mm512_cmp_* with flags
// Generates all 6 comparison operators using mask returns
#define DEFINE_AVX512_COMPARISONS_MASK(type, N, bool_n, intrinsic_suffix)     \
  inline Simd<bool, bool_n> operator<(Simd<type, N> a, Simd<type, N> b) {     \
    return Simd<bool, bool_n>(                                                \
        _mm512_cmp_##intrinsic_suffix##_mask(a.value, b.value, _CMP_LT_OQ));  \
  }                                                                           \
  inline Simd<bool, bool_n> operator<=(Simd<type, N> a, Simd<type, N> b) {    \
    return Simd<bool, bool_n>(                                                \
        _mm512_cmp_##intrinsic_suffix##_mask(a.value, b.value, _CMP_LE_OQ));  \
  }                                                                           \
  inline Simd<bool, bool_n> operator>(Simd<type, N> a, Simd<type, N> b) {     \
    return Simd<bool, bool_n>(                                                \
        _mm512_cmp_##intrinsic_suffix##_mask(a.value, b.value, _CMP_GT_OQ));  \
  }                                                                           \
  inline Simd<bool, bool_n> operator>=(Simd<type, N> a, Simd<type, N> b) {    \
    return Simd<bool, bool_n>(                                                \
        _mm512_cmp_##intrinsic_suffix##_mask(a.value, b.value, _CMP_GE_OQ));  \
  }                                                                           \
  inline Simd<bool, bool_n> operator==(Simd<type, N> a, Simd<type, N> b) {    \
    return Simd<bool, bool_n>(                                                \
        _mm512_cmp_##intrinsic_suffix##_mask(a.value, b.value, _CMP_EQ_OQ));  \
  }                                                                           \
  inline Simd<bool, bool_n> operator!=(Simd<type, N> a, Simd<type, N> b) {    \
    return Simd<bool, bool_n>(                                                \
        _mm512_cmp_##intrinsic_suffix##_mask(a.value, b.value, _CMP_NEQ_OQ)); \
  }

// Binary operation for AVX-512 int16_t using 256-bit AVX2 intrinsics
// Pattern: 512→256 cast, operate, 256→512 cast
#define DEFINE_AVX512_INT16_BINARY_OP(op, avx2_intrinsic) \
  inline Simd<int16_t, 16> operator op(                   \
      Simd<int16_t, 16> a, Simd<int16_t, 16> b) {         \
    __m256i a256 = _mm512_castsi512_si256(a.value);       \
    __m256i b256 = _mm512_castsi512_si256(b.value);       \
    __m256i result = avx2_intrinsic(a256, b256);          \
    return _mm512_castsi256_si512(result);                \
  }

// Binary operation for AVX-512 int8_t using 128-bit SSE intrinsics
// Pattern: 512→128 cast, operate, 128→512 cast
#define DEFINE_AVX512_INT8_BINARY_OP(op, sse_intrinsic) \
  inline Simd<int8_t, 16> operator op(                  \
      Simd<int8_t, 16> a, Simd<int8_t, 16> b) {         \
    __m128i a128 = _mm512_castsi512_si128(a.value);     \
    __m128i b128 = _mm512_castsi512_si128(b.value);     \
    __m128i result = sse_intrinsic(a128, b128);         \
    return _mm512_castsi128_si512(result);              \
  }

// Unary operation for AVX-512 int16_t using 256-bit AVX2 intrinsics
#define DEFINE_AVX512_INT16_UNARY_OP(name, avx2_intrinsic) \
  inline Simd<int16_t, 16> name(Simd<int16_t, 16> a) {     \
    __m256i a256 = _mm512_castsi512_si256(a.value);        \
    __m256i result = avx2_intrinsic(a256);                 \
    return _mm512_castsi256_si512(result);                 \
  }

// Unary operation for AVX-512 int8_t using 128-bit SSE intrinsics
#define DEFINE_AVX512_INT8_UNARY_OP(name, sse_intrinsic) \
  inline Simd<int8_t, 16> name(Simd<int8_t, 16> a) {     \
    __m128i a128 = _mm512_castsi512_si128(a.value);      \
    __m128i result = sse_intrinsic(a128);                \
    return _mm512_castsi128_si512(result);               \
  }

// Binary function for AVX-512 int16_t using 256-bit AVX2 intrinsics
#define DEFINE_AVX512_INT16_BINARY_FUNC(name, avx2_intrinsic)               \
  inline Simd<int16_t, 16> name(Simd<int16_t, 16> a, Simd<int16_t, 16> b) { \
    __m256i a256 = _mm512_castsi512_si256(a.value);                         \
    __m256i b256 = _mm512_castsi512_si256(b.value);                         \
    __m256i result = avx2_intrinsic(a256, b256);                            \
    return _mm512_castsi256_si512(result);                                  \
  }

// Binary function for AVX-512 int8_t using 128-bit SSE intrinsics
#define DEFINE_AVX512_INT8_BINARY_FUNC(name, sse_intrinsic)              \
  inline Simd<int8_t, 16> name(Simd<int8_t, 16> a, Simd<int8_t, 16> b) { \
    __m128i a128 = _mm512_castsi512_si128(a.value);                      \
    __m128i b128 = _mm512_castsi512_si128(b.value);                      \
    __m128i result = sse_intrinsic(a128, b128);                          \
    return _mm512_castsi128_si512(result);                               \
  }

// Binary operation with scalar fallback for AVX-512 int16_t
#define DEFINE_AVX512_INT16_SCALAR_FALLBACK(op)          \
  inline Simd<int16_t, 16> operator op(                  \
      Simd<int16_t, 16> a, Simd<int16_t, 16> b) {        \
    alignas(32) int16_t tmp_a[16], tmp_b[16], tmp_r[16]; \
    __m256i a256 = _mm512_castsi512_si256(a.value);      \
    __m256i b256 = _mm512_castsi512_si256(b.value);      \
    _mm256_store_si256((__m256i*)tmp_a, a256);           \
    _mm256_store_si256((__m256i*)tmp_b, b256);           \
    for (int i = 0; i < 16; ++i) {                       \
      tmp_r[i] = tmp_a[i] op tmp_b[i];                   \
    }                                                    \
    __m256i result = _mm256_load_si256((__m256i*)tmp_r); \
    return _mm512_castsi256_si512(result);               \
  }

// Binary operation with scalar fallback for AVX-512 int8_t
#define DEFINE_AVX512_INT8_SCALAR_FALLBACK(op)          \
  inline Simd<int8_t, 16> operator op(                  \
      Simd<int8_t, 16> a, Simd<int8_t, 16> b) {         \
    alignas(16) int8_t tmp_a[16], tmp_b[16], tmp_r[16]; \
    __m128i a128 = _mm512_castsi512_si128(a.value);     \
    __m128i b128 = _mm512_castsi512_si128(b.value);     \
    _mm_store_si128((__m128i*)tmp_a, a128);             \
    _mm_store_si128((__m128i*)tmp_b, b128);             \
    for (int i = 0; i < 16; ++i) {                      \
      tmp_r[i] = tmp_a[i] op tmp_b[i];                  \
    }                                                   \
    __m128i result = _mm_load_si128((__m128i*)tmp_r);   \
    return _mm512_castsi128_si512(result);              \
  }

// Unary operation with scalar fallback for AVX-512 int16_t
#define DEFINE_AVX512_INT16_UNARY_SCALAR(name, expr)     \
  inline Simd<int16_t, 16> name(Simd<int16_t, 16> a) {   \
    alignas(32) int16_t tmp_a[16], tmp_r[16];            \
    __m256i a256 = _mm512_castsi512_si256(a.value);      \
    _mm256_store_si256((__m256i*)tmp_a, a256);           \
    for (int i = 0; i < 16; ++i) {                       \
      tmp_r[i] = expr;                                   \
    }                                                    \
    __m256i result = _mm256_load_si256((__m256i*)tmp_r); \
    return _mm512_castsi256_si512(result);               \
  }

// Unary operation with scalar fallback for AVX-512 int8_t
#define DEFINE_AVX512_INT8_UNARY_SCALAR(name, expr)   \
  inline Simd<int8_t, 16> name(Simd<int8_t, 16> a) {  \
    alignas(16) int8_t tmp_a[16], tmp_r[16];          \
    __m128i a128 = _mm512_castsi512_si128(a.value);   \
    _mm_store_si128((__m128i*)tmp_a, a128);           \
    for (int i = 0; i < 16; ++i) {                    \
      tmp_r[i] = expr;                                \
    }                                                 \
    __m128i result = _mm_load_si128((__m128i*)tmp_r); \
    return _mm512_castsi128_si512(result);            \
  }

// Reduction for AVX-512 int16_t using scalar loop
#define DEFINE_AVX512_INT16_REDUCTION(name, init, op) \
  inline int16_t name(Simd<int16_t, 16> v) {          \
    alignas(32) int16_t tmp[16];                      \
    __m256i v256 = _mm512_castsi512_si256(v.value);   \
    _mm256_store_si256((__m256i*)tmp, v256);          \
    int16_t result = init;                            \
    for (int i = 0; i < 16; ++i) {                    \
      result = result op tmp[i];                      \
    }                                                 \
    return result;                                    \
  }

// Reduction for AVX-512 int8_t using scalar loop
#define DEFINE_AVX512_INT8_REDUCTION(name, init, op) \
  inline int8_t name(Simd<int8_t, 16> v) {           \
    alignas(16) int8_t tmp[16];                      \
    __m128i v128 = _mm512_castsi512_si128(v.value);  \
    _mm_store_si128((__m128i*)tmp, v128);            \
    int8_t result = init;                            \
    for (int i = 0; i < 16; ++i) {                   \
      result = result op tmp[i];                     \
    }                                                \
    return result;                                   \
  }

namespace mlx::core::simd {

// AVX-512: 512-bit SIMD for floats, doubles, and integers
template <>
inline constexpr int max_size<float> = 16;
template <>
inline constexpr int max_size<double> = 8;
template <>
inline constexpr int max_size<int32_t> = 16;
template <>
inline constexpr int max_size<uint32_t> = 16;
template <>
inline constexpr int max_size<uint8_t> = 16;

// ============================================================================
// Bool specializations using AVX-512 mask registers
// Note: AVX-512 has dedicated mask registers (k0-k7) but we use __mmask types
// ============================================================================

// Bool for 16 floats or 16 int32s (16-bit mask)
template <>
struct Simd<bool, 16> {
  static constexpr int size = 16;
  __mmask16 value;

  Simd() : value(0) {}
  Simd(__mmask16 v) : value(v) {}
  explicit Simd(bool v) : value(v ? 0xFFFF : 0) {}
  Simd(Simd<uint8_t, 16> v);
};

// Bool for 32 float16s or 32 bfloat16s (32-bit mask)
template <>
struct Simd<bool, 32> {
  static constexpr int size = 32;
  __mmask32 value;

  Simd() : value(0) {}
  Simd(__mmask32 v) : value(v) {}
  explicit Simd(bool v) : value(v ? 0xFFFFFFFF : 0) {}
};

// Bool for 8 doubles (8-bit mask)
template <>
struct Simd<bool, 8> {
  static constexpr int size = 8;
  __mmask8 value;

  Simd() : value(0) {}
  Simd(__mmask8 v) : value(v) {}
  explicit Simd(bool v) : value(v ? 0xFF : 0) {}
};

// Bool operators for Simd<bool, 32>
DEFINE_AVX512_BOOL_OPS_MASK(32, __mmask32, mask32)

// Bool operators for Simd<bool, 16>
DEFINE_AVX512_BOOL_OPS_MASK(16, __mmask16, mask16)

// Bool operators for Simd<bool, 8>
DEFINE_AVX512_BOOL_OPS_MASK(8, __mmask8, mask8)

// Bool load/store for 32 elements
template <>
inline Simd<bool, 32> load<bool, 32>(const bool* ptr) {
  __mmask32 mask = 0;
  for (int i = 0; i < 32; ++i) {
    if (ptr[i])
      mask |= (1u << i);
  }
  return Simd<bool, 32>(mask);
}

template <>
inline void store<bool, 32>(bool* ptr, Simd<bool, 32> v) {
  for (int i = 0; i < 32; ++i) {
    ptr[i] = (v.value & (1u << i)) != 0;
  }
}

// Bool load/store for 16 elements
template <>
inline Simd<bool, 16> load<bool, 16>(const bool* ptr) {
  __mmask16 mask = 0;
  for (int i = 0; i < 16; ++i) {
    if (ptr[i])
      mask |= (1 << i);
  }
  return Simd<bool, 16>(mask);
}

template <>
inline void store<bool, 16>(bool* ptr, Simd<bool, 16> v) {
  for (int i = 0; i < 16; ++i) {
    ptr[i] = (v.value & (1 << i)) != 0;
  }
}

// Bool load/store for 8 elements
template <>
inline Simd<bool, 8> load<bool, 8>(const bool* ptr) {
  __mmask8 mask = 0;
  for (int i = 0; i < 8; ++i) {
    if (ptr[i])
      mask |= (1 << i);
  }
  return Simd<bool, 8>(mask);
}

template <>
inline void store<bool, 8>(bool* ptr, Simd<bool, 8> v) {
  for (int i = 0; i < 8; ++i) {
    ptr[i] = (v.value & (1 << i)) != 0;
  }
}

// Bool reductions
inline bool all(Simd<bool, 16> x) {
  return x.value == 0xFFFF;
}

inline bool any(Simd<bool, 16> x) {
  return x.value != 0;
}

inline bool all(Simd<bool, 8> x) {
  return x.value == 0xFF;
}

inline bool any(Simd<bool, 8> x) {
  return x.value != 0;
}

// ============================================================================
// float32x16 (16 floats in 512 bits)
// ============================================================================

template <>
struct Simd<float, 16> {
  static constexpr int size = 16;
  __m512 value;

  Simd() : value(_mm512_setzero_ps()) {}
  Simd(__m512 v) : value(v) {}
  Simd(float v) : value(_mm512_set1_ps(v)) {}
  Simd(Simd<int32_t, 16> v);
  Simd(Simd<uint32_t, 16> v);
  Simd(Simd<bool, 16> v)
      : value(_mm512_castsi512_ps(_mm512_maskz_set1_epi32(v.value, -1))) {}

  float operator[](int idx) const {
    alignas(64) float tmp[16];
    _mm512_store_ps(tmp, value);
    return tmp[idx];
  }
};

// Arithmetic operators
DEFINE_X86_BINARY_OP(+, float, 16, _mm512_add_ps)
DEFINE_X86_BINARY_OP(-, float, 16, _mm512_sub_ps)
DEFINE_X86_BINARY_OP(*, float, 16, _mm512_mul_ps)
DEFINE_X86_BINARY_OP(/, float, 16, _mm512_div_ps)

inline Simd<float, 16> operator-(Simd<float, 16> a) {
  return _mm512_sub_ps(_mm512_setzero_ps(), a.value);
}

// Scalar operators
DEFINE_X86_ARITHMETIC_SCALAR_OVERLOADS(float, 16)

// Scalar operators with float16_t
inline Simd<float, 16> operator+(Simd<float, 16> a, float16_t b) {
  return a + static_cast<float>(b);
}

inline Simd<float, 16> operator+(float16_t a, Simd<float, 16> b) {
  return static_cast<float>(a) + b;
}

inline Simd<float, 16> operator-(Simd<float, 16> a, float16_t b) {
  return a - static_cast<float>(b);
}

inline Simd<float, 16> operator-(float16_t a, Simd<float, 16> b) {
  return static_cast<float>(a) - b;
}

inline Simd<float, 16> operator*(Simd<float, 16> a, float16_t b) {
  return a * static_cast<float>(b);
}

inline Simd<float, 16> operator*(float16_t a, Simd<float, 16> b) {
  return static_cast<float>(a) * b;
}

inline Simd<float, 16> operator/(Simd<float, 16> a, float16_t b) {
  return a / static_cast<float>(b);
}

inline Simd<float, 16> operator/(float16_t a, Simd<float, 16> b) {
  return static_cast<float>(a) / b;
}

// Comparison operators
DEFINE_AVX512_COMPARISONS_MASK(float, 16, 16, ps)

// Math operations
DEFINE_X86_UNARY_OP(abs, float, 16, _mm512_abs_ps)

inline Simd<bool, 16> isnan(Simd<float, 16> v) {
  return Simd<bool, 16>(_mm512_cmp_ps_mask(v.value, v.value, _CMP_UNORD_Q));
}

inline Simd<bool, 16> isinf(Simd<float, 16> v) {
  auto abs_v = abs(v);
  auto inf_v = Simd<float, 16>(std::numeric_limits<float>::infinity());
  return abs_v == inf_v;
}

// Select (needed by minimum/maximum)
inline Simd<float, 16>
select(Simd<bool, 16> mask, Simd<float, 16> a, Simd<float, 16> b) {
  return _mm512_mask_blend_ps(mask.value, b.value, a.value);
}

inline Simd<float, 16> minimum(Simd<float, 16> a, Simd<float, 16> b) {
  auto out = Simd<float, 16>(_mm512_min_ps(a.value, b.value));
  out = select(isnan(b), b, select(isnan(a), a, out));
  return out;
}

inline Simd<float, 16> maximum(Simd<float, 16> a, Simd<float, 16> b) {
  auto out = Simd<float, 16>(_mm512_max_ps(a.value, b.value));
  out = select(isnan(b), b, select(isnan(a), a, out));
  return out;
}

DEFINE_X86_UNARY_OP(sqrt, float, 16, _mm512_sqrt_ps)
DEFINE_X86_UNARY_OP(rsqrt, float, 16, _mm512_rsqrt14_ps)
DEFINE_X86_UNARY_OP(recip, float, 16, _mm512_rcp14_ps)

// FMA operations
inline Simd<float, 16>
fma(Simd<float, 16> a, Simd<float, 16> b, Simd<float, 16> c) {
  return _mm512_fmadd_ps(a.value, b.value, c.value);
}

template <typename U>
inline Simd<float, 16> fma(Simd<float, 16> a, Simd<float, 16> b, U c) {
  return fma(a, b, Simd<float, 16>(c));
}

// Load/store (must be defined before transcendental functions)
template <>
inline Simd<float, 16> load<float, 16>(const float* ptr) {
  return _mm512_loadu_ps(ptr);
}

template <>
inline void store<float, 16>(float* ptr, Simd<float, 16> v) {
  _mm512_storeu_ps(ptr, v.value);
}

// Transcendental functions (scalar fallback)
DEFINE_X86_TRANSCENDENTAL(exp, float, 16, 64, std::exp)
DEFINE_X86_TRANSCENDENTAL(log, float, 16, 64, std::log)
DEFINE_X86_TRANSCENDENTAL(sin, float, 16, 64, std::sin)
DEFINE_X86_TRANSCENDENTAL(cos, float, 16, 64, std::cos)

inline Simd<float, 16> atan2(Simd<float, 16> a, Simd<float, 16> b) {
  alignas(64) float tmp_a[16], tmp_b[16], tmp_r[16];
  store<float, 16>(tmp_a, a);
  store<float, 16>(tmp_b, b);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = std::atan2(tmp_a[i], tmp_b[i]);
  }
  return load<float, 16>(tmp_r);
}

// Additional transcendental functions (scalar fallback)
DEFINE_X86_TRANSCENDENTAL(acos, float, 16, 64, std::acos)
DEFINE_X86_TRANSCENDENTAL(acosh, float, 16, 64, std::acosh)
DEFINE_X86_TRANSCENDENTAL(asin, float, 16, 64, std::asin)
DEFINE_X86_TRANSCENDENTAL(asinh, float, 16, 64, std::asinh)
DEFINE_X86_TRANSCENDENTAL(atan, float, 16, 64, std::atan)
DEFINE_X86_TRANSCENDENTAL(atanh, float, 16, 64, std::atanh)
DEFINE_X86_TRANSCENDENTAL(ceil, float, 16, 64, std::ceil)
DEFINE_X86_TRANSCENDENTAL(cosh, float, 16, 64, std::cosh)
DEFINE_X86_TRANSCENDENTAL(expm1, float, 16, 64, std::expm1)
DEFINE_X86_TRANSCENDENTAL(floor, float, 16, 64, std::floor)
DEFINE_X86_TRANSCENDENTAL(log10, float, 16, 64, std::log10)
DEFINE_X86_TRANSCENDENTAL(sinh, float, 16, 64, std::sinh)
DEFINE_X86_TRANSCENDENTAL(tan, float, 16, 64, std::tan)
DEFINE_X86_TRANSCENDENTAL(tanh, float, 16, 64, std::tanh)
DEFINE_X86_TRANSCENDENTAL(log2, float, 16, 64, std::log2)
DEFINE_X86_TRANSCENDENTAL(rint, float, 16, 64, std::rint)

// Binary transcendental functions (scalar fallback)
inline Simd<float, 16> pow(Simd<float, 16> a, Simd<float, 16> b) {
  alignas(64) float tmp_a[16], tmp_b[16], tmp_r[16];
  store<float, 16>(tmp_a, a);
  store<float, 16>(tmp_b, b);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = std::pow(tmp_a[i], tmp_b[i]);
  }
  return load<float, 16>(tmp_r);
}

// Reductions
DEFINE_X86_REDUCTION(sum, float, 16, _mm512_reduce_add_ps)
DEFINE_X86_REDUCTION(prod, float, 16, _mm512_reduce_mul_ps)
DEFINE_X86_REDUCTION(min, float, 16, _mm512_reduce_min_ps)
DEFINE_X86_REDUCTION(max, float, 16, _mm512_reduce_max_ps)

// ============================================================================
// float64x8 (8 doubles in 512 bits)
// ============================================================================

template <>
struct Simd<double, 8> {
  static constexpr int size = 8;
  __m512d value;

  Simd() : value(_mm512_setzero_pd()) {}
  Simd(__m512d v) : value(v) {}
  Simd(double v) : value(_mm512_set1_pd(v)) {}
  Simd(Simd<bool, 8> v)
      : value(_mm512_castsi512_pd(_mm512_maskz_set1_epi64(v.value, -1LL))) {}

  double operator[](int idx) const {
    alignas(64) double tmp[8];
    _mm512_store_pd(tmp, value);
    return tmp[idx];
  }
};

// Arithmetic operators
DEFINE_X86_BINARY_OP(+, double, 8, _mm512_add_pd)
DEFINE_X86_BINARY_OP(-, double, 8, _mm512_sub_pd)
DEFINE_X86_BINARY_OP(*, double, 8, _mm512_mul_pd)
DEFINE_X86_BINARY_OP(/, double, 8, _mm512_div_pd)

inline Simd<double, 8> operator-(Simd<double, 8> a) {
  return _mm512_sub_pd(_mm512_setzero_pd(), a.value);
}

// Comparison operators
DEFINE_AVX512_COMPARISONS_MASK(double, 8, 8, pd)

// Math operations
DEFINE_X86_UNARY_OP(abs, double, 8, _mm512_abs_pd)

inline Simd<bool, 8> isnan(Simd<double, 8> v) {
  return Simd<bool, 8>(_mm512_cmp_pd_mask(v.value, v.value, _CMP_UNORD_Q));
}

inline Simd<bool, 8> isinf(Simd<double, 8> v) {
  auto abs_v = abs(v);
  auto inf_v = Simd<double, 8>(std::numeric_limits<double>::infinity());
  return abs_v == inf_v;
}

// Select (needed by minimum/maximum)
inline Simd<double, 8>
select(Simd<bool, 8> mask, Simd<double, 8> a, Simd<double, 8> b) {
  return _mm512_mask_blend_pd(mask.value, b.value, a.value);
}

inline Simd<double, 8> minimum(Simd<double, 8> a, Simd<double, 8> b) {
  auto out = Simd<double, 8>(_mm512_min_pd(a.value, b.value));
  out = select(isnan(b), b, select(isnan(a), a, out));
  return out;
}

inline Simd<double, 8> maximum(Simd<double, 8> a, Simd<double, 8> b) {
  auto out = Simd<double, 8>(_mm512_max_pd(a.value, b.value));
  out = select(isnan(b), b, select(isnan(a), a, out));
  return out;
}

DEFINE_X86_UNARY_OP(sqrt, double, 8, _mm512_sqrt_pd)
DEFINE_X86_UNARY_OP(rsqrt, double, 8, _mm512_rsqrt14_pd)
DEFINE_X86_UNARY_OP(recip, double, 8, _mm512_rcp14_pd)

// FMA operations
inline Simd<double, 8>
fma(Simd<double, 8> a, Simd<double, 8> b, Simd<double, 8> c) {
  return _mm512_fmadd_pd(a.value, b.value, c.value);
}

template <typename U>
inline Simd<double, 8> fma(Simd<double, 8> a, Simd<double, 8> b, U c) {
  return fma(a, b, Simd<double, 8>(c));
}

// Load/store (must be defined before transcendental functions)
template <>
inline Simd<double, 8> load<double, 8>(const double* ptr) {
  return _mm512_loadu_pd(ptr);
}

template <>
inline void store<double, 8>(double* ptr, Simd<double, 8> v) {
  _mm512_storeu_pd(ptr, v.value);
}

// Transcendental functions (scalar fallback)
DEFINE_X86_TRANSCENDENTAL(exp, double, 8, 64, std::exp)
DEFINE_X86_TRANSCENDENTAL(log, double, 8, 64, std::log)
DEFINE_X86_TRANSCENDENTAL(sin, double, 8, 64, std::sin)
DEFINE_X86_TRANSCENDENTAL(cos, double, 8, 64, std::cos)

inline Simd<double, 8> atan2(Simd<double, 8> a, Simd<double, 8> b) {
  alignas(64) double tmp_a[8], tmp_b[8], tmp_r[8];
  store<double, 8>(tmp_a, a);
  store<double, 8>(tmp_b, b);
  for (int i = 0; i < 8; ++i) {
    tmp_r[i] = std::atan2(tmp_a[i], tmp_b[i]);
  }
  return load<double, 8>(tmp_r);
}

// Additional transcendental functions (scalar fallback)
DEFINE_X86_TRANSCENDENTAL(acos, double, 8, 64, std::acos)
DEFINE_X86_TRANSCENDENTAL(acosh, double, 8, 64, std::acosh)
DEFINE_X86_TRANSCENDENTAL(asin, double, 8, 64, std::asin)
DEFINE_X86_TRANSCENDENTAL(asinh, double, 8, 64, std::asinh)
DEFINE_X86_TRANSCENDENTAL(atan, double, 8, 64, std::atan)
DEFINE_X86_TRANSCENDENTAL(atanh, double, 8, 64, std::atanh)
DEFINE_X86_TRANSCENDENTAL(ceil, double, 8, 64, std::ceil)
DEFINE_X86_TRANSCENDENTAL(cosh, double, 8, 64, std::cosh)
DEFINE_X86_TRANSCENDENTAL(expm1, double, 8, 64, std::expm1)
DEFINE_X86_TRANSCENDENTAL(floor, double, 8, 64, std::floor)
DEFINE_X86_TRANSCENDENTAL(log10, double, 8, 64, std::log10)
DEFINE_X86_TRANSCENDENTAL(sinh, double, 8, 64, std::sinh)
DEFINE_X86_TRANSCENDENTAL(tan, double, 8, 64, std::tan)
DEFINE_X86_TRANSCENDENTAL(tanh, double, 8, 64, std::tanh)
DEFINE_X86_TRANSCENDENTAL(log2, double, 8, 64, std::log2)
DEFINE_X86_TRANSCENDENTAL(rint, double, 8, 64, std::rint)

// Binary transcendental functions (scalar fallback)
inline Simd<double, 8> pow(Simd<double, 8> a, Simd<double, 8> b) {
  alignas(64) double tmp_a[8], tmp_b[8], tmp_r[8];
  store<double, 8>(tmp_a, a);
  store<double, 8>(tmp_b, b);
  for (int i = 0; i < 8; ++i) {
    tmp_r[i] = std::pow(tmp_a[i], tmp_b[i]);
  }
  return load<double, 8>(tmp_r);
}

// Forward declarations for integer pow (will be defined after int32/uint32
// load/store)
inline Simd<int32_t, 16> pow(Simd<int32_t, 16> a, Simd<int32_t, 16> b);
inline Simd<uint32_t, 16> pow(Simd<uint32_t, 16> a, Simd<uint32_t, 16> b);

// Reductions
DEFINE_X86_REDUCTION(sum, double, 8, _mm512_reduce_add_pd)
DEFINE_X86_REDUCTION(prod, double, 8, _mm512_reduce_mul_pd)
DEFINE_X86_REDUCTION(min, double, 8, _mm512_reduce_min_pd)
DEFINE_X86_REDUCTION(max, double, 8, _mm512_reduce_max_pd)

// ============================================================================
// int32x16 and uint32x16 (16 integers in 512 bits)
// ============================================================================

template <>
struct Simd<int32_t, 16> {
  static constexpr int size = 16;
  __m512i value;

  Simd() : value(_mm512_setzero_si512()) {}
  Simd(__m512i v) : value(v) {}
  Simd(int32_t v) : value(_mm512_set1_epi32(v)) {}
  Simd(Simd<bool, 16> v) : value(_mm512_maskz_set1_epi32(v.value, -1)) {}
  Simd(Simd<uint32_t, 16> v); // Defined after uint32_t struct
  Simd(Simd<uint16_t, 16> v); // Defined after uint16_t struct
  Simd(Simd<uint8_t, 16> v); // Defined after uint8_t struct
  Simd(Simd<int16_t, 16> v); // Defined after int16_t struct
  Simd(Simd<int8_t, 16> v); // Defined after int8_t struct
  Simd(Simd<float, 16> v); // Defined after float struct

  int32_t operator[](int idx) const {
    alignas(64) int32_t tmp[16];
    _mm512_store_si512((__m512i*)tmp, value);
    return tmp[idx];
  }
};

template <>
struct Simd<uint32_t, 16> {
  static constexpr int size = 16;
  __m512i value;

  Simd() : value(_mm512_setzero_si512()) {}
  Simd(__m512i v) : value(v) {}
  Simd(uint32_t v) : value(_mm512_set1_epi32(v)) {}
  Simd(Simd<bool, 16> v) : value(_mm512_maskz_set1_epi32(v.value, -1)) {}
  Simd(Simd<float, 16> v); // Defined after float struct

  uint32_t operator[](int idx) const {
    alignas(64) uint32_t tmp[16];
    _mm512_store_si512((__m512i*)tmp, value);
    return tmp[idx];
  }
};

// Load/store for int32_t and uint32_t (must come before division operators)
template <>
inline Simd<int32_t, 16> load<int32_t, 16>(const int32_t* ptr) {
  return _mm512_loadu_si512((__m512i*)ptr);
}

template <>
inline void store<int32_t, 16>(int32_t* ptr, Simd<int32_t, 16> v) {
  _mm512_storeu_si512((__m512i*)ptr, v.value);
}

template <>
inline Simd<uint32_t, 16> load<uint32_t, 16>(const uint32_t* ptr) {
  return _mm512_loadu_si512((__m512i*)ptr);
}

template <>
inline void store<uint32_t, 16>(uint32_t* ptr, Simd<uint32_t, 16> v) {
  _mm512_storeu_si512((__m512i*)ptr, v.value);
}

// Arithmetic operators for int32_t
DEFINE_X86_BINARY_OP(+, int32_t, 16, _mm512_add_epi32)
DEFINE_X86_BINARY_OP(-, int32_t, 16, _mm512_sub_epi32)
DEFINE_X86_BINARY_OP(*, int32_t, 16, _mm512_mullo_epi32)

inline Simd<int32_t, 16> operator/(Simd<int32_t, 16> a, Simd<int32_t, 16> b) {
  // No AVX-512 integer division, use scalar fallback
  alignas(64) int32_t tmp_a[16], tmp_b[16], tmp_r[16];
  store<int32_t, 16>(tmp_a, a);
  store<int32_t, 16>(tmp_b, b);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = tmp_a[i] / tmp_b[i];
  }
  return load<int32_t, 16>(tmp_r);
}

inline Simd<int32_t, 16> operator-(Simd<int32_t, 16> a) {
  return _mm512_sub_epi32(_mm512_setzero_si512(), a.value);
}

// Scalar operators for int32_t
DEFINE_X86_SCALAR_BINARY_OVERLOADS(+, int32_t, 16)
DEFINE_X86_SCALAR_BINARY_OVERLOADS(-, int32_t, 16)
DEFINE_X86_SCALAR_BINARY_OVERLOADS(*, int32_t, 16)

// Arithmetic operators for uint32_t
DEFINE_X86_BINARY_OP(+, uint32_t, 16, _mm512_add_epi32)
DEFINE_X86_BINARY_OP(-, uint32_t, 16, _mm512_sub_epi32)
DEFINE_X86_BINARY_OP(*, uint32_t, 16, _mm512_mullo_epi32)

inline Simd<uint32_t, 16> operator/(
    Simd<uint32_t, 16> a,
    Simd<uint32_t, 16> b) {
  // No AVX-512 integer division, use scalar fallback
  alignas(64) uint32_t tmp_a[16], tmp_b[16], tmp_r[16];
  store<uint32_t, 16>(tmp_a, a);
  store<uint32_t, 16>(tmp_b, b);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = tmp_a[i] / tmp_b[i];
  }
  return load<uint32_t, 16>(tmp_r);
}

// Scalar operators for uint32_t
DEFINE_X86_SCALAR_BINARY_OVERLOADS(+, uint32_t, 16)
DEFINE_X86_SCALAR_BINARY_OVERLOADS(-, uint32_t, 16)
DEFINE_X86_SCALAR_BINARY_OVERLOADS(*, uint32_t, 16)

// Bitwise operators for int32_t
DEFINE_X86_BINARY_OP(&, int32_t, 16, _mm512_and_si512)
DEFINE_X86_BINARY_OP(|, int32_t, 16, _mm512_or_si512)
DEFINE_X86_BINARY_OP(^, int32_t, 16, _mm512_xor_si512)

inline Simd<int32_t, 16> operator~(Simd<int32_t, 16> a) {
  return _mm512_xor_si512(a.value, _mm512_set1_epi32(-1));
}

// Bitwise operators for uint32_t
DEFINE_X86_BINARY_OP(&, uint32_t, 16, _mm512_and_si512)
inline Simd<uint32_t, 16> operator&(Simd<uint32_t, 16> a, uint32_t b) {
  return a & Simd<uint32_t, 16>(b);
}
DEFINE_X86_BINARY_OP(|, uint32_t, 16, _mm512_or_si512)
DEFINE_X86_BINARY_OP(^, uint32_t, 16, _mm512_xor_si512)

inline Simd<uint32_t, 16> operator~(Simd<uint32_t, 16> a) {
  return _mm512_xor_si512(a.value, _mm512_set1_epi32(-1));
}

// Shift operators for uint32_t
inline Simd<uint32_t, 16> operator<<(Simd<uint32_t, 16> a, int shift) {
  return _mm512_slli_epi32(a.value, shift);
}

inline Simd<uint32_t, 16> operator>>(Simd<uint32_t, 16> a, int shift) {
  return _mm512_srli_epi32(a.value, shift);
}

inline Simd<uint32_t, 16> operator<<(
    Simd<uint32_t, 16> a,
    Simd<uint32_t, 16> b) {
  return _mm512_sllv_epi32(a.value, b.value);
}

inline Simd<uint32_t, 16> operator>>(
    Simd<uint32_t, 16> a,
    Simd<uint32_t, 16> b) {
  return _mm512_srlv_epi32(a.value, b.value);
}

// Shift operators for int32_t
inline Simd<int32_t, 16> operator<<(Simd<int32_t, 16> a, int shift) {
  return _mm512_slli_epi32(a.value, shift);
}

inline Simd<int32_t, 16> operator>>(Simd<int32_t, 16> a, int shift) {
  return _mm512_srai_epi32(a.value, shift);
}

inline Simd<int32_t, 16> operator<<(Simd<int32_t, 16> a, Simd<int32_t, 16> b) {
  return _mm512_sllv_epi32(a.value, b.value);
}

inline Simd<int32_t, 16> operator>>(Simd<int32_t, 16> a, Simd<int32_t, 16> b) {
  return _mm512_srav_epi32(a.value, b.value);
}

// Comparison operators for int32_t
inline Simd<bool, 16> operator<(Simd<int32_t, 16> a, Simd<int32_t, 16> b) {
  return Simd<bool, 16>(_mm512_cmplt_epi32_mask(a.value, b.value));
}

inline Simd<bool, 16> operator<=(Simd<int32_t, 16> a, Simd<int32_t, 16> b) {
  return Simd<bool, 16>(_mm512_cmple_epi32_mask(a.value, b.value));
}

inline Simd<bool, 16> operator>(Simd<int32_t, 16> a, Simd<int32_t, 16> b) {
  return Simd<bool, 16>(_mm512_cmpgt_epi32_mask(a.value, b.value));
}

inline Simd<bool, 16> operator>=(Simd<int32_t, 16> a, Simd<int32_t, 16> b) {
  return Simd<bool, 16>(_mm512_cmpge_epi32_mask(a.value, b.value));
}

inline Simd<bool, 16> operator==(Simd<int32_t, 16> a, Simd<int32_t, 16> b) {
  return Simd<bool, 16>(_mm512_cmpeq_epi32_mask(a.value, b.value));
}

inline Simd<bool, 16> operator!=(Simd<int32_t, 16> a, Simd<int32_t, 16> b) {
  return Simd<bool, 16>(_mm512_cmpneq_epi32_mask(a.value, b.value));
}

// Comparison operators for uint32_t
inline Simd<bool, 16> operator<(Simd<uint32_t, 16> a, Simd<uint32_t, 16> b) {
  return Simd<bool, 16>(_mm512_cmplt_epu32_mask(a.value, b.value));
}

inline Simd<bool, 16> operator<=(Simd<uint32_t, 16> a, Simd<uint32_t, 16> b) {
  return Simd<bool, 16>(_mm512_cmple_epu32_mask(a.value, b.value));
}

inline Simd<bool, 16> operator>(Simd<uint32_t, 16> a, Simd<uint32_t, 16> b) {
  return Simd<bool, 16>(_mm512_cmpgt_epu32_mask(a.value, b.value));
}

inline Simd<bool, 16> operator>=(Simd<uint32_t, 16> a, Simd<uint32_t, 16> b) {
  return Simd<bool, 16>(_mm512_cmpge_epu32_mask(a.value, b.value));
}

inline Simd<bool, 16> operator==(Simd<uint32_t, 16> a, Simd<uint32_t, 16> b) {
  return Simd<bool, 16>(_mm512_cmpeq_epi32_mask(a.value, b.value));
}

inline Simd<bool, 16> operator!=(Simd<uint32_t, 16> a, Simd<uint32_t, 16> b) {
  return Simd<bool, 16>(_mm512_cmpneq_epi32_mask(a.value, b.value));
}

// Math operations for int32_t
DEFINE_X86_UNARY_OP(abs, int32_t, 16, _mm512_abs_epi32)

DEFINE_X86_BINARY_FUNC(minimum, int32_t, 16, _mm512_min_epi32)
DEFINE_X86_BINARY_FUNC(maximum, int32_t, 16, _mm512_max_epi32)

// Math operations for uint32_t
DEFINE_X86_BINARY_FUNC(minimum, uint32_t, 16, _mm512_min_epu32)
DEFINE_X86_BINARY_FUNC(maximum, uint32_t, 16, _mm512_max_epu32)

// Select
inline Simd<int32_t, 16>
select(Simd<bool, 16> mask, Simd<int32_t, 16> a, Simd<int32_t, 16> b) {
  return _mm512_mask_blend_epi32(mask.value, b.value, a.value);
}

inline Simd<uint32_t, 16>
select(Simd<bool, 16> mask, Simd<uint32_t, 16> a, Simd<uint32_t, 16> b) {
  return _mm512_mask_blend_epi32(mask.value, b.value, a.value);
}

// Reductions for int32_t
DEFINE_X86_REDUCTION(sum, int32_t, 16, _mm512_reduce_add_epi32)
DEFINE_X86_REDUCTION_SCALAR(prod, int32_t, 16, 64, 1, *)
DEFINE_X86_REDUCTION(min, int32_t, 16, _mm512_reduce_min_epi32)
DEFINE_X86_REDUCTION(max, int32_t, 16, _mm512_reduce_max_epi32)

// Reductions for uint32_t
DEFINE_X86_REDUCTION(sum, uint32_t, 16, _mm512_reduce_add_epi32)
DEFINE_X86_REDUCTION_SCALAR(prod, uint32_t, 16, 64, 1, *)
DEFINE_X86_REDUCTION(min, uint32_t, 16, _mm512_reduce_min_epu32)
DEFINE_X86_REDUCTION(max, uint32_t, 16, _mm512_reduce_max_epu32)

// Integer pow implementations (using scalar fallback)
inline Simd<int32_t, 16> pow(Simd<int32_t, 16> a, Simd<int32_t, 16> b) {
  alignas(64) int32_t tmp_a[16], tmp_b[16], tmp_r[16];
  store<int32_t, 16>(tmp_a, a);
  store<int32_t, 16>(tmp_b, b);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = static_cast<int32_t>(std::pow(tmp_a[i], tmp_b[i]));
  }
  return load<int32_t, 16>(tmp_r);
}

inline Simd<uint32_t, 16> pow(Simd<uint32_t, 16> a, Simd<uint32_t, 16> b) {
  alignas(64) uint32_t tmp_a[16], tmp_b[16], tmp_r[16];
  store<uint32_t, 16>(tmp_a, a);
  store<uint32_t, 16>(tmp_b, b);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = static_cast<uint32_t>(std::pow(tmp_a[i], tmp_b[i]));
  }
  return load<uint32_t, 16>(tmp_r);
}

// ============================================================================
// float16x16 (16 float16s in 256 bits)
// ============================================================================

template <>
inline constexpr int max_size<float16_t> = 16;

template <>
struct Simd<float16_t, 16> {
  static constexpr int size = 16;
  __m512i value; // Use 512-bit register but only lower 256 bits are used

  Simd() : value(_mm512_setzero_si512()) {}
  Simd(__m512i v) : value(v) {}
  Simd(float16_t v) {
    alignas(32) float16_t tmp[16];
    for (int i = 0; i < 16; ++i)
      tmp[i] = v;
    value = _mm512_castsi256_si512(_mm256_load_si256((__m256i*)tmp));
  }
  // Generic numeric constructor (for float, double, int, etc)
  template <
      typename U,
      typename = std::enable_if_t<
          std::is_arithmetic_v<U> && !std::is_same_v<U, float16_t>>>
  Simd(U v) : Simd(static_cast<float16_t>(v)) {}
  Simd(Simd<float, 16> v); // Conversion from float, defined later

  // Conversion to float
  operator Simd<float, 16>() const {
    alignas(32) float16_t tmp_in[16];
    alignas(64) float tmp_out[16];
    __m256i v256 = _mm512_castsi512_si256(value);
    _mm256_store_si256((__m256i*)tmp_in, v256);
    for (int i = 0; i < 16; ++i) {
      tmp_out[i] = static_cast<float>(tmp_in[i]);
    }
    return load<float, 16>(tmp_out);
  }

  float16_t operator[](int idx) const {
    alignas(32) float16_t tmp[16];
    __m256i v256 = _mm512_castsi512_si256(value);
    _mm256_store_si256((__m256i*)tmp, v256);
    return tmp[idx];
  }
};

// Minimal float16 binary operations (scalar fallback)
inline Simd<float16_t, 16> operator+(
    Simd<float16_t, 16> a,
    Simd<float16_t, 16> b) {
  alignas(32) float16_t tmp_a[16], tmp_b[16], tmp_r[16];
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  _mm256_store_si256((__m256i*)tmp_a, a256);
  _mm256_store_si256((__m256i*)tmp_b, b256);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = static_cast<float16_t>(
        static_cast<float>(tmp_a[i]) + static_cast<float>(tmp_b[i]));
  }
  __m256i result = _mm256_load_si256((__m256i*)tmp_r);
  Simd<float16_t, 16> out;
  out.value = _mm512_castsi256_si512(result);
  return out;
}

// Mixed float/float16 operations - convert to float
inline Simd<float, 16> operator+(Simd<float, 16> a, Simd<float16_t, 16> b) {
  return a + Simd<float, 16>(b);
}

inline Simd<float, 16> operator+(Simd<float16_t, 16> a, Simd<float, 16> b) {
  return Simd<float, 16>(a) + b;
}

inline Simd<float, 16> operator-(Simd<float, 16> a, Simd<float16_t, 16> b) {
  return a - Simd<float, 16>(b);
}

inline Simd<float, 16> operator-(Simd<float16_t, 16> a, Simd<float, 16> b) {
  return Simd<float, 16>(a) - b;
}

inline Simd<float16_t, 16> operator-(
    Simd<float16_t, 16> a,
    Simd<float16_t, 16> b) {
  alignas(32) float16_t tmp_a[16], tmp_b[16], tmp_r[16];
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  _mm256_store_si256((__m256i*)tmp_a, a256);
  _mm256_store_si256((__m256i*)tmp_b, b256);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = static_cast<float16_t>(
        static_cast<float>(tmp_a[i]) - static_cast<float>(tmp_b[i]));
  }
  __m256i result = _mm256_load_si256((__m256i*)tmp_r);
  Simd<float16_t, 16> out;
  out.value = _mm512_castsi256_si512(result);
  return out;
}

inline Simd<float16_t, 16> pow(Simd<float16_t, 16> a, Simd<float16_t, 16> b) {
  alignas(32) float16_t tmp_a[16], tmp_b[16], tmp_r[16];
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  _mm256_store_si256((__m256i*)tmp_a, a256);
  _mm256_store_si256((__m256i*)tmp_b, b256);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = static_cast<float16_t>(
        std::pow(static_cast<float>(tmp_a[i]), static_cast<float>(tmp_b[i])));
  }
  __m256i result = _mm256_load_si256((__m256i*)tmp_r);
  Simd<float16_t, 16> out;
  out.value = _mm512_castsi256_si512(result);
  return out;
}

inline Simd<float16_t, 16> operator*(
    Simd<float16_t, 16> a,
    Simd<float16_t, 16> b) {
  alignas(32) float16_t tmp_a[16], tmp_b[16], tmp_r[16];
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  _mm256_store_si256((__m256i*)tmp_a, a256);
  _mm256_store_si256((__m256i*)tmp_b, b256);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = static_cast<float16_t>(
        static_cast<float>(tmp_a[i]) * static_cast<float>(tmp_b[i]));
  }
  __m256i result = _mm256_load_si256((__m256i*)tmp_r);
  Simd<float16_t, 16> out;
  out.value = _mm512_castsi256_si512(result);
  return out;
}

inline Simd<float16_t, 16> operator*(Simd<float16_t, 16> a, float16_t b) {
  return a * Simd<float16_t, 16>(b);
}

inline Simd<float16_t, 16> operator*(float16_t a, Simd<float16_t, 16> b) {
  return Simd<float16_t, 16>(a) * b;
}

inline Simd<float16_t, 16> operator*(Simd<float16_t, 16> a, float b) {
  return a * Simd<float16_t, 16>(static_cast<float16_t>(b));
}

inline Simd<float16_t, 16> operator*(float a, Simd<float16_t, 16> b) {
  return Simd<float16_t, 16>(static_cast<float16_t>(a)) * b;
}

inline Simd<float16_t, 16> operator*(Simd<float16_t, 16> a, double b) {
  return a * Simd<float16_t, 16>(static_cast<float16_t>(b));
}

inline Simd<float16_t, 16> operator*(double a, Simd<float16_t, 16> b) {
  return Simd<float16_t, 16>(static_cast<float16_t>(a)) * b;
}

inline Simd<float16_t, 16> operator+(Simd<float16_t, 16> a, float16_t b) {
  return a + Simd<float16_t, 16>(static_cast<float16_t>(b));
}

inline Simd<float16_t, 16> operator+(float16_t a, Simd<float16_t, 16> b) {
  return Simd<float16_t, 16>(static_cast<float16_t>(a)) + b;
}

inline Simd<float16_t, 16> operator+(float a, Simd<float16_t, 16> b) {
  return Simd<float16_t, 16>(static_cast<float16_t>(a)) + b;
}

inline Simd<float16_t, 16> operator-(Simd<float16_t, 16> a, float16_t b) {
  return a - Simd<float16_t, 16>(static_cast<float16_t>(b));
}

inline Simd<float16_t, 16> operator-(float16_t a, Simd<float16_t, 16> b) {
  return Simd<float16_t, 16>(static_cast<float16_t>(a)) - b;
}

inline Simd<float16_t, 16> minimum(
    Simd<float16_t, 16> a,
    Simd<float16_t, 16> b) {
  alignas(32) float16_t tmp_a[16], tmp_b[16], tmp_r[16];
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  _mm256_store_si256((__m256i*)tmp_a, a256);
  _mm256_store_si256((__m256i*)tmp_b, b256);
  for (int i = 0; i < 16; ++i) {
    float fa = static_cast<float>(tmp_a[i]);
    float fb = static_cast<float>(tmp_b[i]);
    // NaN propagation: if either is NaN, use that value
    if (std::isnan(fb)) {
      tmp_r[i] = tmp_b[i];
    } else if (std::isnan(fa)) {
      tmp_r[i] = tmp_a[i];
    } else {
      tmp_r[i] = static_cast<float16_t>(fa < fb ? fa : fb);
    }
  }
  __m256i result = _mm256_load_si256((__m256i*)tmp_r);
  Simd<float16_t, 16> out;
  out.value = _mm512_castsi256_si512(result);
  return out;
}

inline Simd<bool, 16> isnan(Simd<float16_t, 16> v) {
  return v != v;
}

inline Simd<float16_t, 16> maximum(
    Simd<float16_t, 16> a,
    Simd<float16_t, 16> b) {
  alignas(32) float16_t tmp_a[16], tmp_b[16], tmp_r[16];
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  _mm256_store_si256((__m256i*)tmp_a, a256);
  _mm256_store_si256((__m256i*)tmp_b, b256);
  for (int i = 0; i < 16; ++i) {
    float fa = static_cast<float>(tmp_a[i]);
    float fb = static_cast<float>(tmp_b[i]);
    // NaN propagation: if either is NaN, use that value
    if (std::isnan(fb)) {
      tmp_r[i] = tmp_b[i];
    } else if (std::isnan(fa)) {
      tmp_r[i] = tmp_a[i];
    } else {
      tmp_r[i] = static_cast<float16_t>(fa > fb ? fa : fb);
    }
  }
  __m256i result = _mm256_load_si256((__m256i*)tmp_r);
  Simd<float16_t, 16> out;
  out.value = _mm512_castsi256_si512(result);
  return out;
}

inline Simd<float16_t, 16> atan2(Simd<float16_t, 16> a, Simd<float16_t, 16> b) {
  alignas(32) float16_t tmp_a[16], tmp_b[16], tmp_r[16];
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  _mm256_store_si256((__m256i*)tmp_a, a256);
  _mm256_store_si256((__m256i*)tmp_b, b256);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = static_cast<float16_t>(
        std::atan2(static_cast<float>(tmp_a[i]), static_cast<float>(tmp_b[i])));
  }
  __m256i result = _mm256_load_si256((__m256i*)tmp_r);
  Simd<float16_t, 16> out;
  out.value = _mm512_castsi256_si512(result);
  return out;
};

inline Simd<float16_t, 16> abs(Simd<float16_t, 16> v) {
  // Clear the sign bit (bit 15) for absolute value
  __m256i v256 = _mm512_castsi512_si256(v.value);
  __m256i mask = _mm256_set1_epi16(0x7FFF);
  __m256i result = _mm256_and_si256(v256, mask);
  return Simd<float16_t, 16>(_mm512_castsi256_si512(result));
}

// Load/store for float16_t (16 elements stored in lower 256 bits)
template <>
inline Simd<float16_t, 16> load<float16_t, 16>(const float16_t* ptr) {
  __m256i v256 = _mm256_loadu_si256((const __m256i*)ptr);
  return _mm512_castsi256_si512(v256);
}

template <>
inline void store<float16_t, 16>(float16_t* ptr, Simd<float16_t, 16> v) {
  __m256i v256 = _mm512_castsi512_si256(v.value);
  _mm256_storeu_si256((__m256i*)ptr, v256);
}

// Comparison operators for float16_t (using scalar fallback)
inline Simd<bool, 16> operator<(Simd<float16_t, 16> a, Simd<float16_t, 16> b) {
  alignas(32) float16_t tmp_a[16], tmp_b[16];
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  _mm256_store_si256((__m256i*)tmp_a, a256);
  _mm256_store_si256((__m256i*)tmp_b, b256);
  alignas(64) int32_t result[16];
  for (int i = 0; i < 16; ++i) {
    result[i] =
        static_cast<float>(tmp_a[i]) < static_cast<float>(tmp_b[i]) ? -1 : 0;
  }
  return load<bool, 16>((const bool*)result);
}

inline Simd<bool, 16> operator<=(Simd<float16_t, 16> a, Simd<float16_t, 16> b) {
  alignas(32) float16_t tmp_a[16], tmp_b[16];
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  _mm256_store_si256((__m256i*)tmp_a, a256);
  _mm256_store_si256((__m256i*)tmp_b, b256);
  alignas(64) int32_t result[16];
  for (int i = 0; i < 16; ++i) {
    result[i] =
        static_cast<float>(tmp_a[i]) <= static_cast<float>(tmp_b[i]) ? -1 : 0;
  }
  return load<bool, 16>((const bool*)result);
}

inline Simd<bool, 16> operator>(Simd<float16_t, 16> a, Simd<float16_t, 16> b) {
  return b < a;
}

inline Simd<bool, 16> operator>=(Simd<float16_t, 16> a, Simd<float16_t, 16> b) {
  return b <= a;
}

inline Simd<bool, 16> operator==(Simd<float16_t, 16> a, Simd<float16_t, 16> b) {
  alignas(32) float16_t tmp_a[16], tmp_b[16];
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  _mm256_store_si256((__m256i*)tmp_a, a256);
  _mm256_store_si256((__m256i*)tmp_b, b256);
  alignas(64) int32_t result[16];
  for (int i = 0; i < 16; ++i) {
    result[i] =
        static_cast<float>(tmp_a[i]) == static_cast<float>(tmp_b[i]) ? -1 : 0;
  }
  return load<bool, 16>((const bool*)result);
}

inline Simd<bool, 16> operator!=(Simd<float16_t, 16> a, Simd<float16_t, 16> b) {
  return !(a == b);
}

// ============================================================================
// uint16x16 (16 uint16s in 256 bits - reusing AVX portion of __m512i)
// ============================================================================

template <>
inline constexpr int max_size<uint16_t> = 16;

template <>
struct Simd<uint16_t, 16> {
  static constexpr int size = 16;
  __m512i value; // Use 512-bit register but only lower 256 bits are used

  Simd() : value(_mm512_setzero_si512()) {}
  Simd(__m512i v) : value(v) {}
  Simd(uint16_t v) {
    __m256i v256 = _mm256_set1_epi16(v);
    value = _mm512_castsi256_si512(v256);
  }
  Simd(Simd<uint8_t, 16> v); // Defined after uint8_t struct
  Simd(Simd<int32_t, 16> v); // Defined after int32_t operators

  uint16_t operator[](int idx) const {
    alignas(64) uint16_t tmp[32];
    _mm512_store_si512((__m512i*)tmp, value);
    return tmp[idx];
  }
};

// Load/store for uint16_t
template <>
inline Simd<uint16_t, 16> load<uint16_t, 16>(const uint16_t* ptr) {
  __m256i v256 = _mm256_loadu_si256((const __m256i*)ptr);
  return _mm512_castsi256_si512(v256);
}

template <>
inline void store<uint16_t, 16>(uint16_t* ptr, Simd<uint16_t, 16> v) {
  __m256i v256 = _mm512_castsi512_si256(v.value);
  _mm256_storeu_si256((__m256i*)ptr, v256);
}

// uint16_t operators
inline Simd<uint16_t, 16> operator&(
    Simd<uint16_t, 16> a,
    Simd<uint16_t, 16> b) {
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  __m256i result = _mm256_and_si256(a256, b256);
  return _mm512_castsi256_si512(result);
}

inline Simd<uint16_t, 16> operator&(Simd<uint16_t, 16> a, uint16_t b) {
  return a & Simd<uint16_t, 16>(b);
}

inline Simd<uint16_t, 16> operator|(
    Simd<uint16_t, 16> a,
    Simd<uint16_t, 16> b) {
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  __m256i result = _mm256_or_si256(a256, b256);
  return _mm512_castsi256_si512(result);
}

inline Simd<uint16_t, 16> operator^(
    Simd<uint16_t, 16> a,
    Simd<uint16_t, 16> b) {
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  __m256i result = _mm256_xor_si256(a256, b256);
  return _mm512_castsi256_si512(result);
}

inline Simd<uint16_t, 16> operator~(Simd<uint16_t, 16> a) {
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i result = _mm256_xor_si256(a256, _mm256_set1_epi16(-1));
  return _mm512_castsi256_si512(result);
}

inline Simd<uint16_t, 16> operator!(Simd<uint16_t, 16> a) {
  alignas(32) uint16_t tmp_a[16], tmp_r[16];
  __m256i a256 = _mm512_castsi512_si256(a.value);
  _mm256_store_si256((__m256i*)tmp_a, a256);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = !tmp_a[i] ? 1 : 0;
  }
  __m256i result = _mm256_load_si256((__m256i*)tmp_r);
  return _mm512_castsi256_si512(result);
}

inline Simd<uint16_t, 16> operator-(Simd<uint16_t, 16> a) {
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i result = _mm256_sub_epi16(_mm256_setzero_si256(), a256);
  return _mm512_castsi256_si512(result);
}

inline Simd<uint16_t, 16> operator+(
    Simd<uint16_t, 16> a,
    Simd<uint16_t, 16> b) {
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  __m256i result = _mm256_add_epi16(a256, b256);
  return _mm512_castsi256_si512(result);
}

inline Simd<uint16_t, 16> operator-(
    Simd<uint16_t, 16> a,
    Simd<uint16_t, 16> b) {
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  __m256i result = _mm256_sub_epi16(a256, b256);
  return _mm512_castsi256_si512(result);
}

inline Simd<uint16_t, 16> operator*(
    Simd<uint16_t, 16> a,
    Simd<uint16_t, 16> b) {
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  __m256i result = _mm256_mullo_epi16(a256, b256);
  return _mm512_castsi256_si512(result);
}

inline Simd<uint16_t, 16> minimum(Simd<uint16_t, 16> a, Simd<uint16_t, 16> b) {
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  __m256i result = _mm256_min_epu16(a256, b256);
  return _mm512_castsi256_si512(result);
}

inline Simd<uint16_t, 16> maximum(Simd<uint16_t, 16> a, Simd<uint16_t, 16> b) {
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  __m256i result = _mm256_max_epu16(a256, b256);
  return _mm512_castsi256_si512(result);
}

inline Simd<uint16_t, 16> operator||(
    Simd<uint16_t, 16> a,
    Simd<uint16_t, 16> b) {
  // Scalar fallback for logical OR
  alignas(32) uint16_t tmp_a[16], tmp_b[16], tmp_r[16];
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  _mm256_store_si256((__m256i*)tmp_a, a256);
  _mm256_store_si256((__m256i*)tmp_b, b256);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = (tmp_a[i] || tmp_b[i]) ? 1 : 0;
  }
  __m256i result = _mm256_load_si256((__m256i*)tmp_r);
  return _mm512_castsi256_si512(result);
}

inline Simd<uint16_t, 16> pow(Simd<uint16_t, 16> a, Simd<uint16_t, 16> b) {
  // Scalar fallback for power
  alignas(32) uint16_t tmp_a[16], tmp_b[16], tmp_r[16];
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  _mm256_store_si256((__m256i*)tmp_a, a256);
  _mm256_store_si256((__m256i*)tmp_b, b256);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = static_cast<uint16_t>(std::pow(tmp_a[i], tmp_b[i]));
  }
  __m256i result = _mm256_load_si256((__m256i*)tmp_r);
  return _mm512_castsi256_si512(result);
}

inline Simd<uint16_t, 16> operator/(
    Simd<uint16_t, 16> a,
    Simd<uint16_t, 16> b) {
  alignas(32) uint16_t tmp_a[16], tmp_b[16], tmp_r[16];
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  _mm256_store_si256((__m256i*)tmp_a, a256);
  _mm256_store_si256((__m256i*)tmp_b, b256);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = tmp_a[i] / tmp_b[i];
  }
  __m256i result = _mm256_load_si256((__m256i*)tmp_r);
  return _mm512_castsi256_si512(result);
}

inline Simd<uint16_t, 16> remainder(
    Simd<uint16_t, 16> a,
    Simd<uint16_t, 16> b) {
  alignas(32) uint16_t tmp_a[16], tmp_b[16], tmp_r[16];
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  _mm256_store_si256((__m256i*)tmp_a, a256);
  _mm256_store_si256((__m256i*)tmp_b, b256);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = tmp_a[i] % tmp_b[i];
  }
  __m256i result = _mm256_load_si256((__m256i*)tmp_r);
  return _mm512_castsi256_si512(result);
}

inline Simd<uint16_t, 16> operator&&(
    Simd<uint16_t, 16> a,
    Simd<uint16_t, 16> b) {
  alignas(32) uint16_t tmp_a[16], tmp_b[16], tmp_r[16];
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  _mm256_store_si256((__m256i*)tmp_a, a256);
  _mm256_store_si256((__m256i*)tmp_b, b256);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = (tmp_a[i] && tmp_b[i]) ? 1 : 0;
  }
  __m256i result = _mm256_load_si256((__m256i*)tmp_r);
  return _mm512_castsi256_si512(result);
}

inline Simd<uint16_t, 16> operator<<(Simd<uint16_t, 16> a, int shift) {
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i result = _mm256_slli_epi16(a256, shift);
  return _mm512_castsi256_si512(result);
}

inline Simd<uint16_t, 16> operator<<(
    Simd<uint16_t, 16> a,
    Simd<uint16_t, 16> b) {
  // AVX2 doesn't have variable shift for 16-bit, use scalar fallback
  alignas(32) uint16_t tmp_a[16], tmp_b[16], tmp_r[16];
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  _mm256_store_si256((__m256i*)tmp_a, a256);
  _mm256_store_si256((__m256i*)tmp_b, b256);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = tmp_a[i] << tmp_b[i];
  }
  __m256i result = _mm256_load_si256((__m256i*)tmp_r);
  return _mm512_castsi256_si512(result);
}

inline Simd<uint16_t, 16> operator>>(Simd<uint16_t, 16> a, int shift) {
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i result = _mm256_srli_epi16(a256, shift);
  return _mm512_castsi256_si512(result);
}

inline Simd<uint16_t, 16> operator>>(
    Simd<uint16_t, 16> a,
    Simd<uint16_t, 16> b) {
  // AVX2 doesn't have variable shift for 16-bit, use scalar fallback
  alignas(32) uint16_t tmp_a[16], tmp_b[16], tmp_r[16];
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  _mm256_store_si256((__m256i*)tmp_a, a256);
  _mm256_store_si256((__m256i*)tmp_b, b256);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = tmp_a[i] >> tmp_b[i];
  }
  __m256i result = _mm256_load_si256((__m256i*)tmp_r);
  return _mm512_castsi256_si512(result);
}

// ============================================================================
// uint8x16 (16 bytes in 128 bits - reusing SSE portion of __m512i)
// Note: AVX-512 doesn't provide major benefits for 8-bit ops, so we keep
// size=16
// ============================================================================

template <>
struct Simd<uint8_t, 16> {
  static constexpr int size = 16;
  __m512i value; // Use 512-bit register but only lower 128 bits are used

  Simd() : value(_mm512_setzero_si512()) {}
  Simd(__m512i v) : value(v) {}
  Simd(uint8_t v) {
    __m128i v128 = _mm_set1_epi8(v);
    value = _mm512_castsi128_si512(v128);
  }
  Simd(Simd<bool, 16> v);
  Simd(Simd<uint32_t, 16> v) {
    // Truncate 16x uint32 to 16x uint8
    __m128i result = _mm512_cvtepi32_epi8(v.value);
    value = _mm512_castsi128_si512(result);
  }

  uint8_t operator[](int idx) const {
    alignas(64) uint8_t tmp[64];
    _mm512_store_si512((__m512i*)tmp, value);
    return tmp[idx];
  }
};

// Load/store for uint8_t
template <>
inline Simd<uint8_t, 16> load<uint8_t, 16>(const uint8_t* ptr) {
  __m128i v128 = _mm_loadu_si128((const __m128i*)ptr);
  return _mm512_castsi128_si512(v128);
}

template <>
inline void store<uint8_t, 16>(uint8_t* ptr, Simd<uint8_t, 16> v) {
  __m128i v128 = _mm512_castsi512_si128(v.value);
  _mm_storeu_si128((__m128i*)ptr, v128);
}

// Conversion constructors (defined after all types are declared)
inline Simd<float, 16>::Simd(Simd<int32_t, 16> v)
    : value(_mm512_cvtepi32_ps(v.value)) {}
inline Simd<float, 16>::Simd(Simd<uint32_t, 16> v)
    : value(_mm512_cvtepu32_ps(v.value)) {}

// float16_t constructor from float
inline Simd<float16_t, 16>::Simd(Simd<float, 16> v) {
  alignas(64) float tmp_in[16];
  alignas(32) float16_t tmp_out[16];
  store<float, 16>(tmp_in, v);
  for (int i = 0; i < 16; ++i) {
    tmp_out[i] = static_cast<float16_t>(tmp_in[i]);
  }
  value = _mm512_castsi256_si512(_mm256_load_si256((__m256i*)tmp_out));
}

// Type conversions (must come before uint8 shift operators)
inline Simd<uint32_t, 16> to_uint32(Simd<uint8_t, 16> v) {
  __m128i v128 = _mm512_castsi512_si128(v.value);
  return _mm512_cvtepu8_epi32(v128);
}

inline Simd<uint8_t, 16> to_uint8(Simd<uint32_t, 16> v) {
  // Truncate 16x uint32 to 16x uint8
  __m128i result = _mm512_cvtepi32_epi8(v.value);
  return _mm512_castsi128_si512(result);
}

// Uint8 operations (basic only - most quantized ops use uint32)
inline Simd<uint8_t, 16> operator&(Simd<uint8_t, 16> a, Simd<uint8_t, 16> b) {
  __m128i a128 = _mm512_castsi512_si128(a.value);
  __m128i b128 = _mm512_castsi512_si128(b.value);
  __m128i result = _mm_and_si128(a128, b128);
  return _mm512_castsi128_si512(result);
}

inline Simd<uint8_t, 16> operator&(Simd<uint8_t, 16> a, uint8_t b) {
  return a & Simd<uint8_t, 16>(b);
}

inline Simd<uint8_t, 16> operator|(Simd<uint8_t, 16> a, Simd<uint8_t, 16> b) {
  __m128i a128 = _mm512_castsi512_si128(a.value);
  __m128i b128 = _mm512_castsi512_si128(b.value);
  __m128i result = _mm_or_si128(a128, b128);
  return _mm512_castsi128_si512(result);
}

inline Simd<uint8_t, 16> operator^(Simd<uint8_t, 16> a, Simd<uint8_t, 16> b) {
  __m128i a128 = _mm512_castsi512_si128(a.value);
  __m128i b128 = _mm512_castsi512_si128(b.value);
  __m128i result = _mm_xor_si128(a128, b128);
  return _mm512_castsi128_si512(result);
}

inline Simd<uint8_t, 16> operator~(Simd<uint8_t, 16> a) {
  __m128i a128 = _mm512_castsi512_si128(a.value);
  __m128i result = _mm_xor_si128(a128, _mm_set1_epi8(-1));
  return _mm512_castsi128_si512(result);
}

inline Simd<uint8_t, 16> operator!(Simd<uint8_t, 16> a) {
  alignas(16) uint8_t tmp_a[16], tmp_r[16];
  __m128i a128 = _mm512_castsi512_si128(a.value);
  _mm_store_si128((__m128i*)tmp_a, a128);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = !tmp_a[i] ? 1 : 0;
  }
  __m128i result = _mm_load_si128((__m128i*)tmp_r);
  return _mm512_castsi128_si512(result);
}

inline Simd<uint8_t, 16> operator-(Simd<uint8_t, 16> a) {
  __m128i a128 = _mm512_castsi512_si128(a.value);
  __m128i result = _mm_sub_epi8(_mm_setzero_si128(), a128);
  return _mm512_castsi128_si512(result);
}

inline Simd<uint8_t, 16> operator+(Simd<uint8_t, 16> a, Simd<uint8_t, 16> b) {
  __m128i a128 = _mm512_castsi512_si128(a.value);
  __m128i b128 = _mm512_castsi512_si128(b.value);
  __m128i result = _mm_add_epi8(a128, b128);
  return _mm512_castsi128_si512(result);
}

inline Simd<uint8_t, 16> operator-(Simd<uint8_t, 16> a, Simd<uint8_t, 16> b) {
  __m128i a128 = _mm512_castsi512_si128(a.value);
  __m128i b128 = _mm512_castsi512_si128(b.value);
  __m128i result = _mm_sub_epi8(a128, b128);
  return _mm512_castsi128_si512(result);
}

inline Simd<uint8_t, 16> operator>>(Simd<uint8_t, 16> a, int shift) {
  // Convert to uint32, shift, then convert back
  auto a32 = to_uint32(a);
  auto shifted = a32 >> shift;
  return to_uint8(shifted);
}

inline Simd<uint8_t, 16> operator>>(Simd<uint8_t, 16> a, Simd<uint8_t, 16> b) {
  // Convert to uint32, shift, then convert back
  auto a32 = to_uint32(a);
  auto b32 = to_uint32(b);
  auto shifted = a32 >> b32;
  return to_uint8(shifted);
}

inline Simd<uint8_t, 16> operator<<(Simd<uint8_t, 16> a, int shift) {
  // Convert to uint32, shift, then convert back
  auto a32 = to_uint32(a);
  auto shifted = a32 << shift;
  return to_uint8(shifted);
}

inline Simd<uint8_t, 16> operator<<(Simd<uint8_t, 16> a, Simd<uint8_t, 16> b) {
  // Convert to uint32, shift, then convert back
  auto a32 = to_uint32(a);
  auto b32 = to_uint32(b);
  auto shifted = a32 << b32;
  return to_uint8(shifted);
}

inline Simd<uint8_t, 16> operator||(Simd<uint8_t, 16> a, Simd<uint8_t, 16> b) {
  alignas(16) uint8_t tmp_a[16], tmp_b[16], tmp_r[16];
  __m128i a128 = _mm512_castsi512_si128(a.value);
  __m128i b128 = _mm512_castsi512_si128(b.value);
  _mm_store_si128((__m128i*)tmp_a, a128);
  _mm_store_si128((__m128i*)tmp_b, b128);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = (tmp_a[i] || tmp_b[i]) ? 1 : 0;
  }
  __m128i result = _mm_load_si128((__m128i*)tmp_r);
  return _mm512_castsi128_si512(result);
}

inline Simd<uint8_t, 16> operator&&(Simd<uint8_t, 16> a, Simd<uint8_t, 16> b) {
  alignas(16) uint8_t tmp_a[16], tmp_b[16], tmp_r[16];
  __m128i a128 = _mm512_castsi512_si128(a.value);
  __m128i b128 = _mm512_castsi512_si128(b.value);
  _mm_store_si128((__m128i*)tmp_a, a128);
  _mm_store_si128((__m128i*)tmp_b, b128);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = (tmp_a[i] && tmp_b[i]) ? 1 : 0;
  }
  __m128i result = _mm_load_si128((__m128i*)tmp_r);
  return _mm512_castsi128_si512(result);
}

inline Simd<bool, 16>::Simd(Simd<uint8_t, 16> v) {
  __m128i v128 = _mm512_castsi512_si128(v.value);
  // Compare each byte with zero to get mask
  __mmask16 mask = _mm_cmpneq_epi8_mask(v128, _mm_setzero_si128());
  value = mask;
}

inline Simd<uint8_t, 16>::Simd(Simd<bool, 16> v) {
  // Convert mask bits to bytes (0 or 0xFF)
  __m128i result = _mm_maskz_set1_epi8(v.value, 0xFF);
  value = _mm512_castsi128_si512(result);
}

// uint16_t constructor from uint8_t (defined after uint8_t struct)
inline Simd<uint16_t, 16>::Simd(Simd<uint8_t, 16> v) {
  // Zero-extend 16x uint8 to 16x uint16
  __m128i v128 = _mm512_castsi512_si128(v.value);
  __m256i v256 = _mm256_cvtepu8_epi16(v128);
  value = _mm512_castsi256_si512(v256);
}

// int32_t constructor from uint16_t
inline Simd<int32_t, 16>::Simd(Simd<uint16_t, 16> v) {
  // Zero-extend 16x uint16 to 16x int32
  __m256i v256 = _mm512_castsi512_si256(v.value);
  value = _mm512_cvtepu16_epi32(v256);
}

// int32_t constructor from uint8_t
inline Simd<int32_t, 16>::Simd(Simd<uint8_t, 16> v) {
  // Zero-extend 16x uint8 to 16x int32
  __m128i v128 = _mm512_castsi512_si128(v.value);
  value = _mm512_cvtepu8_epi32(v128);
}

// int32_t constructor from float
inline Simd<int32_t, 16>::Simd(Simd<float, 16> v)
    : value(_mm512_cvtps_epi32(v.value)) {}

// uint32_t constructor from float
inline Simd<uint32_t, 16>::Simd(Simd<float, 16> v)
    : value(_mm512_cvtps_epu32(v.value)) {}

// uint16_t constructor from int32_t (truncate)
inline Simd<uint16_t, 16>::Simd(Simd<int32_t, 16> v) {
  // Truncate 16x int32 to 16x uint16
  __m256i result = _mm512_cvtepi32_epi16(v.value);
  value = _mm512_castsi256_si512(result);
}

// uint16_t reductions
inline uint16_t sum(Simd<uint16_t, 16> v) {
  __m256i v256 = _mm512_castsi512_si256(v.value);
  alignas(32) uint16_t tmp[16];
  _mm256_store_si256((__m256i*)tmp, v256);
  uint16_t result = 0;
  for (int i = 0; i < 16; ++i) {
    result += tmp[i];
  }
  return result;
}

inline uint16_t prod(Simd<uint16_t, 16> v) {
  __m256i v256 = _mm512_castsi512_si256(v.value);
  alignas(32) uint16_t tmp[16];
  _mm256_store_si256((__m256i*)tmp, v256);
  uint16_t result = 1;
  for (int i = 0; i < 16; ++i) {
    result *= tmp[i];
  }
  return result;
}

inline uint16_t min(Simd<uint16_t, 16> v) {
  __m256i v256 = _mm512_castsi512_si256(v.value);
  alignas(32) uint16_t tmp[16];
  _mm256_store_si256((__m256i*)tmp, v256);
  uint16_t result = tmp[0];
  for (int i = 1; i < 16; ++i) {
    result = result < tmp[i] ? result : tmp[i];
  }
  return result;
}

inline uint16_t max(Simd<uint16_t, 16> v) {
  __m256i v256 = _mm512_castsi512_si256(v.value);
  alignas(32) uint16_t tmp[16];
  _mm256_store_si256((__m256i*)tmp, v256);
  uint16_t result = tmp[0];
  for (int i = 1; i < 16; ++i) {
    result = result > tmp[i] ? result : tmp[i];
  }
  return result;
}

// uint8_t comparison operators (using uint32 conversion)
inline Simd<bool, 16> operator<(Simd<uint8_t, 16> a, Simd<uint8_t, 16> b) {
  auto a32 = to_uint32(a);
  auto b32 = to_uint32(b);
  return a32 < b32;
}

inline Simd<bool, 16> operator<=(Simd<uint8_t, 16> a, Simd<uint8_t, 16> b) {
  auto a32 = to_uint32(a);
  auto b32 = to_uint32(b);
  return a32 <= b32;
}

inline Simd<bool, 16> operator>(Simd<uint8_t, 16> a, Simd<uint8_t, 16> b) {
  auto a32 = to_uint32(a);
  auto b32 = to_uint32(b);
  return a32 > b32;
}

inline Simd<bool, 16> operator>=(Simd<uint8_t, 16> a, Simd<uint8_t, 16> b) {
  auto a32 = to_uint32(a);
  auto b32 = to_uint32(b);
  return a32 >= b32;
}

inline Simd<bool, 16> operator==(Simd<uint8_t, 16> a, Simd<uint8_t, 16> b) {
  auto a32 = to_uint32(a);
  auto b32 = to_uint32(b);
  return a32 == b32;
}

inline Simd<bool, 16> operator!=(Simd<uint8_t, 16> a, Simd<uint8_t, 16> b) {
  auto a32 = to_uint32(a);
  auto b32 = to_uint32(b);
  return a32 != b32;
}

// uint8_t arithmetic operators
inline Simd<uint8_t, 16> operator*(Simd<uint8_t, 16> a, Simd<uint8_t, 16> b) {
  auto a32 = to_uint32(a);
  auto b32 = to_uint32(b);
  return to_uint8(a32 * b32);
}

inline Simd<uint8_t, 16> operator/(Simd<uint8_t, 16> a, Simd<uint8_t, 16> b) {
  auto a32 = to_uint32(a);
  auto b32 = to_uint32(b);
  return to_uint8(a32 / b32);
}

// uint8_t select
inline Simd<uint8_t, 16>
select(Simd<bool, 16> mask, Simd<uint8_t, 16> a, Simd<uint8_t, 16> b) {
  // Convert to uint32 for selection
  auto a32 = to_uint32(a);
  auto b32 = to_uint32(b);
  auto result32 = select(mask, a32, b32);
  return to_uint8(result32);
}

// uint8_t min/max
inline Simd<uint8_t, 16> minimum(Simd<uint8_t, 16> a, Simd<uint8_t, 16> b) {
  __m128i a128 = _mm512_castsi512_si128(a.value);
  __m128i b128 = _mm512_castsi512_si128(b.value);
  __m128i result = _mm_min_epu8(a128, b128);
  return _mm512_castsi128_si512(result);
}

inline Simd<uint8_t, 16> maximum(Simd<uint8_t, 16> a, Simd<uint8_t, 16> b) {
  __m128i a128 = _mm512_castsi512_si128(a.value);
  __m128i b128 = _mm512_castsi512_si128(b.value);
  __m128i result = _mm_max_epu8(a128, b128);
  return _mm512_castsi128_si512(result);
}

// uint8_t reductions
inline uint8_t sum(Simd<uint8_t, 16> v) {
  // Convert to uint32 for reduction
  auto v32 = to_uint32(v);
  return static_cast<uint8_t>(sum(v32));
}

inline uint8_t min(Simd<uint8_t, 16> v) {
  __m128i v128 = _mm512_castsi512_si128(v.value);
  alignas(16) uint8_t tmp[16];
  _mm_store_si128((__m128i*)tmp, v128);
  uint8_t result = tmp[0];
  for (int i = 1; i < 16; ++i) {
    result = result < tmp[i] ? result : tmp[i];
  }
  return result;
}

inline uint8_t max(Simd<uint8_t, 16> v) {
  __m128i v128 = _mm512_castsi512_si128(v.value);
  alignas(16) uint8_t tmp[16];
  _mm_store_si128((__m128i*)tmp, v128);
  uint8_t result = tmp[0];
  for (int i = 1; i < 16; ++i) {
    result = result > tmp[i] ? result : tmp[i];
  }
  return result;
}

// remainder() for all types
inline Simd<float, 16> remainder(Simd<float, 16> a, Simd<float, 16> b) {
  alignas(64) float tmp_a[16], tmp_b[16], tmp_r[16];
  store<float, 16>(tmp_a, a);
  store<float, 16>(tmp_b, b);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = std::remainder(tmp_a[i], tmp_b[i]);
  }
  return load<float, 16>(tmp_r);
}

inline Simd<double, 8> remainder(Simd<double, 8> a, Simd<double, 8> b) {
  alignas(64) double tmp_a[8], tmp_b[8], tmp_r[8];
  store<double, 8>(tmp_a, a);
  store<double, 8>(tmp_b, b);
  for (int i = 0; i < 8; ++i) {
    tmp_r[i] = std::remainder(tmp_a[i], tmp_b[i]);
  }
  return load<double, 8>(tmp_r);
}

inline Simd<int32_t, 16> remainder(Simd<int32_t, 16> a, Simd<int32_t, 16> b) {
  alignas(64) int32_t tmp_a[16], tmp_b[16], tmp_r[16];
  store<int32_t, 16>(tmp_a, a);
  store<int32_t, 16>(tmp_b, b);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = tmp_a[i] % tmp_b[i];
  }
  return load<int32_t, 16>(tmp_r);
}

inline Simd<uint32_t, 16> remainder(
    Simd<uint32_t, 16> a,
    Simd<uint32_t, 16> b) {
  alignas(64) uint32_t tmp_a[16], tmp_b[16], tmp_r[16];
  store<uint32_t, 16>(tmp_a, a);
  store<uint32_t, 16>(tmp_b, b);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = tmp_a[i] % tmp_b[i];
  }
  return load<uint32_t, 16>(tmp_r);
}

inline Simd<uint8_t, 16> remainder(Simd<uint8_t, 16> a, Simd<uint8_t, 16> b) {
  auto a32 = to_uint32(a);
  auto b32 = to_uint32(b);
  return to_uint8(remainder(a32, b32));
}

// log1p() transcendental function
inline Simd<float, 16> log1p(Simd<float, 16> v) {
  alignas(64) float tmp[16];
  store<float, 16>(tmp, v);
  for (int i = 0; i < 16; ++i) {
    tmp[i] = std::log1p(tmp[i]);
  }
  return load<float, 16>(tmp);
}

inline Simd<double, 8> log1p(Simd<double, 8> v) {
  alignas(64) double tmp[8];
  store<double, 8>(tmp, v);
  for (int i = 0; i < 8; ++i) {
    tmp[i] = std::log1p(tmp[i]);
  }
  return load<double, 8>(tmp);
}

// Logical operators for non-bool types (element-wise boolean AND/OR)
// These convert to bool, apply logical op, then convert back

// float logical operators
inline Simd<float, 16> operator&&(Simd<float, 16> a, Simd<float, 16> b) {
  auto a_bool = a != Simd<float, 16>(0.0f);
  auto b_bool = b != Simd<float, 16>(0.0f);
  auto result_bool = a_bool & b_bool;
  return select(result_bool, Simd<float, 16>(1.0f), Simd<float, 16>(0.0f));
}

inline Simd<float, 16> operator||(Simd<float, 16> a, Simd<float, 16> b) {
  auto a_bool = a != Simd<float, 16>(0.0f);
  auto b_bool = b != Simd<float, 16>(0.0f);
  auto result_bool = a_bool | b_bool;
  return select(result_bool, Simd<float, 16>(1.0f), Simd<float, 16>(0.0f));
}

// double logical operators
inline Simd<double, 8> operator&&(Simd<double, 8> a, Simd<double, 8> b) {
  auto a_bool = a != Simd<double, 8>(0.0);
  auto b_bool = b != Simd<double, 8>(0.0);
  auto result_bool = a_bool & b_bool;
  return select(result_bool, Simd<double, 8>(1.0), Simd<double, 8>(0.0));
}

inline Simd<double, 8> operator||(Simd<double, 8> a, Simd<double, 8> b) {
  auto a_bool = a != Simd<double, 8>(0.0);
  auto b_bool = b != Simd<double, 8>(0.0);
  auto result_bool = a_bool | b_bool;
  return select(result_bool, Simd<double, 8>(1.0), Simd<double, 8>(0.0));
}

// int32_t logical operators
inline Simd<int32_t, 16> operator&&(Simd<int32_t, 16> a, Simd<int32_t, 16> b) {
  auto a_bool = a != Simd<int32_t, 16>(0);
  auto b_bool = b != Simd<int32_t, 16>(0);
  auto result_bool = a_bool & b_bool;
  return select(result_bool, Simd<int32_t, 16>(1), Simd<int32_t, 16>(0));
}

inline Simd<int32_t, 16> operator||(Simd<int32_t, 16> a, Simd<int32_t, 16> b) {
  auto a_bool = a != Simd<int32_t, 16>(0);
  auto b_bool = b != Simd<int32_t, 16>(0);
  auto result_bool = a_bool | b_bool;
  return select(result_bool, Simd<int32_t, 16>(1), Simd<int32_t, 16>(0));
}

// uint32_t logical operators
inline Simd<uint32_t, 16> operator&&(
    Simd<uint32_t, 16> a,
    Simd<uint32_t, 16> b) {
  auto a_bool = a != Simd<uint32_t, 16>(0);
  auto b_bool = b != Simd<uint32_t, 16>(0);
  auto result_bool = a_bool & b_bool;
  return select(result_bool, Simd<uint32_t, 16>(1), Simd<uint32_t, 16>(0));
}

inline Simd<uint32_t, 16> operator||(
    Simd<uint32_t, 16> a,
    Simd<uint32_t, 16> b) {
  auto a_bool = a != Simd<uint32_t, 16>(0);
  auto b_bool = b != Simd<uint32_t, 16>(0);
  auto result_bool = a_bool | b_bool;
  return select(result_bool, Simd<uint32_t, 16>(1), Simd<uint32_t, 16>(0));
}

// Logical NOT operators for non-bool types
inline Simd<float, 16> operator!(Simd<float, 16> a) {
  auto zero = Simd<float, 16>(0.0f);
  return select(a == zero, Simd<float, 16>(1.0f), Simd<float, 16>(0.0f));
}

inline Simd<double, 8> operator!(Simd<double, 8> a) {
  auto zero = Simd<double, 8>(0.0);
  return select(a == zero, Simd<double, 8>(1.0), Simd<double, 8>(0.0));
}

inline Simd<int32_t, 16> operator!(Simd<int32_t, 16> a) {
  auto zero = Simd<int32_t, 16>(0);
  return select(a == zero, Simd<int32_t, 16>(1), Simd<int32_t, 16>(0));
}

inline Simd<uint32_t, 16> operator!(Simd<uint32_t, 16> a) {
  auto zero = Simd<uint32_t, 16>(0);
  return select(a == zero, Simd<uint32_t, 16>(1), Simd<uint32_t, 16>(0));
}

// Unary negation for uint32_t
inline Simd<uint32_t, 16> operator-(Simd<uint32_t, 16> a) {
  return _mm512_sub_epi32(_mm512_setzero_si512(), a.value);
}

// pow for uint8_t
inline Simd<uint8_t, 16> pow(Simd<uint8_t, 16> a, Simd<uint8_t, 16> b) {
  auto a32 = to_uint32(a);
  auto b32 = to_uint32(b);
  return to_uint8(pow(a32, b32));
}

// Constructor for Simd<int32_t, 16> from Simd<uint32_t, 16>
inline Simd<int32_t, 16>::Simd(Simd<uint32_t, 16> v) : value(v.value) {}

// ============================================================================
// int16_t and int8_t support (used as index types in scatter/gather operations)
// ============================================================================

// int16_t
template <>
inline constexpr int max_size<int16_t> = 16;

template <>
struct Simd<int16_t, 16> {
  static constexpr int size = 16;
  __m512i value;

  Simd() : value(_mm512_setzero_si512()) {}
  Simd(__m512i v) : value(v) {}
  Simd(int16_t v) {
    __m256i v256 = _mm256_set1_epi16(v);
    value = _mm512_castsi256_si512(v256);
  }
  Simd(Simd<int32_t, 16> v); // Defined after int32 operators (truncate)

  int16_t operator[](int idx) const {
    alignas(64) int16_t tmp[32];
    _mm512_store_si512((__m512i*)tmp, value);
    return tmp[idx];
  }
};

template <>
inline Simd<int16_t, 16> load<int16_t, 16>(const int16_t* ptr) {
  __m256i v256 = _mm256_loadu_si256((const __m256i*)ptr);
  return _mm512_castsi256_si512(v256);
}

template <>
inline void store<int16_t, 16>(int16_t* ptr, Simd<int16_t, 16> v) {
  __m256i v256 = _mm512_castsi512_si256(v.value);
  _mm256_storeu_si256((__m256i*)ptr, v256);
}

// Arithmetic operators for int16_t
DEFINE_AVX512_INT16_BINARY_OP(+, _mm256_add_epi16)
DEFINE_AVX512_INT16_BINARY_OP(-, _mm256_sub_epi16)
DEFINE_AVX512_INT16_BINARY_OP(*, _mm256_mullo_epi16)
DEFINE_AVX512_INT16_SCALAR_FALLBACK(/)

// Bitwise operators for int16_t
DEFINE_AVX512_INT16_BINARY_OP(&, _mm256_and_si256)
DEFINE_AVX512_INT16_BINARY_OP(|, _mm256_or_si256)
DEFINE_AVX512_INT16_BINARY_OP(^, _mm256_xor_si256)

inline Simd<int16_t, 16> operator~(Simd<int16_t, 16> a) {
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i result = _mm256_xor_si256(a256, _mm256_set1_epi16(-1));
  return _mm512_castsi256_si512(result);
}

// Shift operators for int16_t
DEFINE_AVX512_INT16_SCALAR_FALLBACK(<<)
DEFINE_AVX512_INT16_SCALAR_FALLBACK(>>)

// Unary operators for int16_t
inline Simd<int16_t, 16> operator-(Simd<int16_t, 16> a) {
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i result = _mm256_sub_epi16(_mm256_setzero_si256(), a256);
  return _mm512_castsi256_si512(result);
}

DEFINE_AVX512_INT16_UNARY_SCALAR(operator!, !tmp_a[i] ? 1 : 0)

// Min/max/abs for int16_t
DEFINE_AVX512_INT16_BINARY_FUNC(minimum, _mm256_min_epi16)
DEFINE_AVX512_INT16_BINARY_FUNC(maximum, _mm256_max_epi16)
DEFINE_AVX512_INT16_UNARY_OP(abs, _mm256_abs_epi16)

// Reductions for int16_t
DEFINE_AVX512_INT16_REDUCTION(sum, 0, +)
DEFINE_AVX512_INT16_REDUCTION(prod, 1, *)

inline int16_t min(Simd<int16_t, 16> v) {
  alignas(32) int16_t tmp[16];
  __m256i v256 = _mm512_castsi512_si256(v.value);
  _mm256_store_si256((__m256i*)tmp, v256);
  int16_t result = tmp[0];
  for (int i = 1; i < 16; ++i) {
    result = result < tmp[i] ? result : tmp[i];
  }
  return result;
}

inline int16_t max(Simd<int16_t, 16> v) {
  alignas(32) int16_t tmp[16];
  __m256i v256 = _mm512_castsi512_si256(v.value);
  _mm256_store_si256((__m256i*)tmp, v256);
  int16_t result = tmp[0];
  for (int i = 1; i < 16; ++i) {
    result = result > tmp[i] ? result : tmp[i];
  }
  return result;
}

// pow for int16_t
inline Simd<int16_t, 16> pow(Simd<int16_t, 16> a, Simd<int16_t, 16> b) {
  alignas(32) int16_t tmp_a[16], tmp_b[16], tmp_r[16];
  store<int16_t, 16>(tmp_a, a);
  store<int16_t, 16>(tmp_b, b);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = std::pow(tmp_a[i], tmp_b[i]);
  }
  return load<int16_t, 16>(tmp_r);
}

// Comparison operators for int16_t (return Simd<bool, 16>)
inline Simd<bool, 16> operator==(Simd<int16_t, 16> a, Simd<int16_t, 16> b) {
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  __m256i cmp = _mm256_cmpeq_epi16(a256, b256);
  // Convert to 16x int32 masks
  __m512i result = _mm512_cvtepi16_epi32(cmp);
  return Simd<bool, 16>(result);
}

inline Simd<bool, 16> operator!=(Simd<int16_t, 16> a, Simd<int16_t, 16> b) {
  return !(a == b);
}

inline Simd<bool, 16> operator<(Simd<int16_t, 16> a, Simd<int16_t, 16> b) {
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  __m256i cmp = _mm256_cmpgt_epi16(b256, a256); // b > a means a < b
  __m512i result = _mm512_cvtepi16_epi32(cmp);
  return Simd<bool, 16>(result);
}

inline Simd<bool, 16> operator<=(Simd<int16_t, 16> a, Simd<int16_t, 16> b) {
  return !(a > b);
}

inline Simd<bool, 16> operator>(Simd<int16_t, 16> a, Simd<int16_t, 16> b) {
  __m256i a256 = _mm512_castsi512_si256(a.value);
  __m256i b256 = _mm512_castsi512_si256(b.value);
  __m256i cmp = _mm256_cmpgt_epi16(a256, b256);
  __m512i result = _mm512_cvtepi16_epi32(cmp);
  return Simd<bool, 16>(result);
}

inline Simd<bool, 16> operator>=(Simd<int16_t, 16> a, Simd<int16_t, 16> b) {
  return !(a < b);
}

// Logical operators for int16_t
inline Simd<int16_t, 16> operator&&(Simd<int16_t, 16> a, Simd<int16_t, 16> b) {
  alignas(32) int16_t tmp_a[16], tmp_b[16], tmp_r[16];
  store<int16_t, 16>(tmp_a, a);
  store<int16_t, 16>(tmp_b, b);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = (tmp_a[i] && tmp_b[i]) ? 1 : 0;
  }
  return load<int16_t, 16>(tmp_r);
}

inline Simd<int16_t, 16> operator||(Simd<int16_t, 16> a, Simd<int16_t, 16> b) {
  alignas(32) int16_t tmp_a[16], tmp_b[16], tmp_r[16];
  store<int16_t, 16>(tmp_a, a);
  store<int16_t, 16>(tmp_b, b);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = (tmp_a[i] || tmp_b[i]) ? 1 : 0;
  }
  return load<int16_t, 16>(tmp_r);
}

// remainder for int16_t
inline Simd<int16_t, 16> remainder(Simd<int16_t, 16> a, Simd<int16_t, 16> b) {
  alignas(32) int16_t tmp_a[16], tmp_b[16], tmp_r[16];
  store<int16_t, 16>(tmp_a, a);
  store<int16_t, 16>(tmp_b, b);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = tmp_a[i] % tmp_b[i];
  }
  return load<int16_t, 16>(tmp_r);
}

// int8_t
template <>
inline constexpr int max_size<int8_t> = 16;

template <>
struct Simd<int8_t, 16> {
  static constexpr int size = 16;
  __m512i value;

  Simd() : value(_mm512_setzero_si512()) {}
  Simd(__m512i v) : value(v) {}
  Simd(int8_t v) {
    __m128i v128 = _mm_set1_epi8(v);
    value = _mm512_castsi128_si512(v128);
  }
  Simd(Simd<int32_t, 16> v); // Defined after int32 operators (truncate)

  int8_t operator[](int idx) const {
    alignas(64) int8_t tmp[64];
    _mm512_store_si512((__m512i*)tmp, value);
    return tmp[idx];
  }
};

template <>
inline Simd<int8_t, 16> load<int8_t, 16>(const int8_t* ptr) {
  __m128i v128 = _mm_loadu_si128((const __m128i*)ptr);
  return _mm512_castsi128_si512(v128);
}

template <>
inline void store<int8_t, 16>(int8_t* ptr, Simd<int8_t, 16> v) {
  __m128i v128 = _mm512_castsi512_si128(v.value);
  _mm_storeu_si128((__m128i*)ptr, v128);
}

// Arithmetic operators for int8_t
DEFINE_AVX512_INT8_BINARY_OP(+, _mm_add_epi8)
DEFINE_AVX512_INT8_BINARY_OP(-, _mm_sub_epi8)
DEFINE_AVX512_INT8_SCALAR_FALLBACK(*)
DEFINE_AVX512_INT8_SCALAR_FALLBACK(/)

// Bitwise operators for int8_t
DEFINE_AVX512_INT8_BINARY_OP(&, _mm_and_si128)
DEFINE_AVX512_INT8_BINARY_OP(|, _mm_or_si128)
DEFINE_AVX512_INT8_BINARY_OP(^, _mm_xor_si128)

inline Simd<int8_t, 16> operator~(Simd<int8_t, 16> a) {
  __m128i a128 = _mm512_castsi512_si128(a.value);
  __m128i result = _mm_xor_si128(a128, _mm_set1_epi8(-1));
  return _mm512_castsi128_si512(result);
}

// Shift operators for int8_t (scalar fallback)
DEFINE_AVX512_INT8_SCALAR_FALLBACK(<<)
DEFINE_AVX512_INT8_SCALAR_FALLBACK(>>)

// Unary operators for int8_t
inline Simd<int8_t, 16> operator-(Simd<int8_t, 16> a) {
  __m128i a128 = _mm512_castsi512_si128(a.value);
  __m128i result = _mm_sub_epi8(_mm_setzero_si128(), a128);
  return _mm512_castsi128_si512(result);
}

DEFINE_AVX512_INT8_UNARY_SCALAR(operator!, !tmp_a[i] ? 1 : 0)

// Min/max/abs for int8_t
DEFINE_AVX512_INT8_BINARY_FUNC(minimum, _mm_min_epi8)
DEFINE_AVX512_INT8_BINARY_FUNC(maximum, _mm_max_epi8)
DEFINE_AVX512_INT8_UNARY_OP(abs, _mm_abs_epi8)

// Reductions for int8_t
DEFINE_AVX512_INT8_REDUCTION(sum, 0, +)
DEFINE_AVX512_INT8_REDUCTION(prod, 1, *)

inline int8_t min(Simd<int8_t, 16> v) {
  alignas(16) int8_t tmp[16];
  __m128i v128 = _mm512_castsi512_si128(v.value);
  _mm_store_si128((__m128i*)tmp, v128);
  int8_t result = tmp[0];
  for (int i = 1; i < 16; ++i) {
    result = result < tmp[i] ? result : tmp[i];
  }
  return result;
}

inline int8_t max(Simd<int8_t, 16> v) {
  alignas(16) int8_t tmp[16];
  __m128i v128 = _mm512_castsi512_si128(v.value);
  _mm_store_si128((__m128i*)tmp, v128);
  int8_t result = tmp[0];
  for (int i = 1; i < 16; ++i) {
    result = result > tmp[i] ? result : tmp[i];
  }
  return result;
}

// pow for int8_t
inline Simd<int8_t, 16> pow(Simd<int8_t, 16> a, Simd<int8_t, 16> b) {
  alignas(16) int8_t tmp_a[16], tmp_b[16], tmp_r[16];
  store<int8_t, 16>(tmp_a, a);
  store<int8_t, 16>(tmp_b, b);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = std::pow(tmp_a[i], tmp_b[i]);
  }
  return load<int8_t, 16>(tmp_r);
}

// Comparison operators for int8_t (return Simd<bool, 16>)
inline Simd<bool, 16> operator==(Simd<int8_t, 16> a, Simd<int8_t, 16> b) {
  __m128i a128 = _mm512_castsi512_si128(a.value);
  __m128i b128 = _mm512_castsi512_si128(b.value);
  __m128i cmp = _mm_cmpeq_epi8(a128, b128);
  // Convert to 16x int32 masks
  __m512i result = _mm512_cvtepi8_epi32(cmp);
  return Simd<bool, 16>(result);
}

inline Simd<bool, 16> operator!=(Simd<int8_t, 16> a, Simd<int8_t, 16> b) {
  return !(a == b);
}

inline Simd<bool, 16> operator<(Simd<int8_t, 16> a, Simd<int8_t, 16> b) {
  __m128i a128 = _mm512_castsi512_si128(a.value);
  __m128i b128 = _mm512_castsi512_si128(b.value);
  __m128i cmp = _mm_cmpgt_epi8(b128, a128); // b > a means a < b
  __m512i result = _mm512_cvtepi8_epi32(cmp);
  return Simd<bool, 16>(result);
}

inline Simd<bool, 16> operator<=(Simd<int8_t, 16> a, Simd<int8_t, 16> b) {
  return !(a > b);
}

inline Simd<bool, 16> operator>(Simd<int8_t, 16> a, Simd<int8_t, 16> b) {
  __m128i a128 = _mm512_castsi512_si128(a.value);
  __m128i b128 = _mm512_castsi512_si128(b.value);
  __m128i cmp = _mm_cmpgt_epi8(a128, b128);
  __m512i result = _mm512_cvtepi8_epi32(cmp);
  return Simd<bool, 16>(result);
}

inline Simd<bool, 16> operator>=(Simd<int8_t, 16> a, Simd<int8_t, 16> b) {
  return !(a < b);
}

// Logical operators for int8_t
inline Simd<int8_t, 16> operator&&(Simd<int8_t, 16> a, Simd<int8_t, 16> b) {
  alignas(16) int8_t tmp_a[16], tmp_b[16], tmp_r[16];
  store<int8_t, 16>(tmp_a, a);
  store<int8_t, 16>(tmp_b, b);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = (tmp_a[i] && tmp_b[i]) ? 1 : 0;
  }
  return load<int8_t, 16>(tmp_r);
}

inline Simd<int8_t, 16> operator||(Simd<int8_t, 16> a, Simd<int8_t, 16> b) {
  alignas(16) int8_t tmp_a[16], tmp_b[16], tmp_r[16];
  store<int8_t, 16>(tmp_a, a);
  store<int8_t, 16>(tmp_b, b);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = (tmp_a[i] || tmp_b[i]) ? 1 : 0;
  }
  return load<int8_t, 16>(tmp_r);
}

// remainder for int8_t
inline Simd<int8_t, 16> remainder(Simd<int8_t, 16> a, Simd<int8_t, 16> b) {
  alignas(16) int8_t tmp_a[16], tmp_b[16], tmp_r[16];
  store<int8_t, 16>(tmp_a, a);
  store<int8_t, 16>(tmp_b, b);
  for (int i = 0; i < 16; ++i) {
    tmp_r[i] = tmp_a[i] % tmp_b[i];
  }
  return load<int8_t, 16>(tmp_r);
}

// Conversion constructors: int8_t/int16_t → int32_t
inline Simd<int32_t, 16>::Simd(Simd<int8_t, 16> v) {
  __m128i v128 = _mm512_castsi512_si128(v.value);
  value = _mm512_cvtepi8_epi32(v128);
}

inline Simd<int32_t, 16>::Simd(Simd<int16_t, 16> v) {
  __m256i v256 = _mm512_castsi512_si256(v.value);
  value = _mm512_cvtepi16_epi32(v256);
}

// Truncating conversions: int32_t → int16_t/int8_t
inline Simd<int16_t, 16>::Simd(Simd<int32_t, 16> v) {
  // Truncate 16x int32 to 16x int16
  __m256i result = _mm512_cvtepi32_epi16(v.value);
  value = _mm512_castsi256_si512(result);
}

inline Simd<int8_t, 16>::Simd(Simd<int32_t, 16> v) {
  // Truncate 16x int32 to 16x int8
  __m128i result = _mm512_cvtepi32_epi8(v.value);
  value = _mm512_castsi128_si512(result);
}

// ============================================================================
// AVX-512 BF16 (Native BFloat16 Arithmetic with Simd<> Integration)
// ============================================================================
//
// AVX-512 BF16 provides hardware-accelerated BFloat16 operations for ML.
// BFloat16 is a 16-bit format that preserves the exponent range of Float32
// but with reduced mantissa precision (8 bits vs 23 bits).
//
// Format: 1 sign bit + 8 exponent bits + 7 mantissa bits
// Register: 32x BF16 values in 512-bit register (vs 16x FP32)
// Performance: ~2x speedup potential (better bandwidth, FP32 accumulation)
//
// Advantages:
//   - Same exponent range as FP32 (no overflow/underflow issues)
//   - 2x memory bandwidth vs FP32
//   - Direct conversion from/to FP32 (just truncate/zero-extend mantissa)
//
// Available on: Cooper Lake (server, 2020+), Sapphire Rapids (2023+)

#ifdef __AVX512BF16__

// Note: Unlike FP16, BF16 lacks native arithmetic instructions (add, mul,
// etc.). The primary operation is VDPBF16PS (dot product with FP32
// accumulation). For element-wise operations, we convert BF16 ↔ FP32.

template <>
struct Simd<bfloat16_t, 32> {
  static constexpr int size = 32;
  using scalar_t = bfloat16_t;

  __m512bh value;

  Simd() : value((__m512bh)_mm512_setzero_si512()) {}
  Simd(__m512bh v) : value(v) {}

  // Convert from bfloat16_t scalar - convert to FP32 first, then to BF16
  Simd(bfloat16_t v) {
    float f = static_cast<float>(v);
    __m512 fp32_vec = _mm512_set1_ps(f);
    __m256bh half = _mm512_cvtneps_pbh(fp32_vec);
    // Broadcast the 16-wide result to 32-wide
    value = (__m512bh)_mm512_inserti64x4(
        _mm512_castsi256_si512((__m256i)half), (__m256i)half, 1);
  }

  // Convert from other numeric types via FP32
  template <
      typename U,
      typename = std::enable_if_t<
          std::is_arithmetic_v<U> && !std::is_same_v<U, bfloat16_t>>>
  Simd(U v) : Simd(static_cast<bfloat16_t>(v)) {}

  // Array access
  bfloat16_t operator[](int idx) const {
    alignas(64) uint16_t tmp[32];
    _mm512_storeu_epi16(tmp, (__m512i)value);
    // Reconstruct bfloat16_t from raw bits
    union {
      uint16_t u;
      bfloat16_t b;
    } conv;
    conv.u = tmp[idx];
    return conv.b;
  }
};

// Note: max_size<bfloat16_t> would typically be defined elsewhere
// Keep the existing definition for compatibility

// BF16 arithmetic requires conversion to FP32, operate, then convert back
// This is less efficient than FP16 but still useful for memory bandwidth

#define DEFINE_AVX512_BF16_BINARY_OP_VIA_FP32(name, op)                       \
  inline Simd<bfloat16_t, 32> name(                                           \
      Simd<bfloat16_t, 32> a, Simd<bfloat16_t, 32> b) {                       \
    /* Convert BF16 to FP32 (2x 512-bit vectors) */                           \
    __m256bh a_lo = (__m256bh)_mm512_extracti64x4_epi64((__m512i)a.value, 0); \
    __m256bh a_hi = (__m256bh)_mm512_extracti64x4_epi64((__m512i)a.value, 1); \
    __m256bh b_lo = (__m256bh)_mm512_extracti64x4_epi64((__m512i)b.value, 0); \
    __m256bh b_hi = (__m256bh)_mm512_extracti64x4_epi64((__m512i)b.value, 1); \
    __m512 a_lo_f32 = _mm512_cvtpbh_ps(a_lo);                                 \
    __m512 a_hi_f32 = _mm512_cvtpbh_ps(a_hi);                                 \
    __m512 b_lo_f32 = _mm512_cvtpbh_ps(b_lo);                                 \
    __m512 b_hi_f32 = _mm512_cvtpbh_ps(b_hi);                                 \
    /* Perform operation in FP32 */                                           \
    Simd<float, 16> result_lo =                                               \
        op(Simd<float, 16>(a_lo_f32), Simd<float, 16>(b_lo_f32));             \
    Simd<float, 16> result_hi =                                               \
        op(Simd<float, 16>(a_hi_f32), Simd<float, 16>(b_hi_f32));             \
    /* Convert back to BF16 */                                                \
    __m512bh result_bf16 =                                                    \
        _mm512_cvtne2ps_pbh(result_hi.value, result_lo.value);                \
    return result_bf16;                                                       \
  }

#define DEFINE_AVX512_BF16_UNARY_OP_VIA_FP32(name, op)                        \
  inline Simd<bfloat16_t, 32> name(Simd<bfloat16_t, 32> a) {                  \
    /* Convert BF16 to FP32 (2x 512-bit vectors) */                           \
    __m256bh a_lo = (__m256bh)_mm512_extracti64x4_epi64((__m512i)a.value, 0); \
    __m256bh a_hi = (__m256bh)_mm512_extracti64x4_epi64((__m512i)a.value, 1); \
    __m512 a_lo_f32 = _mm512_cvtpbh_ps(a_lo);                                 \
    __m512 a_hi_f32 = _mm512_cvtpbh_ps(a_hi);                                 \
    /* Perform operation in FP32 */                                           \
    Simd<float, 16> result_lo = op(Simd<float, 16>(a_lo_f32));                \
    Simd<float, 16> result_hi = op(Simd<float, 16>(a_hi_f32));                \
    /* Convert back to BF16 */                                                \
    __m512bh result_bf16 =                                                    \
        _mm512_cvtne2ps_pbh(result_hi.value, result_lo.value);                \
    return result_bf16;                                                       \
  }

// Arithmetic operations (via FP32)
DEFINE_AVX512_BF16_BINARY_OP_VIA_FP32(operator+, operator+)
DEFINE_AVX512_BF16_BINARY_OP_VIA_FP32(operator-, operator-)
DEFINE_AVX512_BF16_BINARY_OP_VIA_FP32(operator*, operator*)
DEFINE_AVX512_BF16_BINARY_OP_VIA_FP32(operator/, operator/)
DEFINE_AVX512_BF16_BINARY_OP_VIA_FP32(minimum, minimum)
DEFINE_AVX512_BF16_BINARY_OP_VIA_FP32(maximum, maximum)

// Unary operations (via FP32)
DEFINE_AVX512_BF16_UNARY_OP_VIA_FP32(abs, abs)
DEFINE_AVX512_BF16_UNARY_OP_VIA_FP32(sqrt, sqrt)

inline Simd<bfloat16_t, 32> operator-(Simd<bfloat16_t, 32> a) {
  // Negate by converting to FP32, negating, and converting back
  __m256bh a_lo = (__m256bh)_mm512_extracti64x4_epi64((__m512i)a.value, 0);
  __m256bh a_hi = (__m256bh)_mm512_extracti64x4_epi64((__m512i)a.value, 1);
  __m512 a_lo_f32 = _mm512_cvtpbh_ps(a_lo);
  __m512 a_hi_f32 = _mm512_cvtpbh_ps(a_hi);
  __m512 neg_lo = _mm512_sub_ps(_mm512_setzero_ps(), a_lo_f32);
  __m512 neg_hi = _mm512_sub_ps(_mm512_setzero_ps(), a_hi_f32);
  return _mm512_cvtne2ps_pbh(neg_hi, neg_lo);
}

// FMA using specialized BF16 dot product instruction
// Note: This is actually a dot product, not element-wise FMA
// For true element-wise FMA, we'd need to go via FP32
template <typename T>
Simd<bfloat16_t, 32> fma(Simd<bfloat16_t, 32> a, Simd<bfloat16_t, 32> b, T c) {
  // Convert to FP32, perform FMA, convert back
  __m256bh a_lo = (__m256bh)_mm512_extracti64x4_epi64((__m512i)a.value, 0);
  __m256bh a_hi = (__m256bh)_mm512_extracti64x4_epi64((__m512i)a.value, 1);
  __m256bh b_lo = (__m256bh)_mm512_extracti64x4_epi64((__m512i)b.value, 0);
  __m256bh b_hi = (__m256bh)_mm512_extracti64x4_epi64((__m512i)b.value, 1);

  __m512 a_lo_f32 = _mm512_cvtpbh_ps(a_lo);
  __m512 a_hi_f32 = _mm512_cvtpbh_ps(a_hi);
  __m512 b_lo_f32 = _mm512_cvtpbh_ps(b_lo);
  __m512 b_hi_f32 = _mm512_cvtpbh_ps(b_hi);

  Simd<float, 16> result_lo =
      fma(Simd<float, 16>(a_lo_f32),
          Simd<float, 16>(b_lo_f32),
          static_cast<float>(c));
  Simd<float, 16> result_hi =
      fma(Simd<float, 16>(a_hi_f32),
          Simd<float, 16>(b_hi_f32),
          static_cast<float>(c));

  return _mm512_cvtne2ps_pbh(result_hi.value, result_lo.value);
}

// Reductions - convert to FP32, reduce, convert back
inline bfloat16_t sum(Simd<bfloat16_t, 32> x) {
  __m256bh x_lo = (__m256bh)_mm512_extracti64x4_epi64((__m512i)x.value, 0);
  __m256bh x_hi = (__m256bh)_mm512_extracti64x4_epi64((__m512i)x.value, 1);
  __m512 x_lo_f32 = _mm512_cvtpbh_ps(x_lo);
  __m512 x_hi_f32 = _mm512_cvtpbh_ps(x_hi);
  float sum_f32 =
      sum(Simd<float, 16>(x_lo_f32)) + sum(Simd<float, 16>(x_hi_f32));
  return static_cast<bfloat16_t>(sum_f32);
}

inline bfloat16_t min(Simd<bfloat16_t, 32> x) {
  __m256bh x_lo = (__m256bh)_mm512_extracti64x4_epi64((__m512i)x.value, 0);
  __m256bh x_hi = (__m256bh)_mm512_extracti64x4_epi64((__m512i)x.value, 1);
  __m512 x_lo_f32 = _mm512_cvtpbh_ps(x_lo);
  __m512 x_hi_f32 = _mm512_cvtpbh_ps(x_hi);
  float min_lo = min(Simd<float, 16>(x_lo_f32));
  float min_hi = min(Simd<float, 16>(x_hi_f32));
  return static_cast<bfloat16_t>(std::min(min_lo, min_hi));
}

inline bfloat16_t max(Simd<bfloat16_t, 32> x) {
  __m256bh x_lo = (__m256bh)_mm512_extracti64x4_epi64((__m512i)x.value, 0);
  __m256bh x_hi = (__m256bh)_mm512_extracti64x4_epi64((__m512i)x.value, 1);
  __m512 x_lo_f32 = _mm512_cvtpbh_ps(x_lo);
  __m512 x_hi_f32 = _mm512_cvtpbh_ps(x_hi);
  float max_lo = max(Simd<float, 16>(x_lo_f32));
  float max_hi = max(Simd<float, 16>(x_hi_f32));
  return static_cast<bfloat16_t>(std::max(max_lo, max_hi));
}

inline bfloat16_t prod(Simd<bfloat16_t, 32> x) {
  __m256bh x_lo = (__m256bh)_mm512_extracti64x4_epi64((__m512i)x.value, 0);
  __m256bh x_hi = (__m256bh)_mm512_extracti64x4_epi64((__m512i)x.value, 1);
  __m512 x_lo_f32 = _mm512_cvtpbh_ps(x_lo);
  __m512 x_hi_f32 = _mm512_cvtpbh_ps(x_hi);
  float prod_f32 =
      prod(Simd<float, 16>(x_lo_f32)) * prod(Simd<float, 16>(x_hi_f32));
  return static_cast<bfloat16_t>(prod_f32);
}

// Comparison operators (return Simd<bool, 32> for MLX compatibility)
// Convert to FP32, compare, combine masks
inline Simd<bool, 32> operator==(
    Simd<bfloat16_t, 32> a,
    Simd<bfloat16_t, 32> b) {
  __m256bh a_lo = (__m256bh)_mm512_extracti64x4_epi64((__m512i)a.value, 0);
  __m256bh a_hi = (__m256bh)_mm512_extracti64x4_epi64((__m512i)a.value, 1);
  __m256bh b_lo = (__m256bh)_mm512_extracti64x4_epi64((__m512i)b.value, 0);
  __m256bh b_hi = (__m256bh)_mm512_extracti64x4_epi64((__m512i)b.value, 1);

  __m512 a_lo_f32 = _mm512_cvtpbh_ps(a_lo);
  __m512 a_hi_f32 = _mm512_cvtpbh_ps(a_hi);
  __m512 b_lo_f32 = _mm512_cvtpbh_ps(b_lo);
  __m512 b_hi_f32 = _mm512_cvtpbh_ps(b_hi);

  __mmask16 cmp_lo = _mm512_cmp_ps_mask(a_lo_f32, b_lo_f32, _CMP_EQ_OQ);
  __mmask16 cmp_hi = _mm512_cmp_ps_mask(a_hi_f32, b_hi_f32, _CMP_EQ_OQ);

  return Simd<bool, 32>((__mmask32)cmp_lo | ((__mmask32)cmp_hi << 16));
}

inline Simd<bool, 32> operator<(
    Simd<bfloat16_t, 32> a,
    Simd<bfloat16_t, 32> b) {
  __m256bh a_lo = (__m256bh)_mm512_extracti64x4_epi64((__m512i)a.value, 0);
  __m256bh a_hi = (__m256bh)_mm512_extracti64x4_epi64((__m512i)a.value, 1);
  __m256bh b_lo = (__m256bh)_mm512_extracti64x4_epi64((__m512i)b.value, 0);
  __m256bh b_hi = (__m256bh)_mm512_extracti64x4_epi64((__m512i)b.value, 1);

  __m512 a_lo_f32 = _mm512_cvtpbh_ps(a_lo);
  __m512 a_hi_f32 = _mm512_cvtpbh_ps(a_hi);
  __m512 b_lo_f32 = _mm512_cvtpbh_ps(b_lo);
  __m512 b_hi_f32 = _mm512_cvtpbh_ps(b_hi);

  __mmask16 cmp_lo = _mm512_cmp_ps_mask(a_lo_f32, b_lo_f32, _CMP_LT_OQ);
  __mmask16 cmp_hi = _mm512_cmp_ps_mask(a_hi_f32, b_hi_f32, _CMP_LT_OQ);

  return Simd<bool, 32>((__mmask32)cmp_lo | ((__mmask32)cmp_hi << 16));
}

inline Simd<bool, 32> operator<=(
    Simd<bfloat16_t, 32> a,
    Simd<bfloat16_t, 32> b) {
  __m256bh a_lo = (__m256bh)_mm512_extracti64x4_epi64((__m512i)a.value, 0);
  __m256bh a_hi = (__m256bh)_mm512_extracti64x4_epi64((__m512i)a.value, 1);
  __m256bh b_lo = (__m256bh)_mm512_extracti64x4_epi64((__m512i)b.value, 0);
  __m256bh b_hi = (__m256bh)_mm512_extracti64x4_epi64((__m512i)b.value, 1);

  __m512 a_lo_f32 = _mm512_cvtpbh_ps(a_lo);
  __m512 a_hi_f32 = _mm512_cvtpbh_ps(a_hi);
  __m512 b_lo_f32 = _mm512_cvtpbh_ps(b_lo);
  __m512 b_hi_f32 = _mm512_cvtpbh_ps(b_hi);

  __mmask16 cmp_lo = _mm512_cmp_ps_mask(a_lo_f32, b_lo_f32, _CMP_LE_OQ);
  __mmask16 cmp_hi = _mm512_cmp_ps_mask(a_hi_f32, b_hi_f32, _CMP_LE_OQ);

  return Simd<bool, 32>((__mmask32)cmp_lo | ((__mmask32)cmp_hi << 16));
}

inline Simd<bool, 32> operator>(
    Simd<bfloat16_t, 32> a,
    Simd<bfloat16_t, 32> b) {
  __m256bh a_lo = (__m256bh)_mm512_extracti64x4_epi64((__m512i)a.value, 0);
  __m256bh a_hi = (__m256bh)_mm512_extracti64x4_epi64((__m512i)a.value, 1);
  __m256bh b_lo = (__m256bh)_mm512_extracti64x4_epi64((__m512i)b.value, 0);
  __m256bh b_hi = (__m256bh)_mm512_extracti64x4_epi64((__m512i)b.value, 1);

  __m512 a_lo_f32 = _mm512_cvtpbh_ps(a_lo);
  __m512 a_hi_f32 = _mm512_cvtpbh_ps(a_hi);
  __m512 b_lo_f32 = _mm512_cvtpbh_ps(b_lo);
  __m512 b_hi_f32 = _mm512_cvtpbh_ps(b_hi);

  __mmask16 cmp_lo = _mm512_cmp_ps_mask(a_lo_f32, b_lo_f32, _CMP_GT_OQ);
  __mmask16 cmp_hi = _mm512_cmp_ps_mask(a_hi_f32, b_hi_f32, _CMP_GT_OQ);

  return Simd<bool, 32>((__mmask32)cmp_lo | ((__mmask32)cmp_hi << 16));
}

inline Simd<bool, 32> operator>=(
    Simd<bfloat16_t, 32> a,
    Simd<bfloat16_t, 32> b) {
  __m256bh a_lo = (__m256bh)_mm512_extracti64x4_epi64((__m512i)a.value, 0);
  __m256bh a_hi = (__m256bh)_mm512_extracti64x4_epi64((__m512i)a.value, 1);
  __m256bh b_lo = (__m256bh)_mm512_extracti64x4_epi64((__m512i)b.value, 0);
  __m256bh b_hi = (__m256bh)_mm512_extracti64x4_epi64((__m512i)b.value, 1);

  __m512 a_lo_f32 = _mm512_cvtpbh_ps(a_lo);
  __m512 a_hi_f32 = _mm512_cvtpbh_ps(a_hi);
  __m512 b_lo_f32 = _mm512_cvtpbh_ps(b_lo);
  __m512 b_hi_f32 = _mm512_cvtpbh_ps(b_hi);

  __mmask16 cmp_lo = _mm512_cmp_ps_mask(a_lo_f32, b_lo_f32, _CMP_GE_OQ);
  __mmask16 cmp_hi = _mm512_cmp_ps_mask(a_hi_f32, b_hi_f32, _CMP_GE_OQ);

  return Simd<bool, 32>((__mmask32)cmp_lo | ((__mmask32)cmp_hi << 16));
}

inline Simd<bool, 32> operator!=(
    Simd<bfloat16_t, 32> a,
    Simd<bfloat16_t, 32> b) {
  return !(a == b);
}

// Rounding operations (via FP32)
inline Simd<bfloat16_t, 32> ceil(Simd<bfloat16_t, 32> x) {
  __m256bh x_lo = (__m256bh)_mm512_extracti64x4_epi64((__m512i)x.value, 0);
  __m256bh x_hi = (__m256bh)_mm512_extracti64x4_epi64((__m512i)x.value, 1);
  __m512 x_lo_f32 = _mm512_cvtpbh_ps(x_lo);
  __m512 x_hi_f32 = _mm512_cvtpbh_ps(x_hi);
  Simd<float, 16> result_lo = ceil(Simd<float, 16>(x_lo_f32));
  Simd<float, 16> result_hi = ceil(Simd<float, 16>(x_hi_f32));
  return _mm512_cvtne2ps_pbh(result_hi.value, result_lo.value);
}

inline Simd<bfloat16_t, 32> floor(Simd<bfloat16_t, 32> x) {
  __m256bh x_lo = (__m256bh)_mm512_extracti64x4_epi64((__m512i)x.value, 0);
  __m256bh x_hi = (__m256bh)_mm512_extracti64x4_epi64((__m512i)x.value, 1);
  __m512 x_lo_f32 = _mm512_cvtpbh_ps(x_lo);
  __m512 x_hi_f32 = _mm512_cvtpbh_ps(x_hi);
  Simd<float, 16> result_lo = floor(Simd<float, 16>(x_lo_f32));
  Simd<float, 16> result_hi = floor(Simd<float, 16>(x_hi_f32));
  return _mm512_cvtne2ps_pbh(result_hi.value, result_lo.value);
}

inline Simd<bfloat16_t, 32> rint(Simd<bfloat16_t, 32> x) {
  __m256bh x_lo = (__m256bh)_mm512_extracti64x4_epi64((__m512i)x.value, 0);
  __m256bh x_hi = (__m256bh)_mm512_extracti64x4_epi64((__m512i)x.value, 1);
  __m512 x_lo_f32 = _mm512_cvtpbh_ps(x_lo);
  __m512 x_hi_f32 = _mm512_cvtpbh_ps(x_hi);
  Simd<float, 16> result_lo = rint(Simd<float, 16>(x_lo_f32));
  Simd<float, 16> result_hi = rint(Simd<float, 16>(x_hi_f32));
  return _mm512_cvtne2ps_pbh(result_hi.value, result_lo.value);
}

// Additional unary operations via FP32
inline Simd<bfloat16_t, 32> rsqrt(Simd<bfloat16_t, 32> x) {
  __m256bh x_lo = (__m256bh)_mm512_extracti64x4_epi64((__m512i)x.value, 0);
  __m256bh x_hi = (__m256bh)_mm512_extracti64x4_epi64((__m512i)x.value, 1);
  __m512 x_lo_f32 = _mm512_cvtpbh_ps(x_lo);
  __m512 x_hi_f32 = _mm512_cvtpbh_ps(x_hi);
  Simd<float, 16> result_lo = rsqrt(Simd<float, 16>(x_lo_f32));
  Simd<float, 16> result_hi = rsqrt(Simd<float, 16>(x_hi_f32));
  return _mm512_cvtne2ps_pbh(result_hi.value, result_lo.value);
}

inline Simd<bfloat16_t, 32> recip(Simd<bfloat16_t, 32> x) {
  __m256bh x_lo = (__m256bh)_mm512_extracti64x4_epi64((__m512i)x.value, 0);
  __m256bh x_hi = (__m256bh)_mm512_extracti64x4_epi64((__m512i)x.value, 1);
  __m512 x_lo_f32 = _mm512_cvtpbh_ps(x_lo);
  __m512 x_hi_f32 = _mm512_cvtpbh_ps(x_hi);
  Simd<float, 16> result_lo = recip(Simd<float, 16>(x_lo_f32));
  Simd<float, 16> result_hi = recip(Simd<float, 16>(x_hi_f32));
  return _mm512_cvtne2ps_pbh(result_hi.value, result_lo.value);
}

// Utility functions (matching NEON/MLX patterns)
inline Simd<bool, 32> isnan(Simd<bfloat16_t, 32> v) {
  return v != v; // NaN != NaN
}

inline Simd<bfloat16_t, 32> clamp(
    Simd<bfloat16_t, 32> v,
    Simd<bfloat16_t, 32> min_val,
    Simd<bfloat16_t, 32> max_val) {
  return minimum(maximum(v, min_val), max_val);
}

template <typename MaskT>
inline Simd<bfloat16_t, 32>
select(Simd<MaskT, 32> mask, Simd<bfloat16_t, 32> x, Simd<bfloat16_t, 32> y) {
  // Convert mask to __mmask32 and use integer blend (no native BF16 blend)
  __mmask32 m = mask.value;
  __m512i x_bits = (__m512i)x.value;
  __m512i y_bits = (__m512i)y.value;
  __m512i result = _mm512_mask_blend_epi16(m, y_bits, x_bits);
  return (__m512bh)result;
}

// Logical operations
inline Simd<bfloat16_t, 32> operator!(Simd<bfloat16_t, 32> v) {
  auto zero = Simd<bfloat16_t, 32>(static_cast<bfloat16_t>(0));
  return select(
      v == zero, Simd<bfloat16_t, 32>(static_cast<bfloat16_t>(1)), zero);
}

inline Simd<bfloat16_t, 32> operator&&(
    Simd<bfloat16_t, 32> a,
    Simd<bfloat16_t, 32> b) {
  auto zero = Simd<bfloat16_t, 32>(static_cast<bfloat16_t>(0));
  auto one = Simd<bfloat16_t, 32>(static_cast<bfloat16_t>(1));
  return select((a != zero) && (b != zero), one, zero);
}

inline Simd<bfloat16_t, 32> operator||(
    Simd<bfloat16_t, 32> a,
    Simd<bfloat16_t, 32> b) {
  auto zero = Simd<bfloat16_t, 32>(static_cast<bfloat16_t>(0));
  auto one = Simd<bfloat16_t, 32>(static_cast<bfloat16_t>(1));
  return select((a != zero) || (b != zero), one, zero);
}

// Specialized BF16 dot product instruction (hardware-accelerated)
// This is the main performance benefit of BF16 on AVX-512
inline Simd<float, 16> bf16_dot_product(
    Simd<float, 16> acc,
    Simd<bfloat16_t, 32> a,
    Simd<bfloat16_t, 32> b) {
  return _mm512_dpbf16_ps(acc.value, a.value, b.value);
}

#undef DEFINE_AVX512_BF16_UNARY_OP_VIA_FP32
#undef DEFINE_AVX512_BF16_BINARY_OP_VIA_FP32

#endif // __AVX512BF16__

// ============================================================================
// AVX-512 FP16 (Native Float16 Arithmetic with Simd<> Integration)
// ============================================================================
//
// AVX-512 FP16 provides native hardware Float16 operations.
// This replaces the scalar fallback Simd<float16_t, 16> with native __m512h.
//
// Format: IEEE 754 half-precision (1 sign + 5 exponent + 10 mantissa bits)
// Register: 32x FP16 values in 512-bit register (vs 16x FP32)
// Performance: ~4x speedup potential (2x throughput + 2x bandwidth)
//
// Available on: Intel Sapphire Rapids (2023+), AMD Zen 4 (2022+)

#ifdef __AVX512FP16__

// AVX-512 FP16 native specialization (32-wide, matching NEON's pattern)
template <>
struct Simd<float16_t, 32> {
  static constexpr int size = 32;
  using scalar_t = float16_t;

  __m512h value;

  Simd() : value(_mm512_setzero_ph()) {}

  Simd(__m512h v) : value(v) {}

  Simd(float16_t v)
      : value(_mm512_set1_ph(static_cast<_Float16>(static_cast<float>(v)))) {}

  // Generic numeric constructor
  template <
      typename U,
      typename = std::enable_if_t<
          std::is_arithmetic_v<U> && !std::is_same_v<U, float16_t>>>
  Simd(U v)
      : value(_mm512_set1_ph(static_cast<_Float16>(
            static_cast<float>(static_cast<float16_t>(v))))) {}

  // Array access
  float16_t operator[](int idx) const {
    alignas(64) float16_t tmp[32];
    _mm512_store_ph(tmp, value);
    return tmp[idx];
  }

  float16_t& operator[](int idx) {
    // Note: Non-const access requires load-modify-store pattern
    // This is a limitation of SIMD - use sparingly
    alignas(64) static float16_t tmp[32];
    _mm512_store_ph(tmp, value);
    value = _mm512_load_ph(tmp); // Ensure value is synced
    return tmp[idx];
  }
};

// Note: max_size<float16_t> is already defined as 16 earlier in the file
// When FP16 hardware is available, we provide 32-wide operations
// but keep the existing 16-wide as the "max" for compatibility

// Macros for DRY code (following NEON pattern)
#define DEFINE_AVX512_FP16_UNARY_OP(name, intrinsic)       \
  inline Simd<float16_t, 32> name(Simd<float16_t, 32> a) { \
    return intrinsic(a.value);                             \
  }

#define DEFINE_AVX512_FP16_BINARY_OP(name, intrinsic) \
  inline Simd<float16_t, 32> name(                    \
      Simd<float16_t, 32> a, Simd<float16_t, 32> b) { \
    return intrinsic(a.value, b.value);               \
  }

// Unary operations
DEFINE_AVX512_FP16_UNARY_OP(abs, _mm512_abs_ph)
DEFINE_AVX512_FP16_UNARY_OP(sqrt, _mm512_sqrt_ph)
DEFINE_AVX512_FP16_UNARY_OP(rsqrt, _mm512_rsqrt_ph)
DEFINE_AVX512_FP16_UNARY_OP(recip, _mm512_rcp_ph)

inline Simd<float16_t, 32> operator-(Simd<float16_t, 32> a) {
  return _mm512_sub_ph(_mm512_setzero_ph(), a.value);
}

// Binary arithmetic operations (matching MLX naming)
DEFINE_AVX512_FP16_BINARY_OP(operator+, _mm512_add_ph)
DEFINE_AVX512_FP16_BINARY_OP(operator-, _mm512_sub_ph)
DEFINE_AVX512_FP16_BINARY_OP(operator*, _mm512_mul_ph)
DEFINE_AVX512_FP16_BINARY_OP(operator/, _mm512_div_ph)
DEFINE_AVX512_FP16_BINARY_OP(minimum, _mm512_min_ph)
DEFINE_AVX512_FP16_BINARY_OP(maximum, _mm512_max_ph)

// FMA (fused multiply-add): a * b + c
template <typename T>
Simd<float16_t, 32> fma(Simd<float16_t, 32> a, Simd<float16_t, 32> b, T c) {
  return _mm512_fmadd_ph(a.value, b.value, Simd<float16_t, 32>(c).value);
}

// Comparisons (return Simd<bool, 32> for MLX compatibility)
inline Simd<bool, 32> operator==(Simd<float16_t, 32> a, Simd<float16_t, 32> b) {
  return Simd<bool, 32>(_mm512_cmp_ph_mask(a.value, b.value, _CMP_EQ_OQ));
}

inline Simd<bool, 32> operator<(Simd<float16_t, 32> a, Simd<float16_t, 32> b) {
  return Simd<bool, 32>(_mm512_cmp_ph_mask(a.value, b.value, _CMP_LT_OQ));
}

inline Simd<bool, 32> operator<=(Simd<float16_t, 32> a, Simd<float16_t, 32> b) {
  return Simd<bool, 32>(_mm512_cmp_ph_mask(a.value, b.value, _CMP_LE_OQ));
}

inline Simd<bool, 32> operator>(Simd<float16_t, 32> a, Simd<float16_t, 32> b) {
  return Simd<bool, 32>(_mm512_cmp_ph_mask(a.value, b.value, _CMP_GT_OQ));
}

inline Simd<bool, 32> operator>=(Simd<float16_t, 32> a, Simd<float16_t, 32> b) {
  return Simd<bool, 32>(_mm512_cmp_ph_mask(a.value, b.value, _CMP_GE_OQ));
}

inline Simd<bool, 32> operator!=(Simd<float16_t, 32> a, Simd<float16_t, 32> b) {
  return !(a == b);
}

// Reductions (matching MLX naming: sum, min, max, prod)
inline float16_t sum(Simd<float16_t, 32> x) {
  return static_cast<float16_t>(_mm512_reduce_add_ph(x.value));
}

inline float16_t min(Simd<float16_t, 32> x) {
  return static_cast<float16_t>(_mm512_reduce_min_ph(x.value));
}

inline float16_t max(Simd<float16_t, 32> x) {
  return static_cast<float16_t>(_mm512_reduce_max_ph(x.value));
}

inline float16_t prod(Simd<float16_t, 32> x) {
  return static_cast<float16_t>(_mm512_reduce_mul_ph(x.value));
}

// Rounding operations
inline Simd<float16_t, 32> ceil(Simd<float16_t, 32> x) {
  return _mm512_roundscale_ph(
      x.value, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}

inline Simd<float16_t, 32> floor(Simd<float16_t, 32> x) {
  return _mm512_roundscale_ph(
      x.value, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

inline Simd<float16_t, 32> rint(Simd<float16_t, 32> x) {
  return _mm512_roundscale_ph(
      x.value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

// Utility functions (matching NEON/MLX patterns)
inline Simd<bool, 32> isnan(Simd<float16_t, 32> v) {
  return v != v; // NaN != NaN
}

inline Simd<float16_t, 32> clamp(
    Simd<float16_t, 32> v,
    Simd<float16_t, 32> min_val,
    Simd<float16_t, 32> max_val) {
  return minimum(maximum(v, min_val), max_val);
}

template <typename MaskT>
inline Simd<float16_t, 32>
select(Simd<MaskT, 32> mask, Simd<float16_t, 32> x, Simd<float16_t, 32> y) {
  // Convert mask to __mmask32
  __mmask32 m = mask.value;
  return _mm512_mask_blend_ph(m, y.value, x.value);
}

// Logical operations
inline Simd<float16_t, 32> operator!(Simd<float16_t, 32> v) {
  auto zero = Simd<float16_t, 32>(static_cast<float16_t>(0));
  return select(
      v == zero, Simd<float16_t, 32>(static_cast<float16_t>(1)), zero);
}

inline Simd<float16_t, 32> operator&&(
    Simd<float16_t, 32> a,
    Simd<float16_t, 32> b) {
  auto zero = Simd<float16_t, 32>(static_cast<float16_t>(0));
  auto one = Simd<float16_t, 32>(static_cast<float16_t>(1));
  return select((a != zero) && (b != zero), one, zero);
}

inline Simd<float16_t, 32> operator||(
    Simd<float16_t, 32> a,
    Simd<float16_t, 32> b) {
  auto zero = Simd<float16_t, 32>(static_cast<float16_t>(0));
  auto one = Simd<float16_t, 32>(static_cast<float16_t>(1));
  return select((a != zero) || (b != zero), one, zero);
}

// Clean up macros
#undef DEFINE_AVX512_FP16_UNARY_OP
#undef DEFINE_AVX512_FP16_BINARY_OP

#endif // __AVX512FP16__

} // namespace mlx::core::simd

#endif // __AVX512F__
