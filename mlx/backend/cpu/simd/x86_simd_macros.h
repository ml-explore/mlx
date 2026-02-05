// Copyright © 2026 Apple Inc.

#pragma once

// x86 SIMD common macro definitions.
// Shared macros for generating Simd<T,N> operator overloads and math functions
// across SSE, AVX2, and AVX-512 backends.

// Bool operators using integer SIMD intrinsics
#define DEFINE_X86_BOOL_OPS_VECTOR(N, reg_type, prefix, elem_width)           \
  inline Simd<bool, N> operator!(Simd<bool, N> a) {                           \
    return Simd<bool, N>(                                                     \
        prefix##_xor_##reg_type(a.value, prefix##_set1_epi##elem_width(-1))); \
  }                                                                           \
  inline Simd<bool, N> operator||(Simd<bool, N> a, Simd<bool, N> b) {         \
    return Simd<bool, N>(prefix##_or_##reg_type(a.value, b.value));           \
  }                                                                           \
  inline Simd<bool, N> operator&&(Simd<bool, N> a, Simd<bool, N> b) {         \
    return Simd<bool, N>(prefix##_and_##reg_type(a.value, b.value));          \
  }                                                                           \
  inline Simd<bool, N> operator&(Simd<bool, N> a, Simd<bool, N> b) {          \
    return Simd<bool, N>(prefix##_and_##reg_type(a.value, b.value));          \
  }                                                                           \
  inline Simd<bool, N> operator|(Simd<bool, N> a, Simd<bool, N> b) {          \
    return Simd<bool, N>(prefix##_or_##reg_type(a.value, b.value));           \
  }                                                                           \
  inline Simd<bool, N> operator^(Simd<bool, N> a, Simd<bool, N> b) {          \
    return Simd<bool, N>(prefix##_xor_##reg_type(a.value, b.value));          \
  }                                                                           \
  inline Simd<bool, N> operator==(Simd<bool, N> a, Simd<bool, N> b) {         \
    return !Simd<bool, N>(prefix##_xor_##reg_type(a.value, b.value));         \
  }                                                                           \
  inline Simd<bool, N> operator!=(Simd<bool, N> a, Simd<bool, N> b) {         \
    return !(a == b);                                                         \
  }

// Binary operator with direct intrinsic
#define DEFINE_X86_BINARY_OP(op, type, N, intrinsic)                   \
  inline Simd<type, N> operator op(Simd<type, N> a, Simd<type, N> b) { \
    return intrinsic(a.value, b.value);                                \
  }

// Unary function with direct intrinsic
#define DEFINE_X86_UNARY_OP(name, type, N, intrinsic) \
  inline Simd<type, N> name(Simd<type, N> a) {        \
    return intrinsic(a.value);                        \
  }

// Comparison operators using _mm*_cmp_* with predicate flags.
// != uses _CMP_NEQ_UQ (unordered) so NaN != NaN is true.
#define DEFINE_X86_COMPARISONS_CMP(type, N, prefix, reg_suffix, bool_cast)    \
  inline Simd<bool, N> operator<(Simd<type, N> a, Simd<type, N> b) {          \
    return Simd<bool, N>(                                                     \
        bool_cast(prefix##_cmp_##reg_suffix(a.value, b.value, _CMP_LT_OQ)));  \
  }                                                                           \
  inline Simd<bool, N> operator>(Simd<type, N> a, Simd<type, N> b) {          \
    return Simd<bool, N>(                                                     \
        bool_cast(prefix##_cmp_##reg_suffix(a.value, b.value, _CMP_GT_OQ)));  \
  }                                                                           \
  inline Simd<bool, N> operator<=(Simd<type, N> a, Simd<type, N> b) {         \
    return Simd<bool, N>(                                                     \
        bool_cast(prefix##_cmp_##reg_suffix(a.value, b.value, _CMP_LE_OQ)));  \
  }                                                                           \
  inline Simd<bool, N> operator>=(Simd<type, N> a, Simd<type, N> b) {         \
    return Simd<bool, N>(                                                     \
        bool_cast(prefix##_cmp_##reg_suffix(a.value, b.value, _CMP_GE_OQ)));  \
  }                                                                           \
  inline Simd<bool, N> operator==(Simd<type, N> a, Simd<type, N> b) {         \
    return Simd<bool, N>(                                                     \
        bool_cast(prefix##_cmp_##reg_suffix(a.value, b.value, _CMP_EQ_OQ)));  \
  }                                                                           \
  inline Simd<bool, N> operator!=(Simd<type, N> a, Simd<type, N> b) {         \
    return Simd<bool, N>(                                                     \
        bool_cast(prefix##_cmp_##reg_suffix(a.value, b.value, _CMP_NEQ_UQ))); \
  }

// Transcendental functions via scalar loop (no SIMD intrinsics available)
#define DEFINE_X86_TRANSCENDENTAL(name, type, N, alignment, func) \
  inline Simd<type, N> name(Simd<type, N> a) {                    \
    alignas(alignment) type tmp[N];                               \
    store<type, N>(tmp, a);                                       \
    for (int i = 0; i < N; ++i) {                                 \
      tmp[i] = func(tmp[i]);                                      \
    }                                                             \
    return load<type, N>(tmp);                                    \
  }

// Binary transcendental via scalar loop
#define DEFINE_X86_BINARY_TRANSCENDENTAL(name, type, N, alignment, func) \
  inline Simd<type, N> name(Simd<type, N> a, Simd<type, N> b) {          \
    alignas(alignment) type tmp_a[N], tmp_b[N];                          \
    store<type, N>(tmp_a, a);                                            \
    store<type, N>(tmp_b, b);                                            \
    for (int i = 0; i < N; ++i) {                                        \
      tmp_a[i] = func(tmp_a[i], tmp_b[i]);                               \
    }                                                                    \
    return load<type, N>(tmp_a);                                         \
  }

// Half-type operations (float16/bfloat16): convert to float32, operate,
// convert back. Used for both float16_t and bfloat16_t.
#define DEFINE_X86_HALF_SCALAR_MUL(half_type)                                \
  inline Simd<half_type, 8> operator*(                                       \
      Simd<half_type, 8> a, half_type scalar) {                              \
    return Simd<half_type, 8>(                                               \
        Simd<float, 8>(a) * Simd<float, 8>(static_cast<float>(scalar)));     \
  }                                                                          \
  inline Simd<half_type, 8> operator*(Simd<half_type, 8> a, float scalar) {  \
    return Simd<half_type, 8>(Simd<float, 8>(a) * Simd<float, 8>(scalar));   \
  }                                                                          \
  inline Simd<half_type, 8> operator*(Simd<half_type, 8> a, double scalar) { \
    return Simd<half_type, 8>(                                               \
        Simd<float, 8>(a) * Simd<float, 8>(static_cast<float>(scalar)));     \
  }                                                                          \
  inline Simd<half_type, 8> operator*(Simd<half_type, 8> a, int scalar) {    \
    return Simd<half_type, 8>(                                               \
        Simd<float, 8>(a) * Simd<float, 8>(static_cast<float>(scalar)));     \
  }

// Binary arithmetic operators (+, -, *, /, unary -) via float32
#define DEFINE_X86_HALF_BINARY_ARITHMETIC(half_type)                  \
  inline Simd<half_type, 8> operator+(                                \
      Simd<half_type, 8> a, Simd<half_type, 8> b) {                   \
    return Simd<half_type, 8>(Simd<float, 8>(a) + Simd<float, 8>(b)); \
  }                                                                   \
  inline Simd<half_type, 8> operator-(                                \
      Simd<half_type, 8> a, Simd<half_type, 8> b) {                   \
    return Simd<half_type, 8>(Simd<float, 8>(a) - Simd<float, 8>(b)); \
  }                                                                   \
  inline Simd<half_type, 8> operator*(                                \
      Simd<half_type, 8> a, Simd<half_type, 8> b) {                   \
    return Simd<half_type, 8>(Simd<float, 8>(a) * Simd<float, 8>(b)); \
  }                                                                   \
  inline Simd<half_type, 8> operator/(                                \
      Simd<half_type, 8> a, Simd<half_type, 8> b) {                   \
    return Simd<half_type, 8>(Simd<float, 8>(a) / Simd<float, 8>(b)); \
  }                                                                   \
  inline Simd<half_type, 8> operator-(Simd<half_type, 8> a) {         \
    return Simd<half_type, 8>(-Simd<float, 8>(a));                    \
  }

// Comparison operators via float32 conversion
#define DEFINE_X86_HALF_COMPARISONS_VIA_FLOAT(half_type)                       \
  inline Simd<bool, 8> operator<(Simd<half_type, 8> a, Simd<half_type, 8> b) { \
    return Simd<float, 8>(a) < Simd<float, 8>(b);                              \
  }                                                                            \
  inline Simd<bool, 8> operator<=(                                             \
      Simd<half_type, 8> a, Simd<half_type, 8> b) {                            \
    return Simd<float, 8>(a) <= Simd<float, 8>(b);                             \
  }                                                                            \
  inline Simd<bool, 8> operator>(Simd<half_type, 8> a, Simd<half_type, 8> b) { \
    return b < a;                                                              \
  }                                                                            \
  inline Simd<bool, 8> operator>=(                                             \
      Simd<half_type, 8> a, Simd<half_type, 8> b) {                            \
    return b <= a;                                                             \
  }                                                                            \
  inline Simd<bool, 8> operator==(                                             \
      Simd<half_type, 8> a, Simd<half_type, 8> b) {                            \
    return Simd<float, 8>(a) == Simd<float, 8>(b);                             \
  }                                                                            \
  inline Simd<bool, 8> operator!=(                                             \
      Simd<half_type, 8> a, Simd<half_type, 8> b) {                            \
    return !(a == b);                                                          \
  }

// Single unary function via float32 (e.g., abs, sin, exp)
#define DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, op) \
  inline Simd<half_type, 8> op(Simd<half_type, 8> v) { \
    return Simd<half_type, 8>(op(Simd<float, 8>(v)));  \
  }

// Single binary function via float32 (e.g., atan2, pow, maximum)
#define DEFINE_X86_HALF_BINARY_VIA_FLOAT(half_type, op)                      \
  inline Simd<half_type, 8> op(Simd<half_type, 8> x, Simd<half_type, 8> y) { \
    return Simd<half_type, 8>(op(Simd<float, 8>(x), Simd<float, 8>(y)));     \
  }

// select, fma, isnan for half types
#define DEFINE_X86_HALF_SELECT_FMA_ISNAN(half_type)                       \
  inline Simd<half_type, 8> select(                                       \
      Simd<bool, 8> mask, Simd<half_type, 8> x, Simd<half_type, 8> y) {   \
    return Simd<half_type, 8>(                                            \
        select(mask, Simd<float, 8>(x), Simd<float, 8>(y)));              \
  }                                                                       \
  inline Simd<half_type, 8> fma(                                          \
      Simd<half_type, 8> a, Simd<half_type, 8> b, Simd<half_type, 8> c) { \
    return Simd<half_type, 8>(                                            \
        fma(Simd<float, 8>(a), Simd<float, 8>(b), Simd<float, 8>(c)));    \
  }                                                                       \
  inline Simd<bool, 8> isnan(Simd<half_type, 8> v) {                      \
    return isnan(Simd<float, 8>(v));                                      \
  }

// Logical operators (SIMD vectors and scalars)
#define DEFINE_X86_HALF_LOGICAL_OPS(half_type)                                \
  inline Simd<half_type, 8> operator!(Simd<half_type, 8> v) {                 \
    Simd<bool, 8> mask = v == Simd<float, 8>{0.0f};                           \
    return select(                                                            \
        mask,                                                                 \
        Simd<half_type, 8>{static_cast<half_type>(1)},                        \
        Simd<half_type, 8>{static_cast<half_type>(0)});                       \
  }                                                                           \
  inline Simd<half_type, 8> operator&&(                                       \
      Simd<half_type, 8> a, Simd<half_type, 8> b) {                           \
    Simd<bool, 8> mask =                                                      \
        (a != Simd<float, 8>{0.0f}) && (b != Simd<float, 8>{0.0f});           \
    return select(mask, a, Simd<half_type, 8>{static_cast<half_type>(0)});    \
  }                                                                           \
  inline Simd<half_type, 8> operator||(                                       \
      Simd<half_type, 8> a, Simd<half_type, 8> b) {                           \
    Simd<bool, 8> mask = a != Simd<float, 8>{0.0f};                           \
    return select(mask, a, b);                                                \
  }                                                                           \
  inline half_type operator&&(half_type a, half_type b) {                     \
    return (a != static_cast<half_type>(0) && b != static_cast<half_type>(0)) \
        ? static_cast<half_type>(1)                                           \
        : static_cast<half_type>(0);                                          \
  }                                                                           \
  inline half_type operator||(half_type a, half_type b) {                     \
    return (a != static_cast<half_type>(0) || b != static_cast<half_type>(0)) \
        ? static_cast<half_type>(1)                                           \
        : static_cast<half_type>(0);                                          \
  }                                                                           \
  inline half_type operator!(half_type v) {                                   \
    return v == static_cast<half_type>(0) ? static_cast<half_type>(1)         \
                                          : static_cast<half_type>(0);        \
  }

// All transcendental + math functions for a half type
#define DEFINE_X86_HALF_TRANSCENDENTALS(half_type)       \
  DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, abs)        \
  DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, acos)       \
  DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, acosh)      \
  DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, asin)       \
  DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, asinh)      \
  DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, atan)       \
  DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, atanh)      \
  DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, ceil)       \
  DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, cos)        \
  DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, cosh)       \
  DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, exp)        \
  DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, expm1)      \
  DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, floor)      \
  DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, log)        \
  DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, log2)       \
  DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, log10)      \
  DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, log1p)      \
  DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, rint)       \
  DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, rsqrt)      \
  DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, recip)      \
  DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, sin)        \
  DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, sinh)       \
  DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, sqrt)       \
  DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, tan)        \
  DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, tanh)       \
  DEFINE_X86_HALF_UNARY_VIA_FLOAT(half_type, sigmoid)    \
  DEFINE_X86_HALF_BINARY_VIA_FLOAT(half_type, atan2)     \
  DEFINE_X86_HALF_BINARY_VIA_FLOAT(half_type, remainder) \
  DEFINE_X86_HALF_BINARY_VIA_FLOAT(half_type, pow)       \
  DEFINE_X86_HALF_BINARY_VIA_FLOAT(half_type, maximum)   \
  DEFINE_X86_HALF_BINARY_VIA_FLOAT(half_type, minimum)
