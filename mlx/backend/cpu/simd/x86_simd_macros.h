#pragma once

// ============================================================================
// x86 SIMD Common Macro Definitions
// ============================================================================
// This file provides COMMON macro patterns shared across SSE, AVX2, and
// AVX-512.
//
// Implementation-specific macros (e.g., AVX-512 cast-down patterns) are defined
// in their respective headers (sse_simd.h, avx_simd.h, avx512_simd.h).
//
// Design principles:
// - Follow NEON's DEFINE_* naming convention
// - Only include truly cross-platform patterns
// - Each macro generates inline functions with proper type signatures
//
// Macro categories:
// 1. Bool operations (SSE/AVX2 vector-based)
// 2. Binary operations (direct intrinsic mapping)
// 3. Scalar fallback operations (no intrinsic available)
// 4. Scalar overloads (type + Simd combinations)
// 5. Comparison operations
// 6. Unary operations
// 7. Reduction operations
// 8. Transcendental functions
// ============================================================================

// ============================================================================
// 1. Bool Operations (SSE/AVX2 - using vector registers)
// ============================================================================

// Bool binary operators using integer SIMD intrinsics (AVX2/SSE)
// Generates: &&, ||, &, |, ^, ==, != operators for Simd<bool, N>
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

// ============================================================================
// 2. Binary Operations (Direct Intrinsic)
// ============================================================================

// Binary operation with direct intrinsic support
// Generates: inline Simd<T, N> operator op(Simd<T, N> a, Simd<T, N> b)
#define DEFINE_X86_BINARY_OP(op, type, N, intrinsic)                   \
  inline Simd<type, N> operator op(Simd<type, N> a, Simd<type, N> b) { \
    return intrinsic(a.value, b.value);                                \
  }

// ============================================================================
// 3. Scalar Fallback Operations
// ============================================================================

// Binary operation using scalar loop (no intrinsic available)
// For normal types (int32_t, uint32_t, float, double)
#define DEFINE_X86_BINARY_SCALAR_FALLBACK(op, type, N, alignment)      \
  inline Simd<type, N> operator op(Simd<type, N> a, Simd<type, N> b) { \
    alignas(alignment) type tmp_a[N], tmp_b[N], tmp_r[N];              \
    store<type, N>(tmp_a, a);                                          \
    store<type, N>(tmp_b, b);                                          \
    for (int i = 0; i < N; ++i) {                                      \
      tmp_r[i] = tmp_a[i] op tmp_b[i];                                 \
    }                                                                  \
    return load<type, N>(tmp_r);                                       \
  }

// Unary operation using scalar loop
#define DEFINE_X86_UNARY_SCALAR_FALLBACK(name, type, N, alignment, expr) \
  inline Simd<type, N> name(Simd<type, N> a) {                           \
    alignas(alignment) type tmp_a[N], tmp_r[N];                          \
    store<type, N>(tmp_a, a);                                            \
    for (int i = 0; i < N; ++i) {                                        \
      tmp_r[i] = expr;                                                   \
    }                                                                    \
    return load<type, N>(tmp_r);                                         \
  }

// ============================================================================
// 4. Scalar Overload Operations
// ============================================================================

// Scalar overloads for binary operators (both orderings)
// Generates: Simd op scalar, scalar op Simd
#define DEFINE_X86_SCALAR_BINARY_OVERLOADS(op, type, N)       \
  inline Simd<type, N> operator op(Simd<type, N> a, type b) { \
    return a op Simd<type, N>(b);                             \
  }                                                           \
  inline Simd<type, N> operator op(type a, Simd<type, N> b) { \
    return Simd<type, N>(a) op b;                             \
  }

// All four arithmetic scalar overloads at once
#define DEFINE_X86_ARITHMETIC_SCALAR_OVERLOADS(type, N) \
  DEFINE_X86_SCALAR_BINARY_OVERLOADS(+, type, N)        \
  DEFINE_X86_SCALAR_BINARY_OVERLOADS(-, type, N)        \
  DEFINE_X86_SCALAR_BINARY_OVERLOADS(*, type, N)        \
  DEFINE_X86_SCALAR_BINARY_OVERLOADS(/, type, N)

// ============================================================================
// 5. Comparison Operations
// ============================================================================

// Comparison operator using intrinsic
#define DEFINE_X86_COMPARISON_OP(op, type, N, intrinsic, bool_cast)    \
  inline Simd<bool, N> operator op(Simd<type, N> a, Simd<type, N> b) { \
    return Simd<bool, N>(bool_cast(intrinsic(a.value, b.value)));      \
  }

// All six comparison operators using _mm*_cmp_* with flags
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
        bool_cast(prefix##_cmp_##reg_suffix(a.value, b.value, _CMP_NEQ_OQ))); \
  }

// ============================================================================
// 6. Unary Operations
// ============================================================================

// Unary operation with direct intrinsic support
#define DEFINE_X86_UNARY_OP(name, type, N, intrinsic) \
  inline Simd<type, N> name(Simd<type, N> a) {        \
    return intrinsic(a.value);                        \
  }

// Binary function (not operator) with direct intrinsic support
// Generates: inline Simd<T, N> name(Simd<T, N> a, Simd<T, N> b)
#define DEFINE_X86_BINARY_FUNC(name, type, N, intrinsic)        \
  inline Simd<type, N> name(Simd<type, N> a, Simd<type, N> b) { \
    return intrinsic(a.value, b.value);                         \
  }

// ============================================================================
// 7. Reduction Operations
// ============================================================================

// Reduction using scalar loop
#define DEFINE_X86_REDUCTION_SCALAR(name, type, N, alignment, init, op) \
  inline type name(Simd<type, N> v) {                                   \
    alignas(alignment) type tmp[N];                                     \
    store<type, N>(tmp, v);                                             \
    type result = init;                                                 \
    for (int i = 0; i < N; ++i) {                                       \
      result = result op tmp[i];                                        \
    }                                                                   \
    return result;                                                      \
  }

// Reduction using direct intrinsic
#define DEFINE_X86_REDUCTION(name, type, N, intrinsic) \
  inline type name(Simd<type, N> v) {                  \
    return intrinsic(v.value);                         \
  }

// ============================================================================
// 8. Transcendental Functions
// ============================================================================

// Transcendental function using scalar std lib function
#define DEFINE_X86_TRANSCENDENTAL(name, type, N, alignment, func) \
  inline Simd<type, N> name(Simd<type, N> a) {                    \
    alignas(alignment) type tmp[N];                               \
    store<type, N>(tmp, a);                                       \
    for (int i = 0; i < N; ++i) {                                 \
      tmp[i] = func(tmp[i]);                                      \
    }                                                             \
    return load<type, N>(tmp);                                    \
  }
