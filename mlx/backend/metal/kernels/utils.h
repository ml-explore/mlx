// Copyright © 2023 Apple Inc.

#pragma once

#include <metal_math>
#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/complex.h"

///////////////////////////////////////////////////////////////////////////////
// Type limits utils
///////////////////////////////////////////////////////////////////////////////

template <typename U>
struct Limits {
  static const constant U max = metal::numeric_limits<U>::max();
  static const constant U min = metal::numeric_limits<U>::min();
  static const constant U finite_max = metal::numeric_limits<U>::max();
  static const constant U finite_min = metal::numeric_limits<U>::min();
};

#define instantiate_default_limit(type)                                      \
  template <>                                                                \
  struct Limits<type> {                                                      \
    static constexpr constant type max = metal::numeric_limits<type>::max(); \
    static constexpr constant type min = metal::numeric_limits<type>::min(); \
    static constexpr constant type finite_max =                              \
        metal::numeric_limits<type>::max();                                  \
    static constexpr constant type finite_min =                              \
        metal::numeric_limits<type>::min();                                  \
  };

instantiate_default_limit(uint8_t);
instantiate_default_limit(uint16_t);
instantiate_default_limit(uint32_t);
instantiate_default_limit(uint64_t);
instantiate_default_limit(int8_t);
instantiate_default_limit(int16_t);
instantiate_default_limit(int32_t);
instantiate_default_limit(int64_t);

#define instantiate_float_limit(type)             \
  template <>                                     \
  struct Limits<type> {                           \
    static constexpr constant type max =          \
        metal::numeric_limits<type>::infinity();  \
    static constexpr constant type min =          \
        -metal::numeric_limits<type>::infinity(); \
    static constexpr constant type finite_max =   \
        metal::numeric_limits<type>::max();       \
    static constexpr constant type finite_min =   \
        -metal::numeric_limits<type>::max();      \
  };

instantiate_float_limit(half);
instantiate_float_limit(float);
instantiate_float_limit(bfloat16_t);

template <>
struct Limits<bool> {
  static constexpr constant bool max = true;
  static constexpr constant bool min = false;
};

///////////////////////////////////////////////////////////////////////////////
// Indexing utils
///////////////////////////////////////////////////////////////////////////////

inline size_t elem_to_loc(
    uint elem,
    device const int* shape,
    device const size_t* strides,
    int ndim) {
  size_t loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * strides[i];
    elem /= shape[i];
  }
  return loc;
}

inline size_t elem_to_loc(
    uint elem,
    constant const int* shape,
    constant const size_t* strides,
    int ndim) {
  size_t loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * strides[i];
    elem /= shape[i];
  }
  return loc;
}

template <int NDIM>
inline uint2 elem_to_loc_2_nd(
    uint3 elem,
    constant const int shape[NDIM],
    constant const size_t a_strides[NDIM],
    constant const size_t b_strides[NDIM]) {
  uint2 loc = {
      static_cast<uint>(
          elem.x * a_strides[NDIM - 1] + elem.y * a_strides[NDIM - 2]),
      static_cast<uint>(
          elem.x * b_strides[NDIM - 1] + elem.y * b_strides[NDIM - 2])};
  for (int d = NDIM - 3; d >= 0; --d) {
    uint l = elem.z % shape[d];
    loc.x += l * a_strides[d];
    loc.y += l * b_strides[d];
    elem.z /= shape[d];
  }
  return loc;
}

template <int NDIM>
inline size_t elem_to_loc_nd(
    uint3 elem,
    constant const int shape[NDIM],
    constant const size_t strides[NDIM]) {
  size_t loc = elem.x * strides[NDIM - 1] + elem.y * strides[NDIM - 2];
  for (int d = NDIM - 3; d >= 0; --d) {
    loc += (elem.z % shape[d]) * strides[d];
    elem.z /= shape[d];
  }
  return loc;
}

inline size_t elem_to_loc_1(uint elem, constant const size_t& stride) {
  return elem * stride;
}

inline size_t elem_to_loc_2(uint2 elem, constant const size_t strides[2]) {
  return elem.x * strides[1] + elem.y * strides[0];
}

inline size_t elem_to_loc_3(uint3 elem, constant const size_t strides[3]) {
  return elem.x * strides[2] + elem.y * strides[1] + elem.z * strides[0];
}

// Non templated version to handle arbitrary dims
inline size_t elem_to_loc(
    uint3 elem,
    constant const int* shape,
    constant const size_t* strides,
    int ndim) {
  size_t loc = elem.x * strides[ndim - 1] + elem.y * strides[ndim - 2];
  for (int d = ndim - 3; d >= 0; --d) {
    loc += (elem.z % shape[d]) * strides[d];
    elem.z /= shape[d];
  }
  return loc;
}

inline uint2 elem_to_loc_2_nd(
    uint3 elem,
    constant const int* shape,
    constant const size_t* a_strides,
    constant const size_t* b_strides,
    int ndim) {
  uint2 loc = {
      static_cast<uint>(
          elem.x * a_strides[ndim - 1] + elem.y * a_strides[ndim - 2]),
      static_cast<uint>(
          elem.x * b_strides[ndim - 1] + elem.y * b_strides[ndim - 2])};
  for (int d = ndim - 3; d >= 0; --d) {
    uint l = elem.z % shape[d];
    loc.x += l * a_strides[d];
    loc.y += l * b_strides[d];
    elem.z /= shape[d];
  }
  return loc;
}

template <int NDIM>
inline uint elem_to_loc_nd(
    uint elem,
    device const int* shape,
    device const size_t* strides);

template <>
inline uint elem_to_loc_nd<1>(
    uint elem,
    device const int* shape,
    device const size_t* strides) {
  return (elem % shape[0]) * strides[0];
}

template <>
inline uint elem_to_loc_nd<2>(
    uint elem,
    device const int* shape,
    device const size_t* strides) {
  uint loc = (elem % shape[1]) * strides[1];
  elem /= shape[1];
  loc += (elem % shape[0]) * strides[0];
  return loc;
}

template <>
inline uint elem_to_loc_nd<3>(
    uint elem,
    device const int* shape,
    device const size_t* strides) {
  uint loc = (elem % shape[2]) * strides[2];
  elem /= shape[2];
  loc += (elem % shape[1]) * strides[1];
  elem /= shape[1];
  loc += (elem % shape[0]) * strides[0];
  return loc;
}

template <>
inline uint elem_to_loc_nd<4>(
    uint elem,
    device const int* shape,
    device const size_t* strides) {
  uint loc = (elem % shape[3]) * strides[3];
  elem /= shape[3];
  loc += (elem % shape[2]) * strides[2];
  elem /= shape[2];
  loc += (elem % shape[1]) * strides[1];
  elem /= shape[1];
  loc += (elem % shape[0]) * strides[0];
  return loc;
}

///////////////////////////////////////////////////////////////////////////////
// Calculation utils
///////////////////////////////////////////////////////////////////////////////

/** Compute ceil((float)N/(float)M) */
inline size_t ceildiv(size_t N, size_t M) {
  return (N + M - 1) / M;
}

// https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html#1202
inline float log1p(float x) {
  float xp1 = 1.0f + x;
  if (xp1 == Limits<float>::max) {
    return Limits<float>::max;
  }
  if (xp1 == 1.0f) {
    return x;
  }

  return x * (metal::log(xp1) / (xp1 - 1.0f));
}

inline bfloat16_t log1p(bfloat16_t x) {
  float xp1 = 1.0f + static_cast<float>(x);
  if (xp1 == Limits<float>::max) {
    return Limits<bfloat16_t>::max;
  }
  if (xp1 == 1.0f) {
    return x;
  }

  return bfloat16_t(x * (metal::log(xp1) / (xp1 - 1.0f)));
}

///////////////////////////////////////////////////////////////////////////////
// SIMD shuffle ops
///////////////////////////////////////////////////////////////////////////////

inline uint64_t simd_shuffle_down(uint64_t data, uint16_t delta) {
  return as_type<uint64_t>(
      metal::simd_shuffle_down(as_type<uint2>(data), delta));
}

inline int64_t simd_shuffle_down(int64_t data, uint16_t delta) {
  return as_type<int64_t>(
      metal::simd_shuffle_down(as_type<uint2>(data), delta));
}

inline bool simd_shuffle_down(bool data, uint16_t delta) {
  return simd_shuffle_down(static_cast<uint32_t>(data), delta);
}
