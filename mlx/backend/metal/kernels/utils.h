// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <metal_math>
#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/complex.h"
#include "mlx/backend/metal/kernels/defines.h"

typedef half float16_t;

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

template <>
struct Limits<complex64_t> {
  static constexpr constant complex64_t max = complex64_t(
      metal::numeric_limits<float>::infinity(),
      metal::numeric_limits<float>::infinity());
  static constexpr constant complex64_t min = complex64_t(
      -metal::numeric_limits<float>::infinity(),
      -metal::numeric_limits<float>::infinity());
};

///////////////////////////////////////////////////////////////////////////////
// Indexing utils
///////////////////////////////////////////////////////////////////////////////

#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

///////////////////////////////////////////////////////////////////////////////
// Single Array with generic dims

template <typename stride_t>
METAL_FUNC stride_t elem_to_loc(
    uint elem,
    constant const int* shape,
    constant const stride_t* strides,
    int ndim) {
  stride_t loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * strides[i];
    elem /= shape[i];
  }
  return loc;
}

template <typename stride_t>
METAL_FUNC stride_t elem_to_loc(
    stride_t elem,
    constant const int* shape,
    constant const stride_t* strides,
    int ndim) {
  stride_t loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * strides[i];
    elem /= shape[i];
  }
  return loc;
}

// Non templated version to handle arbitrary dims
template <typename stride_t>
METAL_FUNC stride_t elem_to_loc(
    uint3 elem,
    constant const int* shape,
    constant const stride_t* strides,
    int ndim) {
  stride_t loc = elem.x * strides[ndim - 1] + elem.y * strides[ndim - 2];
  for (int d = ndim - 3; d >= 0; --d) {
    loc += (elem.z % shape[d]) * strides[d];
    elem.z /= shape[d];
  }
  return loc;
}

///////////////////////////////////////////////////////////////////////////////
// Single Array with fixed N dims

template <typename stride_t>
METAL_FUNC stride_t elem_to_loc_1(uint elem, constant const stride_t& stride) {
  return elem * stride;
}

template <typename stride_t>
METAL_FUNC stride_t
elem_to_loc_2(uint2 elem, constant const stride_t strides[2]) {
  return elem.x * strides[1] + elem.y * strides[0];
}

template <typename stride_t>
METAL_FUNC stride_t
elem_to_loc_3(uint3 elem, constant const stride_t strides[3]) {
  return elem.x * strides[2] + elem.y * strides[1] + elem.z * strides[0];
}

///////////////////////////////////////////////////////////////////////////////
// Multiple Arrays with generic dims

template <typename stride_t>
METAL_FUNC ulong2 elem_to_loc_2_nd(
    uint3 elem,
    constant const int* shape,
    constant const stride_t* a_strides,
    constant const stride_t* b_strides,
    int ndim) {
  ulong2 loc = {
      ulong(elem.x * a_strides[ndim - 1] + elem.y * a_strides[ndim - 2]),
      ulong(elem.x * b_strides[ndim - 1] + elem.y * b_strides[ndim - 2])};
  for (int d = ndim - 3; d >= 0; --d) {
    uint l = elem.z % shape[d];
    loc.x += l * a_strides[d];
    loc.y += l * b_strides[d];
    elem.z /= shape[d];
  }
  return loc;
}

METAL_FUNC ulong3 elem_to_loc_3_nd(
    uint3 elem,
    constant const int* shape,
    constant const size_t* a_strides,
    constant const size_t* b_strides,
    constant const size_t* c_strides,
    int ndim) {
  ulong3 loc = {
      elem.x * a_strides[ndim - 1] + elem.y * a_strides[ndim - 2],
      elem.x * b_strides[ndim - 1] + elem.y * b_strides[ndim - 2],
      elem.x * c_strides[ndim - 1] + elem.y * c_strides[ndim - 2]};
  for (int d = ndim - 3; d >= 0; --d) {
    uint l = elem.z % shape[d];
    loc.x += l * a_strides[d];
    loc.y += l * b_strides[d];
    loc.z += l * c_strides[d];
    elem.z /= shape[d];
  }
  return loc;
}

///////////////////////////////////////////////////////////////////////////////
// Elem to loc in a loop utils
///////////////////////////////////////////////////////////////////////////////

template <int dim, typename offset_t = size_t>
struct looped_elem_to_loc {
  looped_elem_to_loc<dim - 1, offset_t> inner_looper;
  offset_t offset{0};
  int index{0};

  void next(const constant int* shape, const constant size_t* strides) {
    index++;
    offset += strides[dim - 1];

    if (index >= shape[dim - 1]) {
      index = 0;
      inner_looper.next(shape, strides);
      offset = inner_looper.offset;
    }
  }

  void next(int n, const constant int* shape, const constant size_t* strides) {
    index += n;
    offset += n * strides[dim - 1];

    if (index >= shape[dim - 1]) {
      int extra = index - shape[dim - 1];
      index = 0;
      inner_looper.next(shape, strides);
      offset = inner_looper.offset;
      if (extra > 0) {
        next(extra, shape, strides);
      }
    }
  }

  offset_t
  location(offset_t, const constant int*, const constant size_t*, int) {
    return offset;
  }
};

template <typename offset_t>
struct looped_elem_to_loc<1, offset_t> {
  offset_t offset{0};

  void next(const constant int*, const constant size_t* strides) {
    offset += strides[0];
  }

  void next(int n, const constant int*, const constant size_t* strides) {
    offset += n * strides[0];
  }

  offset_t
  location(offset_t, const constant int*, const constant size_t*, int) {
    return offset;
  }
};

template <typename offset_t>
struct looped_elem_to_loc<0, offset_t> {
  void next(const constant int*, const constant size_t*) {}
  void next(int, const constant int*, const constant size_t*) {}

  offset_t location(
      offset_t idx,
      const constant int* shape,
      const constant size_t* strides,
      int ndim) {
    return elem_to_loc(idx, shape, strides, ndim);
  }
};

///////////////////////////////////////////////////////////////////////////////
// Calculation utils
///////////////////////////////////////////////////////////////////////////////

/** Compute ceil((float)N/(float)M) */
template <typename T, typename U>
inline T ceildiv(T N, U M) {
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

inline complex64_t simd_shuffle_down(complex64_t data, uint16_t delta) {
  return complex64_t(
      simd_shuffle_down(data.real, delta), simd_shuffle_down(data.imag, delta));
}
