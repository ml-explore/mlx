// Copyright © 2025 Apple Inc.

// This file must not include any host-only code, utilies that work under both
// host and device can be put here.
//
// See more about the requirements at:
// https://docs.nvidia.com/cuda/nvrtc/#language

#pragma once

#include "mlx/backend/cuda/device/config.h"

#include <cuComplex.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda/std/array>
#include <cuda/std/limits>
#include <cuda/std/tuple>

namespace mlx::core::cu {

///////////////////////////////////////////////////////////////////////////////
// CUDA kernel utils
///////////////////////////////////////////////////////////////////////////////

// To pass shape/strides to kernels via constant memory, their size must be
// known at compile time.
using Shape = cuda::std::array<int32_t, MAX_NDIM>;
using Strides = cuda::std::array<int64_t, MAX_NDIM>;

///////////////////////////////////////////////////////////////////////////////
// Type limits utils
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename = void>
struct Limits {
  static constexpr __host__ __device__ T max() {
    return cuda::std::numeric_limits<T>::max();
  }
  static constexpr __host__ __device__ T min() {
    return cuda::std::numeric_limits<T>::min();
  }
  static constexpr __host__ __device__ T finite_max() {
    return cuda::std::numeric_limits<T>::max();
  }
  static constexpr __host__ __device__ T finite_min() {
    return cuda::std::numeric_limits<T>::min();
  }
};

template <typename T>
struct Limits<
    T,
    cuda::std::enable_if_t<
        cuda::std::is_same_v<T, float> || cuda::std::is_same_v<T, double>>> {
  static constexpr __host__ __device__ T max() {
    return cuda::std::numeric_limits<T>::infinity();
  }
  static constexpr __host__ __device__ T min() {
    return -cuda::std::numeric_limits<T>::infinity();
  }
  static constexpr __host__ __device__ T finite_max() {
    return cuda::std::numeric_limits<T>::max();
  }
  static constexpr __host__ __device__ T finite_min() {
    return cuda::std::numeric_limits<T>::lowest();
  }
};

// CUDA 11 does not have host side arithmatic operators for half types.
template <typename T>
struct Limits<
    T,
    cuda::std::enable_if_t<
        cuda::std::is_same_v<T, __half> ||
        cuda::std::is_same_v<T, __nv_bfloat16>>> {
  static constexpr __host__ __device__ T max() {
    return cuda::std::numeric_limits<T>::infinity();
  }
  static constexpr __host__ __device__ T min() {
#if defined(__CUDA_ARCH__) || CUDART_VERSION >= 12000
    return -cuda::std::numeric_limits<T>::infinity();
#else
    return -cuda::std::numeric_limits<float>::infinity();
#endif
  }
  static constexpr __host__ __device__ T finite_max() {
    return cuda::std::numeric_limits<T>::max();
  }
  static constexpr __host__ __device__ T finite_min() {
#if defined(__CUDA_ARCH__) || CUDART_VERSION >= 12000
    return cuda::std::numeric_limits<T>::lowest();
#else
    return cuda::std::numeric_limits<float>::lowest();
#endif
  }
};

template <>
struct Limits<bool> {
  static constexpr __host__ __device__ bool max() {
    return true;
  }
  static constexpr __host__ __device__ bool min() {
    return false;
  }
};

template <>
struct Limits<cuComplex> {
  static constexpr __host__ __device__ cuComplex max() {
    return {Limits<float>::max(), Limits<float>::max()};
  }
  static constexpr __host__ __device__ cuComplex min() {
    return {Limits<float>::min(), Limits<float>::min()};
  }
};

///////////////////////////////////////////////////////////////////////////////
// Indexing utils
///////////////////////////////////////////////////////////////////////////////

template <typename IdxT = int64_t>
inline __host__ __device__ IdxT
elem_to_loc(IdxT elem, const int* shape, const int64_t* strides, int ndim) {
  IdxT loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * IdxT(strides[i]);
    elem /= shape[i];
  }
  return loc;
}

// Optimize when the ndim is known at compile time.
template <int NDIM, typename IdxT = int64_t>
inline __host__ __device__ IdxT
elem_to_loc_nd(IdxT elem, const int* shape, const int64_t* strides) {
  IdxT loc = 0;
#pragma unroll
  for (int i = NDIM - 1; i >= 0; --i) {
    loc += (elem % shape[i]) * IdxT(strides[i]);
    elem /= shape[i];
  }
  return loc;
}

template <int NDIM, typename IdxT = int64_t>
inline __host__ __device__ cuda::std::tuple<IdxT, IdxT> elem_to_loc_nd(
    IdxT elem,
    const int* shape,
    const int64_t* a_strides,
    const int64_t* b_strides) {
  IdxT a_loc = 0;
  IdxT b_loc = 0;
#pragma unroll
  for (int i = NDIM - 1; i >= 0; --i) {
    int dim_idx = elem % shape[i];
    a_loc += dim_idx * a_strides[i];
    b_loc += dim_idx * b_strides[i];
    elem /= shape[i];
  }
  return cuda::std::make_tuple(a_loc, b_loc);
}

template <int NDIM, typename IdxT = int64_t>
inline __host__ __device__ cuda::std::tuple<IdxT, IdxT, IdxT> elem_to_loc_nd(
    IdxT elem,
    const int* shape,
    const int64_t* a_strides,
    const int64_t* b_strides,
    const int64_t* c_strides) {
  IdxT a_loc = 0;
  IdxT b_loc = 0;
  IdxT c_loc = 0;
#pragma unroll
  for (int i = NDIM - 1; i >= 0; --i) {
    int dim_idx = elem % shape[i];
    a_loc += dim_idx * a_strides[i];
    b_loc += dim_idx * b_strides[i];
    c_loc += dim_idx * c_strides[i];
    elem /= shape[i];
  }
  return cuda::std::make_tuple(a_loc, b_loc, c_loc);
}

// Optimized version when ndim is larger than 4.
template <typename IdxT = int64_t>
inline __host__ __device__ IdxT
elem_to_loc_4d(IdxT elem, const int* shape, const int64_t* strides, int ndim) {
  IdxT loc = 0;
  for (int i = ndim - 1; i >= 0; --i) {
    loc += (elem % shape[i]) * IdxT(strides[i]);
    elem /= shape[i];
  }
  return loc;
}

template <typename IdxT = int64_t>
inline __host__ __device__ cuda::std::tuple<IdxT, IdxT> elem_to_loc_4d(
    IdxT elem,
    const int* shape,
    const int64_t* a_strides,
    const int64_t* b_strides,
    int ndim) {
  IdxT a_loc = 0;
  IdxT b_loc = 0;
  for (int i = ndim - 1; i >= 0; --i) {
    int dim_idx = elem % shape[i];
    a_loc += dim_idx * a_strides[i];
    b_loc += dim_idx * b_strides[i];
    elem /= shape[i];
  }
  return cuda::std::make_tuple(a_loc, b_loc);
}

template <typename IdxT = int64_t>
inline __host__ __device__ cuda::std::tuple<IdxT, IdxT, IdxT> elem_to_loc_4d(
    IdxT elem,
    const int* shape,
    const int64_t* a_strides,
    const int64_t* b_strides,
    const int64_t* c_strides,
    int ndim) {
  IdxT a_loc = 0;
  IdxT b_loc = 0;
  IdxT c_loc = 0;
  for (int i = ndim - 1; i >= 0; --i) {
    int dim_idx = elem % shape[i];
    a_loc += dim_idx * a_strides[i];
    b_loc += dim_idx * b_strides[i];
    c_loc += dim_idx * c_strides[i];
    elem /= shape[i];
  }
  return cuda::std::make_tuple(a_loc, b_loc, c_loc);
}

///////////////////////////////////////////////////////////////////////////////
// Elem to loc in a loop utils
///////////////////////////////////////////////////////////////////////////////

template <int DIM, bool General = true, typename OffsetT = size_t>
struct LoopedElemToLoc {
  int dim;
  LoopedElemToLoc<DIM - 1, General, OffsetT> inner_looper;
  OffsetT offset{0};
  int index{0};

  __device__ LoopedElemToLoc(int dim) : dim(dim), inner_looper(dim - 1) {}

  __device__ void next(const int* shape, const int64_t* strides) {
    if (dim == 0) {
      return;
    }
    index++;
    offset += OffsetT(strides[dim - 1]);
    if (index >= shape[dim - 1]) {
      index = 0;
      inner_looper.next(shape, strides);
      offset = inner_looper.offset;
    }
  }

  __device__ void next(int n, const int* shape, const int64_t* strides) {
    if (dim == 0) {
      return;
    }
    index += n;
    offset += n * OffsetT(strides[dim - 1]);

    if (index >= shape[dim - 1]) {
      int extra = index - shape[dim - 1];
      if (extra >= shape[dim - 1]) {
        inner_looper.next(1 + extra / shape[dim - 1], shape, strides);
        extra = extra % shape[dim - 1];
      } else {
        inner_looper.next(shape, strides);
      }
      index = 0;
      offset = inner_looper.offset;
      if (extra > 0) {
        next(extra, shape, strides);
      }
    }
  }

  __device__ OffsetT location() {
    return offset;
  }
};

template <typename OffsetT>
struct LoopedElemToLoc<1, true, OffsetT> {
  int dim;
  OffsetT offset{0};
  int index{0};

  __device__ LoopedElemToLoc(int dim) : dim(dim) {}

  __device__ void next(const int* shape, const int64_t* strides) {
    index++;
    if (dim > 1) {
      offset = elem_to_loc<OffsetT>(index, shape, strides, dim);
    } else {
      offset += OffsetT(strides[0]);
    }
  }

  __device__ void next(int n, const int* shape, const int64_t* strides) {
    index += n;
    if (dim > 1) {
      offset = elem_to_loc<OffsetT>(index, shape, strides, dim);
    } else {
      offset = index * OffsetT(strides[0]);
    }
  }

  __device__ OffsetT location() {
    return offset;
  }
};

template <typename OffsetT>
struct LoopedElemToLoc<1, false, OffsetT> {
  OffsetT offset{0};

  __device__ LoopedElemToLoc(int) {}

  __device__ void next(const int*, const int64_t* strides) {
    offset += OffsetT(strides[0]);
  }

  __device__ void next(int n, const int*, const int64_t* strides) {
    offset += n * OffsetT(strides[0]);
  }

  __device__ OffsetT location() {
    return offset;
  }
};

inline __device__ cuComplex log1p(cuComplex in) {
  float x = cuCrealf(in);
  float y = cuCimagf(in);
  float zabs = sqrt(x * x + y * y);
  float theta = atan2f(y, x + 1);
  if (zabs < 0.5f) {
    float r = x * (2 + x) + y * y;
    if (r == 0) { // handle underflow
      return {x, theta};
    }
    return {0.5f * log1pf(r), theta};
  } else {
    auto z0 = sqrt((x + 1) * (x + 1) + y * y);
    return {log(z0), theta};
  }
}

} // namespace mlx::core::cu
