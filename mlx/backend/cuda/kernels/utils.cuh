// Copyright © 2025 Apple Inc.

// This file must not include any host-only code, utilies that work under both
// host and device can be put here.
//
// See more about the requirements at:
// https://docs.nvidia.com/cuda/nvrtc/#language

#pragma once

#include <cuComplex.h>
#include <cuda/std/array>
#include <cuda/std/limits>
#include <cuda/std/tuple>

namespace mlx::core::cu {

///////////////////////////////////////////////////////////////////////////////
// CUDA kernel utils
///////////////////////////////////////////////////////////////////////////////

// To pass shape/strides to kernels via constant memory, their size must be
// known at compile time.
#define MAX_NDIM 8

using Shape = cuda::std::array<int32_t, MAX_NDIM>;
using Strides = cuda::std::array<int64_t, MAX_NDIM>;

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

// Optimized version when ndim is larger than 4.
template <typename IdxT = int64_t>
inline __host__ __device__ IdxT
elem_to_loc_4d(IdxT elem, const int* shape, const int64_t* strides, int ndim) {
  IdxT loc = elem_to_loc_nd<3>(elem, shape, strides);
  for (int i = ndim - 1; i >= 3; --i) {
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
  auto [a_loc, b_loc] = elem_to_loc_nd<3>(elem, shape, a_strides, b_strides);
  for (int i = ndim - 1; i >= 3; --i) {
    int dim_idx = elem % shape[i];
    a_loc += dim_idx * a_strides[i];
    b_loc += dim_idx * b_strides[i];
    elem /= shape[i];
  }
  return cuda::std::make_tuple(a_loc, b_loc);
}

} // namespace mlx::core::cu
