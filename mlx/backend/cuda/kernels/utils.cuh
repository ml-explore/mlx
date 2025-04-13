// Copyright © 2025 Apple Inc.

#pragma once

#include <cuComplex.h>
#include <cuda/std/array>
#include <cuda/std/limits>

namespace mlx::core::cu {

///////////////////////////////////////////////////////////////////////////////
// CUDA kernel utils
///////////////////////////////////////////////////////////////////////////////

// To pass shape/strides to kernels via constant memory, their size must be
// known at compile time.
#define MAX_NDIM 8

using Shape = cuda::std::array<int32_t, MAX_NDIM>;
using Strides = cuda::std::array<int64_t, MAX_NDIM>;

// Utility to copy data from vector to array in host.
template <typename T>
inline cuda::std::array<T, MAX_NDIM> const_param(const std::vector<T>& vec) {
  if (vec.size() > MAX_NDIM) {
    throw std::runtime_error("ndim can not be larger than 8.");
  }
  cuda::std::array<T, MAX_NDIM> result;
  std::copy_n(vec.begin(), vec.size(), result.begin());
  return result;
}

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

} // namespace mlx::core::cu
