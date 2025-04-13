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

} // namespace mlx::core::cu
