// Copyright © 2025 Apple Inc.

// This file must not include any host-only code, utilies that work under both
// host and device can be put here.
//
// See more about the requirements at:
// https://docs.nvidia.com/cuda/nvrtc/#language

#pragma once

#include "mlx/backend/cuda/kernels/fp16_math.cuh"

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
// Type limits utils
///////////////////////////////////////////////////////////////////////////////

template <typename U>
struct Limits {
  static constexpr U max = cuda::std::numeric_limits<U>::max();
  static constexpr U min = cuda::std::numeric_limits<U>::min();
  static constexpr U finite_max = cuda::std::numeric_limits<U>::max();
  static constexpr U finite_min = cuda::std::numeric_limits<U>::min();
};

template <>
struct Limits<bool> {
  static constexpr bool max = true;
  static constexpr bool min = false;
};

template <>
struct Limits<cuComplex> {
  static constexpr cuComplex max = {
      cuda::std::numeric_limits<float>::infinity(),
      cuda::std::numeric_limits<float>::infinity()};
  static constexpr cuComplex min = {
      -cuda::std::numeric_limits<float>::infinity(),
      -cuda::std::numeric_limits<float>::infinity()};
};

// Like MLX_FORALL_FLOAT_TYPES but use CUDA types.
#define MLX_FORALL_CUDA_FLOAT_TYPES(_) \
  _(float, float32)                    \
  _(double, float64)                   \
  _(__half, float16)                   \
  _(__nv_bfloat16, bfloat16)

// Some CCCL/CUDA combinations do not provide constexpr limits for half types.
#define SPECIALIZE_FloatLimits(CPP_TYPE, DTYPE)                          \
  template <>                                                            \
  struct Limits<CPP_TYPE> {                                              \
    static constexpr CPP_TYPE max = infinite_value<CPP_TYPE>();          \
    static constexpr CPP_TYPE min = negative_infinite_value<CPP_TYPE>(); \
    static constexpr CPP_TYPE finite_max = max_value<CPP_TYPE>();        \
    static constexpr CPP_TYPE finite_min = lowest_value<CPP_TYPE>();     \
  };

MLX_FORALL_CUDA_FLOAT_TYPES(SPECIALIZE_FloatLimits)

#undef SPECIALIZE_FloatLimits

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
  uint index{0};

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

} // namespace mlx::core::cu
