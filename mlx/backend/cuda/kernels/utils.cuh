// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/dtype_utils.cuh"
#include "mlx/backend/cuda/kernels/fp16_math.cuh"

#include <cuComplex.h>
#include <cuda/std/limits>

namespace mlx::core::mxcuda {

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

#define SPECIALIZE_FloatLimits(CPP_TYPE, DTYPE)           \
  template <>                                             \
  struct Limits<CPP_TYPE> {                               \
    static constexpr CPP_TYPE max =                       \
        cuda::std::numeric_limits<CPP_TYPE>::infinity();  \
    static constexpr CPP_TYPE min =                       \
        negative_infinite<CPP_TYPE>();                    \
    static constexpr CPP_TYPE finite_max =                \
        cuda::std::numeric_limits<CPP_TYPE>::max();       \
    static constexpr CPP_TYPE finite_min =                \
        cuda::std::numeric_limits<CPP_TYPE>::min();       \
  };

MLX_FORALL_CUDA_FLOAT_TYPES(SPECIALIZE_FloatLimits)

#undef SPECIALIZE_FloatLimits

} // namespace mlx::core::mxcuda
