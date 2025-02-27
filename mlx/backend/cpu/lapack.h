// Copyright © 2023-2024 Apple Inc.

#pragma once

// Required for Visual Studio.
// https://github.com/OpenMathLib/OpenBLAS/blob/develop/docs/install.md
#ifdef _MSC_VER
#include <complex>
#define LAPACK_COMPLEX_CUSTOM
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#endif

#ifdef MLX_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#include <lapack.h>
#endif

#if defined(LAPACK_GLOBAL) || defined(LAPACK_NAME)

// This is to work around a change in the function signatures of lapack >= 3.9.1
// where functions taking char* also include a strlen argument, see a similar
// change in OpenCV:
// https://github.com/opencv/opencv/blob/1eb061f89de0fb85c4c75a2deeb0f61a961a63ad/cmake/OpenCVFindLAPACK.cmake#L57
#define MLX_LAPACK_FUNC(f) LAPACK_##f

#else

#define MLX_LAPACK_FUNC(f) f##_

#endif

#define INSTANTIATE_LAPACK_TYPES(FUNC)                       \
  template <typename T, typename... Args>                    \
  void FUNC(Args... args) {                                  \
    if constexpr (std::is_same_v<T, float>) {                \
      MLX_LAPACK_FUNC(s##FUNC)(std::forward<Args>(args)...); \
    } else if constexpr (std::is_same_v<T, double>) {        \
      MLX_LAPACK_FUNC(d##FUNC)(std::forward<Args>(args)...); \
    }                                                        \
  }

INSTANTIATE_LAPACK_TYPES(geqrf)
INSTANTIATE_LAPACK_TYPES(orgqr)
INSTANTIATE_LAPACK_TYPES(syevd)
INSTANTIATE_LAPACK_TYPES(potrf)
INSTANTIATE_LAPACK_TYPES(gesvdx)
INSTANTIATE_LAPACK_TYPES(getrf)
INSTANTIATE_LAPACK_TYPES(getri)
INSTANTIATE_LAPACK_TYPES(trtri)
