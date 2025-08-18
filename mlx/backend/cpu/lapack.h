// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <complex>
#define LAPACK_COMPLEX_CUSTOM
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#define lapack_complex_float_real(z) ((z).real())
#define lapack_complex_float_imag(z) ((z).imag())
#define lapack_complex_double_real(z) ((z).real())
#define lapack_complex_double_imag(z) ((z).imag())

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

#define INSTANTIATE_LAPACK_REAL(FUNC)                        \
  template <typename T, typename... Args>                    \
  void FUNC(Args... args) {                                  \
    if constexpr (std::is_same_v<T, float>) {                \
      MLX_LAPACK_FUNC(s##FUNC)(std::forward<Args>(args)...); \
    } else if constexpr (std::is_same_v<T, double>) {        \
      MLX_LAPACK_FUNC(d##FUNC)(std::forward<Args>(args)...); \
    }                                                        \
  }

INSTANTIATE_LAPACK_REAL(geqrf)
INSTANTIATE_LAPACK_REAL(orgqr)
INSTANTIATE_LAPACK_REAL(syevd)
INSTANTIATE_LAPACK_REAL(geev)
INSTANTIATE_LAPACK_REAL(potrf)
INSTANTIATE_LAPACK_REAL(gesdd)
INSTANTIATE_LAPACK_REAL(getrf)
INSTANTIATE_LAPACK_REAL(getri)
INSTANTIATE_LAPACK_REAL(trtri)

#define INSTANTIATE_LAPACK_COMPLEX(FUNC)                            \
  template <typename T, typename... Args>                           \
  void FUNC(Args... args) {                                         \
    if constexpr (std::is_same_v<T, std::complex<float>>) {         \
      MLX_LAPACK_FUNC(c##FUNC)(std::forward<Args>(args)...);        \
    } else if constexpr (std::is_same_v<T, std::complex<double>>) { \
      MLX_LAPACK_FUNC(z##FUNC)(std::forward<Args>(args)...);        \
    }                                                               \
  }

INSTANTIATE_LAPACK_COMPLEX(heevd)
