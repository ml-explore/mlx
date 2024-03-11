// Copyright © 2024 Apple Inc.

#pragma once

#ifdef ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
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
