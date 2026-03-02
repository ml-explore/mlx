// Copyright © 2025 Apple Inc.

// This file is used by both CUDA kernel code and host-only C++ code.

#pragma once

// The maximum dimensions of shape/strides passed as kernel parameters.
#define MAX_NDIM 10

// All existing NVIDIA hardware has a fixed 32 warp size. Though a built-in
// warpSize variable exists, using it would prevent compile-time optimizations.
#define WARP_SIZE 32
