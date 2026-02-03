// Copyright © 2025 Apple Inc.

// This file is used by both HIP kernel code and host-only C++ code.

#pragma once

// The maximum dimensions of shape/strides passed as kernel parameters.
#define MAX_NDIM 10

// AMD GPU warp (wavefront) size varies by architecture:
// - CDNA/GCN (gfx9xx and earlier): 64
// - RDNA (gfx10xx, gfx11xx): 32
//
// The __AMDGCN_WAVEFRONT_SIZE__ macro is defined by the HIP compiler
// based on the target architecture. We use it when available.
#if defined(__AMDGCN_WAVEFRONT_SIZE__)
  #define WARP_SIZE __AMDGCN_WAVEFRONT_SIZE__
#elif defined(__gfx1010__) || defined(__gfx1011__) || defined(__gfx1012__) || \
      defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || \
      defined(__gfx1033__) || defined(__gfx1034__) || defined(__gfx1035__) || \
      defined(__gfx1036__) || defined(__gfx1100__) || defined(__gfx1101__) || \
      defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || \
      defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)
  // RDNA architectures use 32-wide wavefronts
  #define WARP_SIZE 32
#else
  // Default to 64 for CDNA/GCN architectures
  #define WARP_SIZE 64
#endif

namespace mlx::core::rocm {

// Configuration constants for ROCm kernels

// Default thread block size
constexpr int kDefaultBlockSize = 256;

// Maximum threads per block (typical for AMD GPUs)
constexpr int kMaxThreadsPerBlock = 1024;

// Warp size (wavefront size) - use the macro for compile-time value
constexpr int kWarpSize = WARP_SIZE;

// Maximum shared memory per block (in bytes)
constexpr int kMaxSharedMemoryPerBlock = 65536;

// Maximum number of dimensions supported
constexpr int kMaxNdim = 8;

// Reduce constants
constexpr int kReduceBlockSize = 256;
constexpr int kReduceMaxBlocks = 1024;

// Copy constants
constexpr int kCopyBlockSize = 256;

// Softmax constants
constexpr int kSoftmaxBlockSize = 256;

// Layer norm constants
constexpr int kLayerNormBlockSize = 256;

// RMS norm constants
constexpr int kRMSNormBlockSize = 256;

// Attention constants
constexpr int kAttentionBlockSize = 256;

} // namespace mlx::core::rocm
