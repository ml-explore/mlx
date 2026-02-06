// Copyright © 2025 Apple Inc.

// This file is used by both HIP kernel code and host-only C++ code.

#pragma once

// The maximum dimensions of shape/strides passed as kernel parameters.
#define MAX_NDIM 10

// AMD GPU warp (wavefront) size varies by architecture:
// - CDNA/GCN (gfx9xx and earlier): 64
// - RDNA (gfx10xx, gfx11xx, gfx12xx): 32
//
// The __AMDGCN_WAVEFRONT_SIZE__ macro is defined by the HIP compiler
// based on the target architecture. We use it when available for device code.
//
// IMPORTANT: For host code, we need a consistent value that matches the
// compiled device code. Since we compile for specific architectures via
// CMAKE_HIP_ARCHITECTURES, we need to ensure host and device agree.
//
// For now, we default to 32 (RDNA) since that's the most common consumer GPU.
// If targeting CDNA/GCN architectures, change this to 64.
#if defined(__AMDGCN_WAVEFRONT_SIZE__)
// Device code: use the compiler-provided value
#define WARP_SIZE __AMDGCN_WAVEFRONT_SIZE__
#elif defined(__HIP_DEVICE_COMPILE__)
// Device code without wavefront size macro - check architecture macros
#if defined(__gfx1010__) || defined(__gfx1011__) || defined(__gfx1012__) || \
    defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || \
    defined(__gfx1033__) || defined(__gfx1034__) || defined(__gfx1035__) || \
    defined(__gfx1036__) || defined(__gfx1100__) || defined(__gfx1101__) || \
    defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || \
    defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)
#define WARP_SIZE 32
#else
#define WARP_SIZE 64
#endif
#else
// Host code: use a fixed value that matches the target architecture.
// This MUST match the CMAKE_HIP_ARCHITECTURES setting.
// For RDNA (gfx10xx, gfx11xx, gfx12xx): 32
// For CDNA/GCN (gfx9xx): 64
#define WARP_SIZE 32
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
