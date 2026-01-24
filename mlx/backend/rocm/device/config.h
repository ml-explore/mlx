// Copyright © 2025 Apple Inc.

#pragma once

namespace mlx::core::rocm {

// Configuration constants for ROCm kernels

// Default thread block size
constexpr int kDefaultBlockSize = 256;

// Maximum threads per block (typical for AMD GPUs)
constexpr int kMaxThreadsPerBlock = 1024;

// Warp size (wavefront size on AMD GPUs is typically 64)
constexpr int kWarpSize = 64;

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
