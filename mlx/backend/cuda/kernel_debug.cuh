// Copyright © 2025 Apple Inc.
// Debug utilities for CUDA kernel registration diagnostics on Windows

#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <typeinfo>

namespace mlx::core::cu {

// Debug macro to check kernel registration before launch
// Usage: MLX_KERNEL_DEBUG(kernel_ptr, "kernel_name", grid, block);
#ifdef MLX_CUDA_KERNEL_DEBUG

#define MLX_KERNEL_DEBUG(kernel_ptr, kernel_name, grid, block)                \
  do {                                                                        \
    cudaFuncAttributes attr;                                                  \
    cudaError_t check_err = cudaFuncGetAttributes(&attr, kernel_ptr);         \
    if (check_err != cudaSuccess) {                                           \
      fprintf(                                                                \
          stderr,                                                             \
          "[KERNEL DEBUG] %s at %p NOT registered: %s\n",                     \
          kernel_name,                                                        \
          kernel_ptr,                                                         \
          cudaGetErrorString(check_err));                                     \
      fprintf(                                                                \
          stderr,                                                             \
          "[KERNEL DEBUG]   grid=(%u,%u,%u) block=(%u,%u,%u)\n",              \
          (grid).x,                                                           \
          (grid).y,                                                           \
          (grid).z,                                                           \
          (block).x,                                                          \
          (block).y,                                                          \
          (block).z);                                                         \
      fflush(stderr);                                                         \
      cudaGetLastError(); /* clear the error */                               \
    } else {                                                                  \
      fprintf(                                                                \
          stderr,                                                             \
          "[KERNEL DEBUG] %s at %p IS registered (regs=%d, maxThreads=%d)\n", \
          kernel_name,                                                        \
          kernel_ptr,                                                         \
          attr.numRegs,                                                       \
          attr.maxThreadsPerBlock);                                           \
      fflush(stderr);                                                         \
    }                                                                         \
  } while (0)

// Variant that includes type info
#define MLX_KERNEL_DEBUG_T(kernel_ptr, kernel_name, grid, block, ...) \
  do {                                                                \
    cudaFuncAttributes attr;                                          \
    cudaError_t check_err = cudaFuncGetAttributes(&attr, kernel_ptr); \
    if (check_err != cudaSuccess) {                                   \
      fprintf(                                                        \
          stderr,                                                     \
          "[KERNEL DEBUG] %s at %p NOT registered: %s\n",             \
          kernel_name,                                                \
          kernel_ptr,                                                 \
          cudaGetErrorString(check_err));                             \
      fprintf(                                                        \
          stderr,                                                     \
          "[KERNEL DEBUG]   grid=(%u,%u,%u) block=(%u,%u,%u)\n",      \
          (grid).x,                                                   \
          (grid).y,                                                   \
          (grid).z,                                                   \
          (block).x,                                                  \
          (block).y,                                                  \
          (block).z);                                                 \
      fflush(stderr);                                                 \
      cudaGetLastError(); /* clear the error */                       \
    } else {                                                          \
      fprintf(                                                        \
          stderr,                                                     \
          "[KERNEL DEBUG] %s at %p IS registered (regs=%d)\n",        \
          kernel_name,                                                \
          kernel_ptr,                                                 \
          attr.numRegs);                                              \
      fflush(stderr);                                                 \
    }                                                                 \
  } while (0)

#else

#define MLX_KERNEL_DEBUG(kernel_ptr, kernel_name, grid, block) ((void)0)
#define MLX_KERNEL_DEBUG_T(kernel_ptr, kernel_name, grid, block, ...) ((void)0)

#endif // MLX_CUDA_KERNEL_DEBUG

} // namespace mlx::core::cu
