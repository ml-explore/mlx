// Copyright © 2025 Apple Inc.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "mlx/backend/cuda/cuda_utils.h"
#include "mlx/backend/cuda/ptx.cuh"

namespace mlx::core {

__forceinline__ __device__ void copy_2d_to_shared(
    void* dst,
    const CUtensorMap* tensor_map,
    uint32_t tile_x,
    uint32_t tile_y,
    uint32_t num_bytes,
    uint64_t* barrier,
    const bool is_master_thread) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (is_master_thread) {
    // Arrive and tell how many bytes are expected
    ptx::mbarrier_arrive_expect_tx(barrier, num_bytes);
    // Initiate bulk tensor copy
    ptx::cp_async_bulk_tensor_2d_global_to_shared(
        dst, tensor_map, tile_x, tile_y, barrier);
  } else {
    // Other threads just arrive
    ptx::mbarrier_arrive(barrier);
  }
#endif // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

// Host function to create a 2D TMA tensor map descriptor
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
inline void create_2D_tensor_map(
    CUtensorMap* tensorMap,
    void* input_ptr,
    CUtensorMapDataType dtype,
    uint64_t rows,
    uint64_t cols,
    uint32_t tile_y,
    uint32_t tile_x,
    uint64_t stride_bytes,
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE) {
  constexpr uint32_t rank = 2; // 2D
  uint64_t global_dim[rank] = {cols, rows};
  // For row-major layout
  uint64_t strides[rank - 1] = {stride_bytes};
  uint32_t tile_dim[rank] = {tile_x, tile_y};
  uint32_t elem_stride[rank] = {1, 1};

  CHECK_CUDA_ERROR(cuTensorMapEncodeTiled(
      tensorMap,
      dtype,
      rank,
      input_ptr,
      global_dim,
      strides,
      tile_dim,
      elem_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

} // namespace mlx::core
