// Copyright © 2025 Apple Inc.
#pragma once

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include "mlx/backend/cuda/ptx.cuh"

namespace mlx::core {

namespace cu {

constexpr float F8E4M3_MAX = 448.0f;
constexpr float F4E2M1_MAX = 6.0f;

inline __device__ float4 dequant_fp8(uint32_t bits) {
  auto out = *(__nv_fp8x4_e4m3*)(&bits);
  return out.operator float4();
}

inline __device__ float4 dequant_fp4(uint16_t bits) {
  auto out = *(__nv_fp4x4_e2m1*)(&bits);
  return out.operator float4();
}

template <int bits, int wsize = 8>
inline constexpr __device__ short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr __device__ short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

template <typename T>
__device__ __forceinline__ void absmax_x2(T& out, const T& x1, const T& x2) {
  if constexpr (
      (std::is_same<T, __nv_bfloat162>::value) ||
      (std::is_same<T, __half2>::value)) {
    T a = x1;
    T b = x2;
    out = __hmax2(__habs2(a), __habs2(b));
  } else if constexpr (std::is_same<T, float2>::value) {
    float2 a = x1;
    float2 b = x2;
    out.x = fmaxf(fabsf(a.x), fabsf(b.x));
    out.y = fmaxf(fabsf(a.y), fabsf(b.y));
  }
}

__device__ __forceinline__ void copy_2d_to_shared(
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

} // namespace cu

template <typename F>
void dispatch_groups(int group_size, F&& f) {
  switch (group_size) {
    case 32:
      f(std::integral_constant<int, 32>{});
      break;
    case 64:
      f(std::integral_constant<int, 64>{});
      break;
    case 128:
      f(std::integral_constant<int, 128>{});
      break;
  }
}

template <typename F>
void dispatch_bits(int bits, F&& f) {
  switch (bits) {
    case 2:
      f(std::integral_constant<int, 2>{});
      break;
    case 3:
      f(std::integral_constant<int, 3>{});
      break;
    case 4:
      f(std::integral_constant<int, 4>{});
      break;
    case 5:
      f(std::integral_constant<int, 5>{});
      break;
    case 6:
      f(std::integral_constant<int, 6>{});
      break;
    case 8:
      f(std::integral_constant<int, 8>{});
      break;
  }
}

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
