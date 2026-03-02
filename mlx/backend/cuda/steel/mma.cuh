// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/steel/defines.cuh"
#include "mlx/backend/cuda/steel/tiles.cuh"

namespace mlx::core::cu {

/**
 * Fallback mma.
 *
 * We should probably a) implement a fallback or complain about it to the
 * compiler.
 */
template <typename U, typename T>
__device__ inline void
mma_t(Tile16x16<U>& C, Tile16x16<T>& A, Tile16x16<T>& B) {}

/**
 * Multiply the 16x16 bfloat16 tiles and accumulate the result in one 16x16
 * float tile.
 *
 * We actually perform C += A @ B.T
 */
__device__ __forceinline__ void mma_t(
    Tile16x16<float>& C,
    Tile16x16<__nv_bfloat16>& A,
    Tile16x16<__nv_bfloat16>& B) {
#if defined(MLX_CUDA_SM_80_ENABLED)
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%10, %11, %12, %13};"

      // D matrix
      : "+f"(C.values[0].x),
        "+f"(C.values[0].y),
        "+f"(C.values[1].x),
        "+f"(C.values[1].y)

      // A matrix
      : "r"(*(uint32_t*)(&A.values[0])),
        "r"(*(uint32_t*)(&A.values[1])),
        "r"(*(uint32_t*)(&A.values[2])),
        "r"(*(uint32_t*)(&A.values[3])),

        // B matrix
        "r"(*(uint32_t*)(&B.values[0])),
        "r"(*(uint32_t*)(&B.values[2])),

        // C matrix
        "f"(C.values[0].x),
        "f"(C.values[0].y),
        "f"(C.values[1].x),
        "f"(C.values[1].y));
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%10, %11, %12, %13};"

      // D matrix
      : "+f"(C.values[2].x),
        "+f"(C.values[2].y),
        "+f"(C.values[3].x),
        "+f"(C.values[3].y)

      // A matrix
      : "r"(*(uint32_t*)(&A.values[0])),
        "r"(*(uint32_t*)(&A.values[1])),
        "r"(*(uint32_t*)(&A.values[2])),
        "r"(*(uint32_t*)(&A.values[3])),

        // B matrix
        "r"(*(uint32_t*)(&B.values[1])),
        "r"(*(uint32_t*)(&B.values[3])),

        // C matrix
        "f"(C.values[2].x),
        "f"(C.values[2].y),
        "f"(C.values[3].x),
        "f"(C.values[3].y));
#endif
}

/**
 * Multiply larger register tiles by delegating to mma_t.
 */
template <typename U, typename T, int M, int N, int K>
__device__ __forceinline__ void mma_t(
    RegisterTile<U, M, N>& C,
    RegisterTile<T, M, K>& A,
    RegisterTile<T, N, K>& B) {
  constexpr int TILES_M = RegisterTile<T, M, K>::TILES_Y;
  constexpr int TILES_K = RegisterTile<T, M, K>::TILES_X;
  constexpr int TILES_N = RegisterTile<T, N, K>::TILES_Y;

  MLX_UNROLL
  for (int k = 0; k < TILES_K; k++) {
    MLX_UNROLL
    for (int m = 0; m < TILES_M; m++) {
      MLX_UNROLL
      for (int n = 0; n < TILES_N; n++) {
        mma_t(
            C.data[m * TILES_N + n],
            A.data[m * TILES_K + k],
            B.data[n * TILES_K + k]);
      }
    }
  }
}

} // namespace mlx::core::cu
