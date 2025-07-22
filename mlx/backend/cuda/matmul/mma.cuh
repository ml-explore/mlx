// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/matmul/tiles.cuh"

namespace mlx::core::cu {

template <typename TileAccum, typename Tile>
__device__ inline void mma(TileAccum& C, Tile& A, Tile& B) {}

/**
 * Multiply the 16x16 bfloat16 tiles and accumulate the result in one 16x16
 * float tile.
 *
 * We actually perform C += A @ B.T
 */
__device__ inline void mma(
    Tile16x16<float>& C,
    Tile16x16<__nv_bfloat16>& A,
    Tile16x16<__nv_bfloat16>& B) {
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
}

} // namespace mlx::core::cu
