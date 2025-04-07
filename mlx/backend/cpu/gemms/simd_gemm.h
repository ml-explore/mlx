// Copyright © 2025 Apple Inc.
#pragma once

#include "mlx/backend/cpu/simd/simd.h"

namespace mlx::core {

inline int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

template <int block_size, typename T, typename AccT>
void load_block(
    const T* in,
    AccT* out,
    int M,
    int N,
    int i,
    int j,
    bool transpose) {
  if (transpose) {
    for (int ii = 0; ii < block_size && i * block_size + ii < M; ++ii) {
      for (int jj = 0; jj < block_size && j * block_size + jj < N; ++jj) {
        out[jj * block_size + ii] =
            in[(i * block_size + ii) * N + j * block_size + jj];
      }
    }
  } else {
    for (int ii = 0; ii < block_size && i * block_size + ii < M; ++ii) {
      for (int jj = 0; jj < block_size && j * block_size + jj < N; ++jj) {
        out[ii * block_size + jj] =
            in[(i * block_size + ii) * N + j * block_size + jj];
      }
    }
  }
}

template <typename T, typename AccT>
void simd_gemm(
    const T* a,
    const T* b,
    T* c,
    bool a_trans,
    bool b_trans,
    int M,
    int N,
    int K,
    float alpha,
    float beta) {
  constexpr int block_size = 16;
  constexpr int simd_size = simd::max_size<AccT>;
  static_assert(
      (block_size % simd_size) == 0,
      "Block size must be divisible by SIMD size");

  int last_k_block_size = K - block_size * (K / block_size);
  int last_k_simd_block = (last_k_block_size / simd_size) * simd_size;
  for (int i = 0; i < ceildiv(M, block_size); i++) {
    for (int j = 0; j < ceildiv(N, block_size); j++) {
      AccT c_block[block_size * block_size] = {0.0};
      AccT a_block[block_size * block_size];
      AccT b_block[block_size * block_size];

      int k = 0;
      for (; k < K / block_size; k++) {
        // Load a and b blocks
        if (a_trans) {
          load_block<block_size>(a, a_block, K, M, k, i, true);
        } else {
          load_block<block_size>(a, a_block, M, K, i, k, false);
        }
        if (b_trans) {
          load_block<block_size>(b, b_block, N, K, j, k, false);
        } else {
          load_block<block_size>(b, b_block, K, N, k, j, true);
        }

        // Multiply and accumulate
        for (int ii = 0; ii < block_size && i * block_size + ii < M; ++ii) {
          for (int jj = 0; jj < block_size && j * block_size + jj < N; ++jj) {
            for (int kk = 0; kk < block_size; kk += simd_size) {
              auto av =
                  simd::load<AccT, simd_size>(a_block + ii * block_size + kk);
              auto bv =
                  simd::load<AccT, simd_size>(b_block + jj * block_size + kk);
              c_block[ii * block_size + jj] += simd::sum(av * bv);
            }
          }
        }
      }
      if (last_k_block_size) {
        // Load a and b blocks
        if (a_trans) {
          load_block<block_size>(a, a_block, K, M, k, i, true);
        } else {
          load_block<block_size>(a, a_block, M, K, i, k, false);
        }
        if (b_trans) {
          load_block<block_size>(b, b_block, N, K, j, k, false);
        } else {
          load_block<block_size>(b, b_block, K, N, k, j, true);
        }

        // Multiply and accumulate
        for (int ii = 0; ii < block_size && i * block_size + ii < M; ++ii) {
          for (int jj = 0; jj < block_size && j * block_size + jj < N; ++jj) {
            int kk = 0;
            for (; kk < last_k_simd_block; kk += simd_size) {
              auto av =
                  simd::load<AccT, simd_size>(a_block + ii * block_size + kk);
              auto bv =
                  simd::load<AccT, simd_size>(b_block + jj * block_size + kk);
              c_block[ii * block_size + jj] += simd::sum(av * bv);
            }
            for (; kk < last_k_block_size; ++kk) {
              c_block[ii * block_size + jj] +=
                  a_block[ii * block_size + kk] * b_block[jj * block_size + kk];
            }
          }
        }
      }

      // Store
      for (int ii = 0; ii < block_size && i * block_size + ii < M; ++ii) {
        for (int jj = 0; jj < block_size && j * block_size + jj < N; ++jj) {
          auto c_idx = (i * block_size + ii) * N + j * block_size + jj;
          if (beta != 0) {
            c[c_idx] = static_cast<T>(
                alpha * c_block[ii * block_size + jj] + beta * c[c_idx]);
          } else {
            c[c_idx] = static_cast<T>(alpha * c_block[ii * block_size + jj]);
          }
        }
      }
    }
  }
}

} // namespace mlx::core
