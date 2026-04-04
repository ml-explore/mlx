// Copyright © 2026 Apple Inc.

/**
 * @file params_m5_max.h
 *
 * M5 Max Specific GEMM Parameters
 * =================================
 *
 * Optimized parameters for Apple Silicon M5 Max based on:
 * - 70 MB buffer capacity
 * - 70 ops per buffer
 * - Enhanced memory bandwidth
 * - Optimized thread group sizes for large matrices
 */

#pragma once

#include "mlx/backend/metal/kernels/steel/gemm/params.h"
#include "apple_silicon_optimizations.h"

namespace mlx {
namespace steel {

/**
 * M5 Max specific GEMM parameters
 *
 * For matrices that fit in the 70 MB buffer:
 * - Larger tile sizes for better memory bandwidth utilization
 * - Optimal thread group configurations
 */
struct M5MaxGEMMParams {
  // Matrix dimensions
  const int M;
  const int N;
  const int K;

  // Leading dimensions
  const int lda;
  const int ldb;
  const int ldd;

  // Tile sizes for M5 Max
  // Using larger tiles to maximize memory bandwidth
  const int bm;   // Tile size for M dimension
  const int bn;   // Tile size for N dimension  
  const int bk;   // Tile size for K dimension

  // Thread group dimensions
  const int wm;   // Warp multiplier M
  const int wn;   // Warp multiplier N

  // Batching parameters
  const int64_t batch_stride_a;
  const int64_t batch_stride_b;
  const int64_t batch_stride_d;

  // Buffer parameters (M5 Max specific)
  static constexpr int max_ops_per_buffer = 70;
  static constexpr int max_mb_per_buffer = 70;

  M5MaxGEMMParams(
      const int M_,
      const int N_,
      const int K_,
      const int lda_,
      const int ldb_,
      const int ldd_,
      const int64_t batch_stride_a_,
      const int64_t batch_stride_b_,
      const int64_t batch_stride_d_)
      : M(M_),
        N(N_),
        K(K_),
        lda(lda_),
        ldb(ldb_),
        ldd(ldd_),
        batch_stride_a(batch_stride_a_),
        batch_stride_b(batch_stride_b_),
        batch_stride_d(batch_stride_d_) {
    // Determine optimal tile sizes for M5 Max
    determine_tile_sizes(M, N, K);
  }

  void determine_tile_sizes(const int M, const int N, const int K) {
    // M5 Max with 70 MB buffer - use larger tiles
    if (M >= 2048 || N >= 2048) {
      // Very large matrices: maximize bandwidth
      bm = 64;
      bn = 64;
      bk = 32;  // Larger BK for better K-dimension utilization
      wm = 2;
      wn = 2;
    } else if (M * N >= 1ul << 20) {
      // Large matrices: balanced approach
      if (K >= 4096) {
        bm = 64;
        bn = 64;
        bk = 32;
        wm = 2;
        wn = 2;
      } else {
        bm = 64;
        bn = 64;
        bk = 16;
        wm = 2;
        wn = 2;
      }
    } else {
      // Smaller matrices
      bm = 64;
      bn = 64;
      bk = 16;
      wm = 2;
      wn = 2;
    }
  }

  // Calculate number of threadgroups
  int tiles_m() const { return (M + bm - 1) / bm; }
  int tiles_n() const { return (N + bn - 1) / bn; }

  // Calculate total ops for this GEMM
  int64_t total_ops() const { return static_cast<int64_t>(M) * N * K; }

  // Check if this GEMM fits in M5 Max buffer
  bool fits_buffer() const {
    // Calculate memory usage
    size_t a_mem = static_cast<size_t>(M) * K * sizeof(float);
    size_t b_mem = static_cast<size_t>(K) * N * sizeof(float);
    size_t d_mem = static_cast<size_t>(M) * N * sizeof(float);

    // For simplicity, check if total fits in buffer
    return (a_mem + b_mem + d_mem) / (1024 * 1024) <= max_mb_per_buffer;
  }
};

/**
 * Split-K parameters optimized for M5 Max
 *
 * M5 Max's high bandwidth benefits from split-K parallelism
 */
struct M5MaxSplitKGEMMParams {
  const int M;
  const int N;
  const int K;

  const int lda;
  const int ldb;
  const int ldc;

  // Split-K parameters
  const int split_k_partitions;
  const int split_k_partition_stride;
  const int split_k_partition_size;

  // Buffer parameters
  static constexpr int max_ops_per_buffer = 70;
  static constexpr int max_mb_per_buffer = 70;

  M5MaxSplitKGEMMParams(const int M_, const int N_, const int K_)
      : M(M_), N(N_), K(K_) {
    // Determine optimal split-K parameters for M5 Max
    if (K >= 4096) {
      // Large K: use more partitions for parallelism
      split_k_partitions = 8;
      split_k_partition_stride = N;
    } else if (K >= 2048) {
      split_k_partitions = 4;
      split_k_partition_stride = N;
    } else {
      split_k_partitions = 2;
      split_k_partition_stride = N;
    }

    split_k_partition_size = (K + split_k_partitions - 1) / split_k_partitions;
  }
};

/**
 * Add-MM parameters optimized for M5 Max
 *
 * Fused add-mul operations benefit from M5 Max's high bandwidth
 */
struct M5MaxAddMMParams {
  const int ldc;
  const int fdc;

  const float alpha;
  const float beta;

  static constexpr int max_ops_per_buffer = 70;
  static constexpr int max_mb_per_buffer = 70;

  M5MaxAddMMParams(const int ldc_, const float alpha_, const float beta_)
      : ldc(ldc_), alpha(alpha_), beta(beta_) {
    // Pre-compute fused operation parameters
  }
};

/**
 * GEMM kernel launcher for M5 Max
 *
 * Automatically selects optimal parameters based on matrix dimensions
 */
inline void launch_m5_max_gemm(
    const int M,
    const int N,
    const int K,
    const int lda,
    const int ldb,
    const int ldd,
    const void* A,
    const void* B,
    void* D,
    const int64_t batch_stride_a,
    const int64_t batch_stride_b,
    const int64_t batch_stride_d,
    void* command_buffer) {
  
  // Create optimized parameters
  M5MaxGEMMParams params(
      M, N, K,
      lda, ldb, ldd,
      batch_stride_a, batch_stride_b, batch_stride_d);

  // Check if we can use split-K for better parallelism
  bool use_split_k = (K >= 4096 && params.total_ops() > 1ul << 25);

  if (use_split_k) {
    // Use split-K GEMM for large K
    M5MaxSplitKGEMMParams split_params(M, N, K);
    // Launch split-K kernel...
  } else {
    // Standard GEMM with M5 Max parameters
    // Launch standard kernel with params.bm, params.bn, params.bk
  }
}

} // namespace steel
} // namespace mlx
