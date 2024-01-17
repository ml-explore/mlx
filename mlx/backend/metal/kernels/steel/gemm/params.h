// Copyright © 2024 Apple Inc.

#pragma once

///////////////////////////////////////////////////////////////////////////////
// GEMM param classes
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

struct GEMMParams {
  const int M;
  const int N;
  const int K;

  const int lda;
  const int ldb;
  const int ldc;

  const int tiles_n;
  const int tiles_m;

  const int batch_stride_a;
  const int batch_stride_b;
  const int batch_stride_c;

  const int swizzle_log;
  const int gemm_k_iterations_aligned;
};

struct GEMMSpiltKParams {
  const int M;
  const int N;
  const int K;

  const int lda;
  const int ldb;
  const int ldc;

  const int tiles_n;
  const int tiles_m;

  const int split_k_partitions;
  const int split_k_partition_stride;
  const int split_k_partition_size;

  const int gemm_k_iterations_aligned;
};

struct GEMMAddMMParams {
  const int M;
  const int N;
  const int K;

  const int lda;
  const int ldb;
  const int ldc;
  const int ldd;

  const int tiles_n;
  const int tiles_m;

  const int batch_stride_a;
  const int batch_stride_b;
  const int batch_stride_c;
  const int batch_stride_d;

  const int swizzle_log;
  const int gemm_k_iterations_aligned;

  const float alpha;
  const float beta;

  const int fdc;
};

} // namespace steel
} // namespace mlx
