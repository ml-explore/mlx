// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/backend/metal/kernels/steel/host.h"

template <int NDIM>
struct MLXConvParams {
  const int N; // Batch size
  const int C; // In channels
  const int O; // Out channels
  const int iS[NDIM]; // Input spatial dim
  const int wS[NDIM]; // Weight spatial dim
  const int oS[NDIM]; // Output spatial dim
  const int str[NDIM]; // Kernel strides
  const int pad[NDIM]; // Input padding
  const int kdil[NDIM]; // Kernel dilation
  const int idil[NDIM]; // Input dilation
  const size_t in_strides[NDIM + 2]; // In strides
  const size_t wt_strides[NDIM + 2]; // Wt strides
  const size_t out_strides[NDIM + 2]; // Out strides
  const int groups; // Input channel groups
  const bool flip;
};

namespace mlx {
namespace steel {

struct ImplicitGemmConv2DParams {
  const int M;
  const int N;
  const int K;

  const int gemm_k_iterations;

  const int inp_jump_w;
  const int inp_jump_h;
  const int inp_jump_c;
};

} // namespace steel
} // namespace mlx