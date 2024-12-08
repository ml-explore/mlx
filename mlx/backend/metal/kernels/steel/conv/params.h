// Copyright © 2024 Apple Inc.

#pragma once

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
  const int64_t in_strides[NDIM + 2]; // In strides
  const int64_t wt_strides[NDIM + 2]; // Wt strides
  const int64_t out_strides[NDIM + 2]; // Out strides
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

  const int tiles_n;
  const int tiles_m;
  const int swizzle_log;
};

struct Conv2DGeneralJumpParams {
  const int f_wgt_jump_h;
  const int f_wgt_jump_w;

  const int f_out_jump_h;
  const int f_out_jump_w;

  const int adj_out_h;
  const int adj_out_w;
  const int adj_out_hw;
  const int adj_implicit_m;
};

struct Conv2DGeneralBaseInfo {
  int weight_base;
  int weight_size;
};

} // namespace steel
} // namespace mlx
