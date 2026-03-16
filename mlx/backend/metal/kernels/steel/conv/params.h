// Copyright © 2024 Apple Inc.

#pragma once

template <int NDIM>
struct MLXConvParams {
  int N; // Batch size
  int C; // In channels
  int O; // Out channels
  int iS[NDIM]; // Input spatial dim
  int wS[NDIM]; // Weight spatial dim
  int oS[NDIM]; // Output spatial dim
  int str[NDIM]; // Kernel strides
  int pad[NDIM]; // Input padding
  int kdil[NDIM]; // Kernel dilation
  int idil[NDIM]; // Input dilation
  int64_t in_strides[NDIM + 2]; // In strides
  int64_t wt_strides[NDIM + 2]; // Wt strides
  int64_t out_strides[NDIM + 2]; // Out strides
  int groups; // Input channel groups
  bool flip;

  static MLXConvParams<NDIM>
  with_padded_channels(MLXConvParams<NDIM> other, int pad_out, int pad_in) {
    MLXConvParams<NDIM> params = other;

    // Update strides
    for (int i = 0; i < NDIM + 1; i++) {
      params.in_strides[i] =
          (params.in_strides[i] / params.C) * (params.C + pad_in);
      params.wt_strides[i] =
          (params.wt_strides[i] / params.C) * (params.C + pad_in);
      params.out_strides[i] =
          (params.out_strides[i] / params.O) * (params.O + pad_out);
    }
    params.in_strides[NDIM + 1] = 1;
    params.wt_strides[NDIM + 1] = 1;
    params.out_strides[NDIM + 1] = 1;

    // Update channels
    params.C += pad_in;
    params.O += pad_out;

    return params;
  };
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

struct ImplicitGemmConv3DParams {
  const int M;
  const int N;
  const int K;

  const int gemm_k_iterations;

  const int inp_jump_w;
  const int inp_jump_h;
  const int inp_jump_d;
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
