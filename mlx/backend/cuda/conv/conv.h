// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device.h"

namespace mlx::core {

template <int NDIM>
struct ConvParams {
  int N; // Batch size
  int C; // In channels
  int O; // Out channels
  int strides[NDIM];
  int padding[NDIM];
  int kernel_dilation[NDIM];
  int input_dilation[NDIM];
  int groups;
  bool flip;
  int in_spatial_dims[NDIM];
  int wt_spatial_dims[NDIM];
  int out_spatial_dims[NDIM];
  int64_t in_strides[NDIM + 2];
  int64_t wt_strides[NDIM + 2];
  int64_t out_strides[NDIM + 2];
};

void gemm_conv(
    cu::CommandEncoder& encoder,
    const array& in,
    const array& wt,
    array& out,
    const std::vector<int>& strides,
    const std::vector<int>& padding,
    const std::vector<int>& kernel_dilation,
    const std::vector<int>& input_dilation,
    int groups,
    bool flip);

} // namespace mlx::core
