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

  ConvParams(
      const array& in,
      const array& wt,
      const array& out,
      const std::vector<int>& strides,
      const std::vector<int>& padding,
      const std::vector<int>& kernel_dilation,
      const std::vector<int>& input_dilation,
      int groups,
      bool flip)
      : N(in.shape(0)),
        C(in.shape(-1)),
        O(wt.shape(0)),
        groups(groups),
        flip(flip) {
    std::copy_n(strides.begin(), NDIM, this->strides);
    std::copy_n(padding.begin(), NDIM, this->padding);
    std::copy_n(kernel_dilation.begin(), NDIM, this->kernel_dilation);
    std::copy_n(input_dilation.begin(), NDIM, this->input_dilation);
    std::copy_n(in.shape().begin() + 1, NDIM, this->in_spatial_dims);
    std::copy_n(wt.shape().begin() + 1, NDIM, this->wt_spatial_dims);
    std::copy_n(out.shape().begin() + 1, NDIM, this->out_spatial_dims);
    std::copy_n(in.strides().begin(), NDIM + 2, this->in_strides);
  }
};

void gemm_conv(
    cu::CommandEncoder& encoder,
    array in,
    array wt,
    array& out,
    const std::vector<int>& strides,
    const std::vector<int>& padding,
    const std::vector<int>& kernel_dilation,
    const std::vector<int>& input_dilation,
    int groups,
    bool flip,
    Stream s);

} // namespace mlx::core
