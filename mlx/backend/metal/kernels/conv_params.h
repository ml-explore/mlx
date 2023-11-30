// Copyright © 2023 Apple Inc.

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
  const int dil[NDIM]; // Kernel dilation
  const size_t in_strides[NDIM + 2]; // In strides
  const size_t wt_strides[NDIM + 2]; // Wt strides
  const size_t out_strides[NDIM + 2]; // Out strides
};
