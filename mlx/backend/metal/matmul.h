// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/backend/metal/device.h"

namespace mlx::core {

template <bool CHECK_AB = true>
void steel_matmul_regular_axpby(
    const Stream& s,
    metal::Device& d,
    const array& a,
    const array& b,
    const array& c,
    array& out,
    int M,
    int N,
    int K,
    int batch_size_out,
    int lda,
    int ldb,
    int ldd,
    bool transpose_a,
    bool transpose_b,
    std::vector<array>& copies,
    Shape batch_shape,
    Strides batch_strides,
    int64_t A_batch_stride,
    int64_t B_batch_stride,
    int64_t matrix_stride_out,
    int64_t C_batch_stride = 0,
    float alpha = 1.0f,
    float beta = 0.0f);

inline void steel_matmul_regular(
    const Stream& s,
    metal::Device& d,
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    int batch_size_out,
    int lda,
    int ldb,
    int ldd,
    bool transpose_a,
    bool transpose_b,
    std::vector<array>& copies,
    Shape batch_shape,
    Strides batch_strides,
    int64_t A_batch_stride,
    int64_t B_batch_stride,
    int64_t matrix_stride_out) {
  return steel_matmul_regular_axpby<false>(
      /* const Stream& s = */ s,
      /* metal::Device& d = */ d,
      /* const array& a = */ a,
      /* const array& b = */ b,
      /* const array& c = */ b,
      /* array& out = */ out,
      /* int M = */ M,
      /* int N = */ N,
      /* int K = */ K,
      /* int batch_size_out = */ batch_size_out,
      /* int lda = */ lda,
      /* int ldb = */ ldb,
      /* int ldd = */ ldd,
      /* bool transpose_a = */ transpose_a,
      /* bool transpose_b = */ transpose_b,
      /* std::vector<array>& copies = */ copies,
      /* Shape batch_shape = */ batch_shape,
      /* Strides batch_strides = */ batch_strides,
      /* int64_t A_batch_stride = */ A_batch_stride,
      /* int64_t B_batch_stride = */ B_batch_stride,
      /* int64_t matrix_stride_out = */ matrix_stride_out);
}

template <bool CHECK_AB = true>
void steel_matmul_axpby(
    const Stream& s,
    metal::Device& d,
    const array& a,
    const array& b,
    const array& c,
    array& out,
    int M,
    int N,
    int K,
    int batch_size_out,
    int lda,
    int ldb,
    bool transpose_a,
    bool transpose_b,
    std::vector<array>& copies,
    Shape batch_shape = {},
    Strides A_batch_stride = {},
    Strides B_batch_stride = {},
    Strides C_batch_stride = {},
    float alpha = 1.0f,
    float beta = 0.0f);

inline void steel_matmul(
    const Stream& s,
    metal::Device& d,
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    int batch_size_out,
    int lda,
    int ldb,
    bool transpose_a,
    bool transpose_b,
    std::vector<array>& copies,
    Shape batch_shape = {},
    Strides A_batch_stride = {},
    Strides B_batch_stride = {}) {
  return steel_matmul_axpby<false>(
      /* const Stream& s = */ s,
      /* metal::Device& d = */ d,
      /* const array& a = */ a,
      /* const array& b = */ b,
      /* const array& c = */ b,
      /* array& out = */ out,
      /* int M = */ M,
      /* int N = */ N,
      /* int K = */ K,
      /* int batch_size_out = */ batch_size_out,
      /* int lda = */ lda,
      /* int ldb = */ ldb,
      /* bool transpose_a = */ transpose_a,
      /* bool transpose_b = */ transpose_b,
      /* std::vector<array>& copies = */ copies,
      /* Shape batch_shape = */ batch_shape,
      /* Strides A_batch_stride = */ A_batch_stride,
      /* Strides B_batch_stride = */ B_batch_stride);
}

} // namespace mlx::core
