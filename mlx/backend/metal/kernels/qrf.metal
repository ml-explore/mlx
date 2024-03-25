// Copyright © 2024 Apple Inc.

#include <metal_math>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/atomic.h"

namespace {
size_t colmajor_idx(size_t row, size_t col, size_t nrows) {
  return col * nrows + row;
}
} // namespace

[[kernel]] void qrf_reset(
    [[ maybe_unused ]] device const float* a_data,
    [[ maybe_unused ]] device float* y_data,
    device mlx_atomic<float>* yp_data,
    device mlx_atomic<float>* betas_data,
    constant const int& jj,
    [[ maybe_unused ]] constant const int& startc,
    [[ maybe_unused ]] constant const int& m,
    uint i [[thread_position_in_grid]]) {
  if (i == 0) {
    mlx_atomic_store_explicit(betas_data, 0.0f, jj);
    mlx_atomic_store_explicit(yp_data, 0.0f, 0);
  }
}

[[kernel]] void qrf_compute_sq_norm_xc(
    device const float* a_data,
    device float* y_data,
    device mlx_atomic<float>* yp_data,
    [[ maybe_unused ]] device mlx_atomic<float>* betas_data,
    constant const int& jj,
    constant const int& startc,
    constant const int& m,
    uint i [[thread_position_in_grid]]) {
  (void)y_data;

  const int col = startc + jj;

  float val = 0;
  if (i >= static_cast<uint>(col)) {
    const auto x = a_data[colmajor_idx(i, col, m)];
    val = x * x;
  } else {
    val = 0;
  }

  // TODO(nicolov): replace atomics with the existing allreduce kernel.
  mlx_atomic_fetch_add_explicit(yp_data, val, 0);
}

[[kernel]] void qrf_compute_v(
    device const float* a_data,
    device float* y_data,
    device float* yp_data,
    device mlx_atomic<float>* betas_data,
    constant const int& jj,
    constant const int& startc,
    constant const int& m,
    uint i [[thread_position_in_grid]]) {
  const int col = startc + jj;

  const float sq_norm_xc = yp_data[0];
  const float norm_xc = metal::sqrt(sq_norm_xc);

  // Build Householder vector v (stored in the jj-th column of Y).
  device float* v = y_data + colmajor_idx(0, jj, m);

  float new_val = 0;
  float sq_val = 0;
  if (i < static_cast<uint>(col)) {
    new_val = 0;
  } else if (i == static_cast<uint>(col)) {
    const float old_val = a_data[colmajor_idx(i, i, m)];
    new_val = old_val >= 0 ? old_val + norm_xc : old_val - norm_xc;
  } else {
    new_val = a_data[colmajor_idx(i, col, m)];
  }
  v[i] = new_val;
  sq_val = new_val * new_val;

  // TODO(nicolov): replace atomics with the existing allreduce kernel.
  mlx_atomic_fetch_add_explicit(betas_data, sq_val, jj);
}

[[kernel]] void qrf_compute_beta(
    [[ maybe_unused ]] device const float* a_data,
    [[ maybe_unused ]] device const float* y_data,
    [[ maybe_unused ]] device float* yp_data,
    device float* betas_data,
    [[ maybe_unused ]] constant const int& jj,
    [[ maybe_unused ]] constant const int& startc,
    [[ maybe_unused ]] constant const int& m,
    uint i [[thread_position_in_grid]]) {
  if (i == 0) {
    betas_data[jj] = 2.0f / betas_data[jj];
  }
}

[[kernel]] void qrf_reflect_current_block(
    device const float* betas_data,
    device const float* y_data,
    device float* a_data,
    device const float* wp_data,
    constant const int& jj,
    constant const int& startc,
    constant const int& m,
    constant const int& R,
    uint i [[thread_position_in_grid]]) {
  const device float* v = y_data + colmajor_idx(0, jj, m);
  const float beta = betas_data[jj];

  const int col = startc + jj;

  for (int j = 0; j < startc + R - col; j++) {
    a_data[colmajor_idx(i, col + j, m)] -= beta * v[i] * wp_data[j];
  }
}

[[kernel]] void qrf_compute_yp(
    device const float* y_data,
    device float* yp_data,
    constant const int& m,
    constant const int& R,
    uint2 index [[thread_position_in_grid]]) {
  int i = index.x;
  int j = index.y;

  float res = 0;
  for (int k = 0; k < m; k++) {
    res += y_data[colmajor_idx(k, i, m)] * y_data[colmajor_idx(k, j, m)];
  }

  yp_data[colmajor_idx(i, j, R)] = res;
}

[[kernel]] void qrf_compute_w_col(
    device const float* betas_data,
    device const float* y_data,
    device const float* yp_data,
    device float* w_data,
    constant const int& m,
    constant const int& R,
    uint i [[thread_position_in_grid]]) {
  for (int j = 0; j < R; j++) {
    const auto loc = colmajor_idx(i, j, m);
    float z = 0;
    z = 0;
    z -= betas_data[j] * y_data[loc];
    for (int k = 0; k < j; k++) {
      z -= betas_data[j] * w_data[colmajor_idx(i, k, m)] *
          yp_data[colmajor_idx(k, j, R)];
    }
    w_data[loc] = z;
  }
}
