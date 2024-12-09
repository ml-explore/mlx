// Copyright © 2023 Apple Inc.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

template <typename T>
[[kernel]] void axpby_general(
    device const T* x [[buffer(0)]],
    device const T* y [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant const float& alpha [[buffer(3)]],
    constant const float& beta [[buffer(4)]],
    constant const int* shape [[buffer(5)]],
    constant const int64_t* x_strides [[buffer(6)]],
    constant const int64_t* y_strides [[buffer(7)]],
    constant const int& ndim [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
  auto x_offset = elem_to_loc(index, shape, x_strides, ndim);
  auto y_offset = elem_to_loc(index, shape, y_strides, ndim);
  out[index] =
      static_cast<T>(alpha) * x[x_offset] + static_cast<T>(beta) * y[y_offset];
}

template <typename T>
[[kernel]] void axpby_contiguous(
    device const T* x [[buffer(0)]],
    device const T* y [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant const float& alpha [[buffer(3)]],
    constant const float& beta [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  out[index] =
      static_cast<T>(alpha) * x[index] + static_cast<T>(beta) * y[index];
}

// clang-format off
#define instantiate_axpby(type_name, type)                             \
  instantiate_kernel("axpby_general_" #type_name, axpby_general, type) \
  instantiate_kernel(                                                  \
          "axpby_contiguous_" #type_name, axpby_contiguous, type)

instantiate_axpby(float32, float);
instantiate_axpby(float16, half);
instantiate_axpby(bfloat16, bfloat16_t);
instantiate_axpby(complex64, complex64_t);
// clang-format on
