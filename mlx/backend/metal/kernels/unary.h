// Copyright © 2024 Apple Inc.

template <typename T, typename Op>
[[kernel]] void unary_v(
    device const T* in,
    device T* out,
    uint index [[thread_position_in_grid]]) {
  out[index] = Op()(in[index]);
}

template <typename T, typename Op>
[[kernel]] void unary_g(
    device const T* in,
    device T* out,
    device const int* in_shape,
    device const size_t* in_strides,
    device const int& ndim,
    uint index [[thread_position_in_grid]]) {
  auto idx = elem_to_loc(index, in_shape, in_strides, ndim);
  out[index] = Op()(in[idx]);
}
