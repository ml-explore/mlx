// Copyright © 2024 Apple Inc.

template <typename T, typename U, typename Op>
[[kernel]] void unary_v(
    device const T* in,
    device U* out,
    uint index [[thread_position_in_grid]]) {
  out[index] = Op()(in[index]);
}

template <typename T, typename U, typename Op>
[[kernel]] void unary_v2(
    device const T* in,
    device U* out,
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  size_t offset = index.x + grid_dim.x * size_t(index.y);
  out[offset] = Op()(in[offset]);
}

template <typename T, typename U, typename Op, int N = 1>
[[kernel]] void unary_g(
    device const T* in,
    device U* out,
    constant const int* in_shape,
    constant const size_t* in_strides,
    device const int& ndim,
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto idx =
      elem_to_loc({N * index.x, index.y, index.z}, in_shape, in_strides, ndim);
  auto xshape = in_shape[ndim - 1];
  auto xstride = in_strides[ndim - 1];
  size_t out_idx =
      N * index.x + xshape * (index.y + size_t(grid_dim.y) * index.z);
  for (int i = 0; i < N && (int(N * index.x) + i) < xshape; ++i) {
    out[out_idx++] = Op()(in[idx]);
    idx += xstride;
  }
}
