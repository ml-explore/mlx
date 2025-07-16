// Copyright © 2024 Apple Inc.

template <typename T, typename U, typename Op, int N = WorkPerThread<T>::n>
[[kernel]] void unary_v(
    device const T* in,
    device U* out,
    constant uint& size,
    uint index [[thread_position_in_grid]]) {
  index *= N;
  if (N > 1 && index + N > size) {
    for (int i = 0; index + i < size; ++i) {
      out[index + i] = Op()(in[index + i]);
    }
  } else {
    for (int i = 0; i < N; ++i) {
      out[index + i] = Op()(in[index + i]);
    }
  }
}

template <typename T, typename U, typename Op, int N = WorkPerThread<T>::n>
[[kernel]] void unary_v2(
    device const T* in,
    device U* out,
    constant int64_t& size,
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  int64_t offset = N * (index.x + grid_dim.x * int64_t(index.y));
  if (N > 1 && offset + N > size) {
    for (int i = 0; offset + i < size; ++i) {
      out[offset + i] = Op()(in[offset + i]);
    }
  } else {
    for (int i = 0; i < N; ++i) {
      out[offset + i] = Op()(in[offset + i]);
    }
  }
}

template <
    typename T,
    typename U,
    typename Op,
    int N = 1,
    typename IdxT = int64_t>
[[kernel]] void unary_g(
    device const T* in,
    device U* out,
    constant const int* in_shape,
    constant const int64_t* in_strides,
    device const int& ndim,
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto idx = elem_to_loc<IdxT>(
      {N * index.x, index.y, index.z}, in_shape, in_strides, ndim);
  auto xshape = in_shape[ndim - 1];
  IdxT xstride = in_strides[ndim - 1];
  IdxT out_idx = N * index.x + xshape * (index.y + IdxT(grid_dim.y) * index.z);
  for (int i = 0; i < N && (int(N * index.x) + i) < xshape; ++i) {
    out[out_idx++] = Op()(in[idx]);
    idx += xstride;
  }
}
