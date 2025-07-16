// Copyright © 2024 Apple Inc.

template <typename T, typename U, typename Op>
[[kernel]] void binary_ss(
    device const T* a,
    device const T* b,
    device U* c,
    uint index [[thread_position_in_grid]]) {
  c[index] = Op()(a[0], b[0]);
}

template <typename T, typename U, typename Op, int N = WorkPerThread<T>::n>
[[kernel]] void binary_sv(
    device const T* a,
    device const T* b,
    device U* c,
    constant uint& size,
    uint index [[thread_position_in_grid]]) {
  index *= N;
  if (N > 1 && index + N > size) {
    for (int i = 0; index + i < size; ++i) {
      c[index + i] = Op()(a[0], b[index + i]);
    }
  } else {
    for (int i = 0; i < N; ++i) {
      c[index + i] = Op()(a[0], b[index + i]);
    }
  }
}

template <typename T, typename U, typename Op, int N = WorkPerThread<T>::n>
[[kernel]] void binary_vs(
    device const T* a,
    device const T* b,
    device U* c,
    constant uint& size,
    uint index [[thread_position_in_grid]]) {
  index *= N;
  if (N > 1 && index + N > size) {
    for (int i = 0; index + i < size; ++i) {
      c[index + i] = Op()(a[index + i], b[0]);
    }
  } else {
    for (int i = 0; i < N; ++i) {
      c[index + i] = Op()(a[index + i], b[0]);
    }
  }
}

template <typename T, typename U, typename Op, int N = WorkPerThread<T>::n>
[[kernel]] void binary_vv(
    device const T* a,
    device const T* b,
    device U* c,
    constant uint& size,
    uint index [[thread_position_in_grid]]) {
  index *= N;
  if (N > 1 && index + N > size) {
    for (int i = 0; index + i < size; ++i) {
      c[index + i] = Op()(a[index + i], b[index + i]);
    }
  } else {
    for (int i = 0; i < N; ++i) {
      c[index + i] = Op()(a[index + i], b[index + i]);
    }
  }
}

template <typename T, typename U, typename Op, int N = WorkPerThread<T>::n>
[[kernel]] void binary_sv2(
    device const T* a,
    device const T* b,
    device U* c,
    constant int64_t& size,
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  int64_t offset = N * (index.x + grid_dim.x * int64_t(index.y));
  if (N > 1 && offset + N > size) {
    for (int i = 0; offset + i < size; ++i) {
      c[offset + i] = Op()(a[0], b[offset + i]);
    }
  } else {
    for (int i = 0; i < N; ++i) {
      c[offset + i] = Op()(a[0], b[offset + i]);
    }
  }
}

template <typename T, typename U, typename Op, int N = WorkPerThread<T>::n>
[[kernel]] void binary_vs2(
    device const T* a,
    device const T* b,
    device U* c,
    constant int64_t& size,
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  int64_t offset = N * (index.x + grid_dim.x * int64_t(index.y));
  if (N > 1 && offset + N > size) {
    for (int i = 0; offset + i < size; ++i) {
      c[offset + i] = Op()(a[offset + i], b[0]);
    }
  } else {
    for (int i = 0; i < N; ++i) {
      c[offset + i] = Op()(a[offset + i], b[0]);
    }
  }
}

template <typename T, typename U, typename Op, int N = WorkPerThread<T>::n>
[[kernel]] void binary_vv2(
    device const T* a,
    device const T* b,
    device U* c,
    constant int64_t& size,
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  int64_t offset = N * (index.x + grid_dim.x * int64_t(index.y));
  if (N > 1 && offset + N > size) {
    for (int i = 0; offset + i < size; ++i) {
      c[offset + i] = Op()(a[offset + i], b[offset + i]);
    }
  } else {
    for (int i = 0; i < N; ++i) {
      c[offset + i] = Op()(a[offset + i], b[offset + i]);
    }
  }
}

template <typename T, typename U, typename Op, typename IdxT = int64_t>
[[kernel]] void binary_g_nd1(
    device const T* a,
    device const T* b,
    device U* c,
    constant const int64_t& a_stride,
    constant const int64_t& b_stride,
    uint index [[thread_position_in_grid]]) {
  auto a_idx = elem_to_loc_1<IdxT>(index, a_stride);
  auto b_idx = elem_to_loc_1<IdxT>(index, b_stride);
  c[index] = Op()(a[a_idx], b[b_idx]);
}

template <typename T, typename U, typename Op, typename IdxT = int64_t>
[[kernel]] void binary_g_nd2(
    device const T* a,
    device const T* b,
    device U* c,
    constant const int64_t a_strides[2],
    constant const int64_t b_strides[2],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  auto a_idx = elem_to_loc_2<IdxT>(index, a_strides);
  auto b_idx = elem_to_loc_2<IdxT>(index, b_strides);
  IdxT out_idx = index.x + IdxT(grid_dim.x) * index.y;
  c[out_idx] = Op()(a[a_idx], b[b_idx]);
}

template <typename T, typename U, typename Op, typename IdxT = int64_t>
[[kernel]] void binary_g_nd3(
    device const T* a,
    device const T* b,
    device U* c,
    constant const int64_t a_strides[3],
    constant const int64_t b_strides[3],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto a_idx = elem_to_loc_3<IdxT>(index, a_strides);
  auto b_idx = elem_to_loc_3<IdxT>(index, b_strides);
  IdxT out_idx = index.x + grid_dim.x * (index.y + IdxT(grid_dim.y) * index.z);
  c[out_idx] = Op()(a[a_idx], b[b_idx]);
}

template <
    typename T,
    typename U,
    typename Op,
    int N = 1,
    typename IdxT = int64_t>
[[kernel]] void binary_g(
    device const T* a,
    device const T* b,
    device U* c,
    constant const int* shape,
    constant const int64_t* a_strides,
    constant const int64_t* b_strides,
    constant const int& ndim,
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto idx = elem_to_loc_2_nd<IdxT>(
      {N * index.x, index.y, index.z}, shape, a_strides, b_strides, ndim);
  auto xshape = shape[ndim - 1];
  IdxT out_idx = N * index.x + xshape * (index.y + IdxT(grid_dim.y) * index.z);
  IdxT a_xstride = a_strides[ndim - 1];
  IdxT b_xstride = b_strides[ndim - 1];
  for (int i = 0; i < N && (int(N * index.x) + i) < xshape; ++i) {
    c[out_idx++] = Op()(a[idx.x], b[idx.y]);
    idx.x += a_xstride;
    idx.y += b_xstride;
  }
}
