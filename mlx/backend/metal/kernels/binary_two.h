// Copyright © 2024 Apple Inc.

template <typename T, typename U, typename Op>
[[kernel]] void binary_ss(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    uint index [[thread_position_in_grid]]) {
  auto out = Op()(a[0], b[0]);
  c[index] = out[0];
  d[index] = out[1];
}

template <typename T, typename U, typename Op>
[[kernel]] void binary_sv(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    uint index [[thread_position_in_grid]]) {
  auto out = Op()(a[0], b[index]);
  c[index] = out[0];
  d[index] = out[1];
}

template <typename T, typename U, typename Op>
[[kernel]] void binary_vs(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    uint index [[thread_position_in_grid]]) {
  auto out = Op()(a[index], b[0]);
  c[index] = out[0];
  d[index] = out[1];
}

template <typename T, typename U, typename Op>
[[kernel]] void binary_vv(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    uint index [[thread_position_in_grid]]) {
  auto out = Op()(a[index], b[index]);
  c[index] = out[0];
  d[index] = out[1];
}

template <typename T, typename U, typename Op>
[[kernel]] void binary_g_nd1(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    constant const size_t& a_stride,
    constant const size_t& b_stride,
    uint index [[thread_position_in_grid]]) {
  auto a_idx = elem_to_loc_1(index, a_stride);
  auto b_idx = elem_to_loc_1(index, b_stride);
  auto out = Op()(a[a_idx], b[b_idx]);
  c[index] = out[0];
  d[index] = out[1];
}

template <typename T, typename U, typename Op>
[[kernel]] void binary_g_nd2(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    constant const size_t a_strides[2],
    constant const size_t b_strides[2],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  auto a_idx = elem_to_loc_2(index, a_strides);
  auto b_idx = elem_to_loc_2(index, b_strides);
  size_t out_idx = index.x + (size_t)grid_dim.x * index.y;
  auto out = Op()(a[a_idx], b[b_idx]);
  c[out_idx] = out[0];
  d[out_idx] = out[1];
}

template <typename T, typename U, typename Op>
[[kernel]] void binary_g_nd3(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    constant const size_t a_strides[3],
    constant const size_t b_strides[3],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto a_idx = elem_to_loc_3(index, a_strides);
  auto b_idx = elem_to_loc_3(index, b_strides);
  size_t out_idx =
      index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
  auto out = Op()(a[a_idx], b[b_idx]);
  c[out_idx] = out[0];
  d[out_idx] = out[1];
}

template <typename T, typename U, typename Op, int DIM>
[[kernel]] void binary_g_nd(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    constant const int shape[DIM],
    constant const size_t a_strides[DIM],
    constant const size_t b_strides[DIM],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto idx = elem_to_loc_2_nd<DIM>(index, shape, a_strides, b_strides);
  size_t out_idx =
      index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
  auto out = Op()(a[idx.x], b[idx.y]);
  c[out_idx] = out[0];
  d[out_idx] = out[1];
}

template <typename T, typename U, typename Op>
[[kernel]] void binary_g(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    constant const int* shape,
    constant const size_t* a_strides,
    constant const size_t* b_strides,
    constant const int& ndim,
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto idx = elem_to_loc_2_nd(index, shape, a_strides, b_strides, ndim);
  size_t out_idx = index.x + grid_dim.x * (index.y + grid_dim.y * index.z);
  auto out = Op()(a[idx.x], b[idx.y]);
  c[out_idx] = out[0];
  d[out_idx] = out[1];
}
