// Copyright © 2024 Apple Inc.

template <typename T, typename Op>
[[kernel]] void ternary_v(
    device const bool* a,
    device const T* b,
    device const T* c,
    device T* d,
    uint index [[thread_position_in_grid]]) {
  d[index] = Op()(a[index], b[index], c[index]);
}

template <typename T, typename Op>
[[kernel]] void ternary_v2(
    device const bool* a,
    device const T* b,
    device const T* c,
    device T* d,
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  auto offset = index.x + grid_dim.x * int64_t(index.y);
  d[offset] = Op()(a[offset], b[offset], c[offset]);
}

template <typename T, typename Op, typename IdxT = int64_t>
[[kernel]] void ternary_g_nd1(
    device const bool* a,
    device const T* b,
    device const T* c,
    device T* d,
    constant const int64_t& a_strides,
    constant const int64_t& b_strides,
    constant const int64_t& c_strides,
    uint index [[thread_position_in_grid]]) {
  auto a_idx = elem_to_loc_1<IdxT>(index, a_strides);
  auto b_idx = elem_to_loc_1<IdxT>(index, b_strides);
  auto c_idx = elem_to_loc_1<IdxT>(index, c_strides);
  d[index] = Op()(a[a_idx], b[b_idx], c[c_idx]);
}

template <typename T, typename Op, typename IdxT = int64_t>
[[kernel]] void ternary_g_nd2(
    device const bool* a,
    device const T* b,
    device const T* c,
    device T* d,
    constant const int64_t a_strides[2],
    constant const int64_t b_strides[2],
    constant const int64_t c_strides[2],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  auto a_idx = elem_to_loc_2<IdxT>(index, a_strides);
  auto b_idx = elem_to_loc_2<IdxT>(index, b_strides);
  auto c_idx = elem_to_loc_2<IdxT>(index, c_strides);
  IdxT out_idx = index.x + IdxT(grid_dim.x) * index.y;
  d[out_idx] = Op()(a[a_idx], b[b_idx], c[c_idx]);
}

template <typename T, typename Op, typename IdxT = int64_t>
[[kernel]] void ternary_g_nd3(
    device const bool* a,
    device const T* b,
    device const T* c,
    device T* d,
    constant const int64_t a_strides[3],
    constant const int64_t b_strides[3],
    constant const int64_t c_strides[3],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto a_idx = elem_to_loc_3<IdxT>(index, a_strides);
  auto b_idx = elem_to_loc_3<IdxT>(index, b_strides);
  auto c_idx = elem_to_loc_3<IdxT>(index, c_strides);
  IdxT out_idx = index.x + grid_dim.x * (index.y + IdxT(grid_dim.y) * index.z);
  d[out_idx] = Op()(a[a_idx], b[b_idx], c[c_idx]);
}

template <typename T, typename Op, int N = 1, typename IdxT = int64_t>
[[kernel]] void ternary_g(
    device const bool* a,
    device const T* b,
    device const T* c,
    device T* d,
    constant const int* shape,
    constant const int64_t* a_strides,
    constant const int64_t* b_strides,
    constant const int64_t* c_strides,
    constant const int& ndim,
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto idx = elem_to_loc_3_nd<IdxT>(
      {N * index.x, index.y, index.z},
      shape,
      a_strides,
      b_strides,
      c_strides,
      ndim);
  auto xshape = shape[ndim - 1];
  IdxT out_idx = N * index.x + xshape * (index.y + IdxT(grid_dim.y) * index.z);
  IdxT a_xstride = a_strides[ndim - 1];
  IdxT b_xstride = b_strides[ndim - 1];
  IdxT c_xstride = c_strides[ndim - 1];
  for (int i = 0; i < N && (int(N * index.x) + i) < xshape; ++i) {
    d[out_idx++] = Op()(a[idx.x], b[idx.y], c[idx.z]);
    idx.x += a_xstride;
    idx.y += b_xstride;
    idx.z += c_xstride;
  }
}
