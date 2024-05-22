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
[[kernel]] void ternary_g_nd1(
    device const bool* a,
    device const T* b,
    device const T* c,
    device T* d,
    constant const size_t& a_strides,
    constant const size_t& b_strides,
    constant const size_t& c_strides,
    uint index [[thread_position_in_grid]]) {
  auto a_idx = elem_to_loc_1(index, a_strides);
  auto b_idx = elem_to_loc_1(index, b_strides);
  auto c_idx = elem_to_loc_1(index, c_strides);
  d[index] = Op()(a[a_idx], b[b_idx], c[c_idx]);
}

template <typename T, typename Op>
[[kernel]] void ternary_g_nd2(
    device const bool* a,
    device const T* b,
    device const T* c,
    device T* d,
    constant const size_t a_strides[2],
    constant const size_t b_strides[2],
    constant const size_t c_strides[2],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  auto a_idx = elem_to_loc_2(index, a_strides);
  auto b_idx = elem_to_loc_2(index, b_strides);
  auto c_idx = elem_to_loc_2(index, c_strides);
  size_t out_idx = index.x + (size_t)grid_dim.x * index.y;
  d[out_idx] = Op()(a[a_idx], b[b_idx], c[c_idx]);
}

template <typename T, typename Op>
[[kernel]] void ternary_g_nd3(
    device const bool* a,
    device const T* b,
    device const T* c,
    device T* d,
    constant const size_t a_strides[3],
    constant const size_t b_strides[3],
    constant const size_t c_strides[3],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto a_idx = elem_to_loc_3(index, a_strides);
  auto b_idx = elem_to_loc_3(index, b_strides);
  auto c_idx = elem_to_loc_3(index, c_strides);
  size_t out_idx =
      index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
  d[out_idx] = Op()(a[a_idx], b[b_idx], c[c_idx]);
}

template <typename T, typename Op, int DIM>
[[kernel]] void ternary_g_nd(
    device const bool* a,
    device const T* b,
    device const T* c,
    device T* d,
    constant const int shape[DIM],
    constant const size_t a_strides[DIM],
    constant const size_t b_strides[DIM],
    constant const size_t c_strides[DIM],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto idx =
      elem_to_loc_3_nd<DIM>(index, shape, a_strides, b_strides, c_strides);
  size_t out_idx =
      index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
  d[out_idx] = Op()(a[idx.x], b[idx.y], c[idx.z]);
}

template <typename T, typename Op>
[[kernel]] void ternary_g(
    device const bool* a,
    device const T* b,
    device const T* c,
    device T* d,
    constant const int* shape,
    constant const size_t* a_strides,
    constant const size_t* b_strides,
    constant const size_t* c_strides,
    constant const int& ndim,
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto idx =
      elem_to_loc_3_nd(index, shape, a_strides, b_strides, c_strides, ndim);
  size_t out_idx = index.x + grid_dim.x * (index.y + grid_dim.y * index.z);
  d[out_idx] = Op()(a[idx.x], b[idx.y], c[idx.z]);
}
