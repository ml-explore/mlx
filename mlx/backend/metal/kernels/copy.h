// Copyright © 2024 Apple Inc.

template <typename T, typename U>
[[kernel]] void copy_s(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  dst[index] = static_cast<U>(src[0]);
}

template <typename T, typename U>
[[kernel]] void copy_v(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  dst[index] = static_cast<U>(src[index]);
}

template <typename T, typename U>
[[kernel]] void copy_g_nd1(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    constant const int64_t& src_stride [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  auto src_idx = elem_to_loc_1(index, src_stride);
  dst[index] = static_cast<U>(src[src_idx]);
}

template <typename T, typename U>
[[kernel]] void copy_g_nd2(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    constant const int64_t* src_strides [[buffer(3)]],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  auto src_idx = elem_to_loc_2(index, src_strides);
  int64_t dst_idx = index.x + (int64_t)grid_dim.x * index.y;
  dst[dst_idx] = static_cast<U>(src[src_idx]);
}

template <typename T, typename U>
[[kernel]] void copy_g_nd3(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    constant const int64_t* src_strides [[buffer(3)]],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto src_idx = elem_to_loc_3(index, src_strides);
  int64_t dst_idx =
      index.x + (int64_t)grid_dim.x * (index.y + (int64_t)grid_dim.y * index.z);
  dst[dst_idx] = static_cast<U>(src[src_idx]);
}

template <typename T, typename U, int DIM>
[[kernel]] void copy_g_nd(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    constant const int* src_shape [[buffer(2)]],
    constant const int64_t* src_strides [[buffer(3)]],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto src_idx = elem_to_loc_nd<DIM>(index, src_shape, src_strides);
  int64_t dst_idx =
      index.x + (int64_t)grid_dim.x * (index.y + (int64_t)grid_dim.y * index.z);
  dst[dst_idx] = static_cast<U>(src[src_idx]);
}

template <typename T, typename U>
[[kernel]] void copy_g(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    constant const int* src_shape [[buffer(2)]],
    constant const int64_t* src_strides [[buffer(3)]],
    constant const int& ndim [[buffer(5)]],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto src_idx = elem_to_loc(index, src_shape, src_strides, ndim);
  int64_t dst_idx =
      index.x + (int64_t)grid_dim.x * (index.y + (int64_t)grid_dim.y * index.z);
  dst[dst_idx] = static_cast<U>(src[src_idx]);
}

template <typename T, typename U>
[[kernel]] void copy_gg_nd1(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    constant const int64_t& src_stride [[buffer(3)]],
    constant const int64_t& dst_stride [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  auto src_idx = elem_to_loc_1(index, src_stride);
  auto dst_idx = elem_to_loc_1(index, dst_stride);
  dst[dst_idx] = static_cast<U>(src[src_idx]);
}

template <typename T, typename U>
[[kernel]] void copy_gg_nd2(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    constant const int64_t* src_strides [[buffer(3)]],
    constant const int64_t* dst_strides [[buffer(4)]],
    uint2 index [[thread_position_in_grid]]) {
  auto src_idx = elem_to_loc_2(index, src_strides);
  auto dst_idx = elem_to_loc_2(index, dst_strides);
  dst[dst_idx] = static_cast<U>(src[src_idx]);
}

template <typename T, typename U>
[[kernel]] void copy_gg_nd3(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    constant const int64_t* src_strides [[buffer(3)]],
    constant const int64_t* dst_strides [[buffer(4)]],
    uint3 index [[thread_position_in_grid]]) {
  auto src_idx = elem_to_loc_3(index, src_strides);
  auto dst_idx = elem_to_loc_3(index, dst_strides);
  dst[dst_idx] = static_cast<U>(src[src_idx]);
}

template <typename T, typename U, int DIM>
[[kernel]] void copy_gg_nd(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    constant const int* src_shape [[buffer(2)]],
    constant const int64_t* src_strides [[buffer(3)]],
    constant const int64_t* dst_strides [[buffer(4)]],
    uint3 index [[thread_position_in_grid]]) {
  auto src_idx = elem_to_loc_nd<DIM>(index, src_shape, src_strides);
  auto dst_idx = elem_to_loc_nd<DIM>(index, src_shape, dst_strides);
  dst[dst_idx] = static_cast<U>(src[src_idx]);
}

template <typename T, typename U>
[[kernel]] void copy_gg(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    constant const int* src_shape [[buffer(2)]],
    constant const int64_t* src_strides [[buffer(3)]],
    constant const int64_t* dst_strides [[buffer(4)]],
    constant const int& ndim [[buffer(5)]],
    uint3 index [[thread_position_in_grid]]) {
  auto src_idx = elem_to_loc(index, src_shape, src_strides, ndim);
  auto dst_idx = elem_to_loc(index, src_shape, dst_strides, ndim);
  dst[dst_idx] = static_cast<U>(src[src_idx]);
}
