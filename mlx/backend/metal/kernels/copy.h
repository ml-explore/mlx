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
[[kernel]] void copy_s2(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  auto offset = index.x + grid_dim.x * int64_t(index.y);
  dst[offset] = static_cast<U>(src[0]);
}

template <typename T, typename U>
[[kernel]] void copy_v2(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  auto offset = index.x + grid_dim.x * int64_t(index.y);
  dst[offset] = static_cast<U>(src[offset]);
}

template <typename T, typename U, typename IdxT = int64_t>
[[kernel]] void copy_g_nd1(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    constant const int64_t& src_stride [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  auto src_idx = elem_to_loc_1<IdxT>(index, src_stride);
  dst[index] = static_cast<U>(src[src_idx]);
}

template <typename T, typename U, typename IdxT = int64_t>
[[kernel]] void copy_g_nd2(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    constant const int64_t* src_strides [[buffer(3)]],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  auto src_idx = elem_to_loc_2<IdxT>(index, src_strides);
  IdxT dst_idx = index.x + IdxT(grid_dim.x) * index.y;
  dst[dst_idx] = static_cast<U>(src[src_idx]);
}

template <typename T, typename U, typename IdxT = int64_t>
[[kernel]] void copy_g_nd3(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    constant const int64_t* src_strides [[buffer(3)]],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto src_idx = elem_to_loc_3<IdxT>(index, src_strides);
  IdxT dst_idx =
      index.x + IdxT(grid_dim.x) * (index.y + IdxT(grid_dim.y) * index.z);
  dst[dst_idx] = static_cast<U>(src[src_idx]);
}

template <typename T, typename U, int N = 1, typename IdxT = int64_t>
[[kernel]] void copy_g(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    constant const int* src_shape [[buffer(2)]],
    constant const int64_t* src_strides [[buffer(3)]],
    constant const int& ndim [[buffer(5)]],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto src_idx = elem_to_loc<IdxT>(
      {N * index.x, index.y, index.z}, src_shape, src_strides, ndim);
  if (N == 1) {
    IdxT dst_idx =
        index.x + grid_dim.x * (index.y + IdxT(grid_dim.y) * index.z);
    dst[dst_idx] = static_cast<U>(src[src_idx]);
    return;
  }
  auto xshape = src_shape[ndim - 1];
  IdxT dst_idx = N * index.x + xshape * (index.y + IdxT(grid_dim.y) * index.z);
  auto src_xstride = src_strides[ndim - 1];
  for (int i = 0; i < N && (int(N * index.x) + i) < xshape; ++i) {
    dst[dst_idx + i] = static_cast<U>(src[src_idx]);
    src_idx += src_xstride;
  }
}

template <typename T, typename U, typename IdxT = int64_t>
[[kernel]] void copy_gg_nd1(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    constant const int64_t& src_stride [[buffer(3)]],
    constant const int64_t& dst_stride [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  auto src_idx = elem_to_loc_1<IdxT>(index, src_stride);
  auto dst_idx = elem_to_loc_1<IdxT>(index, dst_stride);
  dst[dst_idx] = static_cast<U>(src[src_idx]);
}

template <typename T, typename U, typename IdxT = int64_t>
[[kernel]] void copy_gg_nd2(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    constant const int64_t* src_strides [[buffer(3)]],
    constant const int64_t* dst_strides [[buffer(4)]],
    uint2 index [[thread_position_in_grid]]) {
  auto src_idx = elem_to_loc_2<IdxT>(index, src_strides);
  auto dst_idx = elem_to_loc_2<IdxT>(index, dst_strides);
  dst[dst_idx] = static_cast<U>(src[src_idx]);
}

template <typename T, typename U, typename IdxT = int64_t>
[[kernel]] void copy_gg_nd3(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    constant const int64_t* src_strides [[buffer(3)]],
    constant const int64_t* dst_strides [[buffer(4)]],
    uint3 index [[thread_position_in_grid]]) {
  auto src_idx = elem_to_loc_3<IdxT>(index, src_strides);
  auto dst_idx = elem_to_loc_3<IdxT>(index, dst_strides);
  dst[dst_idx] = static_cast<U>(src[src_idx]);
}

template <typename T, typename U, int N = 1, typename IdxT = int64_t>
[[kernel]] void copy_gg(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    constant const int* src_shape [[buffer(2)]],
    constant const int64_t* src_strides [[buffer(3)]],
    constant const int64_t* dst_strides [[buffer(4)]],
    constant const int& ndim [[buffer(5)]],
    uint3 index [[thread_position_in_grid]]) {
  auto idx = elem_to_loc_2_nd<IdxT>(
      {N * index.x, index.y, index.z},
      src_shape,
      src_strides,
      dst_strides,
      ndim);
  if (N == 1) {
    dst[idx.y] = static_cast<U>(src[idx.x]);
    return;
  }
  IdxT src_xstride = src_strides[ndim - 1];
  IdxT dst_xstride = dst_strides[ndim - 1];
  auto xshape = src_shape[ndim - 1];
  for (int i = 0; i < N && (int(N * index.x) + i) < xshape; ++i) {
    dst[idx.y] = static_cast<U>(src[idx.x]);
    idx.x += src_xstride;
    idx.y += dst_xstride;
  }
}

template <typename T, typename U, typename IdxT = int64_t>
[[kernel]] void copy_gg_dynamic_nd1(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    constant const int64_t& src_stride [[buffer(3)]],
    constant const int64_t& dst_stride [[buffer(4)]],
    constant const int64_t& src_offset [[buffer(6)]],
    constant const int64_t& dst_offset [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
  auto src_idx = elem_to_loc_1<IdxT>(index, src_stride);
  auto dst_idx = elem_to_loc_1<IdxT>(index, dst_stride);
  dst[dst_idx + dst_offset] = src[src_idx + src_offset];
}

template <typename T, typename U, typename IdxT = int64_t>
[[kernel]] void copy_gg_dynamic_nd2(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    constant const int64_t* src_strides [[buffer(3)]],
    constant const int64_t* dst_strides [[buffer(4)]],
    constant const int64_t& src_offset [[buffer(6)]],
    constant const int64_t& dst_offset [[buffer(7)]],
    uint2 index [[thread_position_in_grid]]) {
  auto src_idx = elem_to_loc_2<IdxT>(index, src_strides);
  auto dst_idx = elem_to_loc_2<IdxT>(index, dst_strides);
  dst[dst_idx + dst_offset] = src[src_idx + src_offset];
}

template <typename T, typename U, typename IdxT = int64_t>
[[kernel]] void copy_gg_dynamic_nd3(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    constant const int64_t* src_strides [[buffer(3)]],
    constant const int64_t* dst_strides [[buffer(4)]],
    constant const int64_t& src_offset [[buffer(6)]],
    constant const int64_t& dst_offset [[buffer(7)]],
    uint3 index [[thread_position_in_grid]]) {
  auto src_idx = elem_to_loc_3<IdxT>(index, src_strides);
  auto dst_idx = elem_to_loc_3<IdxT>(index, dst_strides);
  dst[dst_idx + dst_offset] = src[src_idx + src_offset];
}

template <typename T, typename U, int N = 1, typename IdxT = int64_t>
[[kernel]] void copy_gg_dynamic(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    constant const int* src_shape [[buffer(2)]],
    constant const int64_t* src_strides [[buffer(3)]],
    constant const int64_t* dst_strides [[buffer(4)]],
    constant const int& ndim [[buffer(5)]],
    constant const int64_t& src_offset [[buffer(6)]],
    constant const int64_t& dst_offset [[buffer(7)]],
    uint3 index [[thread_position_in_grid]]) {
  src += src_offset;
  dst += dst_offset;
  auto idx = elem_to_loc_2_nd<IdxT>(
      {N * index.x, index.y, index.z},
      src_shape,
      src_strides,
      dst_strides,
      ndim);
  if (N == 1) {
    dst[idx.y] = src[idx.x];
    return;
  }
  IdxT src_xstride = src_strides[ndim - 1];
  IdxT dst_xstride = dst_strides[ndim - 1];
  auto xshape = src_shape[ndim - 1];
  for (int i = 0; i < N && (int(N * index.x) + i) < xshape; ++i) {
    dst[idx.y] = src[idx.x];
    idx.x += src_xstride;
    idx.y += dst_xstride;
  }
}
