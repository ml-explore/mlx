// Copyright © 2026 Apple Inc.

// One thread per element of the values input, each doing an independent binary
// search. Comparing through LessThan from sort.h rather than a raw < is what
// keeps the result consistent with sort, which orders NaNs last.
template <typename T, bool Right>
METAL_FUNC uint
searchsorted_impl(device const T* a, T v, uint n, int64_t a_stride) {
  LessThan<T> lt;
  uint lo = 0;
  uint hi = n;
  while (lo < hi) {
    uint mid = lo + (hi - lo) / 2;
    // signed index, so a reversed view with a negative stride walks the right
    // way
    T m = a[int64_t(mid) * a_stride];
    // left advances on lt(a[mid], v), right on !lt(v, a[mid])
    bool below = Right ? !lt(v, m) : lt(m, v);
    if (below) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// The grid covers the output exactly, so neither kernel bounds checks.
METAL_FUNC int64_t grid_offset(uint3 index, uint3 grid_dim) {
  return index.x +
      grid_dim.x * (int64_t(index.y) + int64_t(grid_dim.y) * index.z);
}

// Contiguous values.
template <typename T, bool Right>
[[kernel]] void searchsorted_v(
    device const T* a [[buffer(0)]],
    device const T* v [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant const uint& n [[buffer(3)]],
    constant const int64_t& a_stride [[buffer(4)]],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto offset = grid_offset(index, grid_dim);
  out[offset] = searchsorted_impl<T, Right>(a, v[offset], n, a_stride);
}

// Strided or broadcast values.
template <typename T, bool Right>
[[kernel]] void searchsorted_g(
    device const T* a [[buffer(0)]],
    device const T* v [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant const uint& n [[buffer(3)]],
    constant const int64_t& a_stride [[buffer(4)]],
    constant const int* v_shape [[buffer(5)]],
    constant const int64_t* v_strides [[buffer(6)]],
    constant const int& ndim [[buffer(7)]],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto offset = grid_offset(index, grid_dim);
  auto loc = elem_to_loc<int64_t>(offset, v_shape, v_strides, ndim);
  out[offset] = searchsorted_impl<T, Right>(a, v[loc], n, a_stride);
}
