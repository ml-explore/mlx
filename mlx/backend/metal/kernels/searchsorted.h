// Copyright © 2026 Apple Inc.

template <typename T, bool Right>
METAL_FUNC uint
searchsorted_impl(device const T* a, T v, uint n, int64_t a_stride) {
  LessThan<T> lt;
  uint lo = 0;
  uint hi = n;
  while (lo < hi) {
    uint mid = lo + (hi - lo) / 2;
    T m = a[int64_t(mid) * a_stride];
    bool below = Right ? !lt(v, m) : lt(m, v);
    if (below) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

template <typename T, bool Right>
[[kernel]] void searchsorted(
    device const T* a [[buffer(0)]],
    device const T* v [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant const uint& n [[buffer(3)]],
    constant const int64_t& a_stride [[buffer(4)]],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto offset =
      index.x + grid_dim.x * (int64_t(index.y) + int64_t(grid_dim.y) * index.z);
  out[offset] = searchsorted_impl<T, Right>(a, v[offset], n, a_stride);
}
