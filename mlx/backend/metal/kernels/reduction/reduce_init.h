// Copyright © 2023-2024 Apple Inc.

template <typename T, typename Op>
[[kernel]] void init_reduce(
    device T* out [[buffer(0)]],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  out[index.x + grid_dim.x * int64_t(index.y)] = Op::init;
}
