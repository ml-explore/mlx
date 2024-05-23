// Copyright © 2023-2024 Apple Inc.

template <typename T, typename Op>
[[kernel]] void init_reduce(
    device T* out [[buffer(0)]],
    uint tid [[thread_position_in_grid]]) {
  out[tid] = Op::init;
}
