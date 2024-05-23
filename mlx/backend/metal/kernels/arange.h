// Copyright © 2023-2024 Apple Inc.
template <typename T>
[[kernel]] void arange(
    constant const T& start,
    constant const T& step,
    device T* out,
    uint index [[thread_position_in_grid]]) {
  out[index] = start + index * step;
}
