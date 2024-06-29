// Copyright © 2024 Apple Inc.

[[kernel]] void make_triangular(
    constant const bool& upper,
    constant const int& m,
    device float* out,
    uint3 index [[thread_position_in_grid]]) {
  const bool should_zero = upper ? index.x > index.y : index.y > index.x;
  if (should_zero) {
    out[index.z * m * m + index.x * m + index.y] = 0;
  }
}
