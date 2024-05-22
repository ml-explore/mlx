// Copyright © 2023 Apple Inc.

#include "mlx/backend/metal/kernels/utils.h"

static constexpr constant uint32_t rotations[2][4] = {
    {13, 15, 26, 6},
    {17, 29, 16, 24}};

union rbits {
  uint2 val;
  uchar4 bytes[2];
};

rbits threefry2x32_hash(const thread uint2& key, uint2 count) {
  uint4 ks = {key.x, key.y, key.x ^ key.y ^ 0x1BD11BDA};

  rbits v;
  v.val.x = count.x + ks[0];
  v.val.y = count.y + ks[1];

  for (int i = 0; i < 5; ++i) {
    for (auto r : rotations[i % 2]) {
      v.val.x += v.val.y;
      v.val.y = (v.val.y << r) | (v.val.y >> (32 - r));
      v.val.y ^= v.val.x;
    }
    v.val.x += ks[(i + 1) % 3];
    v.val.y += ks[(i + 2) % 3] + i + 1;
  }

  return v;
}

[[kernel]] void rbitsc(
    device const uint32_t* keys,
    device char* out,
    device const bool& odd,
    device const uint& bytes_per_key,
    uint2 grid_dim [[threads_per_grid]],
    uint2 index [[thread_position_in_grid]]) {
  auto kidx = 2 * index.x;
  auto key = uint2(keys[kidx], keys[kidx + 1]);
  auto half_size = grid_dim.y - odd;
  out += index.x * bytes_per_key;
  bool drop_last = odd && (index.y == half_size);
  auto count = uint2(index.y, drop_last ? 0 : index.y + grid_dim.y);
  auto bits = threefry2x32_hash(key, count);
  for (int i = 0; i < 4; ++i) {
    out[4 * count.x + i] = bits.bytes[0][i];
  }
  if (!drop_last) {
    if ((index.y + 1) == half_size && (bytes_per_key % 4) > 0) {
      int edge_bytes = (bytes_per_key % 4);
      for (int i = 0; i < edge_bytes; ++i) {
        out[4 * count.y + i] = bits.bytes[1][i];
      }
    } else {
      for (int i = 0; i < 4; ++i) {
        out[4 * count.y + i] = bits.bytes[1][i];
      }
    }
  }
}

[[kernel]] void rbits(
    device const uint32_t* keys,
    device char* out,
    device const bool& odd,
    device const uint& bytes_per_key,
    device const int& ndim,
    device const int* key_shape,
    device const size_t* key_strides,
    uint2 grid_dim [[threads_per_grid]],
    uint2 index [[thread_position_in_grid]]) {
  auto kidx = 2 * index.x;
  auto k1_elem = elem_to_loc(kidx, key_shape, key_strides, ndim);
  auto k2_elem = elem_to_loc(kidx + 1, key_shape, key_strides, ndim);
  auto key = uint2(keys[k1_elem], keys[k2_elem]);
  auto half_size = grid_dim.y - odd;
  out += index.x * bytes_per_key;
  bool drop_last = odd && (index.y == half_size);
  auto count = uint2(index.y, drop_last ? 0 : index.y + grid_dim.y);
  auto bits = threefry2x32_hash(key, count);
  for (int i = 0; i < 4; ++i) {
    out[4 * count.x + i] = bits.bytes[0][i];
  }
  if (!drop_last) {
    if ((index.y + 1) == half_size && (bytes_per_key % 4) > 0) {
      int edge_bytes = (bytes_per_key % 4);
      for (int i = 0; i < edge_bytes; ++i) {
        out[4 * count.y + i] = bits.bytes[1][i];
      }
    } else {
      for (int i = 0; i < 4; ++i) {
        out[4 * count.y + i] = bits.bytes[1][i];
      }
    }
  }
}
