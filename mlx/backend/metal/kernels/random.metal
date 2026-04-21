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
    constant const bool& odd,
    constant const uint& bytes_per_key,
    uint2 grid_dim [[threads_per_grid]],
    uint2 index [[thread_position_in_grid]]) {
  auto kidx = 2 * index.x;
  auto key = uint2(keys[kidx], keys[kidx + 1]);
  auto half_size = grid_dim.y - odd;
  out += index.x * bytes_per_key;
  bool drop_last = odd && (index.y == half_size);
  auto bits = threefry2x32_hash(
      key, uint2(index.y, drop_last ? 0 : index.y + grid_dim.y));
  size_t idx = size_t(index.y) << 2;
  for (int i = 0; i < 4; ++i) {
    out[idx + i] = bits.bytes[0][i];
  }
  if (!drop_last) {
    idx = (drop_last ? 0 : size_t(index.y) + grid_dim.y) << 2;
    if ((index.y + 1) == half_size && (bytes_per_key % 4) > 0) {
      int edge_bytes = (bytes_per_key % 4);
      for (int i = 0; i < edge_bytes; ++i) {
        out[idx + i] = bits.bytes[1][i];
      }
    } else {
      for (int i = 0; i < 4; ++i) {
        out[idx + i] = bits.bytes[1][i];
      }
    }
  }
}

[[kernel]] void rbits(
    device const uint32_t* keys,
    device char* out,
    constant const bool& odd,
    constant const uint& bytes_per_key,
    constant const int& ndim,
    constant const int* key_shape,
    constant const int64_t* key_strides,
    uint2 grid_dim [[threads_per_grid]],
    uint2 index [[thread_position_in_grid]]) {
  auto kidx = 2 * index.x;
  auto k1_elem = elem_to_loc(kidx, key_shape, key_strides, ndim);
  auto k2_elem = elem_to_loc(kidx + 1, key_shape, key_strides, ndim);
  auto key = uint2(keys[k1_elem], keys[k2_elem]);
  auto half_size = grid_dim.y - odd;
  out += size_t(index.x) * bytes_per_key;
  bool drop_last = odd && (index.y == half_size);
  auto bits = threefry2x32_hash(
      key, uint2(index.y, drop_last ? 0 : index.y + grid_dim.y));
  size_t idx = size_t(index.y) << 2;
  for (int i = 0; i < 4; ++i) {
    out[idx + i] = bits.bytes[0][i];
  }
  if (!drop_last) {
    idx = (drop_last ? 0 : size_t(index.y) + grid_dim.y) << 2;
    if ((index.y + 1) == half_size && (bytes_per_key % 4) > 0) {
      int edge_bytes = (bytes_per_key % 4);
      for (int i = 0; i < edge_bytes; ++i) {
        out[idx + i] = bits.bytes[1][i];
      }
    } else {
      for (int i = 0; i < 4; ++i) {
        out[idx + i] = bits.bytes[1][i];
      }
    }
  }
}

// Fused per-thread uniform RNG for half-precision targets. Each thread
// emits TWO output elements at positions y and y + grid_dim.y from a
// single threefry call, matching the rbitsc bit-layout exactly so that
// seed -> output mapping is bit-identical to the vanilla
// bits()/divide()/astype()/affine() pipeline (no fp32 intermediate
// buffer in global memory).
template <typename T>
[[kernel]] void runiformc(
    device const uint32_t* keys,
    device T* out,
    constant const float& lo,
    constant const float& range,
    constant const float& upper_clip,
    uint2 grid_dim [[threads_per_grid]],
    uint2 index [[thread_position_in_grid]]) {
  uint2 key2 = uint2(keys[0], keys[1]);
  uint y = index.y;
  uint half_size = grid_dim.y;
  union rbits hash = threefry2x32_hash(key2, uint2(y, y + half_size));

  // Same exact pattern as Step4 (which worked for the upper_clip read).
  float f0 = float(hash.val.x) / 4294967295.0f;
  f0 = min(f0, upper_clip);
  T t0 = T(f0);
  T r_dt = T(range);
  T lo_dt = T(lo);
  T tr0 = r_dt * t0;
  out[y] = tr0 + lo_dt;

  float f1 = float(hash.val.y) / 4294967295.0f;
  f1 = min(f1, upper_clip);
  T t1 = T(f1);
  T tr1 = r_dt * t1;
  out[y + half_size] = tr1 + lo_dt;
}

#define instantiate_runiformc(tname, type) \
  instantiate_kernel("runiformc_" #tname, runiformc, type)

instantiate_runiformc(float16, half)
instantiate_runiformc(bfloat16, bfloat16_t)
