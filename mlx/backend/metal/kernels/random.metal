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
    constant const ulong& bytes_per_key,
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
    constant const ulong& bytes_per_key,
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

METAL_FUNC void store_signed(device char* out, uint idx, int dtype, ulong value) {
  switch (dtype) {
    case 0: *reinterpret_cast<device char*>(out + idx) = char(value); break;
    case 1: *reinterpret_cast<device short*>(out + idx * 2) = short(value); break;
    case 2: *reinterpret_cast<device int*>(out + idx * 4) = int(value); break;
    default: *reinterpret_cast<device long*>(out + idx * 8) = long(value); break;
  }
}

METAL_FUNC void store_unsigned(device char* out, uint idx, int dtype, ulong value) {
  switch (dtype) {
    case 0: *reinterpret_cast<device bool*>(out + idx) = bool(value); break;
    case 1: *reinterpret_cast<device uchar*>(out + idx) = uchar(value); break;
    case 2: *reinterpret_cast<device ushort*>(out + idx * 2) = ushort(value); break;
    case 3: *reinterpret_cast<device uint*>(out + idx * 4) = uint(value); break;
    default: *reinterpret_cast<device ulong*>(out + idx * 8) = value; break;
  }
}

#define RANDINT_ARGS \
    device const uint32_t* keys [[buffer(0)]], \
    device const long* low [[buffer(1)]], \
    device const long* high [[buffer(2)]], \
    device char* out [[buffer(3)]], \
    constant const ulong& n [[buffer(4)]], \
    constant const ulong& elems_per_key [[buffer(5)]], \
    constant const int& bounds_ndim [[buffer(6)]], \
    constant const int* bounds_shape [[buffer(7)]], \
    constant const int64_t* low_strides [[buffer(8)]], \
    constant const int64_t* high_strides [[buffer(9)]], \
    constant const int& key_ndim [[buffer(10)]], \
    constant const int* key_shape [[buffer(11)]], \
    constant const int64_t* key_strides [[buffer(12)]], \
    constant const int& dtype [[buffer(13)]], \
    uint idx [[thread_position_in_grid]]

[[kernel]] void randint_signed(RANDINT_ARGS) {
  if (idx >= n) return;
  auto li = elem_to_loc((long)idx, bounds_shape, low_strides, bounds_ndim);
  auto hi = elem_to_loc((long)idx, bounds_shape, high_strides, bounds_ndim);
  long lo = low[li];
  long high_value = high[hi];
  if (high_value <= lo) {
    store_signed(out, idx, dtype, as_type<ulong>(lo));
    return;
  }
  ulong lo_bits = as_type<ulong>(lo);
  ulong width = as_type<ulong>(high_value) - lo_bits;
  if (width == 1) {
    store_signed(out, idx, dtype, lo_bits);
    return;
  }
  ulong key_index = idx / elems_per_key;
  auto k1 = elem_to_loc((long)(2 * key_index), key_shape, key_strides, key_ndim);
  auto k2 = elem_to_loc((long)(2 * key_index + 1), key_shape, key_strides, key_ndim);
  uint2 key = uint2(keys[k1], keys[k2]);
  uint2 count = uint2(idx, 0);
  ulong result = 0;
  if (width <= 0xffffffffUL) {
    uint w = uint(width);
    uint remainder = -w % w;
    while (true) {
      auto bits = threefry2x32_hash(key, count++);
      if (bits.val.x >= remainder) {
        result = bits.val.x % w;
        break;
      }
    }
  } else {
    ulong remainder = -width % width;
    while (true) {
      auto bits = threefry2x32_hash(key, count++);
      ulong sample = ulong(bits.val.x) | (ulong(bits.val.y) << 32);
      if (sample >= remainder) {
        result = sample % width;
        break;
      }
    }
  }
  store_signed(out, idx, dtype, lo_bits + result);
}

[[kernel]] void randint_unsigned(
    device const uint32_t* keys [[buffer(0)]],
    device const ulong* low [[buffer(1)]],
    device const ulong* high [[buffer(2)]],
    device char* out [[buffer(3)]],
    constant const ulong& n [[buffer(4)]],
    constant const ulong& elems_per_key [[buffer(5)]],
    constant const int& bounds_ndim [[buffer(6)]],
    constant const int* bounds_shape [[buffer(7)]],
    constant const int64_t* low_strides [[buffer(8)]],
    constant const int64_t* high_strides [[buffer(9)]],
    constant const int& key_ndim [[buffer(10)]],
    constant const int* key_shape [[buffer(11)]],
    constant const int64_t* key_strides [[buffer(12)]],
    constant const int& dtype [[buffer(13)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx >= n) return;
  auto li = elem_to_loc((long)idx, bounds_shape, low_strides, bounds_ndim);
  auto hi = elem_to_loc((long)idx, bounds_shape, high_strides, bounds_ndim);
  ulong lo = low[li];
  ulong high_value = high[hi];
  if (high_value <= lo) {
    store_unsigned(out, idx, dtype, lo);
    return;
  }
  ulong width = high_value - lo;
  if (width == 1) {
    store_unsigned(out, idx, dtype, lo);
    return;
  }
  ulong key_index = idx / elems_per_key;
  auto k1 = elem_to_loc((long)(2 * key_index), key_shape, key_strides, key_ndim);
  auto k2 = elem_to_loc((long)(2 * key_index + 1), key_shape, key_strides, key_ndim);
  uint2 key = uint2(keys[k1], keys[k2]);
  uint2 count = uint2(idx, 0);
  ulong result = 0;
  if (width <= 0xffffffffUL) {
    uint w = uint(width);
    uint remainder = -w % w;
    while (true) {
      auto bits = threefry2x32_hash(key, count++);
      if (bits.val.x >= remainder) {
        result = bits.val.x % w;
        break;
      }
    }
  } else {
    ulong remainder = -width % width;
    while (true) {
      auto bits = threefry2x32_hash(key, count++);
      ulong sample = ulong(bits.val.x) | (ulong(bits.val.y) << 32);
      if (sample >= remainder) {
        result = sample % width;
        break;
      }
    }
  }
  store_unsigned(out, idx, dtype, lo + result);
}

#undef RANDINT_ARGS
