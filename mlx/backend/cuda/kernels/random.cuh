// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/kernels/utils.cuh"

namespace mlx::core::cu {

__constant__ constexpr uint32_t rotations[2][4] = {
    {13, 15, 26, 6},
    {17, 29, 16, 24}};

union rbits {
  uint2 val;
  uint8_t bytes[2][4];
};

__device__ rbits threefry2x32_hash(uint2 key, uint2 count) {
  uint32_t ks[] = {key.x, key.y, key.x ^ key.y ^ 0x1BD11BDA};

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

__global__ void rbitsc(
    const uint32_t* keys,
    uint8_t* out,
    const __grid_constant__ dim3 grid_dim,
    const __grid_constant__ bool odd,
    const __grid_constant__ uint32_t bytes_per_key) {
  uint2 index{
      blockIdx.x * blockDim.x + threadIdx.x,
      blockIdx.y * blockDim.y + threadIdx.y};
  if (index.x >= grid_dim.x || index.y >= grid_dim.y) {
    return;
  }

  auto kidx = 2 * index.x;
  auto key = uint2{keys[kidx], keys[kidx + 1]};
  auto half_size = grid_dim.y - odd;
  out += index.x * bytes_per_key;
  bool drop_last = odd && (index.y == half_size);
  auto bits = threefry2x32_hash(
      key, uint2{index.y, drop_last ? 0 : index.y + grid_dim.y});
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

__global__ void rbits(
    const uint32_t* keys,
    uint8_t* out,
    const __grid_constant__ dim3 grid_dim,
    const __grid_constant__ bool odd,
    const __grid_constant__ uint32_t bytes_per_key,
    const __grid_constant__ int32_t ndim,
    const __grid_constant__ Shape key_shape,
    const __grid_constant__ Strides key_strides) {
  uint2 index{
      blockIdx.x * blockDim.x + threadIdx.x,
      blockIdx.y * blockDim.y + threadIdx.y};
  if (index.x >= grid_dim.x || index.y >= grid_dim.y) {
    return;
  }

  auto kidx = 2 * index.x;
  auto k1_elem = elem_to_loc(kidx, key_shape.data(), key_strides.data(), ndim);
  auto k2_elem =
      elem_to_loc(kidx + 1, key_shape.data(), key_strides.data(), ndim);
  auto key = uint2{keys[k1_elem], keys[k2_elem]};
  auto half_size = grid_dim.y - odd;
  out += size_t(index.x) * bytes_per_key;
  bool drop_last = odd && (index.y == half_size);
  auto bits = threefry2x32_hash(
      key, uint2{index.y, drop_last ? 0 : index.y + grid_dim.y});
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

} // namespace mlx::core::cu
