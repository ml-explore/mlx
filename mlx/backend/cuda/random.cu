// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/primitives.h"

#include <cooperative_groups.h>
#include <nvtx3/nvtx3.hpp>

#include <cassert>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

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
    dim3 grid_dims,
    bool odd,
    uint32_t bytes_per_key) {
  auto grid = cg::this_grid();
  uint thread_index = grid.thread_rank();
  uint index_x = thread_index % grid_dims.x;
  uint index_y = thread_index / grid_dims.x;
  if (index_x >= grid_dims.x || index_y >= grid_dims.y) {
    return;
  }

  auto kidx = 2 * index_x;
  auto key = uint2{keys[kidx], keys[kidx + 1]};
  auto half_size = grid_dims.y - odd;
  out += index_x * bytes_per_key;
  bool drop_last = odd && (index_y == half_size);
  auto bits = threefry2x32_hash(
      key, uint2{index_y, drop_last ? 0 : index_y + grid_dims.y});
  size_t idx = size_t(index_y) << 2;
  for (int i = 0; i < 4; ++i) {
    out[idx + i] = bits.bytes[0][i];
  }
  if (!drop_last) {
    idx = (drop_last ? 0 : size_t(index_y) + grid_dims.y) << 2;
    if ((index_y + 1) == half_size && (bytes_per_key % 4) > 0) {
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
    dim3 grid_dims,
    bool odd,
    uint32_t bytes_per_key,
    int32_t ndim,
    const __grid_constant__ Shape key_shape,
    const __grid_constant__ Strides key_strides) {
  auto grid = cg::this_grid();
  uint thread_index = grid.thread_rank();
  uint index_x = thread_index % grid_dims.x;
  uint index_y = thread_index / grid_dims.x;
  if (index_x >= grid_dims.x || index_y >= grid_dims.y) {
    return;
  }

  auto kidx = 2 * index_x;
  auto k1_elem = elem_to_loc(kidx, key_shape.data(), key_strides.data(), ndim);
  auto k2_elem =
      elem_to_loc(kidx + 1, key_shape.data(), key_strides.data(), ndim);
  auto key = uint2{keys[k1_elem], keys[k2_elem]};
  auto half_size = grid_dims.y - odd;
  out += size_t(index_x) * bytes_per_key;
  bool drop_last = odd && (index_y == half_size);
  auto bits = threefry2x32_hash(
      key, uint2{index_y, drop_last ? 0 : index_y + grid_dims.y});
  size_t idx = size_t(index_y) << 2;
  for (int i = 0; i < 4; ++i) {
    out[idx + i] = bits.bytes[0][i];
  }
  if (!drop_last) {
    idx = (drop_last ? 0 : size_t(index_y) + grid_dims.y) << 2;
    if ((index_y + 1) == half_size && (bytes_per_key % 4) > 0) {
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

} // namespace cu

void RandomBits::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("RandomBits::eval_gpu");
  assert(inputs.size() == 1);

  // keys has shape (N1, ..., NK, 2)
  // out has shape (N1, ..., NK, M1, M2, ...)
  auto& keys = inputs[0];
  uint32_t num_keys = keys.size() / 2;

  uint32_t elems_per_key = out.size() / num_keys;
  uint32_t bytes_per_key = out.itemsize() * elems_per_key;
  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  uint32_t out_per_key = (bytes_per_key + 4 - 1) / 4;
  uint32_t half_size = out_per_key / 2;
  bool odd = out_per_key % 2;

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(keys);
  encoder.set_output_array(out);
  dim3 grid_dims{num_keys, half_size + odd};
  int64_t total = grid_dims.x * grid_dims.y;
  int32_t threads_y = 1;
  while ((total / threads_y) >= (1U << 31)) {
    threads_y *= 2;
  }
  int32_t threads_x = cuda::ceil_div(total, threads_y);
  auto [grid, block] = get_grid_and_block(threads_x, threads_y, 1);
  auto& stream = encoder.stream();
  if (keys.flags().row_contiguous) {
    encoder.add_kernel_node(
        cu::rbitsc,
        grid,
        block,
        keys.data<uint32_t>(),
        out.data<uint8_t>(),
        grid_dims,
        odd,
        bytes_per_key);
  } else {
    encoder.add_kernel_node(
        cu::rbits,
        grid,
        block,
        keys.data<uint32_t>(),
        out.data<uint8_t>(),
        grid_dims,
        odd,
        bytes_per_key,
        keys.ndim(),
        const_param(keys.shape()),
        const_param(keys.strides()));
  }
}

} // namespace mlx::core
