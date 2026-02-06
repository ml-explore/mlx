// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>

#include <cassert>

namespace mlx::core {

namespace rocm {

__constant__ constexpr uint32_t rotations[2][4] = {
    {13, 15, 26, 6},
    {17, 29, 16, 24}};

union rbits_union {
  uint2 val;
  uint8_t bytes[2][4];
};

__device__ rbits_union threefry2x32_hash(uint2 key, uint2 count) {
  uint32_t ks[] = {key.x, key.y, key.x ^ key.y ^ 0x1BD11BDA};

  rbits_union v;
  v.val.x = count.x + ks[0];
  v.val.y = count.y + ks[1];

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 4; ++j) {
      uint32_t r = rotations[i % 2][j];
      v.val.x += v.val.y;
      v.val.y = (v.val.y << r) | (v.val.y >> (32 - r));
      v.val.y ^= v.val.x;
    }
    v.val.x += ks[(i + 1) % 3];
    v.val.y += ks[(i + 2) % 3] + i + 1;
  }

  return v;
}

__global__ void rbitsc_kernel(
    const uint32_t* keys,
    uint8_t* out,
    uint32_t grid_dims_x,
    uint32_t grid_dims_y,
    bool odd,
    uint32_t bytes_per_key) {
  uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  uint index_x = thread_index % grid_dims_x;
  uint index_y = thread_index / grid_dims_x;
  if (index_x >= grid_dims_x || index_y >= grid_dims_y) {
    return;
  }

  auto kidx = 2 * index_x;
  auto key = make_uint2(keys[kidx], keys[kidx + 1]);
  auto half_size = grid_dims_y - odd;
  out += index_x * bytes_per_key;
  bool drop_last = odd && (index_y == half_size);
  auto bits = threefry2x32_hash(
      key, make_uint2(index_y, drop_last ? 0 : index_y + grid_dims_y));
  size_t idx = size_t(index_y) << 2;
  for (int i = 0; i < 4; ++i) {
    out[idx + i] = bits.bytes[0][i];
  }
  if (!drop_last) {
    idx = (drop_last ? 0 : size_t(index_y) + grid_dims_y) << 2;
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

__device__ int64_t elem_to_loc_random(
    int64_t elem,
    const int* shape,
    const int64_t* strides,
    int ndim) {
  int64_t loc = 0;
  for (int i = ndim - 1; i >= 0; --i) {
    loc += (elem % shape[i]) * strides[i];
    elem /= shape[i];
  }
  return loc;
}

__global__ void rbits_kernel(
    const uint32_t* keys,
    uint8_t* out,
    uint32_t grid_dims_x,
    uint32_t grid_dims_y,
    bool odd,
    uint32_t bytes_per_key,
    int32_t ndim,
    const int* key_shape,
    const int64_t* key_strides) {
  uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  uint index_x = thread_index % grid_dims_x;
  uint index_y = thread_index / grid_dims_x;
  if (index_x >= grid_dims_x || index_y >= grid_dims_y) {
    return;
  }

  auto kidx = 2 * index_x;
  auto k1_elem = elem_to_loc_random(kidx, key_shape, key_strides, ndim);
  auto k2_elem = elem_to_loc_random(kidx + 1, key_shape, key_strides, ndim);
  auto key = make_uint2(keys[k1_elem], keys[k2_elem]);
  auto half_size = grid_dims_y - odd;
  out += size_t(index_x) * bytes_per_key;
  bool drop_last = odd && (index_y == half_size);
  auto bits = threefry2x32_hash(
      key, make_uint2(index_y, drop_last ? 0 : index_y + grid_dims_y));
  size_t idx = size_t(index_y) << 2;
  for (int i = 0; i < 4; ++i) {
    out[idx + i] = bits.bytes[0][i];
  }
  if (!drop_last) {
    idx = (drop_last ? 0 : size_t(index_y) + grid_dims_y) << 2;
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

} // namespace rocm

void RandomBits::eval_gpu(const std::vector<array>& inputs, array& out) {
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
  auto& encoder = rocm::get_command_encoder(s);
  encoder.set_input_array(keys);
  encoder.set_output_array(out);
  
  uint32_t grid_dims_x = num_keys;
  uint32_t grid_dims_y = half_size + odd;
  int64_t total = static_cast<int64_t>(grid_dims_x) * grid_dims_y;
  
  int block_size = 256;
  int num_blocks = (total + block_size - 1) / block_size;
  num_blocks = std::min(num_blocks, 65535);
  
  encoder.launch_kernel([&](hipStream_t stream) {
    if (keys.flags().row_contiguous) {
      hipLaunchKernelGGL(
          rocm::rbitsc_kernel,
          dim3(num_blocks), dim3(block_size), 0, stream,
          keys.data<uint32_t>(),
          out.data<uint8_t>(),
          grid_dims_x,
          grid_dims_y,
          odd,
          bytes_per_key);
    } else {
      // Need to copy shape and strides to device
      array shape_arr({keys.ndim()}, int32);
      array strides_arr({keys.ndim()}, int64);
      shape_arr.set_data(allocator::malloc(shape_arr.nbytes()));
      strides_arr.set_data(allocator::malloc(strides_arr.nbytes()));
      encoder.add_temporary(shape_arr);
      encoder.add_temporary(strides_arr);
      
      (void)hipMemcpyAsync(shape_arr.data<int32_t>(), keys.shape().data(), 
                     keys.ndim() * sizeof(int32_t), hipMemcpyHostToDevice, stream);
      (void)hipMemcpyAsync(strides_arr.data<int64_t>(), keys.strides().data(),
                     keys.ndim() * sizeof(int64_t), hipMemcpyHostToDevice, stream);
      
      hipLaunchKernelGGL(
          rocm::rbits_kernel,
          dim3(num_blocks), dim3(block_size), 0, stream,
          keys.data<uint32_t>(),
          out.data<uint8_t>(),
          grid_dims_x,
          grid_dims_y,
          odd,
          bytes_per_key,
          keys.ndim(),
          shape_arr.data<int32_t>(),
          strides_arr.data<int64_t>());
    }
  });
}

} // namespace mlx::core
