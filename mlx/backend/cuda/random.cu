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
  uint32_t thread_index = grid.thread_rank();
  uint32_t index_x = thread_index % grid_dims.x;
  uint32_t index_y = thread_index / grid_dims.x;
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
  uint32_t thread_index = grid.thread_rank();
  uint32_t index_x = thread_index % grid_dims.x;
  uint32_t index_y = thread_index / grid_dims.x;
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

// Fused per-thread uniform RNG for half-precision targets. Mirrors the
// Metal kernel in mlx/backend/metal/kernels/random.metal: each thread
// emits TWO output elements at positions y and y + half_size from a
// single threefry call, matching the rbitsc bit-layout exactly so that
// seed -> output mapping is bit-identical to the vanilla
// bits()/divide()/astype()/affine() pipeline (no fp32 intermediate
// buffer in global memory).
//
// NOTE: This CUDA path is untested on this branch (no NVIDIA hardware
// available during development). Algorithmically it is the direct CUDA
// transcription of the validated Metal kernel; the C++ dispatch logic
// in the host-side primitives.cpp is shared. CI / maintainer review
// required to confirm correctness on a CUDA-capable device.
template <typename T>
__global__ void runiformc_fused(
    const uint32_t* keys,
    T* out,
    uint32_t half_size,
    float lo,
    float range,
    float upper_clip) {
  uint32_t y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= half_size) {
    return;
  }
  uint2 key2 = uint2{keys[0], keys[1]};
  rbits hash = threefry2x32_hash(key2, uint2{y, y + half_size});

  float f0 = float(hash.val.x) / 4294967295.0f;
  if (f0 > upper_clip) f0 = upper_clip;
  T t0 = T(f0);
  T r_dt = T(range);
  T lo_dt = T(lo);
  T tr0 = r_dt * t0;
  out[y] = tr0 + lo_dt;

  float f1 = float(hash.val.y) / 4294967295.0f;
  if (f1 > upper_clip) f1 = upper_clip;
  T t1 = T(f1);
  T tr1 = r_dt * t1;
  out[y + half_size] = tr1 + lo_dt;
}

} // namespace cu

void RandomUniform::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("RandomUniform::eval_gpu");
  assert(inputs.size() == 1);
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);
  out.set_data(cu::malloc_async(out.nbytes(), encoder));
  if (out.size() == 0) {
    return;
  }
  size_t N = out.size();
  if (N % 2 != 0) {
    throw std::runtime_error(
        "[RandomUniform::eval_gpu] CUDA fast path requires even N; "
        "fall back to vanilla pipeline at the random.cpp level.");
  }
  auto& keys = inputs[0];
  encoder.set_input_array(keys);
  encoder.set_output_array(out);
  size_t half = N / 2;
  if (half >= UINT32_MAX) {
    throw std::runtime_error(
        "[RandomUniform::eval_gpu] half output size exceeds UINT32_MAX");
  }
  float lo = low_;
  float range = high_ - low_;
  float upper_clip =
      (dtype_ == bfloat16) ? 0.99609375f : 0.99951171875f;
  uint32_t threads = 256;
  uint32_t blocks =
      static_cast<uint32_t>((half + threads - 1) / threads);
  if (dtype_ == bfloat16) {
    encoder.add_kernel_node(
        cu::runiformc_fused<__nv_bfloat16>,
        dim3{blocks},
        dim3{threads},
        gpu_ptr<uint32_t>(keys),
        gpu_ptr<__nv_bfloat16>(out),
        static_cast<uint32_t>(half),
        lo,
        range,
        upper_clip);
  } else {
    encoder.add_kernel_node(
        cu::runiformc_fused<__half>,
        dim3{blocks},
        dim3{threads},
        gpu_ptr<uint32_t>(keys),
        gpu_ptr<__half>(out),
        static_cast<uint32_t>(half),
        lo,
        range,
        upper_clip);
  }
}

void RandomBits::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("RandomBits::eval_gpu");
  assert(inputs.size() == 1);

  // keys has shape (N1, ..., NK, 2)
  // out has shape (N1, ..., NK, M1, M2, ...)
  auto& keys = inputs[0];
  size_t num_keys = keys.size() / 2;

  size_t elems_per_key = out.size() / num_keys;
  size_t bytes_per_key = out.itemsize() * elems_per_key;
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);
  out.set_data(cu::malloc_async(out.nbytes(), encoder));
  if (out.size() == 0) {
    return;
  }

  size_t out_per_key = (bytes_per_key + 4 - 1) / 4;
  size_t half_size = out_per_key / 2;

  bool odd = out_per_key % 2;
  if ((half_size + odd) >= UINT32_MAX || num_keys >= UINT32_MAX) {
    throw std::runtime_error("[RandomBits::eval_gpu] Large size unsupported");
  }

  encoder.set_input_array(keys);
  encoder.set_output_array(out);
  int64_t total = num_keys * (half_size + odd);
  uint32_t threads_y = 1;
  while ((total / threads_y) >= UINT_MAX) {
    threads_y *= 2;
  }
  uint32_t threads_x = cuda::ceil_div(total, threads_y);

  dim3 grid_dims{
      static_cast<uint32_t>(num_keys), static_cast<uint32_t>(half_size + odd)};
  auto [grid, block] = get_grid_and_block(threads_x, threads_y, 1);
  auto& stream = encoder.stream();
  if (keys.flags().row_contiguous) {
    encoder.add_kernel_node(
        cu::rbitsc,
        grid,
        block,
        gpu_ptr<uint32_t>(keys),
        gpu_ptr<uint8_t>(out),
        grid_dims,
        odd,
        bytes_per_key);
  } else {
    encoder.add_kernel_node(
        cu::rbits,
        grid,
        block,
        gpu_ptr<uint32_t>(keys),
        gpu_ptr<uint8_t>(out),
        grid_dims,
        odd,
        bytes_per_key,
        keys.ndim(),
        const_param(keys.shape()),
        const_param(keys.strides()));
  }
}

} // namespace mlx::core
