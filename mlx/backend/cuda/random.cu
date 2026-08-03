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
    uint64_t bytes_per_key) {
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
    uint64_t bytes_per_key,
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

} // namespace cu

namespace cu {

/* RandomInt kernels — per-element bounds via elem_to_loc */
template <typename T>
__global__ void randint_kernel_signed(
    const uint32_t* keys,
    const int64_t* low,
    const int64_t* high,
    T* out,
    uint64_t n,
    uint64_t elems_per_key,
    int bounds_ndim,
    const __grid_constant__ Shape bounds_shape,
    const __grid_constant__ Strides low_strides,
    const __grid_constant__ Strides high_strides,
    int key_ndim,
    const __grid_constant__ Shape key_shape,
    const __grid_constant__ Strides key_strides) {
  auto grid = cg::this_grid();
  for (uint64_t idx = grid.thread_rank(); idx < n; idx += grid.size()) {
    int64_t lo_idx = elem_to_loc(
        static_cast<int64_t>(idx),
        bounds_shape.data(),
        low_strides.data(),
        bounds_ndim);
    int64_t hi_idx = elem_to_loc(
        static_cast<int64_t>(idx),
        bounds_shape.data(),
        high_strides.data(),
        bounds_ndim);
    int64_t si_lo = low[lo_idx];
    int64_t si_hi = high[hi_idx];

    if (si_hi <= si_lo) {
      out[idx] = static_cast<T>(si_lo);
      continue;
    }

    uint64_t lo_u = *reinterpret_cast<const uint64_t*>(&si_lo);
    uint64_t hi_u = *reinterpret_cast<const uint64_t*>(&si_hi);
    uint64_t width = hi_u - lo_u;

    if (width == 1) {
      out[idx] = static_cast<T>(si_lo);
      continue;
    }

    int64_t key_idx = static_cast<int64_t>(idx / elems_per_key);
    int64_t k1_loc = elem_to_loc(
        2 * key_idx, key_shape.data(), key_strides.data(), key_ndim);
    int64_t k2_loc = elem_to_loc(
        2 * key_idx + 1, key_shape.data(), key_strides.data(), key_ndim);
    auto key = uint2{keys[k1_loc], keys[k2_loc]};
    uint2 count = {
        static_cast<uint32_t>(idx), static_cast<uint32_t>(idx >> 32)};

    uint64_t result = 0;
    if (width <= UINT32_MAX) {
      uint32_t uwidth = static_cast<uint32_t>(width);
      uint32_t remainder = -uwidth % uwidth;
      while (true) {
        auto rb = threefry2x32_hash(key, count);
        count.x++;
        count.y++;
        if (rb.val.x >= remainder) {
          result = rb.val.x % uwidth;
          break;
        }
      }
    } else {
      uint64_t uwidth = width;
      uint64_t remainder = -uwidth % uwidth;
      while (true) {
        auto rb = threefry2x32_hash(key, count);
        count.x++;
        count.y++;
        uint64_t sample = static_cast<uint64_t>(rb.val.x) |
            (static_cast<uint64_t>(rb.val.y) << 32);
        if (sample >= remainder) {
          result = sample % uwidth;
          break;
        }
      }
    }

    /* Form result in uint64 to avoid signed overflow. */
    out[idx] = static_cast<T>(static_cast<int64_t>(lo_u + result));
  }
}

template <typename T>
__global__ void randint_kernel_unsigned(
    const uint32_t* keys,
    const uint64_t* low,
    const uint64_t* high,
    T* out,
    uint64_t n,
    uint64_t elems_per_key,
    int bounds_ndim,
    const __grid_constant__ Shape bounds_shape,
    const __grid_constant__ Strides low_strides,
    const __grid_constant__ Strides high_strides,
    int key_ndim,
    const __grid_constant__ Shape key_shape,
    const __grid_constant__ Strides key_strides) {
  auto grid = cg::this_grid();
  for (uint64_t idx = grid.thread_rank(); idx < n; idx += grid.size()) {
    int64_t lo_idx = elem_to_loc(
        static_cast<int64_t>(idx),
        bounds_shape.data(),
        low_strides.data(),
        bounds_ndim);
    int64_t hi_idx = elem_to_loc(
        static_cast<int64_t>(idx),
        bounds_shape.data(),
        high_strides.data(),
        bounds_ndim);
    uint64_t lo_val = low[lo_idx];
    uint64_t hi_val = high[hi_idx];

    if (hi_val <= lo_val) {
      out[idx] = static_cast<T>(lo_val);
      continue;
    }

    uint64_t width = hi_val - lo_val;

    if (width == 1) {
      out[idx] = static_cast<T>(lo_val);
      continue;
    }

    int64_t key_idx = static_cast<int64_t>(idx / elems_per_key);
    int64_t k1_loc = elem_to_loc(
        2 * key_idx, key_shape.data(), key_strides.data(), key_ndim);
    int64_t k2_loc = elem_to_loc(
        2 * key_idx + 1, key_shape.data(), key_strides.data(), key_ndim);
    auto key = uint2{keys[k1_loc], keys[k2_loc]};
    uint2 count = {
        static_cast<uint32_t>(idx), static_cast<uint32_t>(idx >> 32)};

    uint64_t result = 0;
    if (width <= UINT32_MAX) {
      uint32_t uwidth = static_cast<uint32_t>(width);
      uint32_t remainder = -uwidth % uwidth;
      while (true) {
        auto rb = threefry2x32_hash(key, count);
        count.x++;
        count.y++;
        if (rb.val.x >= remainder) {
          result = rb.val.x % uwidth;
          break;
        }
      }
    } else {
      uint64_t uwidth = width;
      uint64_t remainder = -uwidth % uwidth;
      while (true) {
        auto rb = threefry2x32_hash(key, count);
        count.x++;
        count.y++;
        uint64_t sample = static_cast<uint64_t>(rb.val.x) |
            (static_cast<uint64_t>(rb.val.y) << 32);
        if (sample >= remainder) {
          result = sample % uwidth;
          break;
        }
      }
    }

    out[idx] = static_cast<T>(lo_val + result);
  }
}

__global__ void randint_kernel_bool_unsigned(
    const uint32_t* keys,
    bool* out,
    uint64_t n,
    uint64_t elems_per_key,
    int key_ndim,
    const __grid_constant__ Shape key_shape,
    const __grid_constant__ Strides key_strides) {
  auto grid = cg::this_grid();
  for (uint64_t idx = grid.thread_rank(); idx < n; idx += grid.size()) {
    int64_t key_idx = static_cast<int64_t>(idx / elems_per_key);
    int64_t k1_loc = elem_to_loc(
        2 * key_idx, key_shape.data(), key_strides.data(), key_ndim);
    int64_t k2_loc = elem_to_loc(
        2 * key_idx + 1, key_shape.data(), key_strides.data(), key_ndim);
    auto key = uint2{keys[k1_loc], keys[k2_loc]};
    uint2 count = {
        static_cast<uint32_t>(idx), static_cast<uint32_t>(idx >> 32)};
    auto rb = threefry2x32_hash(key, count);
    out[idx] = static_cast<bool>(rb.val.x & 1);
  }
}

} // namespace cu

void RandomInt::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("RandomInt::eval_gpu");
  assert(inputs.size() == 3);

  auto& keys = inputs[0];
  auto& low_in = inputs[1];
  auto& high_in = inputs[2];

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);
  out.set_data(cu::malloc_async(out.nbytes(), encoder));
  if (out.size() == 0) {
    return;
  }

  size_t n = out.size();
  int blocks = std::min(static_cast<int>(cuda::ceil_div(n, 256)), 65536);
  dim3 grid(blocks);
  dim3 block(256);

  encoder.set_input_array(keys);
  encoder.set_input_array(low_in);
  encoder.set_input_array(high_in);
  encoder.set_output_array(out);

  size_t num_keys = keys.size() / 2;
  uint64_t elems_per_key = n / num_keys;

  bool is_signed = (low_in.dtype().val() == Dtype::Val::int64);

  auto bshape = const_param(low_in.shape());
  auto low_strides = const_param(low_in.strides());
  auto high_strides = const_param(high_in.strides());
  auto kshape = const_param(keys.shape());
  auto kstrides = const_param(keys.strides());
  int bounds_ndim = low_in.ndim();
  int key_ndim = keys.ndim();

  if (is_signed) {
    auto low_ptr = gpu_ptr<const int64_t>(low_in);
    auto high_ptr = gpu_ptr<const int64_t>(high_in);
    switch (out.dtype().val()) {
      case Dtype::Val::int8:
        encoder.add_kernel_node(
            cu::randint_kernel_signed<int8_t>,
            grid,
            block,
            gpu_ptr<uint32_t>(keys),
            low_ptr,
            high_ptr,
            gpu_ptr<int8_t>(out),
            static_cast<uint64_t>(n),
            elems_per_key,
            bounds_ndim,
            bshape,
            low_strides,
            high_strides,
            key_ndim,
            kshape,
            kstrides);
        break;
      case Dtype::Val::int16:
        encoder.add_kernel_node(
            cu::randint_kernel_signed<int16_t>,
            grid,
            block,
            gpu_ptr<uint32_t>(keys),
            low_ptr,
            high_ptr,
            gpu_ptr<int16_t>(out),
            static_cast<uint64_t>(n),
            elems_per_key,
            bounds_ndim,
            bshape,
            low_strides,
            high_strides,
            key_ndim,
            kshape,
            kstrides);
        break;
      case Dtype::Val::int32:
        encoder.add_kernel_node(
            cu::randint_kernel_signed<int32_t>,
            grid,
            block,
            gpu_ptr<uint32_t>(keys),
            low_ptr,
            high_ptr,
            gpu_ptr<int32_t>(out),
            static_cast<uint64_t>(n),
            elems_per_key,
            bounds_ndim,
            bshape,
            low_strides,
            high_strides,
            key_ndim,
            kshape,
            kstrides);
        break;
      case Dtype::Val::int64:
        encoder.add_kernel_node(
            cu::randint_kernel_signed<int64_t>,
            grid,
            block,
            gpu_ptr<uint32_t>(keys),
            low_ptr,
            high_ptr,
            gpu_ptr<int64_t>(out),
            static_cast<uint64_t>(n),
            elems_per_key,
            bounds_ndim,
            bshape,
            low_strides,
            high_strides,
            key_ndim,
            kshape,
            kstrides);
        break;
      default:
        throw std::runtime_error(
            "[RandomInt::eval_gpu] Unexpected signed dtype for randint.");
    }
  } else {
    auto low_ptr = gpu_ptr<const uint64_t>(low_in);
    auto high_ptr = gpu_ptr<const uint64_t>(high_in);
    switch (out.dtype().val()) {
      case Dtype::Val::bool_:
        encoder.add_kernel_node(
            cu::randint_kernel_bool_unsigned,
            grid,
            block,
            gpu_ptr<uint32_t>(keys),
            gpu_ptr<bool>(out),
            static_cast<uint64_t>(n),
            elems_per_key,
            key_ndim,
            kshape,
            kstrides);
        break;
      case Dtype::Val::uint8:
        encoder.add_kernel_node(
            cu::randint_kernel_unsigned<uint8_t>,
            grid,
            block,
            gpu_ptr<uint32_t>(keys),
            low_ptr,
            high_ptr,
            gpu_ptr<uint8_t>(out),
            static_cast<uint64_t>(n),
            elems_per_key,
            bounds_ndim,
            bshape,
            low_strides,
            high_strides,
            key_ndim,
            kshape,
            kstrides);
        break;
      case Dtype::Val::uint16:
        encoder.add_kernel_node(
            cu::randint_kernel_unsigned<uint16_t>,
            grid,
            block,
            gpu_ptr<uint32_t>(keys),
            low_ptr,
            high_ptr,
            gpu_ptr<uint16_t>(out),
            static_cast<uint64_t>(n),
            elems_per_key,
            bounds_ndim,
            bshape,
            low_strides,
            high_strides,
            key_ndim,
            kshape,
            kstrides);
        break;
      case Dtype::Val::uint32:
        encoder.add_kernel_node(
            cu::randint_kernel_unsigned<uint32_t>,
            grid,
            block,
            gpu_ptr<uint32_t>(keys),
            low_ptr,
            high_ptr,
            gpu_ptr<uint32_t>(out),
            static_cast<uint64_t>(n),
            elems_per_key,
            bounds_ndim,
            bshape,
            low_strides,
            high_strides,
            key_ndim,
            kshape,
            kstrides);
        break;
      case Dtype::Val::uint64:
        encoder.add_kernel_node(
            cu::randint_kernel_unsigned<uint64_t>,
            grid,
            block,
            gpu_ptr<uint32_t>(keys),
            low_ptr,
            high_ptr,
            gpu_ptr<uint64_t>(out),
            static_cast<uint64_t>(n),
            elems_per_key,
            bounds_ndim,
            bshape,
            low_strides,
            high_strides,
            key_ndim,
            kshape,
            kstrides);
        break;
      default:
        throw std::runtime_error(
            "[RandomInt::eval_gpu] Unexpected unsigned dtype for randint.");
    }
  }
}

void RandomBits::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("RandomBits::eval_gpu");
  assert(inputs.size() == 1);

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
