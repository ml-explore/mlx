// Copyright © 2025 Apple Inc.

#pragma once

#include <numeric>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/config.h"
#include "mlx/backend/rocm/device/utils.hpp"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

// WARP_SIZE is defined in device/config.h based on target architecture

template <size_t N>
struct uint_by_size;
template <>
struct uint_by_size<2> {
  using type = uint16_t;
};
template <>
struct uint_by_size<4> {
  using type = uint32_t;
};
template <>
struct uint_by_size<8> {
  using type = unsigned long long int;
};

template <typename T, typename Op>
__device__ void atomic_reduce(T* x, T y) {
  if constexpr (sizeof(T) == 1) {
    using U = uint16_t;
    U* x_int = (U*)((char*)x - ((size_t)x % 2));
    int shift = ((char*)x - (char*)x_int) * 8;
    int mask = 0xff << shift;
    U old_val, new_val;
    do {
      old_val = *x_int;
      T result = Op{}(static_cast<T>((old_val >> shift) & 0xff), y);
      new_val = (old_val & ~mask) | (result << shift);
    } while (atomicCAS(x_int, old_val, new_val) != old_val);
  } else {
    using U = typename uint_by_size<sizeof(T)>::type;
    U* x_int = (U*)(x);
    U old_val, new_val;
    do {
      old_val = *x_int;
      T result = Op{}(*((T*)&old_val), y);
      new_val = *((U*)&result);
    } while (atomicCAS(x_int, old_val, new_val) != old_val);
  }
}

// Warp-level reduction using shuffle
template <typename T, typename Op>
__device__ T warp_reduce(T val, Op op) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val = op(val, __shfl_down(val, offset));
  }
  return val;
}

// Block-level reduction
template <typename T, int N, typename Op>
__device__ void block_reduce(
    T (&vals)[N],
    T* smem,
    Op op,
    T init,
    int block_size) {
  int lane = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;
  int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;

  // First reduce within each warp
  for (int i = 0; i < N; i++) {
    vals[i] = warp_reduce(vals[i], op);
  }

  // Store warp results to shared memory
  if (lane == 0) {
    for (int i = 0; i < N; i++) {
      smem[warp_id * N + i] = vals[i];
    }
  }
  __syncthreads();

  // Final reduction by first warp
  if (warp_id == 0) {
    for (int i = 0; i < N; i++) {
      vals[i] = (lane < num_warps) ? smem[lane * N + i] : init;
    }
    for (int i = 0; i < N; i++) {
      vals[i] = warp_reduce(vals[i], op);
    }
  }
}

} // namespace rocm

// Allocate output with same layout as input (for reduce operations)
inline void allocate_same_layout(
    array& out,
    const array& in,
    const std::vector<int>& axes,
    rocm::CommandEncoder& encoder) {
  if (in.flags().row_contiguous) {
    out.set_data(allocator::malloc(out.nbytes()));
    return;
  }

  if (out.ndim() < in.ndim()) {
    throw std::runtime_error(
        "Reduction without keepdims only supported for row-contiguous inputs");
  }

  // Calculate the transpositions applied to in in order to apply them to out.
  std::vector<int> axis_order(in.ndim());
  std::iota(axis_order.begin(), axis_order.end(), 0);
  std::sort(axis_order.begin(), axis_order.end(), [&](int left, int right) {
    return in.strides(left) > in.strides(right);
  });

  // Transpose the shape and calculate the strides
  Shape out_shape(in.ndim());
  Strides out_strides(in.ndim(), 1);
  for (int i = 0; i < in.ndim(); i++) {
    out_shape[i] = out.shape(axis_order[i]);
  }
  for (int i = in.ndim() - 2; i >= 0; i--) {
    out_strides[i] = out_shape[i + 1] * out_strides[i + 1];
  }

  // Reverse the axis order to get the final strides
  Strides final_strides(in.ndim());
  for (int i = 0; i < in.ndim(); i++) {
    final_strides[axis_order[i]] = out_strides[i];
  }

  // Calculate the resulting contiguity and do the memory allocation
  auto [data_size, rc, cc] = check_contiguity(out.shape(), final_strides);
  auto fl = in.flags();
  fl.row_contiguous = rc;
  fl.col_contiguous = cc;
  fl.contiguous = true;
  out.set_data(
      allocator::malloc(out.nbytes()),
      data_size,
      final_strides,
      fl,
      allocator::free);
}

} // namespace mlx::core
