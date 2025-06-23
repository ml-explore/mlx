// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device/utils.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

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

// TODO: Should make a custom complex type
template <typename U, typename T>
inline __device__ U __cast(T x) {
  return static_cast<U>(x);
}

template <>
inline __device__ bool __cast<bool, cuComplex>(cuComplex x) {
  return x.x != 0 && x.y != 0;
}

template <>
inline __device__ cuComplex __cast<cuComplex, bool>(bool x) {
  return x ? make_cuFloatComplex(1, 1) : make_cuFloatComplex(0, 0);
}

template <typename T, int N, typename Block, typename Warp, typename Op>
inline __device__ void
block_reduce(Block block, Warp warp, T (&vals)[N], T* smem, Op op, T init) {
  // First reduce in the current warp
  for (int i = 0; i < N; i++) {
    vals[i] = cg::reduce(warp, vals[i], op);
  }

  // Reduce across warps
  if (warp.meta_group_size() > 1) {
    if (warp.thread_rank() == 0) {
      for (int i = 0; i < N; i++) {
        smem[warp.meta_group_rank() * N + i] = vals[i];
      }
    }
    block.sync();
    if (warp.thread_rank() < warp.meta_group_size()) {
      for (int i = 0; i < N; i++) {
        vals[i] = smem[warp.thread_rank() * N + i];
      }
    } else {
      for (int i = 0; i < N; i++) {
        vals[i] = init;
      }
    }
    for (int i = 0; i < N; i++) {
      vals[i] = cg::reduce(warp, vals[i], op);
    }
  }
}

} // namespace cu

inline void allocate_same_layout(
    array& out,
    const array& in,
    const std::vector<int>& axes) {
  // Initialize out such that it matches in's layout. Basically we keep any
  // transpositions as it were and that allows us either to skip finding the
  // location of the output that matches the input or simply contiguous read or
  // writes.
  auto out_strides = in.strides();
  for (auto ax : axes) {
    for (auto& s : out_strides) {
      if (s > in.strides(ax) && in.strides(ax) > 0) {
        s /= in.shape(ax);
      }
    }
  }
  auto [data_size, rc, cc] = check_contiguity(out.shape(), out_strides);
  auto fl = in.flags();
  fl.row_contiguous = rc;
  fl.col_contiguous = cc;
  fl.contiguous = data_size == out.size();
  out.set_data(
      allocator::malloc(out.nbytes()),
      data_size,
      out_strides,
      fl,
      allocator::free);
}

} // namespace mlx::core
