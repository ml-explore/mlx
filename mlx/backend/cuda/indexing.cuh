// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/kernels/fp16_math.cuh"

#include <thrust/for_each.h>
#include <cuda/std/utility>

namespace mlx::core::cu {

// Only allow int32 as index type if MLX_FAST_COMPILE is defined.
#if defined(MLX_FAST_COMPILE)
#define MLX_SWITCH_INDEX_TYPES_CHECKED(TYPE, NAME, CTYPE_ALIAS, ...)         \
  if (TYPE == ::mlx::core::int32) {                                          \
    using CTYPE_ALIAS = int32_t;                                             \
    __VA_ARGS__;                                                             \
  } else if (TYPE == ::mlx::core::uint32) {                                  \
    using CTYPE_ALIAS = uint32_t;                                            \
    __VA_ARGS__;                                                             \
  } else {                                                                   \
    throw std::invalid_argument(fmt::format(                                 \
        "Can not use dtype {} as index for {} when MLX_FAST_COMPILE is on.", \
        dtype_to_string(TYPE),                                               \
        NAME));                                                              \
  }
#else
#define MLX_SWITCH_INDEX_TYPES_CHECKED(TYPE, NAME, CTYPE_ALIAS, ...) \
  MLX_SWITCH_INT_TYPES_CHECKED(TYPE, NAME, CTYPE_ALIAS, __VA_ARGS__)
#endif

// Dispatch dynamic nidx to constexpr.
#if defined(MLX_FAST_COMPILE)
#define MLX_SWITCH_NIDX(nidx, NIDX, ...) \
  {                                      \
    assert(nidx <= 16);                  \
    constexpr uint32_t NIDX = 16;        \
    __VA_ARGS__;                         \
  }
#else
#define MLX_SWITCH_NIDX(nidx, NIDX, ...)                                     \
  if (nidx <= 2) {                                                           \
    constexpr uint32_t NIDX = 2;                                             \
    __VA_ARGS__;                                                             \
  } else if (nidx <= 16) {                                                   \
    constexpr uint32_t NIDX = 16;                                            \
    __VA_ARGS__;                                                             \
  } else {                                                                   \
    throw std::runtime_error(                                                \
        fmt::format("Indices array can not have more than {} items", nidx)); \
  }
#endif

// Dispatch dynamic idx_ndim to constexpr.
#define MORE_THAN_ONE MAX_NDIM
#define MLX_SWITCH_IDX_NDIM(idx_ndim, IDX_NDIM, ...) \
  if (idx_ndim == 0) {                               \
    constexpr uint32_t IDX_NDIM = 0;                 \
    __VA_ARGS__;                                     \
  } else if (idx_ndim == 1) {                        \
    constexpr uint32_t IDX_NDIM = 1;                 \
    __VA_ARGS__;                                     \
  } else {                                           \
    constexpr uint32_t IDX_NDIM = MORE_THAN_ONE;     \
    __VA_ARGS__;                                     \
  }

// Like thrust::scatter but accept custom op.
template <typename Policy, typename It, typename Idx, typename Out, typename Op>
void scatter_n(Policy& policy, It begin, size_t size, Idx idx, Out out, Op op) {
  thrust::for_each_n(
      policy,
      thrust::make_zip_iterator(begin, idx),
      size,
      [out, op] __device__(auto item) {
        op(&out[thrust::get<1>(item)], thrust::get<0>(item));
      });
}

// Convert an absolute index to positions in a 3d grid, assuming the index is
// calculated with:
// index = x * dim1 * dim2 + y * dim2 + z
template <typename T>
inline __device__ cuda::std::tuple<T, T, T>
index_to_dims(T index, size_t dim1, size_t dim2) {
  T x = index / (dim1 * dim2);
  T y = (index % (dim1 * dim2)) / dim2;
  T z = index % dim2;
  return cuda::std::make_tuple(x, y, z);
}

// Get absolute index from possible negative index.
template <typename IdxT>
inline __device__ auto absolute_index(IdxT idx, int32_t size) {
  if constexpr (cuda::std::is_unsigned_v<IdxT>) {
    return idx;
  } else {
    return static_cast<int32_t>(idx < 0 ? idx + size : idx);
  }
}

// An op that takes an index of |src|, and returns the corresponding index value
// from |idx| which is the indices at |axis|.
template <typename IdxT, bool IdxC, bool SrcC = true, typename LocT = int64_t>
struct IndexOp {
  const IdxT* idx;
  int32_t ndim;
  Shape shape;
  Strides src_strides;
  Strides idx_strides;
  int32_t src_axis_size;
  int32_t idx_axis_size;
  int64_t src_axis_stride;
  int64_t idx_axis_stride;
  size_t size_post;

  IndexOp(const array& idx, const array& src, int32_t axis)
      : idx(idx.data<IdxT>()),
        ndim(static_cast<int32_t>(src.ndim()) - 1),
        shape(const_param(remove_index(idx.shape(), axis))),
        src_strides(const_param(remove_index(src.strides(), axis))),
        idx_strides(const_param(remove_index(idx.strides(), axis))),
        src_axis_size(src.shape(axis)),
        idx_axis_size(idx.shape(axis)),
        src_axis_stride(src.strides(axis)),
        idx_axis_stride(idx.strides(axis)) {
    size_post = 1;
    for (int i = axis + 1; i < idx.ndim(); ++i) {
      size_post *= idx.shape(i);
    }
  }

  __device__ LocT operator()(size_t index) {
    auto [x, y, z] = index_to_dims(index, idx_axis_size, size_post);

    LocT elem_idx = x * size_post;

    LocT idx_loc = y * idx_axis_stride;
    if constexpr (IdxC) {
      idx_loc += elem_idx * idx_axis_size + z;
    } else {
      idx_loc +=
          elem_to_loc(elem_idx + z, shape.data(), idx_strides.data(), ndim);
    }

    auto idx_val = absolute_index(idx[idx_loc], src_axis_size);

    LocT src_idx = idx_val * src_axis_stride;
    if constexpr (SrcC) {
      src_idx += elem_idx * src_axis_size + z;
    } else {
      src_idx +=
          elem_to_loc(elem_idx + z, shape.data(), src_strides.data(), ndim);
    }
    return src_idx;
  }
};

// Concatenated |idx| arrays.
template <typename T, size_t NIDX, size_t IDX_NDIM, typename LocT = int64_t>
struct Indices {
  size_t size;
  size_t ndim;
  cuda::std::array<const T*, NIDX> buffers;
  cuda::std::array<bool, NIDX> row_contiguous;
  cuda::std::array<int32_t, NIDX * IDX_NDIM> shapes;
  cuda::std::array<int64_t, NIDX * IDX_NDIM> strides;

  template <typename Iter>
  Indices(Iter begin, Iter end) {
    size = end - begin;
    ndim = size > 0 ? begin->ndim() : 0;
    for (size_t i = 0; i < size; ++i) {
      const array& arr = *(begin + i);
      buffers[i] = arr.data<T>();
      row_contiguous[i] = arr.flags().row_contiguous;
      std::copy_n(arr.shape().begin(), ndim, shapes.begin() + i * ndim);
      std::copy_n(arr.strides().begin(), ndim, strides.begin() + i * ndim);
    }
  }

  __device__ auto operator()(size_t i, size_t x, size_t y) {
    LocT idx_loc;
    if constexpr (IDX_NDIM == 0) {
      idx_loc = 0;
    } else {
      idx_loc = x * strides[ndim * i];
      if constexpr (IDX_NDIM == MORE_THAN_ONE) {
        if (row_contiguous[i]) {
          idx_loc += y;
        } else {
          size_t offset = ndim * i + 1;
          idx_loc += elem_to_loc(
              y, shapes.data() + offset, strides.data() + offset, ndim - 1);
        }
      }
    }
    return buffers[i][idx_loc];
  }
};

// An op that takes an index of |src|, and returns the corresponding index value
// from |indices| located at |axes|.
template <typename IdxT, size_t NIDX, size_t IDX_NDIM, typename LocT = int64_t>
struct IndicesOp {
  size_t ndim;
  Shape shape;
  Strides strides;
  Shape slice_sizes;
  Shape axes;
  Indices<IdxT, NIDX, IDX_NDIM, LocT> indices;
  size_t n_dim0;
  size_t slice_size;

  template <typename Iter>
  IndicesOp(
      const array& src,
      const std::vector<int32_t>& slice_sizes,
      const std::vector<int32_t>& axes,
      Iter idx_begin,
      Iter idx_end)
      : ndim(src.ndim()),
        shape(const_param(src.shape())),
        strides(const_param(src.strides())),
        slice_sizes(const_param(slice_sizes)),
        axes(const_param(axes)),
        indices(idx_begin, idx_end) {
    n_dim0 = 1;
    size_t dim0 = 1;
    if (indices.ndim >= 1) {
      dim0 = idx_begin->shape(0);
    }
    if (indices.ndim >= 2) {
      n_dim0 = idx_begin->size() / dim0;
    }

    slice_size = 1;
    for (size_t s : slice_sizes) {
      slice_size *= s;
    }
  }

  __device__ LocT operator()(size_t index) {
    auto [x, y, z] = index_to_dims(index, n_dim0, slice_size);

    LocT src_idx = 0;
    for (size_t i = 0; i < indices.size; ++i) {
      auto ax = axes[i];
      auto idx_val = absolute_index(indices(i, x, y), shape[ax]);
      src_idx += static_cast<LocT>(idx_val) * strides[ax];
    }

    LocT src_offset = elem_to_loc(z, slice_sizes.data(), strides.data(), ndim);
    return src_idx + src_offset;
  }
};

} // namespace mlx::core::cu
