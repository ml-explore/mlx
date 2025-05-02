// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/dtype_utils.cuh"
#include "mlx/backend/cuda/kernels/fp16_math.cuh"
#include "mlx/backend/cuda/kernels/utils.cuh"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>
#include <thrust/gather.h>

namespace mlx::core {

namespace cu {

// Dispatch dynamic nidx to constexpr.
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

// Convert an absolute index to positions in a 3d grid.
template <typename T>
struct IndexToDims {
  T dim0;
  T dim1;
  T dim2;

  __device__ cuda::std::tuple<T, T, T> index_to_dims(T index) {
    T x = index / (dim1 * dim2);
    T y = (index % (dim1 * dim2)) / dim2;
    T z = index % dim2;
    return cuda::std::make_tuple(x, y, z);
  }
};

// Get absolute index from possible negative index.
template <typename IdxT>
inline __device__ auto absolute_index(IdxT idx, int32_t size) {
  if constexpr (cuda::std::is_unsigned_v<IdxT>) {
    return idx;
  } else {
    return static_cast<int32_t>(idx < 0 ? idx + size : idx);
  }
}

template <typename T, size_t NIDX, size_t IDX_NDIM>
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
};

template <typename IdxT, size_t NIDX, size_t IDX_NDIM, typename LocT = int64_t>
struct IndexingOp {
  IndexToDims<size_t> dims;
  size_t ndim;
  Shape shape;
  Strides strides;
  Shape slice_sizes;
  Shape axes;
  Indices<IdxT, NIDX, IDX_NDIM> indices;

  __device__ LocT operator()(size_t idx) {
    auto [x, y, z] = dims.index_to_dims(idx);

    LocT src_idx = 0;
    for (size_t i = 0; i < indices.size; ++i) {
      LocT idx_loc;
      if constexpr (IDX_NDIM == 0) {
        idx_loc = 0;
      } else {
        idx_loc = x * indices.strides[indices.ndim * i];
        if constexpr (IDX_NDIM == MORE_THAN_ONE) {
          if (indices.row_contiguous[i]) {
            idx_loc += y;
          } else {
            size_t offset = indices.ndim * i + 1;
            idx_loc += elem_to_loc(
                y,
                indices.shapes.data() + offset,
                indices.strides.data() + offset,
                indices.ndim - 1);
          }
        }
      }
      auto ax = axes[i];
      auto idx_val = absolute_index(indices.buffers[i][idx_loc], shape[ax]);
      src_idx += static_cast<LocT>(idx_val) * strides[ax];
    }

    LocT src_offset = elem_to_loc(z, slice_sizes.data(), strides.data(), ndim);
    return src_offset + src_idx;
  }
};

} // namespace cu

void Gather::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Gather::eval_gpu");
  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);
  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  encoder.set_output_array(out);

  const auto& src = inputs[0];
  bool has_indices = inputs.size() > 1;
  auto idx_dtype = has_indices ? inputs[1].dtype() : bool_;
  auto idx_ndim = has_indices ? inputs[1].ndim() : 0;

  size_t dim0 = 1;
  size_t dim1 = 1;
  if (has_indices) {
    if (inputs[1].ndim() >= 1) {
      dim0 = inputs[1].shape(0);
    }
    if (inputs[1].ndim() >= 2) {
      dim1 = inputs[1].size() / dim0;
    }
  }
  size_t dim2 = 1;
  for (size_t s : slice_sizes_) {
    dim2 *= s;
  }

  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_ALL_TYPES(idx_dtype, CTYPE_IDX, {
      using IndexType = cuda_type_t<CTYPE_IDX>;
      if constexpr (cuda::std::is_integral_v<IndexType>) {
        MLX_SWITCH_NIDX(inputs.size() - 1, NIDX, {
          MLX_SWITCH_IDX_NDIM(idx_ndim, IDX_NDIM, {
            MLX_SWITCH_ALL_TYPES(out.dtype(), CTYPE_DATA, {
              using DataType = cuda_type_t<CTYPE_DATA>;
              auto map_begin = thrust::make_transform_iterator(
                  thrust::make_counting_iterator(0),
                  cu::IndexingOp<IndexType, NIDX, IDX_NDIM>{
                      {dim0, dim1, dim2},
                      src.ndim(),
                      cu::const_param(src.shape()),
                      cu::const_param(src.strides()),
                      cu::const_param(slice_sizes_),
                      cu::const_param(axes_),
                      {inputs.begin() + 1, inputs.end()}});
              thrust::gather(
                  cu::thrust_policy(stream),
                  map_begin,
                  map_begin + out.size(),
                  src.data<DataType>(),
                  out.data<DataType>());
            });
          });
        });
      } else {
        throw std::runtime_error(fmt::format(
            "Can not use dtype {} as index.", dtype_to_string(idx_dtype)));
      }
    });
  });
}

} // namespace mlx::core
