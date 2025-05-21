// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/indexing.cuh"
#include "mlx/backend/cuda/iterators/general_iterator.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/kernels/scatter_ops.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>
#include <thrust/device_ptr.h>

namespace mlx::core {

// Like MLX_SWITCH_ALL_TYPES but for reductions.
#define MLX_SWITCH_SCATTER_OP(REDUCE, REDUCE_ALIAS, ...)         \
  if (REDUCE == Scatter::Sum) {                                  \
    using REDUCE_ALIAS = mlx::core::cu::ScatterSum<DataType>;    \
    __VA_ARGS__;                                                 \
  } else if (REDUCE == Scatter::Prod) {                          \
    using REDUCE_ALIAS = mlx::core::cu::ScatterProd<DataType>;   \
    __VA_ARGS__;                                                 \
  } else if (REDUCE == Scatter::Max) {                           \
    using REDUCE_ALIAS = mlx::core::cu::ScatterMax<DataType>;    \
    __VA_ARGS__;                                                 \
  } else if (REDUCE == Scatter::Min) {                           \
    using REDUCE_ALIAS = mlx::core::cu::ScatterMin<DataType>;    \
    __VA_ARGS__;                                                 \
  } else {                                                       \
    using REDUCE_ALIAS = mlx::core::cu::ScatterAssign<DataType>; \
    __VA_ARGS__;                                                 \
  }

void Scatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Gather::eval_gpu");
  auto& upd = inputs.back();

  // Copy src into out.
  CopyType copy_type;
  if (inputs[0].data_size() == 1) {
    copy_type = CopyType::Scalar;
  } else if (inputs[0].flags().row_contiguous) {
    copy_type = CopyType::Vector;
  } else {
    copy_type = CopyType::General;
  }
  copy_gpu(inputs[0], out, copy_type);

  // Empty update.
  if (upd.size() == 0) {
    return;
  }

  size_t nidx = axes_.size();
  auto idx_dtype = nidx > 0 ? inputs[1].dtype() : int32;
  auto idx_ndim = nidx > 0 ? inputs[1].ndim() : 0;

  Shape upd_shape_post(upd.shape().begin() + idx_ndim, upd.shape().end());

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);
  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  encoder.set_output_array(out);
  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_INDEX_TYPES_CHECKED(idx_dtype, "scatter", CTYPE_IDX, {
      using IndexType = cuda_type_t<CTYPE_IDX>;
      MLX_SWITCH_NIDX(nidx, NIDX, {
        MLX_SWITCH_IDX_NDIM(idx_ndim, IDX_NDIM, {
          MLX_SWITCH_ALL_TYPES(out.dtype(), CTYPE_DATA, {
            using DataType = cuda_type_t<CTYPE_DATA>;
            MLX_SWITCH_SCATTER_OP(reduce_type_, SCATTER_OP, {
              auto policy = cu::thrust_policy(stream);
              auto upd_ptr = thrust::device_pointer_cast(upd.data<DataType>());
              auto size = upd.size();
              auto idx_begin = thrust::make_transform_iterator(
                  thrust::make_counting_iterator(0),
                  cu::IndicesOp<IndexType, NIDX, IDX_NDIM>(
                      out,
                      upd_shape_post,
                      axes_,
                      inputs.begin() + 1,
                      inputs.begin() + 1 + nidx));
              auto out_ptr = out.data<DataType>();
              SCATTER_OP op;
              if (upd.flags().row_contiguous) {
                cu::scatter_n(policy, upd_ptr, size, idx_begin, out_ptr, op);
              } else {
                auto upd_begin = cu::make_general_iterator<int64_t>(
                    upd_ptr, upd.shape(), upd.strides());
                cu::scatter_n(policy, upd_begin, size, idx_begin, out_ptr, op);
              }
            });
          });
        });
      });
    });
  });
}

} // namespace mlx::core
