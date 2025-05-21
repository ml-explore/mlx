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
#define MLX_SWITCH_SCATTER_AXIS_OP(REDUCE, REDUCE_ALIAS, ...)    \
  if (REDUCE == ScatterAxis::Sum) {                              \
    using REDUCE_ALIAS = mlx::core::cu::ScatterSum<DataType>;    \
    __VA_ARGS__;                                                 \
  } else {                                                       \
    using REDUCE_ALIAS = mlx::core::cu::ScatterAssign<DataType>; \
    __VA_ARGS__;                                                 \
  }

void ScatterAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("ScatterAxis::eval_gpu");
  auto& src = inputs[0];
  auto& idx = inputs[1];
  auto& upd = inputs[2];

  // Copy src into out.
  CopyType copy_type;
  if (src.data_size() == 1) {
    copy_type = CopyType::Scalar;
  } else if (src.flags().row_contiguous) {
    copy_type = CopyType::Vector;
  } else {
    copy_type = CopyType::General;
  }
  copy_gpu(src, out, copy_type);

  // Empty update.
  if (upd.size() == 0) {
    return;
  }

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(idx);
  encoder.set_input_array(upd);
  encoder.set_output_array(out);
  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_INDEX_TYPES_CHECKED(idx.dtype(), "scatter_axis", CTYPE_IDX, {
      using IndexType = cuda_type_t<CTYPE_IDX>;
      MLX_SWITCH_ALL_TYPES(out.dtype(), CTYPE_DATA, {
        using DataType = cuda_type_t<CTYPE_DATA>;
        MLX_SWITCH_BOOL(idx.flags().row_contiguous, IDX_CONT, {
          MLX_SWITCH_SCATTER_AXIS_OP(reduce_type_, SCATTER_OP, {
            auto policy = cu::thrust_policy(stream);
            auto upd_ptr = thrust::device_pointer_cast(upd.data<DataType>());
            auto size = upd.size();
            auto idx_begin = thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                cu::IndexOp<IndexType, IDX_CONT>(idx, out, axis_));
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
}

} // namespace mlx::core
