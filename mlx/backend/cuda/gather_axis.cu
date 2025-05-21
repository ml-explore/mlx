// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/indexing.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>
#include <thrust/gather.h>

namespace mlx::core {

void GatherAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("GatherAxis::eval_gpu");
  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  auto& src = inputs[0];
  auto& idx = inputs[1];

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(src);
  encoder.set_input_array(idx);
  encoder.set_output_array(out);
  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_INDEX_TYPES_CHECKED(idx.dtype(), "gather_axis", CTYPE_IDX, {
      using IndexType = cuda_type_t<CTYPE_IDX>;
      MLX_SWITCH_ALL_TYPES(out.dtype(), CTYPE_DATA, {
        using DataType = cuda_type_t<CTYPE_DATA>;
        MLX_SWITCH_BOOL(src.flags().row_contiguous, SRC_CONT, {
          MLX_SWITCH_BOOL(idx.flags().row_contiguous, IDX_CONT, {
            auto idx_begin = thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                cu::IndexOp<IndexType, IDX_CONT, SRC_CONT>(idx, src, axis_));
            thrust::gather(
                cu::thrust_policy(stream),
                idx_begin,
                idx_begin + idx.size(),
                src.data<DataType>(),
                out.data<DataType>());
          });
        });
      });
    });
  });
}

} // namespace mlx::core
