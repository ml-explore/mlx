// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/indexing.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>
#include <thrust/gather.h>

namespace mlx::core {

void Gather::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Gather::eval_gpu");
  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  const auto& src = inputs[0];
  size_t nidx = inputs.size() - 1;
  auto idx_dtype = nidx > 0 ? inputs[1].dtype() : int32;
  auto idx_ndim = nidx > 0 ? inputs[1].ndim() : 0;

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);
  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  encoder.set_output_array(out);
  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_INDEX_TYPES_CHECKED(idx_dtype, "gather", CTYPE_IDX, {
      using IndexType = cuda_type_t<CTYPE_IDX>;
      MLX_SWITCH_NIDX(inputs.size() - 1, NIDX, {
        MLX_SWITCH_IDX_NDIM(idx_ndim, IDX_NDIM, {
          MLX_SWITCH_ALL_TYPES(out.dtype(), CTYPE_DATA, {
            using DataType = cuda_type_t<CTYPE_DATA>;
            auto idx_begin = thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                cu::IndicesOp<IndexType, NIDX, IDX_NDIM>(
                    src,
                    slice_sizes_,
                    axes_,
                    inputs.begin() + 1,
                    inputs.end()));
            thrust::gather(
                cu::thrust_policy(stream),
                idx_begin,
                idx_begin + out.size(),
                src.data<DataType>(),
                out.data<DataType>());
          });
        });
      });
    });
  });
}

} // namespace mlx::core
