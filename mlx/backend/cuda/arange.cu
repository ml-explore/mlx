// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/fp16_math.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <cooperative_groups.h>
#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename T, typename IdxT, int N_WRITES>
__global__ void arange(T* out, IdxT size, T start, T step) {
  IdxT index = cg::this_grid().thread_rank();

  if ((index + 1) * N_WRITES > size) {
    for (IdxT i = index * N_WRITES; i < size; ++i) {
      out[i] = start + i * step;
    }
  } else {
    AlignedVector<T, N_WRITES> out_vec;
#pragma unroll
    for (int i = 0; i < N_WRITES; ++i) {
      out_vec[i] = start + (index * N_WRITES + i) * step;
    }

    store_vector<N_WRITES>(out, index, out_vec);
  }
}

} // namespace cu

void Arange::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Arange::eval_gpu");
  if (out.size() == 0) {
    return;
  }
  out.set_data(allocator::malloc(out.nbytes()));

  auto& encoder = cu::get_command_encoder(stream());
  encoder.set_output_array(out);

  dispatch_int_float_types(out.dtype(), "Arange", [&](auto type_tag) {
    using CTYPE = MLX_GET_TYPE(type_tag);
    using OutType = cuda_type_t<CTYPE>;
    constexpr int N_WRITES = 16 / sizeof(OutType);
    dispatch_bool(out.data_size() > INT32_MAX, [&](auto large) {
      using IdxT = std::conditional_t<large(), int64_t, int32_t>;
      auto [num_blocks, block_dims] = get_launch_args(out, large(), N_WRITES);
      encoder.add_kernel_node(
          cu::arange<OutType, IdxT, N_WRITES>,
          num_blocks,
          block_dims,
          0,
          out.data<OutType>(),
          out.data_size(),
          static_cast<CTYPE>(start_),
          static_cast<CTYPE>(start_ + step_) - static_cast<CTYPE>(start_));
    });
  });
}

} // namespace mlx::core
