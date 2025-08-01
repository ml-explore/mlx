// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/reduce/reduce.cuh"

#include <cooperative_groups.h>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename T, typename U, typename Op>
__global__ void init_reduce(U* out, size_t size) {
  auto index = cg::this_grid().thread_rank();
  if (index < size) {
    out[index] = ReduceInit<Op, T>::value();
  }
}

} // namespace cu

void init_reduce(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type) {
  // Allocate if needed
  if (out.data_shared_ptr() == nullptr) {
    out.set_data(allocator::malloc(out.nbytes()));
  }

  encoder.set_output_array(out);
  dispatch_all_types(in.dtype(), [&](auto type_tag) {
    dispatch_reduce_ops(reduce_type, [&](auto reduce_type_tag) {
      using OP = MLX_GET_TYPE(reduce_type_tag);
      using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
      using U = typename cu::ReduceResult<OP, T>::type;
      auto kernel = cu::init_reduce<T, U, OP>;
      dim3 grid = get_2d_grid_dims(out.shape(), out.strides());
      dim3 block(grid.x < 1024 ? grid.x : 1024, 1, 1);
      grid.x = (grid.x + 1023) / 1024;
      encoder.add_kernel_node(
          kernel, grid, block, 0, out.data<U>(), out.size());
    });
  });
}

} // namespace mlx::core
