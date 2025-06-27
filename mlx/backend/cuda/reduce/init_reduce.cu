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
  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_ALL_TYPES(in.dtype(), CTYPE, {
      MLX_SWITCH_REDUCE_OPS(reduce_type, OP, {
        using T = cuda_type_t<CTYPE>;
        using U = cu::ReduceResult<OP, T>::type;
        auto kernel = cu::init_reduce<T, U, OP>;
        dim3 grid = get_2d_grid_dims(out.shape(), out.strides());
        dim3 block(grid.x < 1024 ? grid.x : 1024, 1, 1);
        grid.x = (grid.x + 1023) / 1024;
        kernel<<<grid, block, 0, stream>>>(out.data<U>(), out.size());
      });
    });
  });
}

} // namespace mlx::core
