// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/copy/copy.cuh"

#include <cooperative_groups.h>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename In, typename Out, typename IdxT>
__global__ void copy_s(const In* in, Out* out, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();
  if (index < size) {
    out[index] = CastOp<In, Out>{}(in[0]);
  }
}

template <typename In, typename Out, typename IdxT>
__global__ void copy_v(const In* in, Out* out, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();
  if (index < size) {
    out[index] = CastOp<In, Out>{}(in[index]);
  }
}

} // namespace cu

void copy_contiguous(
    cu::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t in_offset,
    int64_t out_offset) {
  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_COPY_TYPES(in, out, InType, OutType, {
      MLX_SWITCH_BOOL(out.data_size() > UINT32_MAX, LARGE, {
        using IdxT = std::conditional_t<LARGE, int64_t, uint32_t>;
        auto kernel = cu::copy_s<InType, OutType, IdxT>;
        if (ctype == CopyType::Vector) {
          kernel = cu::copy_v<InType, OutType, IdxT>;
        }
        auto [num_blocks, block_dims] = get_launch_args(kernel, out, LARGE);
        kernel<<<num_blocks, block_dims, 0, stream>>>(
            in.data<InType>() + in_offset,
            out.data<OutType>() + out_offset,
            out.data_size());
      });
    });
  });
}

} // namespace mlx::core
