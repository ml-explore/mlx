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
  dispatch_all_types(in.dtype(), [&](auto in_type_tag) {
    dispatch_all_types(out.dtype(), [&](auto out_type_tag) {
      dispatch_bool(out.data_size() > UINT32_MAX, [&](auto large) {
        using InType = cuda_type_t<MLX_GET_TYPE(in_type_tag)>;
        using OutType = cuda_type_t<MLX_GET_TYPE(out_type_tag)>;
        using IdxT = std::conditional_t<large(), int64_t, uint32_t>;
        auto kernel = cu::copy_s<InType, OutType, IdxT>;
        if (ctype == CopyType::Vector) {
          kernel = cu::copy_v<InType, OutType, IdxT>;
        }
        auto [num_blocks, block_dims] = get_launch_args(
            kernel, out.data_size(), out.shape(), out.strides(), large());
        encoder.add_kernel_node(
            kernel,
            num_blocks,
            block_dims,
            in.data<InType>() + in_offset,
            out.data<OutType>() + out_offset,
            out.data_size());
      });
    });
  });
}

} // namespace mlx::core
