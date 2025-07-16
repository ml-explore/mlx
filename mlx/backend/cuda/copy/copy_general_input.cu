// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/copy/copy.cuh"

#include <cooperative_groups.h>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename In, typename Out, typename IdxT, int NDIM>
__global__ void copy_g_nd(
    const In* in,
    Out* out,
    IdxT size,
    const __grid_constant__ cuda::std::array<int32_t, NDIM> shape,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> strides_in) {
  IdxT index = cg::this_grid().thread_rank();
  if (index < size) {
    IdxT idx_in = elem_to_loc_nd<NDIM>(index, shape.data(), strides_in.data());
    out[index] = CastOp<In, Out>{}(in[idx_in]);
  }
}

template <typename In, typename Out, typename IdxT>
__global__ void copy_g(
    const In* in,
    Out* out,
    IdxT size,
    const __grid_constant__ Shape shape,
    const __grid_constant__ Strides strides_in,
    int ndim) {
  IdxT index = cg::this_grid().thread_rank();
  if (index < size) {
    IdxT idx_in = elem_to_loc_4d(index, shape.data(), strides_in.data(), ndim);
    out[index] = CastOp<In, Out>{}(in[idx_in]);
  }
}

} // namespace cu

void copy_general_input(
    cu::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t offset_in,
    int64_t offset_out,
    const Shape& shape,
    const Strides& strides_in) {
  dispatch_all_types(in.dtype(), [&](auto in_type_tag) {
    dispatch_all_types(out.dtype(), [&](auto out_type_tag) {
      dispatch_bool(
          in.data_size() > INT32_MAX || out.data_size() > INT32_MAX,
          [&](auto large) {
            using InType = cuda_type_t<MLX_GET_TYPE(in_type_tag)>;
            using OutType = cuda_type_t<MLX_GET_TYPE(out_type_tag)>;
            using IdxT = std::conditional_t<large(), int64_t, int32_t>;
            const InType* in_ptr = in.data<InType>() + offset_in;
            OutType* out_ptr = out.data<OutType>() + offset_out;
            int ndim = shape.size();
            if (ndim <= 3) {
              dispatch_1_2_3(ndim, [&](auto dims_constant) {
                auto kernel =
                    cu::copy_g_nd<InType, OutType, IdxT, dims_constant()>;
                auto [num_blocks, block_dims] =
                    get_launch_args(kernel, out, large());
                encoder.add_kernel_node(
                    kernel,
                    num_blocks,
                    block_dims,
                    in_ptr,
                    out_ptr,
                    out.size(),
                    const_param<dims_constant()>(shape),
                    const_param<dims_constant()>(strides_in));
              });
            } else { // ndim >= 4
              auto kernel = cu::copy_g<InType, OutType, IdxT>;
              auto [num_blocks, block_dims] =
                  get_launch_args(kernel, out, large());
              encoder.add_kernel_node(
                  kernel,
                  num_blocks,
                  block_dims,
                  in_ptr,
                  out_ptr,
                  out.size(),
                  const_param(shape),
                  const_param(strides_in),
                  ndim);
            }
          });
    });
  });
}

} // namespace mlx::core
