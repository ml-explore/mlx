// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/copy/copy.cuh"

#include <cooperative_groups.h>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename In, typename Out, typename IdxT, int NDIM>
__global__ void copy_gg_dynamic_nd(
    const In* in,
    Out* out,
    IdxT size,
    const __grid_constant__ cuda::std::array<int32_t, NDIM> shape,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> strides_in,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> strides_out,
    const int64_t* offset_in,
    const int64_t* offset_out) {
  IdxT index = cg::this_grid().thread_rank();
  if (index < size) {
    auto [idx_in, idx_out] = elem_to_loc_nd<NDIM>(
        index, shape.data(), strides_in.data(), strides_out.data());
    out[idx_out + *offset_out] = CastOp<In, Out>{}(in[idx_in + *offset_in]);
  }
}

template <typename In, typename Out, typename IdxT>
__global__ void copy_gg_dynamic(
    const In* in,
    Out* out,
    IdxT size,
    const __grid_constant__ Shape shape,
    const __grid_constant__ Strides strides_in,
    const __grid_constant__ Strides strides_out,
    int ndim,
    const int64_t* offset_in,
    const int64_t* offset_out) {
  IdxT index = cg::this_grid().thread_rank();
  if (index < size) {
    auto [idx_in, idx_out] = elem_to_loc_4d(
        index, shape.data(), strides_in.data(), strides_out.data(), ndim);
    out[idx_out + *offset_out] = CastOp<In, Out>{}(in[idx_in + *offset_in]);
  }
}

} // namespace cu

void copy_general_dynamic(
    cu::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t offset_in,
    int64_t offset_out,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out,
    const array& dynamic_offset_in,
    const array& dynamic_offset_out) {
  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_COPY_TYPES(in, out, InType, OutType, {
      const InType* in_ptr = in.data<InType>() + offset_in;
      OutType* out_ptr = out.data<OutType>() + offset_out;
      bool large = in.data_size() > UINT32_MAX || out.data_size() > UINT32_MAX;
      MLX_SWITCH_BOOL(large, LARGE, {
        using IdxT = std::conditional_t<LARGE, int64_t, uint32_t>;
        int ndim = shape.size();
        if (ndim <= 3) {
          MLX_SWITCH_1_2_3(ndim, NDIM, {
            auto kernel = cu::copy_gg_dynamic_nd<InType, OutType, IdxT, NDIM>;
            auto [num_blocks, block_dims] = get_launch_args(kernel, out, large);
            kernel<<<num_blocks, block_dims, 0, stream>>>(
                in_ptr,
                out_ptr,
                out.data_size(),
                const_param<NDIM>(shape),
                const_param<NDIM>(strides_in),
                const_param<NDIM>(strides_out),
                dynamic_offset_in.data<int64_t>(),
                dynamic_offset_out.data<int64_t>());
          });
        } else { // ndim >= 4
          auto kernel = cu::copy_gg_dynamic<InType, OutType, IdxT>;
          auto [num_blocks, block_dims] = get_launch_args(kernel, out, large);
          kernel<<<num_blocks, block_dims, 0, stream>>>(
              in_ptr,
              out_ptr,
              out.data_size(),
              const_param(shape),
              const_param(strides_in),
              const_param(strides_out),
              ndim,
              dynamic_offset_in.data<int64_t>(),
              dynamic_offset_out.data<int64_t>());
        }
      });
    });
  });
}

} // namespace mlx::core
