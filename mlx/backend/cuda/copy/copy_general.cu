// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/copy/copy.cuh"

#include <cooperative_groups.h>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename In, typename Out, typename IdxT, int NDIM, int N_READS>
__global__ void copy_gg_nd(
    const In* in,
    Out* out,
    IdxT size_rest,
    const __grid_constant__ cuda::std::array<int32_t, NDIM> shape,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> strides_in,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> strides_out) {
  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();
  IdxT index_rest =
      grid.block_index().y * block.dim_threads().y + block.thread_index().y;
  if (index_rest >= size_rest) {
    return;
  }

  auto shape_x = shape[NDIM - 1];
  auto in_stride_x = strides_in[NDIM - 1];
  auto out_stride_x = strides_out[NDIM - 1];
  IdxT index_x =
      grid.block_index().x * block.dim_threads().x + block.thread_index().x;
  auto [idx_in, idx_out] = elem_to_loc_nd<NDIM>(
      index_rest * shape_x,
      shape.data(),
      strides_in.data(),
      strides_out.data());

  auto in_vec =
      load_vector<N_READS>(in + idx_in, index_x, shape_x, in_stride_x, In(0));
  AlignedVector<Out, N_READS> out_vec;
#pragma unroll
  for (int i = 0; i < N_READS; ++i) {
    out_vec[i] = CastOp<In, Out>{}(in_vec[i]);
  }
  store_vector(out + idx_out, index_x, out_vec, shape_x, out_stride_x);
}

template <typename In, typename Out, typename IdxT, int N_READS>
__global__ void copy_gg(
    const In* in,
    Out* out,
    IdxT size_rest,
    const __grid_constant__ Shape shape,
    const __grid_constant__ Strides strides_in,
    const __grid_constant__ Strides strides_out,
    int ndim) {
  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();
  IdxT index_rest =
      grid.block_index().y * block.dim_threads().y + block.thread_index().y;
  if (index_rest >= size_rest) {
    return;
  }

  auto shape_x = shape[ndim - 1];
  auto in_stride_x = strides_in[ndim - 1];
  auto out_stride_x = strides_out[ndim - 1];
  IdxT index_x =
      grid.block_index().x * block.dim_threads().x + block.thread_index().x;
  auto [idx_in, idx_out] = elem_to_loc(
      index_rest * shape_x,
      shape.data(),
      strides_in.data(),
      strides_out.data(),
      ndim);

  auto in_vec =
      load_vector<N_READS>(in + idx_in, index_x, shape_x, in_stride_x, In(0));
  AlignedVector<Out, N_READS> out_vec;
#pragma unroll
  for (int i = 0; i < N_READS; ++i) {
    out_vec[i] = CastOp<In, Out>{}(in_vec[i]);
  }
  store_vector(out + idx_out, index_x, out_vec, shape_x, out_stride_x);
}

} // namespace cu

void copy_general(
    cu::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t offset_in,
    int64_t offset_out,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out) {
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
            size_t data_size = 1;
            for (auto& s : shape)
              data_size *= s;

            int work_per_thread = 1;
            auto dim0 = ndim > 0 ? shape.back() : 1;
            auto rest = data_size / dim0;
            if (dim0 >= 4) {
              work_per_thread = 4;
            }

            dim0 = (dim0 + work_per_thread - 1) / work_per_thread;
            auto block_dims = get_block_dims(dim0, rest, 1);
            uint32_t num_blocks_x = cuda::ceil_div(dim0, block_dims.x);
            uint32_t num_blocks_y = cuda::ceil_div(rest, block_dims.y);

            if (ndim <= 3) {
              dispatch_1_2_3(ndim, [&](auto ndim_constant) {
                auto kernel =
                    cu::copy_gg_nd<InType, OutType, IdxT, ndim_constant(), 1>;
                if (work_per_thread == 4) {
                  kernel =
                      cu::copy_gg_nd<InType, OutType, IdxT, ndim_constant(), 4>;
                }
                encoder.add_kernel_node(
                    kernel,
                    {num_blocks_x, num_blocks_y},
                    block_dims,
                    0,
                    in_ptr,
                    out_ptr,
                    rest,
                    const_param<ndim_constant()>(shape),
                    const_param<ndim_constant()>(strides_in),
                    const_param<ndim_constant()>(strides_out));
              });
            } else { // ndim >= 4
              auto kernel = cu::copy_gg<InType, OutType, IdxT, 1>;
              if (work_per_thread == 4) {
                kernel = cu::copy_gg<InType, OutType, IdxT, 4>;
              }
              encoder.add_kernel_node(
                  kernel,
                  {num_blocks_x, num_blocks_y},
                  block_dims,
                  0,
                  in_ptr,
                  out_ptr,
                  rest,
                  const_param(shape),
                  const_param(strides_in),
                  const_param(strides_out),
                  ndim);
            }
          });
    });
  });
}

} // namespace mlx::core
