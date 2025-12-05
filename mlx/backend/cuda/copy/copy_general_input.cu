// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/copy/copy.cuh"

#include <cooperative_groups.h>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename In, typename Out, typename IdxT, int NDIM, int N_READS>
__global__ void copy_g_nd(
    const In* in,
    Out* out,
    IdxT size_rest,
    const __grid_constant__ cuda::std::array<int32_t, NDIM> shape,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> strides) {
  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();
  IdxT index_rest =
      grid.block_index().y * block.dim_threads().y + block.thread_index().y;
  if (index_rest >= size_rest) {
    return;
  }

  auto shape_x = shape[NDIM - 1];
  auto stride_x = strides[NDIM - 1];
  IdxT index_x =
      grid.block_index().x * block.dim_threads().x + block.thread_index().x;
  auto idx =
      elem_to_loc_nd<NDIM>(index_rest * shape_x, shape.data(), strides.data());
  auto in_vec =
      load_vector<N_READS>(in + idx, index_x, shape_x, stride_x, In(0));
  AlignedVector<Out, N_READS> out_vec;
#pragma unroll
  for (int i = 0; i < N_READS; ++i) {
    out_vec[i] = CastOp<In, Out>{}(in_vec[i]);
  }
  store_vector(out + shape_x * index_rest, index_x, out_vec, shape_x);
}

template <typename In, typename Out, typename IdxT, int N_READS>
__global__ void copy_g(
    const In* in,
    Out* out,
    IdxT size_rest,
    const __grid_constant__ Shape shape,
    const __grid_constant__ Strides strides,
    int ndim) {
  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();
  IdxT index_rest =
      grid.block_index().y * block.dim_threads().y + block.thread_index().y;
  if (index_rest >= size_rest) {
    return;
  }

  auto shape_x = shape[ndim - 1];
  auto stride_x = strides[ndim - 1];
  IdxT index_x =
      grid.block_index().x * block.dim_threads().x + block.thread_index().x;
  auto idx =
      elem_to_loc(index_rest * shape_x, shape.data(), strides.data(), ndim);
  auto in_vec =
      load_vector<N_READS>(in + idx, index_x, shape_x, stride_x, In(0));
  AlignedVector<Out, N_READS> out_vec;
#pragma unroll
  for (int i = 0; i < N_READS; ++i) {
    out_vec[i] = CastOp<In, Out>{}(in_vec[i]);
  }
  store_vector(out + shape_x * index_rest, index_x, out_vec, shape_x);
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
            const InType* in_ptr = gpu_ptr<InType>(in) + offset_in;
            OutType* out_ptr = gpu_ptr<OutType>(out) + offset_out;
            int ndim = shape.size();

            int work_per_thread = 8;
            auto dim0 = ndim > 0 ? shape.back() : 1;
            auto rest = out.size() / dim0;
            if (dim0 >= 4 && dim0 < 8) {
              work_per_thread = 4;
            } else if (dim0 < 4) {
              work_per_thread = 1;
            }
            dim0 = (dim0 + work_per_thread - 1) / work_per_thread;
            auto block_dims = get_block_dims(dim0, rest, 1);
            uint32_t num_blocks_x = cuda::ceil_div(dim0, block_dims.x);
            uint32_t num_blocks_y = cuda::ceil_div(rest, block_dims.y);

            if (ndim <= 3) {
              dispatch_1_2_3(ndim, [&](auto dims_constant) {
                auto kernel =
                    cu::copy_g_nd<InType, OutType, IdxT, dims_constant(), 1>;
                if (work_per_thread == 8) {
                  kernel =
                      cu::copy_g_nd<InType, OutType, IdxT, dims_constant(), 8>;
                } else if (work_per_thread == 4) {
                  kernel =
                      cu::copy_g_nd<InType, OutType, IdxT, dims_constant(), 4>;
                }
                encoder.add_kernel_node(
                    kernel,
                    {num_blocks_x, num_blocks_y},
                    block_dims,
                    0,
                    in_ptr,
                    out_ptr,
                    rest,
                    const_param<dims_constant()>(shape),
                    const_param<dims_constant()>(strides_in));
              });
            } else { // ndim >= 4
              auto kernel = cu::copy_g<InType, OutType, IdxT, 1>;
              if (work_per_thread == 8) {
                kernel = cu::copy_g<InType, OutType, IdxT, 8>;
              } else if (work_per_thread == 4) {
                kernel = cu::copy_g<InType, OutType, IdxT, 4>;
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
                  ndim);
            }
          });
    });
  });
}

} // namespace mlx::core
