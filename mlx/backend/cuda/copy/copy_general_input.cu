// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/copy/copy.cuh"

#include <cooperative_groups.h>

namespace mlx::core {
static constexpr int TILE_SIZE = 16;

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

template <typename In, typename Out, int N_READS>
__global__ void
copy_col_row(const In* in, Out* out, int64_t rows, int64_t cols) {
  __shared__ Out
      tile[N_READS * TILE_SIZE][N_READS * TILE_SIZE + 4 / sizeof(Out)];

  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();

  auto tile_row = grid.block_index().x * TILE_SIZE * N_READS;
  auto tile_col = grid.block_index().y * TILE_SIZE * N_READS;

  auto tidx = block.thread_index().x;
  auto tidy = N_READS * block.thread_index().y;

  auto in_ptr = in + (tile_col + tidy) * rows + tile_row;

#pragma unroll
  for (int i = 0; i < N_READS; ++i) {
    if ((tile_col + tidy + i) < cols) {
      auto in_vec = load_vector<N_READS>(in_ptr, tidx, rows - tile_row, In(0));
#pragma unroll
      for (int j = 0; j < N_READS; ++j) {
        tile[N_READS * tidx + j][tidy + i] = CastOp<In, Out>{}(in_vec[j]);
      }
      in_ptr += rows;
    }
  }

  block.sync();

  auto out_ptr = out + (tile_row + tidy) * cols + tile_col;

#pragma unroll
  for (int i = 0; i < N_READS; ++i) {
    if ((tile_row + tidy + i) < rows) {
      AlignedVector<Out, N_READS> out_vec;
#pragma unroll
      for (int j = 0; j < N_READS; ++j) {
        out_vec[j] = tile[tidy + i][N_READS * tidx + j];
      }
      store_vector(out_ptr, tidx, out_vec, cols - tile_col);
      out_ptr += cols;
    }
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
      using InType = cuda_type_t<MLX_GET_TYPE(in_type_tag)>;
      using OutType = cuda_type_t<MLX_GET_TYPE(out_type_tag)>;
      const InType* in_ptr = gpu_ptr<InType>(in) + offset_in;
      OutType* out_ptr = gpu_ptr<OutType>(out) + offset_out;
      int ndim = shape.size();

      // Column contiguous to row contiguous specialization
      if (ndim == 2 && strides_in[0] == 1 && strides_in[1] == shape[0]) {
        constexpr int work_per_thread =
            std::min(static_cast<int>(16 / sizeof(OutType)), 8);
        dim3 block_dims = {TILE_SIZE, TILE_SIZE};
        uint32_t num_blocks_x =
            cuda::ceil_div(shape[0], TILE_SIZE * work_per_thread);
        uint32_t num_blocks_y =
            cuda::ceil_div(shape[1], TILE_SIZE * work_per_thread);
        auto kernel = cu::copy_col_row<InType, OutType, work_per_thread>;
        encoder.add_kernel_node(
            kernel,
            {num_blocks_x, num_blocks_y},
            block_dims,
            0,
            in_ptr,
            out_ptr,
            int64_t(shape[0]),
            int64_t(shape[1]));
        return;
      }

      dispatch_bool(
          in.data_size() > INT32_MAX || out.data_size() > INT32_MAX,
          [&](auto large) {
            using IdxT = std::conditional_t<large(), int64_t, int32_t>;

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
