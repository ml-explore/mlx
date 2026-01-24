// Copyright © 2025 Apple Inc.
#include "mlx/backend/common/ternary.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/ternary_ops.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <cooperative_groups.h>
#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename Op, typename T, typename IdxT, int N_READS>
__global__ void
ternary_v(const bool* a, const T* b, const T* c, T* out, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();

  if ((index + 1) * N_READS > size) {
    for (IdxT i = index * N_READS; i < size; ++i) {
      out[i] = Op{}(a[i], b[i], c[i]);
    }
  } else {
    auto a_vec = load_vector<N_READS>(a, index);
    auto b_vec = load_vector<N_READS>(b, index);
    auto c_vec = load_vector<N_READS>(c, index);

    AlignedVector<T, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = Op{}(a_vec[i], b_vec[i], c_vec[i]);
    }

    store_vector<N_READS>(out, index, out_vec);
  }
}

template <typename Op, typename T, typename IdxT, int NDIM, int N_READS>
__global__ void ternary_g_nd(
    const bool* a,
    const T* b,
    const T* c,
    T* out,
    IdxT size_rest,
    const __grid_constant__ cuda::std::array<int32_t, NDIM> shape,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> a_strides,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> b_strides,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> c_strides) {
  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();
  IdxT index_rest =
      grid.block_index().y * block.dim_threads().y + block.thread_index().y;
  if (index_rest >= size_rest) {
    return;
  }

  auto shape_x = shape[NDIM - 1];
  auto a_stride_x = a_strides[NDIM - 1];
  auto b_stride_x = b_strides[NDIM - 1];
  auto c_stride_x = c_strides[NDIM - 1];
  IdxT index_x =
      grid.block_index().x * block.dim_threads().x + block.thread_index().x;
  auto [a_idx, b_idx, c_idx] = elem_to_loc_nd<NDIM>(
      index_rest * shape_x,
      shape.data(),
      a_strides.data(),
      b_strides.data(),
      c_strides.data());
  auto a_vec =
      load_vector<N_READS>(a + a_idx, index_x, shape_x, a_stride_x, false);
  auto b_vec =
      load_vector<N_READS>(b + b_idx, index_x, shape_x, b_stride_x, T(0));
  auto c_vec =
      load_vector<N_READS>(c + c_idx, index_x, shape_x, c_stride_x, T(0));

  AlignedVector<T, N_READS> out_vec;
#pragma unroll
  for (int i = 0; i < N_READS; ++i) {
    out_vec[i] = Op{}(a_vec[i], b_vec[i], c_vec[i]);
  }
  store_vector(out + shape_x * index_rest, index_x, out_vec, shape_x);
}

template <typename Op, typename T, typename IdxT, int N_READS>
__global__ void ternary_g(
    const bool* a,
    const T* b,
    const T* c,
    T* out,
    IdxT size_rest,
    const __grid_constant__ Shape shape,
    const __grid_constant__ Strides a_strides,
    const __grid_constant__ Strides b_strides,
    const __grid_constant__ Strides c_strides,
    int ndim) {
  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();
  IdxT index_rest =
      grid.block_index().y * block.dim_threads().y + block.thread_index().y;
  if (index_rest >= size_rest) {
    return;
  }

  auto shape_x = shape[ndim - 1];
  auto a_stride_x = a_strides[ndim - 1];
  auto b_stride_x = b_strides[ndim - 1];
  auto c_stride_x = c_strides[ndim - 1];
  IdxT index_x =
      grid.block_index().x * block.dim_threads().x + block.thread_index().x;
  auto [a_idx, b_idx, c_idx] = elem_to_loc(
      index_rest * shape_x,
      shape.data(),
      a_strides.data(),
      b_strides.data(),
      c_strides.data(),
      ndim);
  auto a_vec =
      load_vector<N_READS>(a + a_idx, index_x, shape_x, a_stride_x, false);
  auto b_vec =
      load_vector<N_READS>(b + b_idx, index_x, shape_x, b_stride_x, T(0));
  auto c_vec =
      load_vector<N_READS>(c + c_idx, index_x, shape_x, c_stride_x, T(0));

  AlignedVector<T, N_READS> out_vec;
#pragma unroll
  for (int i = 0; i < N_READS; ++i) {
    out_vec[i] = Op{}(a_vec[i], b_vec[i], c_vec[i]);
  }
  store_vector(out + shape_x * index_rest, index_x, out_vec, shape_x);
}

} // namespace cu

template <typename Op>
void ternary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    const Stream& s) {
  const auto& a = inputs[0];
  const auto& b = inputs[1];
  const auto& c = inputs[2];
  if (out.size() == 0) {
    return;
  }

  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(c);
  encoder.set_output_array(out);
  dispatch_all_types(out.dtype(), [&](auto type_tag) {
    using DType = cuda_type_t<MLX_GET_TYPE(type_tag)>;

    auto topt = get_ternary_op_type(a, b, c);
    if (topt == TernaryOpType::General) {
      dispatch_bool(
          a.data_size() > INT32_MAX || b.data_size() > INT32_MAX ||
              c.data_size() > INT32_MAX || out.data_size() > INT32_MAX,
          [&](auto large) {
            using IdxT = std::conditional_t<large(), int64_t, int32_t>;
            Shape shape;
            std::vector<Strides> strides;
            std::tie(shape, strides) = collapse_contiguous_dims(a, b, c, out);
            auto& a_strides = strides[0];
            auto& b_strides = strides[1];
            auto& c_strides = strides[2];
            int ndim = shape.size();
            int work_per_thread = 1;
            auto dim0 = ndim > 0 ? shape.back() : 1;
            auto rest = out.size() / dim0;
            if (dim0 >= 4) {
              work_per_thread = 4;
            }
            dim0 = (dim0 + work_per_thread - 1) / work_per_thread;
            auto block_dims = get_block_dims(dim0, rest, 1);
            uint32_t num_blocks_x = cuda::ceil_div(dim0, block_dims.x);
            uint32_t num_blocks_y = cuda::ceil_div(rest, block_dims.y);

            if (ndim <= 3) {
              dispatch_1_2_3(ndim, [&](auto dims_constant) {
                auto kernel =
                    cu::ternary_g_nd<Op, DType, IdxT, dims_constant(), 1>;
                if (work_per_thread == 4) {
                  kernel =
                      cu::ternary_g_nd<Op, DType, IdxT, dims_constant(), 4>;
                }
                encoder.add_kernel_node(
                    kernel,
                    {num_blocks_x, num_blocks_y},
                    block_dims,
                    0,
                    a.data<bool>(),
                    b.data<DType>(),
                    c.data<DType>(),
                    out.data<DType>(),
                    rest,
                    const_param<dims_constant()>(shape),
                    const_param<dims_constant()>(a_strides),
                    const_param<dims_constant()>(b_strides),
                    const_param<dims_constant()>(c_strides));
              });
            } else {
              auto kernel = cu::ternary_g<Op, DType, IdxT, 1>;
              if (work_per_thread == 4) {
                kernel = cu::ternary_g<Op, DType, IdxT, 4>;
              }
              encoder.add_kernel_node(
                  kernel,
                  {num_blocks_x, num_blocks_y},
                  block_dims,
                  0,
                  a.data<bool>(),
                  b.data<DType>(),
                  c.data<DType>(),
                  out.data<DType>(),
                  rest,
                  const_param(shape),
                  const_param(a_strides),
                  const_param(b_strides),
                  const_param(c_strides),
                  ndim);
            }
          });
    } else {
      dispatch_bool(out.data_size() > UINT32_MAX, [&](auto large) {
        using IdxT = std::conditional_t<large(), int64_t, uint32_t>;
        constexpr int N_READS = 16 / sizeof(DType);
        auto [num_blocks, block_dims] = get_launch_args(
            out.data_size(), out.shape(), out.strides(), large(), N_READS);
        encoder.add_kernel_node(
            cu::ternary_v<Op, DType, IdxT, N_READS>,
            num_blocks,
            block_dims,
            0,
            a.data<bool>(),
            b.data<DType>(),
            c.data<DType>(),
            out.data<DType>(),
            out.data_size());
      });
    }
  });
}

template <typename Op>
void ternary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const Stream& s) {
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto& c = inputs[2];
  auto topt = get_ternary_op_type(a, b, c);
  set_ternary_op_output_data(a, b, c, out, topt);
  ternary_op_gpu_inplace<Op>(inputs, out, s);
}

void Select::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Select::eval_gpu");
  auto& s = out.primitive().stream();
  ternary_op_gpu<cu::Select>(inputs, out, s);
}

} // namespace mlx::core
