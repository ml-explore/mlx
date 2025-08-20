// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/binary.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/binary_ops.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <cooperative_groups.h>
#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ void binary_ss(const In* a, const In* b, Out* out, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();

  if ((index + 1) * N_READS > size) {
    for (int i = index * N_READS; i < size; ++i) {
      out[i] = Op{}(a[0], b[0]);
    }
  } else {
    AlignedVector<Out, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = Op{}(a[0], b[0]);
    }

    store_vector<N_READS>(out, index, out_vec);
  }
}

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ void binary_sv(const In* a, const In* b, Out* out, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();

  if ((index + 1) * N_READS > size) {
    for (IdxT i = index * N_READS; i < size; ++i) {
      out[i] = Op{}(a[0], b[i]);
    }
  } else {
    auto b_vec = load_vector<N_READS>(b, index);

    AlignedVector<Out, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = Op{}(a[0], b_vec[i]);
    }

    store_vector<N_READS>(out, index, out_vec);
  }
}

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ void binary_vs(const In* a, const In* b, Out* out, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();

  if ((index + 1) * N_READS > size) {
    for (IdxT i = index * N_READS; i < size; ++i) {
      out[i] = Op{}(a[i], b[0]);
    }
  } else {
    auto a_vec = load_vector<N_READS>(a, index);

    AlignedVector<Out, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = Op{}(a_vec[i], b[0]);
    }

    store_vector<N_READS>(out, index, out_vec);
  }
}

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ void binary_vv(const In* a, const In* b, Out* out, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();

  if ((index + 1) * N_READS > size) {
    for (IdxT i = index * N_READS; i < size; ++i) {
      out[i] = Op{}(a[i], b[i]);
    }
  } else {
    auto a_vec = load_vector<N_READS>(a, index);
    auto b_vec = load_vector<N_READS>(b, index);

    AlignedVector<Out, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = Op{}(a_vec[i], b_vec[i]);
    }

    store_vector<N_READS>(out, index, out_vec);
  }
}

template <
    typename Op,
    typename In,
    typename Out,
    typename IdxT,
    int NDIM,
    int N_READS>
__global__ void binary_g_nd(
    const In* a,
    const In* b,
    Out* out,
    IdxT size_rest,
    const __grid_constant__ cuda::std::array<int32_t, NDIM> shape,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> a_strides,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> b_strides) {
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
  IdxT index_x =
      grid.block_index().x * block.dim_threads().x + block.thread_index().x;
  auto [a_idx, b_idx] = elem_to_loc_nd<NDIM>(
      index_rest * shape_x, shape.data(), a_strides.data(), b_strides.data());
  auto a_vec =
      load_vector<N_READS>(a + a_idx, index_x, shape_x, a_stride_x, In(0));
  auto b_vec =
      load_vector<N_READS>(b + b_idx, index_x, shape_x, b_stride_x, In(0));

  AlignedVector<Out, N_READS> out_vec;
#pragma unroll
  for (int i = 0; i < N_READS; ++i) {
    out_vec[i] = Op{}(a_vec[i], b_vec[i]);
  }
  store_vector(out + shape_x * index_rest, index_x, out_vec, shape_x);
}

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ void binary_g(
    const In* a,
    const In* b,
    Out* out,
    IdxT size_rest,
    const __grid_constant__ Shape shape,
    const __grid_constant__ Strides a_strides,
    const __grid_constant__ Strides b_strides,
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
  IdxT index_x =
      grid.block_index().x * block.dim_threads().x + block.thread_index().x;
  auto [a_idx, b_idx] = elem_to_loc(
      index_rest * shape_x,
      shape.data(),
      a_strides.data(),
      b_strides.data(),
      ndim);
  auto a_vec =
      load_vector<N_READS>(a + a_idx, index_x, shape_x, a_stride_x, In(0));
  auto b_vec =
      load_vector<N_READS>(b + b_idx, index_x, shape_x, b_stride_x, In(0));

  AlignedVector<Out, N_READS> out_vec;
#pragma unroll
  for (int i = 0; i < N_READS; ++i) {
    out_vec[i] = Op{}(a_vec[i], b_vec[i]);
  }
  store_vector(out + shape_x * index_rest, index_x, out_vec, shape_x);
}

template <typename Op, typename In, typename Out>
constexpr bool supports_binary_op() {
  if (std::is_same_v<Op, Add> || std::is_same_v<Op, Divide> ||
      std::is_same_v<Op, Maximum> || std::is_same_v<Op, Minimum> ||
      std::is_same_v<Op, Multiply> || std::is_same_v<Op, Subtract> ||
      std::is_same_v<Op, Power> || std::is_same_v<Op, Remainder>) {
    return std::is_same_v<In, Out>;
  }
  if (std::is_same_v<Op, Equal> || std::is_same_v<Op, Greater> ||
      std::is_same_v<Op, GreaterEqual> || std::is_same_v<Op, Less> ||
      std::is_same_v<Op, LessEqual> || std::is_same_v<Op, NotEqual>) {
    return std::is_same_v<Out, bool>;
  }
  if (std::is_same_v<Op, LogicalAnd> || std::is_same_v<Op, LogicalOr>) {
    return std::is_same_v<Out, bool> && std::is_same_v<In, bool>;
  }
  if (std::is_same_v<Op, NaNEqual>) {
    return std::is_same_v<Out, bool> && is_inexact_v<In>;
  }
  if (std::is_same_v<Op, LogAddExp>) {
    return std::is_same_v<In, Out> && is_inexact_v<In>;
  }
  if (std::is_same_v<Op, ArcTan2>) {
    return std::is_same_v<In, Out> && is_floating_v<In>;
  }
  if (std::is_same_v<Op, BitwiseAnd> || std::is_same_v<Op, BitwiseOr> ||
      std::is_same_v<Op, BitwiseXor>) {
    return std::is_same_v<In, Out> && std::is_integral_v<In>;
  }
  if (std::is_same_v<Op, LeftShift> || std::is_same_v<Op, RightShift>) {
    return std::is_same_v<In, Out> && std::is_integral_v<In> &&
        !std::is_same_v<In, bool>;
  }
  return false;
}

} // namespace cu

template <typename Op>
void binary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    const char* op,
    const Stream& s) {
  assert(inputs.size() > 1);
  const auto& a = inputs[0];
  const auto& b = inputs[1];
  if (out.size() == 0) {
    return;
  }

  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);
  dispatch_all_types(a.dtype(), [&](auto in_type_tag) {
    dispatch_all_types(out.dtype(), [&](auto out_type_tag) {
      using CTYPE_IN = MLX_GET_TYPE(in_type_tag);
      using CTYPE_OUT = MLX_GET_TYPE(out_type_tag);
      if constexpr (cu::supports_binary_op<Op, CTYPE_IN, CTYPE_OUT>()) {
        using InType = cuda_type_t<CTYPE_IN>;
        using OutType = cuda_type_t<CTYPE_OUT>;
        auto bopt = get_binary_op_type(a, b);
        if (bopt == BinaryOpType::General) {
          dispatch_bool(
              a.data_size() > INT32_MAX || b.data_size() > INT32_MAX ||
                  out.data_size() > INT32_MAX,
              [&](auto large) {
                using IdxT = std::conditional_t<large(), int64_t, int32_t>;
                Shape shape;
                std::vector<Strides> strides;
                std::tie(shape, strides) = collapse_contiguous_dims(a, b, out);
                auto& a_strides = strides[0];
                auto& b_strides = strides[1];
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
                    auto kernel = cu::binary_g_nd<
                        Op,
                        InType,
                        OutType,
                        IdxT,
                        dims_constant(),
                        1>;
                    if (work_per_thread == 4) {
                      kernel = cu::binary_g_nd<
                          Op,
                          InType,
                          OutType,
                          IdxT,
                          dims_constant(),
                          4>;
                    }
                    encoder.add_kernel_node(
                        kernel,
                        {num_blocks_x, num_blocks_y},
                        block_dims,
                        0,
                        a.data<InType>(),
                        b.data<InType>(),
                        out.data<OutType>(),
                        rest,
                        const_param<dims_constant()>(shape),
                        const_param<dims_constant()>(a_strides),
                        const_param<dims_constant()>(b_strides));
                  });
                } else {
                  auto kernel = cu::binary_g<Op, InType, OutType, IdxT, 1>;
                  if (work_per_thread == 4) {
                    kernel = cu::binary_g<Op, InType, OutType, IdxT, 4>;
                  }
                  encoder.add_kernel_node(
                      kernel,
                      {num_blocks_x, num_blocks_y},
                      block_dims,
                      0,
                      a.data<InType>(),
                      b.data<InType>(),
                      out.data<OutType>(),
                      rest,
                      const_param(shape),
                      const_param(a_strides),
                      const_param(b_strides),
                      ndim);
                }
              });
        } else {
          dispatch_bool(out.data_size() > UINT32_MAX, [&](auto large) {
            using IdxT = std::conditional_t<large(), int64_t, uint32_t>;
            constexpr int N_READS = 16 / sizeof(InType);
            auto kernel = cu::binary_ss<Op, InType, OutType, IdxT, N_READS>;
            if (bopt == BinaryOpType::ScalarVector) {
              kernel = cu::binary_sv<Op, InType, OutType, IdxT, N_READS>;
            } else if (bopt == BinaryOpType::VectorScalar) {
              kernel = cu::binary_vs<Op, InType, OutType, IdxT, N_READS>;
            } else if (bopt == BinaryOpType::VectorVector) {
              kernel = cu::binary_vv<Op, InType, OutType, IdxT, N_READS>;
            }
            auto [num_blocks, block_dims] = get_launch_args(
                out.data_size(), out.shape(), out.strides(), large(), N_READS);
            encoder.add_kernel_node(
                kernel,
                num_blocks,
                block_dims,
                0,
                a.data<InType>(),
                b.data<InType>(),
                out.data<OutType>(),
                out.data_size());
          });
        }
      } else {
        throw std::runtime_error(fmt::format(
            "Can not do binary op {} on inputs of {} with result of {}.",
            op,
            dtype_to_string(a.dtype()),
            dtype_to_string(out.dtype())));
      }
    });
  });
}

template <typename Op>
void binary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const char* op,
    const Stream& s) {
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, out, bopt);
  binary_op_gpu_inplace<Op>(inputs, out, op, s);
}

#define BINARY_GPU(func)                                              \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    nvtx3::scoped_range r(#func "::eval_gpu");                        \
    auto& s = out.primitive().stream();                               \
    binary_op_gpu<cu::func>(inputs, out, name(), s);                  \
  }

} // namespace mlx::core
