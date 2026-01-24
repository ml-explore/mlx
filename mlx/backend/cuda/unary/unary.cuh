// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/unary.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/unary_ops.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <cooperative_groups.h>
#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ void unary_v(const In* in, Out* out, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();

  if ((index + 1) * N_READS > size) {
    for (IdxT i = index * N_READS; i < size; ++i) {
      out[i] = Op{}(in[i]);
    }
  } else {
    auto in_vec = load_vector<N_READS>(in, index);

    AlignedVector<Out, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = Op{}(in_vec[i]);
    }

    store_vector<N_READS>(out, index, out_vec);
  }
}

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ void unary_g(
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
    out_vec[i] = Op{}(in_vec[i]);
  }
  store_vector(out + shape_x * index_rest, index_x, out_vec, shape_x);
}

template <typename Op, typename In, typename Out>
constexpr bool supports_unary_op() {
  if (std::is_same_v<Op, Abs> || std::is_same_v<Op, Negative> ||
      std::is_same_v<Op, Sign> || std::is_same_v<Op, Square>) {
    return std::is_same_v<In, Out>;
  }
  if (std::is_same_v<Op, ArcCosh> || std::is_same_v<Op, ArcSinh> ||
      std::is_same_v<Op, ArcTanh> || std::is_same_v<Op, Erf> ||
      std::is_same_v<Op, ErfInv> || std::is_same_v<Op, Expm1> ||
      std::is_same_v<Op, Sigmoid>) {
    return std::is_same_v<In, Out> && is_floating_v<In>;
  }
  if (std::is_same_v<Op, BitwiseInvert>) {
    return std::is_same_v<In, Out> && std::is_integral_v<In> &&
        !std::is_same_v<In, bool>;
  }
  if (std::is_same_v<Op, Ceil> || std::is_same_v<Op, Floor>) {
    return std::is_same_v<In, Out> && !mlx::core::is_complex_v<In>;
  }
  if (std::is_same_v<Op, Conjugate>) {
    return std::is_same_v<In, Out> && mlx::core::is_complex_v<In>;
  }
  if (std::is_same_v<Op, ArcCos> || std::is_same_v<Op, ArcSin> ||
      std::is_same_v<Op, ArcTan> || std::is_same_v<Op, Cos> ||
      std::is_same_v<Op, Cosh> || std::is_same_v<Op, Exp> ||
      std::is_same_v<Op, Log> || std::is_same_v<Op, Log2> ||
      std::is_same_v<Op, Log10> || std::is_same_v<Op, Log1p> ||
      std::is_same_v<Op, Round> || std::is_same_v<Op, Rsqrt> ||
      std::is_same_v<Op, Sqrt> || std::is_same_v<Op, Sin> ||
      std::is_same_v<Op, Sinh> || std::is_same_v<Op, Tan> ||
      std::is_same_v<Op, Tanh>) {
    return std::is_same_v<In, Out> && is_inexact_v<In>;
  }
  if (std::is_same_v<Op, Imag> || std::is_same_v<Op, Real>) {
    return mlx::core::is_complex_v<In> && std::is_same_v<Out, float>;
  }
  if (std::is_same_v<Op, LogicalNot>) {
    return std::is_same_v<In, Out> && std::is_same_v<In, bool>;
  }
  return false;
}

} // namespace cu

template <typename Op>
void unary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    const char* op,
    const Stream& s) {
  auto& in = inputs[0];
  if (in.size() == 0) {
    return;
  }
  bool contig = in.flags().contiguous;
  bool large;
  if (!contig) {
    large = in.data_size() > INT32_MAX || out.size() > INT32_MAX;
  } else {
    large = in.data_size() > UINT32_MAX;
  }

  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  dispatch_all_types(in.dtype(), [&](auto in_type_tag) {
    dispatch_all_types(out.dtype(), [&](auto out_type_tag) {
      using CTYPE_IN = MLX_GET_TYPE(in_type_tag);
      using CTYPE_OUT = MLX_GET_TYPE(out_type_tag);
      if constexpr (cu::supports_unary_op<Op, CTYPE_IN, CTYPE_OUT>()) {
        dispatch_bool(large, [&](auto large) {
          using InType = cuda_type_t<CTYPE_IN>;
          using OutType = cuda_type_t<CTYPE_OUT>;
          if (contig) {
            using IdxT = std::conditional_t<large(), int64_t, uint32_t>;
            constexpr int N_READS = 16 / sizeof(OutType);
            auto [num_blocks, block_dims] = get_launch_args(
                out.data_size(), out.shape(), out.strides(), large, N_READS);
            encoder.add_kernel_node(
                cu::unary_v<Op, InType, OutType, IdxT, N_READS>,
                num_blocks,
                block_dims,
                0,
                in.data<InType>(),
                out.data<OutType>(),
                out.data_size());
          } else {
            using IdxT = std::conditional_t<large(), int64_t, int32_t>;
            auto [shape, strides] = collapse_contiguous_dims(in);
            auto ndim = shape.size();
            int work_per_thread = 1;
            auto kernel = cu::unary_g<Op, InType, OutType, IdxT, 1>;
            auto dim0 = ndim > 0 ? shape.back() : 1;
            auto rest = out.size() / dim0;
            if (dim0 >= 4) {
              kernel = cu::unary_g<Op, InType, OutType, IdxT, 4>;
              work_per_thread = 4;
            }
            dim0 = (dim0 + work_per_thread - 1) / work_per_thread;
            auto block_dims = get_block_dims(dim0, rest, 1);
            uint32_t num_blocks_x = cuda::ceil_div(dim0, block_dims.x);
            uint32_t num_blocks_y = cuda::ceil_div(rest, block_dims.y);
            encoder.add_kernel_node(
                kernel,
                {num_blocks_x, num_blocks_y},
                block_dims,
                0,
                in.data<InType>(),
                out.data<OutType>(),
                rest,
                const_param(shape),
                const_param(strides),
                ndim);
          }
        });
      } else {
        throw std::runtime_error(fmt::format(
            "Can not do unary op {} on input of {} with output of {}.",
            op,
            dtype_to_string(in.dtype()),
            dtype_to_string(out.dtype())));
      }
    });
  });
}

template <typename Op>
void unary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const char* op,
    const Stream& s) {
  set_unary_output_data(inputs[0], out);
  unary_op_gpu_inplace<Op>(inputs, out, op, s);
}

#define UNARY_GPU(func)                                               \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    nvtx3::scoped_range r(#func "::eval_gpu");                        \
    auto& s = out.primitive().stream();                               \
    unary_op_gpu<cu::func>(inputs, out, name(), s);                   \
  }

} // namespace mlx::core
