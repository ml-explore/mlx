// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/unary.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/cucomplex_math.cuh"
#include "mlx/backend/cuda/device/unary_ops.cuh"
#include "mlx/backend/cuda/iterators/general_iterator.cuh"
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
      out_vec.val[i] = Op{}(in_vec.val[i]);
    }

    store_vector<N_READS>(out, index, out_vec);
  }
}

template <typename Op, typename In, typename Out, typename IdxT>
__global__ void unary_g(
    const In* in,
    Out* out,
    IdxT size,
    const __grid_constant__ Shape shape,
    const __grid_constant__ Strides strides,
    int ndim) {
  IdxT index = cg::this_grid().thread_rank();
  if (index < size) {
    auto idx = elem_to_loc_4d(index, shape.data(), strides.data(), ndim);
    out[index] = Op{}(in[idx]);
  }
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
    return std::is_same_v<In, Out> && !std::is_same_v<In, complex64_t>;
  }
  if (std::is_same_v<Op, Conjugate>) {
    return std::is_same_v<In, Out> && std::is_same_v<In, complex64_t>;
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
    return std::is_same_v<In, complex64_t> && std::is_same_v<Out, float>;
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
    const std::string& op,
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
            // TODO: Choose optimized value based on type size.
            constexpr int N_READS = 4;
            auto kernel = cu::unary_v<Op, InType, OutType, IdxT, N_READS>;
            auto [num_blocks, block_dims] = get_launch_args(
                kernel,
                out.data_size(),
                out.shape(),
                out.strides(),
                large,
                N_READS);
            encoder.add_kernel_node(
                kernel,
                num_blocks,
                block_dims,
                in.data<InType>(),
                out.data<OutType>(),
                out.data_size());
          } else {
            using IdxT = std::conditional_t<large(), int64_t, int32_t>;
            auto [shape, strides] = collapse_contiguous_dims(in);
            auto kernel = cu::unary_g<Op, InType, OutType, IdxT>;
            auto [num_blocks, block_dims] = get_launch_args(kernel, out, large);
            encoder.add_kernel_node(
                kernel,
                num_blocks,
                block_dims,
                in.data<InType>(),
                out.data<OutType>(),
                out.data_size(),
                const_param(shape),
                const_param(strides),
                shape.size());
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
    const std::string& op,
    const Stream& s) {
  set_unary_output_data(inputs[0], out);
  unary_op_gpu_inplace<Op>(inputs, out, op, s);
}

#define UNARY_GPU(func)                                                 \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) {   \
    nvtx3::scoped_range r(#func "::eval_gpu");                          \
    auto& s = out.primitive().stream();                                 \
    unary_op_gpu<cu::func>(inputs, out, get_primitive_string(this), s); \
  }

UNARY_GPU(Abs)
UNARY_GPU(ArcCos)
UNARY_GPU(ArcCosh)
UNARY_GPU(ArcSin)
UNARY_GPU(ArcSinh)
UNARY_GPU(ArcTan)
UNARY_GPU(ArcTanh)
UNARY_GPU(BitwiseInvert)
UNARY_GPU(Ceil)
UNARY_GPU(Conjugate)
UNARY_GPU(Cos)
UNARY_GPU(Cosh)
UNARY_GPU(Erf)
UNARY_GPU(ErfInv)
UNARY_GPU(Exp)
UNARY_GPU(Expm1)
UNARY_GPU(Floor)
UNARY_GPU(Imag)
UNARY_GPU(Log1p)
UNARY_GPU(LogicalNot)
UNARY_GPU(Negative)
UNARY_GPU(Real)
UNARY_GPU(Sigmoid)
UNARY_GPU(Sign)
UNARY_GPU(Sin)
UNARY_GPU(Sinh)
UNARY_GPU(Square)
UNARY_GPU(Tan)
UNARY_GPU(Tanh)

void Log::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Log::eval_gpu");
  auto& s = out.primitive().stream();
  auto op = get_primitive_string(this);
  switch (base_) {
    case Base::e:
      unary_op_gpu<cu::Log>(inputs, out, op, s);
      break;
    case Base::two:
      unary_op_gpu<cu::Log2>(inputs, out, op, s);
      break;
    case Base::ten:
      unary_op_gpu<cu::Log10>(inputs, out, op, s);
      break;
  }
}

void Round::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Round::eval_gpu");
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  auto& s = out.primitive().stream();
  if (issubdtype(in.dtype(), inexact)) {
    unary_op_gpu<cu::Round>(inputs, out, get_primitive_string(this), s);
  } else {
    // No-op integer types
    out.copy_shared_buffer(in);
  }
}

void Sqrt::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Sort::eval_gpu");
  auto& s = out.primitive().stream();
  if (recip_) {
    unary_op_gpu<cu::Rsqrt>(inputs, out, "Rsqrt", s);
  } else {
    unary_op_gpu<cu::Sqrt>(inputs, out, "Sqrt", s);
  }
}

} // namespace mlx::core
