// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/unary.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/dtype_utils.cuh"
#include "mlx/backend/cuda/kernels/cucomplex_math.cuh"
#include "mlx/backend/cuda/kernels/iterators/general_iterator.cuh"
#include "mlx/backend/cuda/kernels/unary_ops.cuh"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

namespace mlx::core {

namespace cu {

template <typename Op, typename In, typename Out>
constexpr bool supports_unary_op() {
  if (std::is_same_v<Op, Abs> || std::is_same_v<Op, Negative> ||
      std::is_same_v<Op, Sign>) {
    return std::is_same_v<In, Out>;
  }
  if (std::is_same_v<Op, ArcCos> || std::is_same_v<Op, ArcCosh> ||
      std::is_same_v<Op, ArcSin> || std::is_same_v<Op, ArcSinh> ||
      std::is_same_v<Op, ArcTan> || std::is_same_v<Op, ArcTanh> ||
      std::is_same_v<Op, Erf> || std::is_same_v<Op, ErfInv> ||
      std::is_same_v<Op, Expm1> || std::is_same_v<Op, Log1p> ||
      std::is_same_v<Op, Log> || std::is_same_v<Op, Log2> ||
      std::is_same_v<Op, Log10> || std::is_same_v<Op, Sigmoid> ||
      std::is_same_v<Op, Sqrt> || std::is_same_v<Op, Rsqrt>) {
    return std::is_same_v<In, Out> && is_floating_v<In>;
  }
  if (std::is_same_v<Op, BitwiseInvert>) {
    return std::is_same_v<In, Out> && std::is_integral_v<In> &&
        !std::is_same_v<In, bool>;
  }
  if (std::is_same_v<Op, Ceil> || std::is_same_v<Op, Floor> ||
      std::is_same_v<Op, Square>) {
    return std::is_same_v<In, Out> && !std::is_same_v<In, complex64_t>;
  }
  if (std::is_same_v<Op, Conjugate>) {
    return std::is_same_v<In, Out> && std::is_same_v<In, complex64_t>;
  }
  if (std::is_same_v<Op, Cos> || std::is_same_v<Op, Cosh> ||
      std::is_same_v<Op, Exp> || std::is_same_v<Op, Round> ||
      std::is_same_v<Op, Sin> || std::is_same_v<Op, Sinh> ||
      std::is_same_v<Op, Tan> || std::is_same_v<Op, Tanh>) {
    return std::is_same_v<In, Out> &&
        (is_floating_v<In> || std::is_same_v<In, complex64_t>);
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

  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_ALL_TYPES(in.dtype(), CTYPE_IN, {
      MLX_SWITCH_ALL_TYPES(out.dtype(), CTYPE_OUT, {
        if constexpr (cu::supports_unary_op<Op, CTYPE_IN, CTYPE_OUT>()) {
          using InType = cuda_type_t<CTYPE_IN>;
          using OutType = cuda_type_t<CTYPE_OUT>;
          auto policy = cu::thrust_policy(stream);
          auto in_ptr = thrust::device_pointer_cast(in.data<InType>());
          auto out_ptr = thrust::device_pointer_cast(out.data<OutType>());
          if (in.flags().contiguous) {
            thrust::transform(
                policy, in_ptr, in_ptr + in.data_size(), out_ptr, Op());
          } else {
            auto [shape, strides] = collapse_contiguous_dims(in);
            auto [in_begin, in_end] = cu::make_general_iterators<int64_t>(
                in_ptr, in.data_size(), shape, strides);
            thrust::transform(policy, in_begin, in_end, out_ptr, Op());
          }
        } else {
          throw std::runtime_error(fmt::format(
              "Can not do unary op {} on input of {} with output of {}.",
              op,
              dtype_to_string(in.dtype()),
              dtype_to_string(out.dtype())));
        }
      });
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
  auto& s = out.primitive().stream();
  if (recip_) {
    unary_op_gpu<cu::Rsqrt>(inputs, out, "Rsqrt", s);
  } else {
    unary_op_gpu<cu::Sqrt>(inputs, out, "Sqrt", s);
  }
}

} // namespace mlx::core
