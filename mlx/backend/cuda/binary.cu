// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/binary.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/iterators/general_iterator.cuh"
#include "mlx/backend/cuda/iterators/repeat_iterator.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/kernels/binary_ops.cuh"
#include "mlx/backend/cuda/kernels/cucomplex_math.cuh"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

namespace mlx::core {

namespace cu {

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
    return std::is_same_v<Out, bool> &&
        (is_floating_v<In> || std::is_same_v<In, complex64_t>);
  }
  if (std::is_same_v<Op, LogAddExp> || std::is_same_v<Op, ArcTan2>) {
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
    std::vector<array>& outputs,
    std::string_view op,
    const Stream& s) {
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto& out = outputs[0];
  if (out.size() == 0) {
    return;
  }

  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);
  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_ALL_TYPES(a.dtype(), CTYPE_IN, {
      MLX_SWITCH_ALL_TYPES(out.dtype(), CTYPE_OUT, {
        if constexpr (cu::supports_binary_op<Op, CTYPE_IN, CTYPE_OUT>()) {
          using InType = cuda_type_t<CTYPE_IN>;
          using OutType = cuda_type_t<CTYPE_OUT>;
          auto policy = cu::thrust_policy(stream);
          auto a_ptr = thrust::device_pointer_cast(a.data<InType>());
          auto b_ptr = thrust::device_pointer_cast(b.data<InType>());
          auto out_ptr = thrust::device_pointer_cast(out.data<OutType>());

          auto bopt = get_binary_op_type(a, b);
          if (bopt == BinaryOpType::ScalarScalar) {
            auto a_begin = cu::repeat_iterator(a_ptr);
            auto a_end = a_begin + out.data_size();
            auto b_begin = cu::repeat_iterator(b_ptr);
            thrust::transform(policy, a_begin, a_end, b_begin, out_ptr, Op());
          } else if (bopt == BinaryOpType::ScalarVector) {
            auto a_begin = cu::repeat_iterator(a_ptr);
            auto a_end = a_begin + out.data_size();
            auto b_begin = b_ptr;
            thrust::transform(policy, a_begin, a_end, b_begin, out_ptr, Op());
          } else if (bopt == BinaryOpType::VectorScalar) {
            auto a_begin = a_ptr;
            auto a_end = a_begin + out.data_size();
            auto b_begin = cu::repeat_iterator(b_ptr);
            thrust::transform(policy, a_begin, a_end, b_begin, out_ptr, Op());
          } else if (bopt == BinaryOpType::VectorVector) {
            auto a_begin = a_ptr;
            auto a_end = a_begin + out.data_size();
            auto b_begin = b_ptr;
            thrust::transform(policy, a_begin, a_end, b_begin, out_ptr, Op());
          } else {
            auto [shape, strides] = collapse_contiguous_dims(a, b, out);
            auto [a_begin, a_end] = cu::make_general_iterators<int64_t>(
                a_ptr, out.data_size(), shape, strides[0]);
            auto [b_begin, b_end] = cu::make_general_iterators<int64_t>(
                b_ptr, out.data_size(), shape, strides[1]);
            thrust::transform(policy, a_begin, a_end, b_begin, out_ptr, Op());
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
  });
}

template <typename Op>
void binary_op_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    std::string_view op,
    const Stream& s) {
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, outputs[0], bopt);
  set_binary_op_output_data(a, b, outputs[1], bopt);
  binary_op_gpu_inplace<Op>(inputs, outputs, op, s);
}

template <typename Op>
void binary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    std::string_view op,
    const Stream& s) {
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, out, bopt);
  std::vector<array> outputs{out};
  binary_op_gpu_inplace<Op>(inputs, outputs, op, s);
}

#define BINARY_GPU(func)                                                 \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) {    \
    nvtx3::scoped_range r(#func "::eval_gpu");                           \
    auto& s = out.primitive().stream();                                  \
    binary_op_gpu<cu::func>(inputs, out, get_primitive_string(this), s); \
  }

#define BINARY_GPU_MULTI(func)                                               \
  void func::eval_gpu(                                                       \
      const std::vector<array>& inputs, std::vector<array>& outputs) {       \
    nvtx3::scoped_range r(#func "::eval_gpu");                               \
    auto& s = outputs[0].primitive().stream();                               \
    binary_op_gpu<cu::func>(inputs, outputs, get_primitive_string(this), s); \
  }

BINARY_GPU(Add)
BINARY_GPU(ArcTan2)
BINARY_GPU(Divide)
BINARY_GPU(Remainder)
BINARY_GPU(Equal)
BINARY_GPU(Greater)
BINARY_GPU(GreaterEqual)
BINARY_GPU(Less)
BINARY_GPU(LessEqual)
BINARY_GPU(LogicalAnd)
BINARY_GPU(LogicalOr)
BINARY_GPU(LogAddExp)
BINARY_GPU(Maximum)
BINARY_GPU(Minimum)
BINARY_GPU(Multiply)
BINARY_GPU(NotEqual)
BINARY_GPU(Power)
BINARY_GPU(Subtract)

void BitwiseBinary::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("BitwiseBinary::eval_gpu");
  auto& s = out.primitive().stream();
  auto op = get_primitive_string(this);
  switch (op_) {
    case BitwiseBinary::And:
      binary_op_gpu<cu::BitwiseAnd>(inputs, out, op, s);
      break;
    case BitwiseBinary::Or:
      binary_op_gpu<cu::BitwiseOr>(inputs, out, op, s);
      break;
    case BitwiseBinary::Xor:
      binary_op_gpu<cu::BitwiseXor>(inputs, out, op, s);
      break;
    case BitwiseBinary::LeftShift:
      binary_op_gpu<cu::LeftShift>(inputs, out, op, s);
      break;
    case BitwiseBinary::RightShift:
      binary_op_gpu<cu::RightShift>(inputs, out, op, s);
      break;
  }
}

} // namespace mlx::core
