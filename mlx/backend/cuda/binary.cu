// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/binary.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/dtype_utils.cuh"
#include "mlx/backend/cuda/kernels/binary.cuh"
#include "mlx/primitives.h"

#define BINARY_GPU(func)                                                     \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) {        \
    auto& s = out.primitive().stream();                                      \
    binary_op_gpu<mxcuda::func>(inputs, out, get_primitive_string(this), s); \
  }

#define BINARY_GPU_MULTI(func)                                         \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    auto& s = outputs[0].primitive().stream();                         \
    binary_op_gpu<mxcuda::func>(                                       \
        inputs, outputs, get_primitive_string(this), s);               \
  }

namespace mlx::core {

namespace {

template <typename T>
inline constexpr bool is_floating_v =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>;

template <typename Op, typename In, typename Out>
constexpr bool is_supported_binary_op() {
  if (std::is_same_v<Op, mxcuda::Add> || std::is_same_v<Op, mxcuda::Divide> ||
      std::is_same_v<Op, mxcuda::Maximum> ||
      std::is_same_v<Op, mxcuda::Minimum> ||
      std::is_same_v<Op, mxcuda::Multiply> ||
      std::is_same_v<Op, mxcuda::Subtract> ||
      std::is_same_v<Op, mxcuda::Power> ||
      std::is_same_v<Op, mxcuda::Remainder>) {
    return std::is_same_v<In, Out>;
  }
  if (std::is_same_v<Op, mxcuda::Equal> ||
      std::is_same_v<Op, mxcuda::Greater> ||
      std::is_same_v<Op, mxcuda::GreaterEqual> ||
      std::is_same_v<Op, mxcuda::Less> ||
      std::is_same_v<Op, mxcuda::LessEqual> ||
      std::is_same_v<Op, mxcuda::NotEqual>) {
    return std::is_same_v<Out, bool>;
  }
  if (std::is_same_v<Op, mxcuda::LogicalAnd> ||
      std::is_same_v<Op, mxcuda::LogicalOr>) {
    return std::is_same_v<Out, bool> && std::is_same_v<In, bool>;
  }
  if (std::is_same_v<Op, mxcuda::NaNEqual>) {
    return std::is_same_v<Out, bool> &&
        (is_floating_v<In> || std::is_same_v<In, complex64_t>);
  }
  if (std::is_same_v<Op, mxcuda::LogAddExp> ||
      std::is_same_v<Op, mxcuda::ArcTan2>) {
    return std::is_same_v<In, Out> && is_floating_v<In>;
  }
  if (std::is_same_v<Op, mxcuda::BitwiseAnd> ||
      std::is_same_v<Op, mxcuda::BitwiseOr> ||
      std::is_same_v<Op, mxcuda::BitwiseXor>) {
    return std::is_same_v<In, Out> && std::is_integral_v<In>;
  }
  if (std::is_same_v<Op, mxcuda::LeftShift> ||
      std::is_same_v<Op, mxcuda::RightShift>) {
    return std::is_same_v<In, Out> && std::is_integral_v<In> &&
        !std::is_same_v<In, bool>;
  }
  return false;
}

template <typename Op>
void binary_op_gpu_inplace(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    std::string_view op,
    const Stream& s) {
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto bopt = get_binary_op_type(a, b);

  auto& out = outputs[0];
  if (out.size() == 0) {
    return;
  }

  // Try to collapse contiguous dims
  auto maybe_collapse = [bopt, &a, &b, &out]() {
    if (bopt == BinaryOpType::General) {
      auto [shape, strides] = collapse_contiguous_dims(a, b, out);
      return std::make_tuple(shape, strides[0], strides[1], strides[2]);
    } else {
      decltype(a.strides()) e{};
      return std::make_tuple(decltype(a.shape()){}, e, e, e);
    }
  };
  auto [shape, strides_a, strides_b, strides_out] = maybe_collapse();

  bool large;
  auto ndim = shape.size();
  int work_per_thread;
  if (bopt == BinaryOpType::General) {
    large = a.data_size() > INT32_MAX || b.data_size() > INT32_MAX ||
        out.size() > INT32_MAX;
    work_per_thread = large ? 4 : 2;
  } else {
    large = out.data_size() > UINT32_MAX;
    work_per_thread = 1;
  }

  std::ignore = work_per_thread;

  auto& encoder = mxcuda::get_command_encoder(s);
  encoder.set_input_array(a, b);
  for (auto& out : outputs) {
    encoder.set_output_array(out);
  }
  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_ALL_TYPES(a.dtype(), CTYPE_IN, [&]() {
      MLX_SWITCH_ALL_TYPES(out.dtype(), CTYPE_OUT, [&]() {
        if constexpr (is_supported_binary_op<Op, CTYPE_IN, CTYPE_OUT>()) {
          using InType = cuda_type_t<CTYPE_IN>;
          using OutType = cuda_type_t<CTYPE_OUT>;
          if (bopt == BinaryOpType::General) {
            throw std::runtime_error(
                "General binary op not implemented for CUDA backend.");
          } else {
            int num_threads = std::min(
                out.data_size(), mxcuda::max_threads_per_block(s.device));
            dim3 num_blocks = large
                ? get_2d_num_blocks(out.shape(), out.strides(), num_threads)
                : dim3(ceil_div(out.data_size(), num_threads));
            switch (bopt) {
              case BinaryOpType::ScalarScalar:
                mxcuda::binary_ss<Op><<<num_blocks, num_threads, 0, stream>>>(
                    a.data<InType>(),
                    b.data<InType>(),
                    out.data<OutType>(),
                    out.data_size());
                break;
              case BinaryOpType::ScalarVector:
                mxcuda::binary_sv<Op><<<num_blocks, num_threads, 0, stream>>>(
                    a.data<InType>(),
                    b.data<InType>(),
                    out.data<OutType>(),
                    out.data_size());
                break;
              case BinaryOpType::VectorScalar:
                mxcuda::binary_vs<Op><<<num_blocks, num_threads, 0, stream>>>(
                    a.data<InType>(),
                    b.data<InType>(),
                    out.data<OutType>(),
                    out.data_size());
                break;
              case BinaryOpType::VectorVector:
                mxcuda::binary_vv<Op><<<num_blocks, num_threads, 0, stream>>>(
                    a.data<InType>(),
                    b.data<InType>(),
                    out.data<OutType>(),
                    out.data_size());
                break;
            }
          }
        } else {
          throw std::runtime_error(fmt::format(
              "Can not do binary op {} on inputs of dtype {} with result of {}",
              op,
              dtype_to_string(a.dtype()),
              dtype_to_string(out.dtype())));
        }
      });
    });
  });
}

template <typename Op>
void binary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    std::string_view op,
    const Stream& s) {
  std::vector<array> outputs = {out};
  binary_op_gpu_inplace<Op>(inputs, outputs, op, s);
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
  binary_op_gpu_inplace<Op>(inputs, out, op, s);
}

} // namespace

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
  auto& s = out.primitive().stream();
  auto op = get_primitive_string(this);
  switch (op_) {
    case BitwiseBinary::And:
      binary_op_gpu<mxcuda::BitwiseAnd>(inputs, out, op, s);
      break;
    case BitwiseBinary::Or:
      binary_op_gpu<mxcuda::BitwiseOr>(inputs, out, op, s);
      break;
    case BitwiseBinary::Xor:
      binary_op_gpu<mxcuda::BitwiseXor>(inputs, out, op, s);
      break;
    case BitwiseBinary::LeftShift:
      binary_op_gpu<mxcuda::LeftShift>(inputs, out, op, s);
      break;
    case BitwiseBinary::RightShift:
      binary_op_gpu<mxcuda::RightShift>(inputs, out, op, s);
      break;
  }
}

} // namespace mlx::core
