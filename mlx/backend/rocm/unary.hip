// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/unary.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/hip_complex_math.hpp"
#include "mlx/backend/rocm/device/unary_ops.hpp"
#include "mlx/backend/rocm/iterators/general_iterator.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>
#include <rocthrust/device_ptr.h>
#include <rocthrust/transform.h>

namespace mlx::core {

namespace rocm {

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
      std::is_same_v<Op, Expm1> || std::is_same_v<Op, Sigmoid> ||
      std::is_same_v<Op, Sqrt> || std::is_same_v<Op, Rsqrt>) {
    return std::is_same_v<In, Out> && is_floating_v<In>;
  }
  if (std::is_same_v<Op, Log> || std::is_same_v<Op, Log2> ||
      std::is_same_v<Op, Log10> || std::is_same_v<Op, Log1p>) {
    return std::is_same_v<In, Out> && is_inexact_v<In>;
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

} // namespace rocm

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

  auto& encoder = rocm::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  encoder.launch_kernel([&](hipStream_t stream) {
    MLX_SWITCH_ALL_TYPES(in.dtype(), CTYPE_IN, {
      MLX_SWITCH_ALL_TYPES(out.dtype(), CTYPE_OUT, {
        if constexpr (rocm::supports_unary_op<Op, CTYPE_IN, CTYPE_OUT>()) {
          using InType = hip_type_t<CTYPE_IN>;
          using OutType = hip_type_t<CTYPE_OUT>;
          auto policy = rocm::thrust_policy(stream);
          auto in_ptr = rocthrust::device_pointer_cast(in.data<InType>());
          auto out_ptr = rocthrust::device_pointer_cast(out.data<OutType>());
          if (in.flags().contiguous) {
            rocthrust::transform(
                policy, in_ptr, in_ptr + in.data_size(), out_ptr, Op());
          } else {
            auto [shape, strides] = collapse_contiguous_dims(in);
            auto [in_begin, in_end] = rocm::make_general_iterators<int64_t>(
                in_ptr, in.size(), shape, strides);
            rocthrust::transform(policy, in_begin, in_end, out_ptr, Op());
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
    auto& s = out.primitive().stream();                                 \
    unary_op_gpu<rocm::func>(inputs, out, get_primitive_string(this), s); \
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
      unary_op_gpu<rocm::Log>(inputs, out, op, s);
      break;
    case Base::two:
      unary_op_gpu<rocm::Log2>(inputs, out, op, s);
      break;
    case Base::ten:
      unary_op_gpu<rocm::Log10>(inputs, out, op, s);
      break;
  }
}

void Round::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  auto& s = out.primitive().stream();
  if (issubdtype(in.dtype(), inexact)) {
    unary_op_gpu<rocm::Round>(inputs, out, get_primitive_string(this), s);
  } else {
    // No-op integer types
    out.copy_shared_buffer(in);
  }
}

void Sqrt::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = out.primitive().stream();
  if (recip_) {
    unary_op_gpu<rocm::Rsqrt>(inputs, out, "Rsqrt", s);
  } else {
    unary_op_gpu<rocm::Sqrt>(inputs, out, "Sqrt", s);
  }
}

} // namespace mlx::core

__global__ void relu_kernel(float* input, float* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = fmaxf(0.0f, input[idx]);
  }
}

__global__ void sigmoid_kernel(float* input, float* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = 1.0f / (1.0f + expf(-input[idx]));
  }
}

void launch_relu(float* input, float* output, int n, hipStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(relu_kernel, dim3(blocks), dim3(threads), 0, stream, input, output, n);
}

void launch_sigmoid(float* input, float* output, int n, hipStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(sigmoid_kernel, dim3(blocks), dim3(threads), 0, stream, input, output, n);
}

} // namespace mlx::core::rocm 