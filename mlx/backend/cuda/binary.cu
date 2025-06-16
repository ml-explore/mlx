// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/binary.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/binary_ops.cuh"
#include "mlx/backend/cuda/device/cucomplex_math.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <cooperative_groups.h>
#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename Op, typename In, typename Out, typename IdxT>
__global__ void binary_ss(const In* a, const In* b, Out* out, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();
  if (index < size) {
    out[index] = Op{}(a[0], b[0]);
  }
}

template <typename Op, typename In, typename Out, typename IdxT>
__global__ void binary_sv(const In* a, const In* b, Out* out, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();
  if (index < size) {
    out[index] = Op{}(a[0], b[index]);
  }
}

template <typename Op, typename In, typename Out, typename IdxT>
__global__ void binary_vs(const In* a, const In* b, Out* out, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();
  if (index < size) {
    out[index] = Op{}(a[index], b[0]);
  }
}

template <typename Op, typename In, typename Out, typename IdxT>
__global__ void binary_vv(const In* a, const In* b, Out* out, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();
  if (index < size) {
    out[index] = Op{}(a[index], b[index]);
  }
}

template <typename Op, typename In, typename Out, typename IdxT, int NDIM>
__global__ void binary_g_nd(
    const In* a,
    const In* b,
    Out* out,
    IdxT size,
    const __grid_constant__ cuda::std::array<int32_t, NDIM> shape,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> a_strides,
    const __grid_constant__ cuda::std::array<int64_t, NDIM> b_strides) {
  IdxT index = cg::this_grid().thread_rank();
  if (index < size) {
    auto [a_idx, b_idx] = elem_to_loc_nd<NDIM>(
        index, shape.data(), a_strides.data(), b_strides.data());
    out[index] = Op{}(a[a_idx], b[b_idx]);
  }
}

template <typename Op, typename In, typename Out, typename IdxT>
__global__ void binary_g(
    const In* a,
    const In* b,
    Out* out,
    IdxT size,
    const __grid_constant__ Shape shape,
    const __grid_constant__ Strides a_strides,
    const __grid_constant__ Strides b_strides,
    int ndim) {
  IdxT index = cg::this_grid().thread_rank();
  if (index < size) {
    auto [a_idx, b_idx] = elem_to_loc_4d(
        index, shape.data(), a_strides.data(), b_strides.data(), ndim);
    out[index] = Op{}(a[a_idx], b[b_idx]);
  }
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
  assert(inputs.size() > 1);
  const auto& a = inputs[0];
  const auto& b = inputs[1];
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

          auto bopt = get_binary_op_type(a, b);
          if (bopt == BinaryOpType::General) {
            auto [shape, strides] = collapse_contiguous_dims(a, b, out);
            auto& a_strides = strides[0];
            auto& b_strides = strides[1];
            bool large = a.data_size() > UINT32_MAX ||
                b.data_size() > UINT32_MAX || out.data_size() > UINT32_MAX;
            MLX_SWITCH_BOOL(large, LARGE, {
              using IdxT = std::conditional_t<LARGE, int64_t, uint32_t>;
              int ndim = shape.size();
              if (ndim <= 3) {
                MLX_SWITCH_1_2_3(ndim, NDIM, {
                  auto kernel =
                      &cu::binary_g_nd<Op, InType, OutType, IdxT, NDIM>;
                  auto [num_blocks, block_dims] =
                      get_launch_args(kernel, out, large);
                  kernel<<<num_blocks, block_dims, 0, stream>>>(
                      a.data<InType>(),
                      b.data<InType>(),
                      out.data<OutType>(),
                      out.size(),
                      const_param<NDIM>(shape),
                      const_param<NDIM>(a_strides),
                      const_param<NDIM>(b_strides));
                });
              } else {
                auto kernel = cu::binary_g<Op, InType, OutType, IdxT>;
                auto [num_blocks, block_dims] =
                    get_launch_args(kernel, out, large);
                kernel<<<num_blocks, block_dims, 0, stream>>>(
                    a.data<InType>(),
                    b.data<InType>(),
                    out.data<OutType>(),
                    out.size(),
                    const_param(shape),
                    const_param(a_strides),
                    const_param(b_strides),
                    ndim);
              }
            });
          } else {
            MLX_SWITCH_BOOL(out.data_size() > UINT32_MAX, LARGE, {
              using IdxT = std::conditional_t<LARGE, int64_t, uint32_t>;
              auto kernel = cu::binary_ss<Op, InType, OutType, IdxT>;
              if (bopt == BinaryOpType::ScalarVector) {
                kernel = cu::binary_sv<Op, InType, OutType, IdxT>;
              } else if (bopt == BinaryOpType::VectorScalar) {
                kernel = cu::binary_vs<Op, InType, OutType, IdxT>;
              } else if (bopt == BinaryOpType::VectorVector) {
                kernel = cu::binary_vv<Op, InType, OutType, IdxT>;
              }
              auto [num_blocks, block_dims] = get_launch_args(
                  kernel, out.data_size(), out.shape(), out.strides(), LARGE);
              kernel<<<num_blocks, block_dims, 0, stream>>>(
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

void Equal::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Equal::eval_gpu");
  auto& s = out.primitive().stream();
  auto op = get_primitive_string(this);
  if (equal_nan_) {
    binary_op_gpu<cu::NaNEqual>(inputs, out, op, s);
  } else {
    binary_op_gpu<cu::Equal>(inputs, out, op, s);
  }
}

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
