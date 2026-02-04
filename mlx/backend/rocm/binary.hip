// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/binary.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/binary_ops.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ void binary_ss(const In* a, const In* b, Out* out, IdxT size) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  IdxT stride = blockDim.x * gridDim.x;

  for (IdxT i = index * N_READS; i < size; i += stride * N_READS) {
    if (i + N_READS <= size) {
      #pragma unroll
      for (int j = 0; j < N_READS; ++j) {
        out[i + j] = Op{}(a[0], b[0]);
      }
    } else {
      for (IdxT j = i; j < size; ++j) {
        out[j] = Op{}(a[0], b[0]);
      }
    }
  }
}

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ void binary_sv(const In* a, const In* b, Out* out, IdxT size) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  IdxT stride = blockDim.x * gridDim.x;

  for (IdxT i = index * N_READS; i < size; i += stride * N_READS) {
    if (i + N_READS <= size) {
      #pragma unroll
      for (int j = 0; j < N_READS; ++j) {
        out[i + j] = Op{}(a[0], b[i + j]);
      }
    } else {
      for (IdxT j = i; j < size; ++j) {
        out[j] = Op{}(a[0], b[j]);
      }
    }
  }
}

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ void binary_vs(const In* a, const In* b, Out* out, IdxT size) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  IdxT stride = blockDim.x * gridDim.x;

  for (IdxT i = index * N_READS; i < size; i += stride * N_READS) {
    if (i + N_READS <= size) {
      #pragma unroll
      for (int j = 0; j < N_READS; ++j) {
        out[i + j] = Op{}(a[i + j], b[0]);
      }
    } else {
      for (IdxT j = i; j < size; ++j) {
        out[j] = Op{}(a[j], b[0]);
      }
    }
  }
}

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ void binary_vv(const In* a, const In* b, Out* out, IdxT size) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  IdxT stride = blockDim.x * gridDim.x;

  for (IdxT i = index * N_READS; i < size; i += stride * N_READS) {
    if (i + N_READS <= size) {
      #pragma unroll
      for (int j = 0; j < N_READS; ++j) {
        out[i + j] = Op{}(a[i + j], b[i + j]);
      }
    } else {
      for (IdxT j = i; j < size; ++j) {
        out[j] = Op{}(a[j], b[j]);
      }
    }
  }
}

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ void binary_g(
    const In* a,
    const In* b,
    Out* out,
    IdxT size_rest,
    const int* shape,
    const int64_t* a_strides,
    const int64_t* b_strides,
    int ndim) {
  IdxT index_rest = blockIdx.y * blockDim.y + threadIdx.y;
  if (index_rest >= size_rest) {
    return;
  }

  auto shape_x = shape[ndim - 1];
  auto a_stride_x = a_strides[ndim - 1];
  auto b_stride_x = b_strides[ndim - 1];
  IdxT index_x = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Compute base offsets for this row
  IdxT a_idx = 0, b_idx = 0;
  IdxT tmp = index_rest * shape_x;
  for (int i = ndim - 1; i >= 0; --i) {
    IdxT coord = tmp % shape[i];
    a_idx += coord * a_strides[i];
    b_idx += coord * b_strides[i];
    tmp /= shape[i];
  }
  
  // Process elements in this row
  for (IdxT i = index_x * N_READS; i < shape_x; i += blockDim.x * gridDim.x * N_READS) {
    if (i + N_READS <= shape_x) {
      #pragma unroll
      for (int j = 0; j < N_READS; ++j) {
        IdxT a_offset = a_idx + (i + j) * a_stride_x;
        IdxT b_offset = b_idx + (i + j) * b_stride_x;
        out[shape_x * index_rest + i + j] = Op{}(a[a_offset], b[b_offset]);
      }
    } else {
      for (IdxT j = i; j < shape_x; ++j) {
        IdxT a_offset = a_idx + j * a_stride_x;
        IdxT b_offset = b_idx + j * b_stride_x;
        out[shape_x * index_rest + j] = Op{}(a[a_offset], b[b_offset]);
      }
    }
  }
}

template <typename Op, typename In, typename Out>
constexpr bool supports_binary_op() {
  if constexpr (std::is_same_v<Op, Add> || std::is_same_v<Op, Divide> ||
      std::is_same_v<Op, Maximum> || std::is_same_v<Op, Minimum> ||
      std::is_same_v<Op, Multiply> || std::is_same_v<Op, Subtract> ||
      std::is_same_v<Op, Power> || std::is_same_v<Op, Remainder>) {
    return std::is_same_v<In, Out>;
  }
  if constexpr (std::is_same_v<Op, Equal> || std::is_same_v<Op, NotEqual>) {
    return std::is_same_v<Out, bool>;
  }
  if constexpr (std::is_same_v<Op, Greater> ||
      std::is_same_v<Op, GreaterEqual> || std::is_same_v<Op, Less> ||
      std::is_same_v<Op, LessEqual>) {
    return std::is_same_v<Out, bool> && !is_complex_v<In>;
  }
  if constexpr (std::is_same_v<Op, LogicalAnd> || std::is_same_v<Op, LogicalOr>) {
    return std::is_same_v<Out, bool> && std::is_same_v<In, bool>;
  }
  if constexpr (std::is_same_v<Op, NaNEqual>) {
    return std::is_same_v<Out, bool>;
  }
  if constexpr (std::is_same_v<Op, LogAddExp>) {
    return std::is_same_v<In, Out>;
  }
  if constexpr (std::is_same_v<Op, ArcTan2>) {
    return std::is_same_v<In, Out> && std::is_floating_point_v<In>;
  }
  if constexpr (std::is_same_v<Op, BitwiseAnd> || std::is_same_v<Op, BitwiseOr> ||
      std::is_same_v<Op, BitwiseXor>) {
    return std::is_same_v<In, Out> && std::is_integral_v<In>;
  }
  if constexpr (std::is_same_v<Op, LeftShift> || std::is_same_v<Op, RightShift>) {
    return std::is_same_v<In, Out> && std::is_integral_v<In> &&
        !std::is_same_v<In, bool>;
  }
  return false;
}

} // namespace rocm

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

  auto& encoder = rocm::get_command_encoder(s);
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);
  
  auto bopt = get_binary_op_type(a, b);
  bool large = out.data_size() > UINT32_MAX;
  
  // Simple dispatch for common types
  auto launch_kernel = [&](auto a_ptr, auto b_ptr, auto out_ptr, auto size) {
    using InType = std::remove_pointer_t<decltype(a_ptr)>;
    using OutType = std::remove_pointer_t<decltype(out_ptr)>;
    
    if constexpr (!rocm::supports_binary_op<Op, InType, OutType>()) {
      throw std::runtime_error(
          std::string("Unsupported type for binary op ") + op);
    } else {
      constexpr int N_READS = 4;
      int block_size = 256;
      int num_blocks = (size + block_size * N_READS - 1) / (block_size * N_READS);
      num_blocks = std::max(1, std::min(num_blocks, 65535));
      
      encoder.launch_kernel([&](hipStream_t stream) {
        if (bopt == BinaryOpType::ScalarScalar) {
          if (large) {
            hipLaunchKernelGGL(
                (rocm::binary_ss<Op, InType, OutType, int64_t, N_READS>),
                dim3(num_blocks), dim3(block_size), 0, stream,
                a_ptr, b_ptr, out_ptr, static_cast<int64_t>(size));
          } else {
            hipLaunchKernelGGL(
                (rocm::binary_ss<Op, InType, OutType, uint32_t, N_READS>),
                dim3(num_blocks), dim3(block_size), 0, stream,
                a_ptr, b_ptr, out_ptr, static_cast<uint32_t>(size));
          }
        } else if (bopt == BinaryOpType::ScalarVector) {
          if (large) {
            hipLaunchKernelGGL(
                (rocm::binary_sv<Op, InType, OutType, int64_t, N_READS>),
                dim3(num_blocks), dim3(block_size), 0, stream,
                a_ptr, b_ptr, out_ptr, static_cast<int64_t>(size));
          } else {
            hipLaunchKernelGGL(
                (rocm::binary_sv<Op, InType, OutType, uint32_t, N_READS>),
                dim3(num_blocks), dim3(block_size), 0, stream,
                a_ptr, b_ptr, out_ptr, static_cast<uint32_t>(size));
          }
        } else if (bopt == BinaryOpType::VectorScalar) {
          if (large) {
            hipLaunchKernelGGL(
                (rocm::binary_vs<Op, InType, OutType, int64_t, N_READS>),
                dim3(num_blocks), dim3(block_size), 0, stream,
                a_ptr, b_ptr, out_ptr, static_cast<int64_t>(size));
          } else {
            hipLaunchKernelGGL(
                (rocm::binary_vs<Op, InType, OutType, uint32_t, N_READS>),
                dim3(num_blocks), dim3(block_size), 0, stream,
                a_ptr, b_ptr, out_ptr, static_cast<uint32_t>(size));
          }
        } else {
          if (large) {
            hipLaunchKernelGGL(
                (rocm::binary_vv<Op, InType, OutType, int64_t, N_READS>),
                dim3(num_blocks), dim3(block_size), 0, stream,
                a_ptr, b_ptr, out_ptr, static_cast<int64_t>(size));
          } else {
            hipLaunchKernelGGL(
                (rocm::binary_vv<Op, InType, OutType, uint32_t, N_READS>),
                dim3(num_blocks), dim3(block_size), 0, stream,
                a_ptr, b_ptr, out_ptr, static_cast<uint32_t>(size));
          }
        }
      });
    }
  };
  
  // Type dispatch
  switch (a.dtype()) {
    case float32:
      if (out.dtype() == bool_) {
        launch_kernel(a.data<float>(), b.data<float>(), out.data<bool>(), out.data_size());
      } else {
        launch_kernel(a.data<float>(), b.data<float>(), out.data<float>(), out.data_size());
      }
      break;
    case float16:
      if (out.dtype() == bool_) {
        launch_kernel(a.data<__half>(), b.data<__half>(), out.data<bool>(), out.data_size());
      } else {
        launch_kernel(a.data<__half>(), b.data<__half>(), out.data<__half>(), out.data_size());
      }
      break;
    case bfloat16:
      if (out.dtype() == bool_) {
        launch_kernel(a.data<hip_bfloat16>(), b.data<hip_bfloat16>(), out.data<bool>(), out.data_size());
      } else {
        launch_kernel(a.data<hip_bfloat16>(), b.data<hip_bfloat16>(), out.data<hip_bfloat16>(), out.data_size());
      }
      break;
    case int32:
      if (out.dtype() == bool_) {
        launch_kernel(a.data<int32_t>(), b.data<int32_t>(), out.data<bool>(), out.data_size());
      } else {
        launch_kernel(a.data<int32_t>(), b.data<int32_t>(), out.data<int32_t>(), out.data_size());
      }
      break;
    case int64:
      if (out.dtype() == bool_) {
        launch_kernel(a.data<int64_t>(), b.data<int64_t>(), out.data<bool>(), out.data_size());
      } else {
        launch_kernel(a.data<int64_t>(), b.data<int64_t>(), out.data<int64_t>(), out.data_size());
      }
      break;
    case uint32:
      if (out.dtype() == bool_) {
        launch_kernel(a.data<uint32_t>(), b.data<uint32_t>(), out.data<bool>(), out.data_size());
      } else {
        launch_kernel(a.data<uint32_t>(), b.data<uint32_t>(), out.data<uint32_t>(), out.data_size());
      }
      break;
    case uint64:
      if (out.dtype() == bool_) {
        launch_kernel(a.data<uint64_t>(), b.data<uint64_t>(), out.data<bool>(), out.data_size());
      } else {
        launch_kernel(a.data<uint64_t>(), b.data<uint64_t>(), out.data<uint64_t>(), out.data_size());
      }
      break;
    case int8:
      if (out.dtype() == bool_) {
        launch_kernel(a.data<int8_t>(), b.data<int8_t>(), out.data<bool>(), out.data_size());
      } else {
        launch_kernel(a.data<int8_t>(), b.data<int8_t>(), out.data<int8_t>(), out.data_size());
      }
      break;
    case uint8:
      if (out.dtype() == bool_) {
        launch_kernel(a.data<uint8_t>(), b.data<uint8_t>(), out.data<bool>(), out.data_size());
      } else {
        launch_kernel(a.data<uint8_t>(), b.data<uint8_t>(), out.data<uint8_t>(), out.data_size());
      }
      break;
    case bool_:
      launch_kernel(a.data<bool>(), b.data<bool>(), out.data<bool>(), out.data_size());
      break;
    default:
      throw std::runtime_error(
          std::string("Unsupported type for binary op ") + op);
  }
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

#define BINARY_GPU(prim)                                              \
  void prim::eval_gpu(const std::vector<array>& inputs, array& out) { \
    auto& s = out.primitive().stream();                               \
    binary_op_gpu<rocm::prim>(inputs, out, name(), s);                \
  }

BINARY_GPU(Add)
BINARY_GPU(ArcTan2)
BINARY_GPU(Divide)
BINARY_GPU(Greater)
BINARY_GPU(GreaterEqual)
BINARY_GPU(Less)
BINARY_GPU(LessEqual)
BINARY_GPU(LogAddExp)
BINARY_GPU(LogicalAnd)
BINARY_GPU(LogicalOr)
BINARY_GPU(Maximum)
BINARY_GPU(Minimum)
BINARY_GPU(Multiply)
BINARY_GPU(NotEqual)
BINARY_GPU(Power)
BINARY_GPU(Remainder)
BINARY_GPU(Subtract)

#undef BINARY_GPU

void Equal::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = out.primitive().stream();
  if (equal_nan_) {
    binary_op_gpu<rocm::NaNEqual>(inputs, out, name(), s);
  } else {
    binary_op_gpu<rocm::Equal>(inputs, out, name(), s);
  }
}

void BitwiseBinary::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = out.primitive().stream();
  switch (op_) {
    case BitwiseBinary::And:
      binary_op_gpu<rocm::BitwiseAnd>(inputs, out, name(), s);
      break;
    case BitwiseBinary::Or:
      binary_op_gpu<rocm::BitwiseOr>(inputs, out, name(), s);
      break;
    case BitwiseBinary::Xor:
      binary_op_gpu<rocm::BitwiseXor>(inputs, out, name(), s);
      break;
    case BitwiseBinary::LeftShift:
      binary_op_gpu<rocm::LeftShift>(inputs, out, name(), s);
      break;
    case BitwiseBinary::RightShift:
      binary_op_gpu<rocm::RightShift>(inputs, out, name(), s);
      break;
  }
}

void DivMod::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // DivMod outputs two arrays: quotient and remainder
  auto& s = outputs[0].primitive().stream();
  auto& a = inputs[0];
  auto& b = inputs[1];
  
  // Set output data
  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, outputs[0], bopt);
  set_binary_op_output_data(a, b, outputs[1], bopt);
  
  // Compute floor divide for first output
  binary_op_gpu_inplace<rocm::FloorDivide>(inputs, outputs[0], "FloorDivide", s);
  
  // Compute remainder for second output
  binary_op_gpu_inplace<rocm::Remainder>(inputs, outputs[1], "Remainder", s);
}

} // namespace mlx::core
