// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/binary.h"
#include "mlx/backend/common/utils.h"
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
    IdxT size,
    const int* shape,
    const int64_t* a_strides,
    const int64_t* b_strides,
    int ndim) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) {
    return;
  }

  // Compute offsets using elem_to_loc style
  IdxT a_idx = 0, b_idx = 0;
  IdxT tmp = index;
  for (int i = ndim - 1; i >= 0 && tmp > 0; --i) {
    IdxT coord = tmp % shape[i];
    a_idx += coord * a_strides[i];
    b_idx += coord * b_strides[i];
    tmp /= shape[i];
  }
  
  out[index] = Op{}(a[a_idx], b[b_idx]);
}

template <typename Op, typename In, typename Out>
constexpr bool supports_binary_op() {
  if constexpr (std::is_same_v<Op, Add> || std::is_same_v<Op, Divide> ||
      std::is_same_v<Op, Maximum> || std::is_same_v<Op, Minimum> ||
      std::is_same_v<Op, Multiply> || std::is_same_v<Op, Subtract> ||
      std::is_same_v<Op, Power> || std::is_same_v<Op, Remainder> ||
      std::is_same_v<Op, FloorDivide>) {
    return std::is_same_v<In, Out>;
  } else if constexpr (std::is_same_v<Op, Equal> || std::is_same_v<Op, NotEqual>) {
    return std::is_same_v<Out, bool>;
  } else if constexpr (std::is_same_v<Op, Greater> ||
      std::is_same_v<Op, GreaterEqual> || std::is_same_v<Op, Less> ||
      std::is_same_v<Op, LessEqual>) {
    return std::is_same_v<Out, bool> && !is_complex_v<In>;
  } else if constexpr (std::is_same_v<Op, LogicalAnd> || std::is_same_v<Op, LogicalOr>) {
    return std::is_same_v<Out, bool> && std::is_same_v<In, bool>;
  } else if constexpr (std::is_same_v<Op, NaNEqual>) {
    return std::is_same_v<Out, bool>;
  } else if constexpr (std::is_same_v<Op, LogAddExp>) {
    return std::is_same_v<In, Out>;
  } else if constexpr (std::is_same_v<Op, ArcTan2>) {
    return std::is_same_v<In, Out> && std::is_floating_point_v<In>;
  } else if constexpr (std::is_same_v<Op, BitwiseAnd> || std::is_same_v<Op, BitwiseOr> ||
      std::is_same_v<Op, BitwiseXor>) {
    return std::is_same_v<In, Out> && std::is_integral_v<In>;
  } else if constexpr (std::is_same_v<Op, LeftShift> || std::is_same_v<Op, RightShift>) {
    return std::is_same_v<In, Out> && std::is_integral_v<In> &&
        !std::is_same_v<In, bool>;
  } else {
    return false;
  }
}

} // namespace rocm

namespace rocm {

// Helper to launch general binary kernel
template <typename Op, typename InType, typename OutType, typename ShapeType, typename StridesVecType>
void launch_binary_general(
    CommandEncoder& encoder,
    const array& a,
    const array& b,
    array& out,
    const ShapeType& shape,
    const StridesVecType& strides_vec) {
  auto& strides_a = strides_vec[0];
  auto& strides_b = strides_vec[1];
  int ndim = shape.size();
  size_t data_size = out.size();
  
  array shape_arr({ndim}, int32, nullptr, {});
  array strides_a_arr({ndim}, int64, nullptr, {});
  array strides_b_arr({ndim}, int64, nullptr, {});
  shape_arr.set_data(allocator::malloc(shape_arr.nbytes()));
  strides_a_arr.set_data(allocator::malloc(strides_a_arr.nbytes()));
  strides_b_arr.set_data(allocator::malloc(strides_b_arr.nbytes()));
  encoder.add_temporary(shape_arr);
  encoder.add_temporary(strides_a_arr);
  encoder.add_temporary(strides_b_arr);
  
  // Need to copy shape and strides data before the lambda captures them
  std::vector<int32_t> shape_copy(shape.begin(), shape.end());
  std::vector<int64_t> strides_a_copy(strides_a.begin(), strides_a.end());
  std::vector<int64_t> strides_b_copy(strides_b.begin(), strides_b.end());
  
  encoder.launch_kernel([=, &a, &b, &out, &shape_arr, &strides_a_arr, &strides_b_arr](hipStream_t stream) {
    (void)hipMemcpyAsync(
        shape_arr.data<int32_t>(),
        shape_copy.data(),
        ndim * sizeof(int32_t),
        hipMemcpyHostToDevice,
        stream);
    (void)hipMemcpyAsync(
        strides_a_arr.data<int64_t>(),
        strides_a_copy.data(),
        ndim * sizeof(int64_t),
        hipMemcpyHostToDevice,
        stream);
    (void)hipMemcpyAsync(
        strides_b_arr.data<int64_t>(),
        strides_b_copy.data(),
        ndim * sizeof(int64_t),
        hipMemcpyHostToDevice,
        stream);
    
    int block_size = 256;
    int num_blocks = (data_size + block_size - 1) / block_size;
    
    hipLaunchKernelGGL(
        (binary_g<Op, InType, OutType, int64_t, 1>),
        dim3(num_blocks), dim3(block_size), 0, stream,
        a.data<InType>(), b.data<InType>(), out.data<OutType>(),
        static_cast<int64_t>(data_size),
        shape_arr.data<int32_t>(),
        strides_a_arr.data<int64_t>(),
        strides_b_arr.data<int64_t>(),
        ndim);
  });
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
  };
  
  // Type dispatch
  switch (a.dtype()) {
    case float32:
      if (bopt == BinaryOpType::General) {
        auto [shape, strides_vec] = collapse_contiguous_dims(a, b, out);
        if (out.dtype() == bool_) {
          rocm::launch_binary_general<Op, float, bool>(encoder, a, b, out, shape, strides_vec);
        } else {
          rocm::launch_binary_general<Op, float, float>(encoder, a, b, out, shape, strides_vec);
        }
      } else if (out.dtype() == bool_) {
        launch_kernel(a.data<float>(), b.data<float>(), out.data<bool>(), out.data_size());
      } else {
        launch_kernel(a.data<float>(), b.data<float>(), out.data<float>(), out.data_size());
      }
      break;
    case float16:
      if (bopt == BinaryOpType::General) {
        auto [shape, strides_vec] = collapse_contiguous_dims(a, b, out);
        if (out.dtype() == bool_) {
          rocm::launch_binary_general<Op, __half, bool>(encoder, a, b, out, shape, strides_vec);
        } else {
          rocm::launch_binary_general<Op, __half, __half>(encoder, a, b, out, shape, strides_vec);
        }
      } else if (out.dtype() == bool_) {
        launch_kernel(a.data<__half>(), b.data<__half>(), out.data<bool>(), out.data_size());
      } else {
        launch_kernel(a.data<__half>(), b.data<__half>(), out.data<__half>(), out.data_size());
      }
      break;
    case bfloat16:
      if (bopt == BinaryOpType::General) {
        auto [shape, strides_vec] = collapse_contiguous_dims(a, b, out);
        if (out.dtype() == bool_) {
          rocm::launch_binary_general<Op, hip_bfloat16, bool>(encoder, a, b, out, shape, strides_vec);
        } else {
          rocm::launch_binary_general<Op, hip_bfloat16, hip_bfloat16>(encoder, a, b, out, shape, strides_vec);
        }
      } else if (out.dtype() == bool_) {
        launch_kernel(a.data<hip_bfloat16>(), b.data<hip_bfloat16>(), out.data<bool>(), out.data_size());
      } else {
        launch_kernel(a.data<hip_bfloat16>(), b.data<hip_bfloat16>(), out.data<hip_bfloat16>(), out.data_size());
      }
      break;
    case int32:
      if (bopt == BinaryOpType::General) {
        auto [shape, strides_vec] = collapse_contiguous_dims(a, b, out);
        if (out.dtype() == bool_) {
          rocm::launch_binary_general<Op, int32_t, bool>(encoder, a, b, out, shape, strides_vec);
        } else {
          rocm::launch_binary_general<Op, int32_t, int32_t>(encoder, a, b, out, shape, strides_vec);
        }
      } else if (out.dtype() == bool_) {
        launch_kernel(a.data<int32_t>(), b.data<int32_t>(), out.data<bool>(), out.data_size());
      } else {
        launch_kernel(a.data<int32_t>(), b.data<int32_t>(), out.data<int32_t>(), out.data_size());
      }
      break;
    case int64:
      if (bopt == BinaryOpType::General) {
        auto [shape, strides_vec] = collapse_contiguous_dims(a, b, out);
        if (out.dtype() == bool_) {
          rocm::launch_binary_general<Op, int64_t, bool>(encoder, a, b, out, shape, strides_vec);
        } else {
          rocm::launch_binary_general<Op, int64_t, int64_t>(encoder, a, b, out, shape, strides_vec);
        }
      } else if (out.dtype() == bool_) {
        launch_kernel(a.data<int64_t>(), b.data<int64_t>(), out.data<bool>(), out.data_size());
      } else {
        launch_kernel(a.data<int64_t>(), b.data<int64_t>(), out.data<int64_t>(), out.data_size());
      }
      break;
    case uint32:
      if (bopt == BinaryOpType::General) {
        auto [shape, strides_vec] = collapse_contiguous_dims(a, b, out);
        if (out.dtype() == bool_) {
          rocm::launch_binary_general<Op, uint32_t, bool>(encoder, a, b, out, shape, strides_vec);
        } else {
          rocm::launch_binary_general<Op, uint32_t, uint32_t>(encoder, a, b, out, shape, strides_vec);
        }
      } else if (out.dtype() == bool_) {
        launch_kernel(a.data<uint32_t>(), b.data<uint32_t>(), out.data<bool>(), out.data_size());
      } else {
        launch_kernel(a.data<uint32_t>(), b.data<uint32_t>(), out.data<uint32_t>(), out.data_size());
      }
      break;
    case uint64:
      if (bopt == BinaryOpType::General) {
        auto [shape, strides_vec] = collapse_contiguous_dims(a, b, out);
        if (out.dtype() == bool_) {
          rocm::launch_binary_general<Op, uint64_t, bool>(encoder, a, b, out, shape, strides_vec);
        } else {
          rocm::launch_binary_general<Op, uint64_t, uint64_t>(encoder, a, b, out, shape, strides_vec);
        }
      } else if (out.dtype() == bool_) {
        launch_kernel(a.data<uint64_t>(), b.data<uint64_t>(), out.data<bool>(), out.data_size());
      } else {
        launch_kernel(a.data<uint64_t>(), b.data<uint64_t>(), out.data<uint64_t>(), out.data_size());
      }
      break;
    case int8:
      if (bopt == BinaryOpType::General) {
        auto [shape, strides_vec] = collapse_contiguous_dims(a, b, out);
        if (out.dtype() == bool_) {
          rocm::launch_binary_general<Op, int8_t, bool>(encoder, a, b, out, shape, strides_vec);
        } else {
          rocm::launch_binary_general<Op, int8_t, int8_t>(encoder, a, b, out, shape, strides_vec);
        }
      } else if (out.dtype() == bool_) {
        launch_kernel(a.data<int8_t>(), b.data<int8_t>(), out.data<bool>(), out.data_size());
      } else {
        launch_kernel(a.data<int8_t>(), b.data<int8_t>(), out.data<int8_t>(), out.data_size());
      }
      break;
    case uint8:
      if (bopt == BinaryOpType::General) {
        auto [shape, strides_vec] = collapse_contiguous_dims(a, b, out);
        if (out.dtype() == bool_) {
          rocm::launch_binary_general<Op, uint8_t, bool>(encoder, a, b, out, shape, strides_vec);
        } else {
          rocm::launch_binary_general<Op, uint8_t, uint8_t>(encoder, a, b, out, shape, strides_vec);
        }
      } else if (out.dtype() == bool_) {
        launch_kernel(a.data<uint8_t>(), b.data<uint8_t>(), out.data<bool>(), out.data_size());
      } else {
        launch_kernel(a.data<uint8_t>(), b.data<uint8_t>(), out.data<uint8_t>(), out.data_size());
      }
      break;
    case bool_:
      if (bopt == BinaryOpType::General) {
        auto [shape, strides_vec] = collapse_contiguous_dims(a, b, out);
        rocm::launch_binary_general<Op, bool, bool>(encoder, a, b, out, shape, strides_vec);
      } else {
        launch_kernel(a.data<bool>(), b.data<bool>(), out.data<bool>(), out.data_size());
      }
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
