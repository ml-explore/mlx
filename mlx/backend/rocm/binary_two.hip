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

// Use DivMod from binary_ops.hpp

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ void binary_two_ss(
    const In* a,
    const In* b,
    Out* out_a,
    Out* out_b,
    IdxT size) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  IdxT stride = blockDim.x * gridDim.x;
  
  Op op;
  for (IdxT i = index * N_READS; i < size; i += stride * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && (i + j) < size; ++j) {
      auto result = op(a[0], b[0]);
      out_a[i + j] = result[0];
      out_b[i + j] = result[1];
    }
  }
}

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ void binary_two_sv(
    const In* a,
    const In* b,
    Out* out_a,
    Out* out_b,
    IdxT size) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  IdxT stride = blockDim.x * gridDim.x;
  
  Op op;
  for (IdxT i = index * N_READS; i < size; i += stride * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && (i + j) < size; ++j) {
      auto result = op(a[0], b[i + j]);
      out_a[i + j] = result[0];
      out_b[i + j] = result[1];
    }
  }
}

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ void binary_two_vs(
    const In* a,
    const In* b,
    Out* out_a,
    Out* out_b,
    IdxT size) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  IdxT stride = blockDim.x * gridDim.x;
  
  Op op;
  for (IdxT i = index * N_READS; i < size; i += stride * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && (i + j) < size; ++j) {
      auto result = op(a[i + j], b[0]);
      out_a[i + j] = result[0];
      out_b[i + j] = result[1];
    }
  }
}

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ void binary_two_vv(
    const In* a,
    const In* b,
    Out* out_a,
    Out* out_b,
    IdxT size) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  IdxT stride = blockDim.x * gridDim.x;
  
  Op op;
  for (IdxT i = index * N_READS; i < size; i += stride * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && (i + j) < size; ++j) {
      auto result = op(a[i + j], b[i + j]);
      out_a[i + j] = result[0];
      out_b[i + j] = result[1];
    }
  }
}

template <typename Op, typename In, typename Out, typename IdxT>
__global__ void binary_two_g(
    const In* a,
    const In* b,
    Out* out_a,
    Out* out_b,
    IdxT size,
    const int* shape,
    const int64_t* a_strides,
    const int64_t* b_strides,
    int ndim) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) return;
  
  // Compute input indices
  int64_t a_idx = 0;
  int64_t b_idx = 0;
  IdxT tmp = index;
  for (int i = ndim - 1; i >= 0; --i) {
    int coord = tmp % shape[i];
    a_idx += coord * a_strides[i];
    b_idx += coord * b_strides[i];
    tmp /= shape[i];
  }
  
  Op op;
  auto result = op(a[a_idx], b[b_idx]);
  out_a[index] = result[0];
  out_b[index] = result[1];
}

template <typename Op, typename In, typename Out>
constexpr bool supports_binary_two_op() {
  if constexpr (std::is_same_v<Op, DivMod>) {
    return std::is_same_v<In, Out> && (std::is_integral_v<Out> || std::is_floating_point_v<Out>);
  }
  return false;
}

} // namespace rocm

template <typename Op>
void binary_two_op_gpu_inplace(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    const char* op_name,
    const Stream& s) {
  assert(inputs.size() > 1);
  const auto& a = inputs[0];
  const auto& b = inputs[1];
  auto& out_a = outputs[0];
  auto& out_b = outputs[1];
  auto bopt = get_binary_op_type(a, b);
  auto& encoder = rocm::get_command_encoder(s);
  
  set_binary_op_output_data(
      a, b, out_a, bopt, [&](auto n) { return allocator::malloc(n); });
  set_binary_op_output_data(
      a, b, out_b, bopt, [&](auto n) { return allocator::malloc(n); });
  
  if (out_a.size() == 0) {
    return;
  }
  
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out_a);
  encoder.set_output_array(out_b);
  
  constexpr int N_READS = 4;
  int block_size = 256;
  size_t size = out_a.data_size();
  int num_blocks = std::min((size + block_size * N_READS - 1) / (block_size * N_READS), (size_t)65535);
  
  encoder.launch_kernel([&](hipStream_t stream) {
    #define LAUNCH_BINARY_TWO(T, OP_TYPE) \
      switch (bopt) { \
        case BinaryOpType::ScalarScalar: \
          hipLaunchKernelGGL( \
              (rocm::binary_two_ss<rocm::OP_TYPE, T, T, int64_t, N_READS>), \
              dim3(num_blocks), dim3(block_size), 0, stream, \
              a.data<T>(), b.data<T>(), out_a.data<T>(), out_b.data<T>(), \
              static_cast<int64_t>(size)); \
          break; \
        case BinaryOpType::ScalarVector: \
          hipLaunchKernelGGL( \
              (rocm::binary_two_sv<rocm::OP_TYPE, T, T, int64_t, N_READS>), \
              dim3(num_blocks), dim3(block_size), 0, stream, \
              a.data<T>(), b.data<T>(), out_a.data<T>(), out_b.data<T>(), \
              static_cast<int64_t>(size)); \
          break; \
        case BinaryOpType::VectorScalar: \
          hipLaunchKernelGGL( \
              (rocm::binary_two_vs<rocm::OP_TYPE, T, T, int64_t, N_READS>), \
              dim3(num_blocks), dim3(block_size), 0, stream, \
              a.data<T>(), b.data<T>(), out_a.data<T>(), out_b.data<T>(), \
              static_cast<int64_t>(size)); \
          break; \
        case BinaryOpType::VectorVector: \
          hipLaunchKernelGGL( \
              (rocm::binary_two_vv<rocm::OP_TYPE, T, T, int64_t, N_READS>), \
              dim3(num_blocks), dim3(block_size), 0, stream, \
              a.data<T>(), b.data<T>(), out_a.data<T>(), out_b.data<T>(), \
              static_cast<int64_t>(size)); \
          break; \
        default: \
          throw std::runtime_error("Unsupported binary op type for binary_two"); \
      }
    
    if constexpr (std::is_same_v<Op, rocm::DivMod>) {
      switch (a.dtype()) {
        case float32: LAUNCH_BINARY_TWO(float, DivMod); break;
        case int32: LAUNCH_BINARY_TWO(int32_t, DivMod); break;
        case int64: LAUNCH_BINARY_TWO(int64_t, DivMod); break;
        default:
          throw std::runtime_error("Unsupported type for DivMod");
      }
    }
    #undef LAUNCH_BINARY_TWO
  });
}

template <typename Op>
void binary_two_op_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    const char* op_name,
    const Stream& s) {
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, outputs[0], bopt);
  set_binary_op_output_data(a, b, outputs[1], bopt);
  binary_two_op_gpu_inplace<Op>(inputs, outputs, op_name, s);
}

void DivMod::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = outputs[0].primitive().stream();
  binary_two_op_gpu<rocm::DivMod>(inputs, outputs, name(), s);
}

} // namespace mlx::core
