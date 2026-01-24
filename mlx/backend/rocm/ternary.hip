// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/ternary.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/ternary_ops.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

template <typename Op, typename T, typename IdxT, int N_READS>
__global__ void
ternary_v(const bool* a, const T* b, const T* c, T* out, IdxT size) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  IdxT stride = blockDim.x * gridDim.x;

  for (IdxT i = index * N_READS; i < size; i += stride * N_READS) {
    if (i + N_READS <= size) {
      #pragma unroll
      for (int j = 0; j < N_READS; ++j) {
        out[i + j] = Op{}(a[i + j], b[i + j], c[i + j]);
      }
    } else {
      for (IdxT j = i; j < size; ++j) {
        out[j] = Op{}(a[j], b[j], c[j]);
      }
    }
  }
}

template <typename Op, typename T, typename IdxT, int N_READS>
__global__ void ternary_g(
    const bool* a,
    const T* b,
    const T* c,
    T* out,
    IdxT size_rest,
    const int* shape,
    const int64_t* a_strides,
    const int64_t* b_strides,
    const int64_t* c_strides,
    int ndim) {
  IdxT index_rest = blockIdx.y * blockDim.y + threadIdx.y;
  if (index_rest >= size_rest) {
    return;
  }

  auto shape_x = shape[ndim - 1];
  auto a_stride_x = a_strides[ndim - 1];
  auto b_stride_x = b_strides[ndim - 1];
  auto c_stride_x = c_strides[ndim - 1];
  IdxT index_x = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Compute base offsets for this row
  IdxT a_idx = 0, b_idx = 0, c_idx = 0;
  IdxT tmp = index_rest * shape_x;
  for (int i = ndim - 1; i >= 0; --i) {
    IdxT coord = tmp % shape[i];
    a_idx += coord * a_strides[i];
    b_idx += coord * b_strides[i];
    c_idx += coord * c_strides[i];
    tmp /= shape[i];
  }
  
  // Process elements in this row
  for (IdxT i = index_x * N_READS; i < shape_x; i += blockDim.x * gridDim.x * N_READS) {
    if (i + N_READS <= shape_x) {
      #pragma unroll
      for (int j = 0; j < N_READS; ++j) {
        IdxT a_offset = a_idx + (i + j) * a_stride_x;
        IdxT b_offset = b_idx + (i + j) * b_stride_x;
        IdxT c_offset = c_idx + (i + j) * c_stride_x;
        out[shape_x * index_rest + i + j] = Op{}(a[a_offset], b[b_offset], c[c_offset]);
      }
    } else {
      for (IdxT j = i; j < shape_x; ++j) {
        IdxT a_offset = a_idx + j * a_stride_x;
        IdxT b_offset = b_idx + j * b_stride_x;
        IdxT c_offset = c_idx + j * c_stride_x;
        out[shape_x * index_rest + j] = Op{}(a[a_offset], b[b_offset], c[c_offset]);
      }
    }
  }
}

} // namespace rocm

template <typename Op>
void ternary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    const Stream& s) {
  const auto& a = inputs[0];
  const auto& b = inputs[1];
  const auto& c = inputs[2];
  if (out.size() == 0) {
    return;
  }

  auto& encoder = rocm::get_command_encoder(s);
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(c);
  encoder.set_output_array(out);
  
  auto topt = get_ternary_op_type(a, b, c);
  bool large = out.data_size() > UINT32_MAX;
  
  // Simple dispatch for common types
  auto launch_kernel = [&](auto b_ptr, auto c_ptr, auto out_ptr, auto size) {
    using DType = std::remove_pointer_t<decltype(b_ptr)>;
    
    constexpr int N_READS = 4;
    int block_size = 256;
    int num_blocks = (size + block_size * N_READS - 1) / (block_size * N_READS);
    num_blocks = std::min(num_blocks, 65535);
    
    encoder.launch_kernel([&](hipStream_t stream) {
      if (large) {
        hipLaunchKernelGGL(
            (rocm::ternary_v<Op, DType, int64_t, N_READS>),
            dim3(num_blocks), dim3(block_size), 0, stream,
            a.data<bool>(), b_ptr, c_ptr, out_ptr, static_cast<int64_t>(size));
      } else {
        hipLaunchKernelGGL(
            (rocm::ternary_v<Op, DType, uint32_t, N_READS>),
            dim3(num_blocks), dim3(block_size), 0, stream,
            a.data<bool>(), b_ptr, c_ptr, out_ptr, static_cast<uint32_t>(size));
      }
    });
  };
  
  // Type dispatch
  switch (out.dtype()) {
    case float32:
      launch_kernel(b.data<float>(), c.data<float>(), out.data<float>(), out.data_size());
      break;
    case float16:
      launch_kernel(b.data<__half>(), c.data<__half>(), out.data<__half>(), out.data_size());
      break;
    case bfloat16:
      launch_kernel(b.data<__hip_bfloat16>(), c.data<__hip_bfloat16>(), out.data<__hip_bfloat16>(), out.data_size());
      break;
    case int32:
      launch_kernel(b.data<int32_t>(), c.data<int32_t>(), out.data<int32_t>(), out.data_size());
      break;
    case int64:
      launch_kernel(b.data<int64_t>(), c.data<int64_t>(), out.data<int64_t>(), out.data_size());
      break;
    case uint32:
      launch_kernel(b.data<uint32_t>(), c.data<uint32_t>(), out.data<uint32_t>(), out.data_size());
      break;
    case uint64:
      launch_kernel(b.data<uint64_t>(), c.data<uint64_t>(), out.data<uint64_t>(), out.data_size());
      break;
    case int8:
      launch_kernel(b.data<int8_t>(), c.data<int8_t>(), out.data<int8_t>(), out.data_size());
      break;
    case uint8:
      launch_kernel(b.data<uint8_t>(), c.data<uint8_t>(), out.data<uint8_t>(), out.data_size());
      break;
    case bool_:
      launch_kernel(b.data<bool>(), c.data<bool>(), out.data<bool>(), out.data_size());
      break;
    default:
      throw std::runtime_error(fmt::format(
          "Unsupported type {} for ternary op.",
          dtype_to_string(out.dtype())));
  }
}

template <typename Op>
void ternary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const Stream& s) {
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto& c = inputs[2];
  auto topt = get_ternary_op_type(a, b, c);
  set_ternary_op_output_data(a, b, c, out, topt);
  ternary_op_gpu_inplace<Op>(inputs, out, s);
}

void Select::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = out.primitive().stream();
  ternary_op_gpu<rocm::Select>(inputs, out, s);
}

} // namespace mlx::core
