// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/ternary.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/ternary_ops.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <type_traits>

namespace mlx::core {

namespace rocm {

// Helper function to copy a value byte-by-byte
template <typename T>
__device__ __forceinline__ void copy_value(T* dst, const T* src) {
  // Use unsigned short for 2-byte types, unsigned int for 4-byte, etc.
  if constexpr (sizeof(T) == 1) {
    *reinterpret_cast<unsigned char*>(dst) = *reinterpret_cast<const unsigned char*>(src);
  } else if constexpr (sizeof(T) == 2) {
    *reinterpret_cast<unsigned short*>(dst) = *reinterpret_cast<const unsigned short*>(src);
  } else if constexpr (sizeof(T) == 4) {
    *reinterpret_cast<unsigned int*>(dst) = *reinterpret_cast<const unsigned int*>(src);
  } else if constexpr (sizeof(T) == 8) {
    *reinterpret_cast<unsigned long long*>(dst) = *reinterpret_cast<const unsigned long long*>(src);
  } else {
    // Fallback for other sizes
    for (size_t i = 0; i < sizeof(T); ++i) {
      reinterpret_cast<unsigned char*>(dst)[i] = reinterpret_cast<const unsigned char*>(src)[i];
    }
  }
}

template <typename Op, typename T, typename IdxT, int N_READS>
__global__ void
ternary_v(const bool* a, const T* b, const T* c, T* out, IdxT size) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  IdxT stride = blockDim.x * gridDim.x;

  for (IdxT i = index * N_READS; i < size; i += stride * N_READS) {
    if (i + N_READS <= size) {
      #pragma unroll
      for (int j = 0; j < N_READS; ++j) {
        bool cond = a[i + j];
        const T* src = cond ? &b[i + j] : &c[i + j];
        copy_value(&out[i + j], src);
      }
    } else {
      for (IdxT j = i; j < size; ++j) {
        bool cond = a[j];
        const T* src = cond ? &b[j] : &c[j];
        copy_value(&out[j], src);
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
  IdxT a_offset = 0;
  IdxT b_offset = 0;
  IdxT c_offset = 0;
  IdxT out_offset = index_rest * shape_x;
  
  IdxT idx = index_rest;
  for (int d = ndim - 2; d >= 0; --d) {
    IdxT coord = idx % shape[d];
    idx /= shape[d];
    a_offset += coord * a_strides[d];
    b_offset += coord * b_strides[d];
    c_offset += coord * c_strides[d];
  }

  for (IdxT i = index_x * N_READS; i < shape_x; i += blockDim.x * gridDim.x * N_READS) {
    if (i + N_READS <= shape_x) {
      #pragma unroll
      for (int j = 0; j < N_READS; ++j) {
        bool cond = a[a_offset + (i + j) * a_stride_x];
        const T* src = cond ? &b[b_offset + (i + j) * b_stride_x] : &c[c_offset + (i + j) * c_stride_x];
        copy_value(&out[out_offset + i + j], src);
      }
    } else {
      for (IdxT j = i; j < shape_x; ++j) {
        bool cond = a[a_offset + j * a_stride_x];
        const T* src = cond ? &b[b_offset + j * b_stride_x] : &c[c_offset + j * c_stride_x];
        copy_value(&out[out_offset + j], src);
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
  
  auto& encoder = rocm::get_command_encoder(s);
  
  constexpr int N_READS = 4;
  int block_size = 256;
  
  auto launch_kernel = [&](auto* b_ptr, auto* c_ptr, auto* out_ptr, size_t size) {
    using T = std::remove_pointer_t<decltype(out_ptr)>;
    int num_blocks = (size + block_size * N_READS - 1) / (block_size * N_READS);
    
    encoder.launch_kernel([&](hipStream_t stream) {
      hipLaunchKernelGGL(
          (rocm::ternary_v<Op, T, int64_t, N_READS>),
          dim3(num_blocks), dim3(block_size), 0, stream,
          a.data<bool>(), b_ptr, c_ptr, out_ptr, static_cast<int64_t>(size));
    });
  };
  
  switch (out.dtype()) {
    case float32:
      launch_kernel(b.data<float>(), c.data<float>(), out.data<float>(), out.data_size());
      break;
    case float16:
      launch_kernel(b.data<__half>(), c.data<__half>(), out.data<__half>(), out.data_size());
      break;
    case bfloat16:
      launch_kernel(b.data<hip_bfloat16>(), c.data<hip_bfloat16>(), out.data<hip_bfloat16>(), out.data_size());
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
      throw std::runtime_error(
          std::string("Unsupported type for ternary op: ") + dtype_to_string(out.dtype()));
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
  auto& s = stream();
  ternary_op_gpu<rocm::Select>(inputs, out, s);
}

} // namespace mlx::core
