// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/ternary.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/rocm/allocator.h"
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
    hip_array<int32_t, MAX_NDIM> shape,
    hip_array<int64_t, MAX_NDIM> a_strides,
    hip_array<int64_t, MAX_NDIM> b_strides,
    hip_array<int64_t, MAX_NDIM> c_strides,
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
  
  // Compute base offsets using elem_to_loc style calculation
  IdxT elem = index_rest * shape_x;
  IdxT a_offset = 0;
  IdxT b_offset = 0;
  IdxT c_offset = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    IdxT coord = elem % shape[i];
    elem /= shape[i];
    a_offset += coord * a_strides[i];
    b_offset += coord * b_strides[i];
    c_offset += coord * c_strides[i];
  }
  
  IdxT out_offset = index_rest * shape_x;

  for (IdxT i = index_x * N_READS; i < shape_x; i += blockDim.x * gridDim.x * N_READS) {
    if (i + N_READS <= shape_x) {
      #pragma unroll
      for (int j = 0; j < N_READS; ++j) {
        bool cond = a[a_offset + (i + j) * a_stride_x];
        T b_val = b[b_offset + (i + j) * b_stride_x];
        T c_val = c[c_offset + (i + j) * c_stride_x];
        out[out_offset + i + j] = Op{}(cond, b_val, c_val);
      }
    } else {
      for (IdxT j = i; j < shape_x; ++j) {
        bool cond = a[a_offset + j * a_stride_x];
        T b_val = b[b_offset + j * b_stride_x];
        T c_val = c[c_offset + j * c_stride_x];
        out[out_offset + j] = Op{}(cond, b_val, c_val);
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
  
  constexpr int N_READS = 4;
  int block_size = 256;
  
  auto topt = get_ternary_op_type(a, b, c);
  
  dispatch_all_types(out.dtype(), [&](auto type_tag) {
    using CTYPE = MLX_GET_TYPE(type_tag);
    using DType = hip_type_t<CTYPE>;
    
    if (topt == TernaryOpType::VectorVectorVector ||
        topt == TernaryOpType::ScalarScalarScalar) {
      // Contiguous case - use ternary_v
      auto size = out.data_size();
      int num_blocks = (size + block_size * N_READS - 1) / (block_size * N_READS);
      num_blocks = std::min(num_blocks, 65535);
      
      int64_t size_arg = static_cast<int64_t>(size);
      encoder.add_kernel_node(
          &rocm::ternary_v<Op, DType, int64_t, N_READS>,
          dim3(num_blocks), dim3(block_size), 0,
          gpu_ptr<const bool>(a), gpu_ptr<const DType>(b), gpu_ptr<const DType>(c),
          gpu_ptr<DType>(out), size_arg);
    } else {
      // General case - use ternary_g with strided access
      Shape shape_vec;
      std::vector<Strides> strides_vec;
      std::tie(shape_vec, strides_vec) = collapse_contiguous_dims(a, b, c, out);
      auto& a_strides_vec = strides_vec[0];
      auto& b_strides_vec = strides_vec[1];
      auto& c_strides_vec = strides_vec[2];
      int ndim = shape_vec.size();

      rocm::hip_array<int32_t, MAX_NDIM> shape_arg = {};
      rocm::hip_array<int64_t, MAX_NDIM> a_strides_arg = {};
      rocm::hip_array<int64_t, MAX_NDIM> b_strides_arg = {};
      rocm::hip_array<int64_t, MAX_NDIM> c_strides_arg = {};
      for (int i = 0; i < ndim; i++) {
        shape_arg.data_[i] = static_cast<int32_t>(shape_vec[i]);
        a_strides_arg.data_[i] = a_strides_vec[i];
        b_strides_arg.data_[i] = b_strides_vec[i];
        c_strides_arg.data_[i] = c_strides_vec[i];
      }

      int dim0 = ndim > 0 ? shape_vec.back() : 1;
      size_t rest = out.size() / dim0;
      
      int work_per_thread = (dim0 >= 4) ? 4 : 1;
      dim0 = (dim0 + work_per_thread - 1) / work_per_thread;
      
      int block_x = std::min(dim0, 32);
      int block_y = std::min(static_cast<int>(rest), 256 / block_x);
      int num_blocks_x = (dim0 + block_x - 1) / block_x;
      int num_blocks_y = (rest + block_y - 1) / block_y;
      
      int64_t rest_arg = static_cast<int64_t>(rest);
      if (work_per_thread == 4) {
        encoder.add_kernel_node(
            &rocm::ternary_g<Op, DType, int64_t, 4>,
            dim3(num_blocks_x, num_blocks_y), dim3(block_x, block_y), 0,
            gpu_ptr<const bool>(a), gpu_ptr<const DType>(b), gpu_ptr<const DType>(c),
            gpu_ptr<DType>(out),
            rest_arg,
            shape_arg,
            a_strides_arg,
            b_strides_arg,
            c_strides_arg,
            ndim);
      } else {
        encoder.add_kernel_node(
            &rocm::ternary_g<Op, DType, int64_t, 1>,
            dim3(num_blocks_x, num_blocks_y), dim3(block_x, block_y), 0,
            gpu_ptr<const bool>(a), gpu_ptr<const DType>(b), gpu_ptr<const DType>(c),
            gpu_ptr<DType>(out),
            rest_arg,
            shape_arg,
            a_strides_arg,
            b_strides_arg,
            c_strides_arg,
            ndim);
      }
    }
  });
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
