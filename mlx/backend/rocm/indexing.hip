// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/jit_module.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>

#include <cassert>
#include <numeric>
#include <sstream>

namespace mlx::core {

namespace rocm {

// Simple gather kernel for axis-based gather
template <typename T, typename IdxT>
__global__ void gather_axis_kernel(
    const T* src,
    const IdxT* idx,
    T* out,
    int64_t idx_size_pre,
    int64_t idx_size_axis,
    int64_t idx_size_post,
    int64_t src_axis_size,
    int64_t src_axis_stride,
    int64_t idx_axis_stride,
    int64_t out_axis_stride) {
  int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = idx_size_pre * idx_size_axis * idx_size_post;
  if (gid >= total) return;
  
  // Decompose index
  int64_t post = gid % idx_size_post;
  int64_t axis = (gid / idx_size_post) % idx_size_axis;
  int64_t pre = gid / (idx_size_post * idx_size_axis);
  
  // Get index value
  int64_t idx_offset = pre * idx_size_axis * idx_size_post + axis * idx_size_post + post;
  IdxT idx_val = idx[idx_offset * idx_axis_stride / idx_size_post];
  
  // Handle negative indices
  if (idx_val < 0) {
    idx_val += src_axis_size;
  }
  
  // Compute source and output offsets
  int64_t src_offset = pre * src_axis_stride * src_axis_size + 
                       idx_val * src_axis_stride + post;
  int64_t out_offset = pre * out_axis_stride * idx_size_axis +
                       axis * out_axis_stride + post;
  
  out[out_offset] = src[src_offset];
}

// Simple scatter kernel for axis-based scatter
template <typename T, typename IdxT, bool IS_SUM>
__global__ void scatter_axis_kernel(
    const T* upd,
    const IdxT* idx,
    T* out,
    int64_t idx_size_pre,
    int64_t idx_size_axis,
    int64_t idx_size_post,
    int64_t out_axis_size,
    int64_t upd_axis_stride,
    int64_t idx_axis_stride,
    int64_t out_axis_stride) {
  int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = idx_size_pre * idx_size_axis * idx_size_post;
  if (gid >= total) return;
  
  // Decompose index
  int64_t post = gid % idx_size_post;
  int64_t axis = (gid / idx_size_post) % idx_size_axis;
  int64_t pre = gid / (idx_size_post * idx_size_axis);
  
  // Get index value
  int64_t idx_offset = pre * idx_size_axis * idx_size_post + axis * idx_size_post + post;
  IdxT idx_val = idx[idx_offset * idx_axis_stride / idx_size_post];
  
  // Handle negative indices
  if (idx_val < 0) {
    idx_val += out_axis_size;
  }
  
  // Compute update and output offsets
  int64_t upd_offset = pre * upd_axis_stride * idx_size_axis +
                       axis * upd_axis_stride + post;
  int64_t out_offset = pre * out_axis_stride * out_axis_size +
                       idx_val * out_axis_stride + post;
  
  if constexpr (IS_SUM) {
    atomicAdd(&out[out_offset], upd[upd_offset]);
  } else {
    out[out_offset] = upd[upd_offset];
  }
}

} // namespace rocm

void Gather::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() > 0);
  const auto& src = inputs[0];

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  // For now, only support simple cases
  // Full implementation requires JIT compilation
  throw std::runtime_error("Gather::eval_gpu requires JIT compilation support for ROCm - use GatherAxis instead");
}

void Scatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() > 1);
  auto& upd = inputs.back();

  // Copy src into out
  CopyType copy_type;
  if (inputs[0].data_size() == 1) {
    copy_type = CopyType::Scalar;
  } else if (inputs[0].flags().row_contiguous) {
    copy_type = CopyType::Vector;
  } else {
    copy_type = CopyType::General;
  }
  copy_gpu(inputs[0], out, copy_type);

  if (upd.size() == 0) {
    return;
  }

  // Full implementation requires JIT compilation
  throw std::runtime_error("Scatter::eval_gpu requires JIT compilation support for ROCm - use ScatterAxis instead");
}

void GatherAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() > 1);
  const auto& src = inputs[0];
  const auto& idx = inputs[1];

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);
  
  encoder.set_input_array(src);
  encoder.set_input_array(idx);
  encoder.set_output_array(out);

  size_t idx_size_pre = 1;
  size_t idx_size_post = 1;
  for (int i = 0; i < axis_; ++i) {
    idx_size_pre *= idx.shape(i);
  }
  for (int i = axis_ + 1; i < idx.ndim(); ++i) {
    idx_size_post *= idx.shape(i);
  }
  size_t idx_size_axis = idx.shape(axis_);
  
  int64_t total = idx_size_pre * idx_size_axis * idx_size_post;
  int block_size = 256;
  int num_blocks = (total + block_size - 1) / block_size;
  
  encoder.launch_kernel([&](hipStream_t stream) {
    switch (src.dtype()) {
      case float32:
        hipLaunchKernelGGL(
            (rocm::gather_axis_kernel<float, int32_t>),
            dim3(num_blocks), dim3(block_size), 0, stream,
            src.data<float>(), idx.data<int32_t>(), out.data<float>(),
            idx_size_pre, idx_size_axis, idx_size_post,
            src.shape(axis_), src.strides(axis_), idx.strides(axis_),
            out.strides(axis_));
        break;
      case int32:
        hipLaunchKernelGGL(
            (rocm::gather_axis_kernel<int32_t, int32_t>),
            dim3(num_blocks), dim3(block_size), 0, stream,
            src.data<int32_t>(), idx.data<int32_t>(), out.data<int32_t>(),
            idx_size_pre, idx_size_axis, idx_size_post,
            src.shape(axis_), src.strides(axis_), idx.strides(axis_),
            out.strides(axis_));
        break;
      case float16:
        hipLaunchKernelGGL(
            (rocm::gather_axis_kernel<__half, int32_t>),
            dim3(num_blocks), dim3(block_size), 0, stream,
            src.data<__half>(), idx.data<int32_t>(), out.data<__half>(),
            idx_size_pre, idx_size_axis, idx_size_post,
            src.shape(axis_), src.strides(axis_), idx.strides(axis_),
            out.strides(axis_));
        break;
      default:
        throw std::runtime_error("Unsupported dtype for GatherAxis");
    }
  });
}

void ScatterAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() > 2);
  const auto& src = inputs[0];
  const auto& idx = inputs[1];
  const auto& upd = inputs[2];

  // Copy src into out
  CopyType copy_type;
  if (src.data_size() == 1) {
    copy_type = CopyType::Scalar;
  } else if (src.flags().row_contiguous) {
    copy_type = CopyType::Vector;
  } else {
    copy_type = CopyType::General;
  }
  copy_gpu(src, out, copy_type);

  if (upd.size() == 0) {
    return;
  }

  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);
  
  encoder.set_input_array(upd);
  encoder.set_input_array(idx);
  encoder.set_output_array(out);

  size_t idx_size_pre = 1;
  size_t idx_size_post = 1;
  for (int i = 0; i < axis_; ++i) {
    idx_size_pre *= idx.shape(i);
  }
  for (int i = axis_ + 1; i < idx.ndim(); ++i) {
    idx_size_post *= idx.shape(i);
  }
  size_t idx_size_axis = idx.shape(axis_);
  
  int64_t total = idx_size_pre * idx_size_axis * idx_size_post;
  int block_size = 256;
  int num_blocks = (total + block_size - 1) / block_size;
  
  bool is_sum = (reduce_type_ == ScatterAxis::Sum);
  
  encoder.launch_kernel([&](hipStream_t stream) {
    if (is_sum) {
      switch (upd.dtype()) {
        case float32:
          hipLaunchKernelGGL(
              (rocm::scatter_axis_kernel<float, int32_t, true>),
              dim3(num_blocks), dim3(block_size), 0, stream,
              upd.data<float>(), idx.data<int32_t>(), out.data<float>(),
              idx_size_pre, idx_size_axis, idx_size_post,
              out.shape(axis_), upd.strides(axis_), idx.strides(axis_),
              out.strides(axis_));
          break;
        default:
          throw std::runtime_error("Unsupported dtype for ScatterAxis Sum");
      }
    } else {
      switch (upd.dtype()) {
        case float32:
          hipLaunchKernelGGL(
              (rocm::scatter_axis_kernel<float, int32_t, false>),
              dim3(num_blocks), dim3(block_size), 0, stream,
              upd.data<float>(), idx.data<int32_t>(), out.data<float>(),
              idx_size_pre, idx_size_axis, idx_size_post,
              out.shape(axis_), upd.strides(axis_), idx.strides(axis_),
              out.strides(axis_));
          break;
        case int32:
          hipLaunchKernelGGL(
              (rocm::scatter_axis_kernel<int32_t, int32_t, false>),
              dim3(num_blocks), dim3(block_size), 0, stream,
              upd.data<int32_t>(), idx.data<int32_t>(), out.data<int32_t>(),
              idx_size_pre, idx_size_axis, idx_size_post,
              out.shape(axis_), upd.strides(axis_), idx.strides(axis_),
              out.strides(axis_));
          break;
        case float16:
          hipLaunchKernelGGL(
              (rocm::scatter_axis_kernel<__half, int32_t, false>),
              dim3(num_blocks), dim3(block_size), 0, stream,
              upd.data<__half>(), idx.data<int32_t>(), out.data<__half>(),
              idx_size_pre, idx_size_axis, idx_size_post,
              out.shape(axis_), upd.strides(axis_), idx.strides(axis_),
              out.strides(axis_));
          break;
        default:
          throw std::runtime_error("Unsupported dtype for ScatterAxis Assign");
      }
    }
  });
}

} // namespace mlx::core
