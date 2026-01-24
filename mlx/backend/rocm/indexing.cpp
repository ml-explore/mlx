// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>
#include <fmt/format.h>

#include <cassert>
#include <numeric>

namespace mlx::core {

namespace rocm {

// Gather kernel - gathers elements from src using indices
template <typename T, typename IdxT, int NIDX>
__global__ void gather_kernel(
    const T* src,
    T* out,
    const void** indices,
    IdxT out_size,
    const int* src_shape,
    const int64_t* src_strides,
    int src_ndim,
    const int* slice_sizes,
    int slice_size,
    const int* axes,
    const int* idx_shapes,
    const int64_t* idx_strides,
    int idx_ndim) {
  IdxT gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= out_size) return;
  
  // Compute output coordinates
  IdxT out_idx = gid / slice_size;
  IdxT slice_idx = gid % slice_size;
  
  // Compute source index
  int64_t src_offset = 0;
  
  // Add contributions from indices
  for (int i = 0; i < NIDX; ++i) {
    // Get the index value
    IdxT idx_offset = 0;
    IdxT tmp = out_idx;
    for (int d = idx_ndim - 1; d >= 0; --d) {
      IdxT coord = tmp % idx_shapes[i * idx_ndim + d];
      idx_offset += coord * idx_strides[i * idx_ndim + d];
      tmp /= idx_shapes[i * idx_ndim + d];
    }
    
    const int32_t* idx_ptr = static_cast<const int32_t*>(indices[i]);
    int32_t idx_val = idx_ptr[idx_offset];
    src_offset += idx_val * src_strides[axes[i]];
  }
  
  // Add contribution from slice position
  IdxT tmp = slice_idx;
  for (int d = src_ndim - 1; d >= 0; --d) {
    IdxT coord = tmp % slice_sizes[d];
    src_offset += coord * src_strides[d];
    tmp /= slice_sizes[d];
  }
  
  out[gid] = src[src_offset];
}

// Scatter kernel - scatters update values into out using indices
template <typename T, typename IdxT, int NIDX, typename Op>
__global__ void scatter_kernel(
    const T* upd,
    T* out,
    const void** indices,
    IdxT upd_size,
    const int* upd_shape,
    const int64_t* upd_strides,
    int upd_ndim,
    IdxT upd_post_idx_size,
    const int* out_shape,
    const int64_t* out_strides,
    int out_ndim,
    const int* axes,
    const int* idx_shapes,
    const int64_t* idx_strides,
    int idx_ndim,
    Op op) {
  IdxT gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= upd_size) return;
  
  // Compute update coordinates
  IdxT idx_part = gid / upd_post_idx_size;
  IdxT post_part = gid % upd_post_idx_size;
  
  // Compute output index
  int64_t out_offset = 0;
  
  // Add contributions from indices
  for (int i = 0; i < NIDX; ++i) {
    IdxT idx_offset = 0;
    IdxT tmp = idx_part;
    for (int d = idx_ndim - 1; d >= 0; --d) {
      IdxT coord = tmp % idx_shapes[i * idx_ndim + d];
      idx_offset += coord * idx_strides[i * idx_ndim + d];
      tmp /= idx_shapes[i * idx_ndim + d];
    }
    
    const int32_t* idx_ptr = static_cast<const int32_t*>(indices[i]);
    int32_t idx_val = idx_ptr[idx_offset];
    out_offset += idx_val * out_strides[axes[i]];
  }
  
  // Add contribution from post-index position
  IdxT tmp = post_part;
  for (int d = out_ndim - 1; d >= idx_ndim; --d) {
    IdxT coord = tmp % out_shape[d];
    out_offset += coord * out_strides[d];
    tmp /= out_shape[d];
  }
  
  // Compute update offset
  int64_t upd_offset = 0;
  tmp = gid;
  for (int d = upd_ndim - 1; d >= 0; --d) {
    IdxT coord = tmp % upd_shape[d];
    upd_offset += coord * upd_strides[d];
    tmp /= upd_shape[d];
  }
  
  // Apply operation
  op(out + out_offset, upd[upd_offset]);
}

// Scatter operations
struct ScatterAssign {
  template <typename T>
  __device__ void operator()(T* dst, T val) const {
    *dst = val;
  }
};

struct ScatterSum {
  template <typename T>
  __device__ void operator()(T* dst, T val) const {
    atomicAdd(dst, val);
  }
};

struct ScatterMax {
  template <typename T>
  __device__ void operator()(T* dst, T val) const {
    // Atomic max for floats needs special handling
    T old = *dst;
    while (val > old) {
      T assumed = old;
      old = atomicCAS(reinterpret_cast<unsigned int*>(dst),
                      __float_as_uint(assumed),
                      __float_as_uint(val));
      if (old == assumed) break;
    }
  }
};

struct ScatterMin {
  template <typename T>
  __device__ void operator()(T* dst, T val) const {
    T old = *dst;
    while (val < old) {
      T assumed = old;
      old = atomicCAS(reinterpret_cast<unsigned int*>(dst),
                      __float_as_uint(assumed),
                      __float_as_uint(val));
      if (old == assumed) break;
    }
  }
};

struct ScatterProd {
  template <typename T>
  __device__ void operator()(T* dst, T val) const {
    // Atomic multiply needs CAS loop
    T old = *dst;
    T assumed;
    do {
      assumed = old;
      old = atomicCAS(reinterpret_cast<unsigned int*>(dst),
                      __float_as_uint(assumed),
                      __float_as_uint(assumed * val));
    } while (old != assumed);
  }
};

} // namespace rocm

void Gather::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() > 0);
  const auto& src = inputs[0];

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  int nidx = inputs.size() - 1;
  
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);
  
  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  encoder.set_output_array(out);
  
  // For now, use a simple fallback implementation
  // A full implementation would need JIT compilation for arbitrary nidx
  if (nidx > 4) {
    throw std::runtime_error("Gather with more than 4 index arrays not yet supported on ROCm");
  }
  
  uint32_t slice_size = std::accumulate(
      slice_sizes_.begin(), slice_sizes_.end(), 1, std::multiplies<uint32_t>());
  
  // Simple implementation: copy to CPU, do gather, copy back
  // This is a placeholder - a proper implementation would use the kernel above
  throw std::runtime_error("Gather::eval_gpu requires JIT compilation support for ROCm");
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

  // Empty update
  if (upd.size() == 0) {
    return;
  }

  int nidx = axes_.size();
  
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);
  
  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  encoder.set_output_array(out);
  
  // For now, throw error - proper implementation needs JIT
  throw std::runtime_error("Scatter::eval_gpu requires JIT compilation support for ROCm");
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
  
  // For now, throw error - proper implementation needs specialized kernel
  throw std::runtime_error("GatherAxis::eval_gpu not yet fully implemented for ROCm");
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

  // Empty update
  if (upd.size() == 0) {
    return;
  }

  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);
  
  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  encoder.set_output_array(out);
  
  // For now, throw error - proper implementation needs specialized kernel
  throw std::runtime_error("ScatterAxis::eval_gpu not yet fully implemented for ROCm");
}

} // namespace mlx::core
