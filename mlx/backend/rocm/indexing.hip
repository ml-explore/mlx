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

// General gather kernel - handles arbitrary indexing
template <typename T, typename IdxT, int NIDX>
__global__ void gather_general_kernel(
    const T* src,
    T* out,
    int64_t size,
    const int32_t* src_shape,
    const int64_t* src_strides,
    int32_t src_ndim,
    const int32_t* slice_sizes,
    uint32_t slice_size,
    const int32_t* axes,
    const IdxT* const* indices,
    const int32_t* indices_shape,
    const int64_t* indices_strides,
    int32_t idx_ndim) {
  int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (out_idx >= size) {
    return;
  }

  int64_t src_elem = out_idx % slice_size;
  int64_t idx_elem = out_idx / slice_size;

  // Compute source location from slice element
  int64_t src_loc = 0;
  int64_t tmp = src_elem;
  for (int i = src_ndim - 1; i >= 0; --i) {
    src_loc += (tmp % slice_sizes[i]) * src_strides[i];
    tmp /= slice_sizes[i];
  }

  // Add index contributions
  for (int i = 0; i < NIDX; ++i) {
    // Compute index location
    int64_t idx_loc = 0;
    int64_t tmp_idx = idx_elem;
    for (int j = idx_ndim - 1; j >= 0; --j) {
      idx_loc += (tmp_idx % indices_shape[i * idx_ndim + j]) * indices_strides[i * idx_ndim + j];
      tmp_idx /= indices_shape[i * idx_ndim + j];
    }
    
    int32_t axis = axes[i];
    IdxT idx_val = indices[i][idx_loc];
    
    // Handle negative indices
    if (idx_val < 0) {
      idx_val += src_shape[axis];
    }
    
    src_loc += idx_val * src_strides[axis];
  }

  out[out_idx] = src[src_loc];
}

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

// General scatter kernel - handles arbitrary indexing
template <typename T, typename IdxT, int NIDX, int ReduceType>
__global__ void scatter_general_kernel(
    const T* upd,
    T* out,
    int64_t upd_size,
    const int32_t* upd_shape,
    const int64_t* upd_strides,
    int32_t upd_ndim,
    int64_t upd_post_idx_size,
    const int32_t* out_shape,
    const int64_t* out_strides,
    int32_t out_ndim,
    const int32_t* axes,
    const IdxT* const* indices,
    const int32_t* indices_shape,
    const int64_t* indices_strides,
    int32_t idx_ndim) {
  int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= upd_size) {
    return;
  }

  // Compute update location
  int64_t upd_loc = 0;
  int64_t tmp = gid;
  for (int i = upd_ndim - 1; i >= 0; --i) {
    upd_loc += (tmp % upd_shape[i]) * upd_strides[i];
    tmp /= upd_shape[i];
  }

  int64_t idx_elem = gid / upd_post_idx_size;
  int64_t out_elem = gid % upd_post_idx_size;

  // Compute output location from out_elem
  int64_t out_loc = 0;
  tmp = out_elem;
  for (int i = out_ndim - 1; i >= 0; --i) {
    out_loc += (tmp % out_shape[i]) * out_strides[i];
    tmp /= out_shape[i];
  }

  // Add index contributions
  for (int i = 0; i < NIDX; ++i) {
    // Compute index location
    int64_t idx_loc = 0;
    int64_t tmp_idx = idx_elem;
    for (int j = idx_ndim - 1; j >= 0; --j) {
      idx_loc += (tmp_idx % indices_shape[i * idx_ndim + j]) * indices_strides[i * idx_ndim + j];
      tmp_idx /= indices_shape[i * idx_ndim + j];
    }
    
    int32_t axis = axes[i];
    IdxT idx_val = indices[i][idx_loc];
    
    // Handle negative indices
    if (idx_val < 0) {
      idx_val += out_shape[axis];
    }
    
    out_loc += idx_val * out_strides[axis];
  }

  T val = upd[upd_loc];
  
  // Apply reduce operation
  if constexpr (ReduceType == 0) { // Assign
    out[out_loc] = val;
  } else if constexpr (ReduceType == 1) { // Sum
    // Use appropriate atomic based on type
    if constexpr (std::is_same_v<T, float>) {
      atomicAdd(&out[out_loc], val);
    } else if constexpr (std::is_same_v<T, int32_t>) {
      atomicAdd(&out[out_loc], val);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      atomicAdd(&out[out_loc], val);
    } else if constexpr (std::is_same_v<T, int64_t>) {
      atomicAdd(reinterpret_cast<unsigned long long*>(&out[out_loc]), 
                static_cast<unsigned long long>(val));
    } else if constexpr (std::is_same_v<T, uint64_t>) {
      atomicAdd(&out[out_loc], val);
    } else {
      // Fallback for types without atomic support
      out[out_loc] += val;
    }
  } else if constexpr (ReduceType == 2) { // Prod
    out[out_loc] *= val;
  } else if constexpr (ReduceType == 3) { // Max
    // Use atomicMax where available
    if constexpr (std::is_same_v<T, int32_t>) {
      atomicMax(&out[out_loc], val);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      atomicMax(&out[out_loc], val);
    } else {
      // Fallback
      if (val > out[out_loc]) out[out_loc] = val;
    }
  } else if constexpr (ReduceType == 4) { // Min
    if constexpr (std::is_same_v<T, int32_t>) {
      atomicMin(&out[out_loc], val);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      atomicMin(&out[out_loc], val);
    } else {
      if (val < out[out_loc]) out[out_loc] = val;
    }
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

  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  int nidx = inputs.size() - 1;
  int32_t idx_ndim = nidx > 0 ? inputs[1].ndim() : 0;

  uint32_t slice_size = std::accumulate(
      slice_sizes_.begin(), slice_sizes_.end(), 1, std::multiplies<uint32_t>());

  // Prepare device memory for parameters
  std::vector<int32_t> h_src_shape(src.shape().begin(), src.shape().end());
  std::vector<int64_t> h_src_strides(src.strides().begin(), src.strides().end());
  std::vector<int32_t> h_slice_sizes(slice_sizes_.begin(), slice_sizes_.end());
  std::vector<int32_t> h_axes(axes_.begin(), axes_.end());

  // Prepare indices pointers and metadata
  std::vector<const void*> h_indices(nidx);
  std::vector<int32_t> h_indices_shape(nidx * std::max(idx_ndim, 1));
  std::vector<int64_t> h_indices_strides(nidx * std::max(idx_ndim, 1));

  for (int i = 0; i < nidx; ++i) {
    h_indices[i] = inputs[i + 1].data<void>();
    for (int j = 0; j < idx_ndim; ++j) {
      h_indices_shape[i * idx_ndim + j] = inputs[i + 1].shape(j);
      h_indices_strides[i * idx_ndim + j] = inputs[i + 1].strides(j);
    }
  }

  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  encoder.set_output_array(out);

  int64_t total = out.size();
  int block_size = 256;
  int num_blocks = (total + block_size - 1) / block_size;

  // Allocate device memory for parameters
  int32_t* d_src_shape;
  int64_t* d_src_strides;
  int32_t* d_slice_sizes;
  int32_t* d_axes;
  const void** d_indices;
  int32_t* d_indices_shape;
  int64_t* d_indices_strides;

  hipMalloc(&d_src_shape, h_src_shape.size() * sizeof(int32_t));
  hipMalloc(&d_src_strides, h_src_strides.size() * sizeof(int64_t));
  hipMalloc(&d_slice_sizes, h_slice_sizes.size() * sizeof(int32_t));
  hipMalloc(&d_axes, h_axes.size() * sizeof(int32_t));
  hipMalloc(&d_indices, h_indices.size() * sizeof(void*));
  hipMalloc(&d_indices_shape, h_indices_shape.size() * sizeof(int32_t));
  hipMalloc(&d_indices_strides, h_indices_strides.size() * sizeof(int64_t));

  hipMemcpy(d_src_shape, h_src_shape.data(), h_src_shape.size() * sizeof(int32_t), hipMemcpyHostToDevice);
  hipMemcpy(d_src_strides, h_src_strides.data(), h_src_strides.size() * sizeof(int64_t), hipMemcpyHostToDevice);
  hipMemcpy(d_slice_sizes, h_slice_sizes.data(), h_slice_sizes.size() * sizeof(int32_t), hipMemcpyHostToDevice);
  hipMemcpy(d_axes, h_axes.data(), h_axes.size() * sizeof(int32_t), hipMemcpyHostToDevice);
  hipMemcpy(d_indices, h_indices.data(), h_indices.size() * sizeof(void*), hipMemcpyHostToDevice);
  hipMemcpy(d_indices_shape, h_indices_shape.data(), h_indices_shape.size() * sizeof(int32_t), hipMemcpyHostToDevice);
  hipMemcpy(d_indices_strides, h_indices_strides.data(), h_indices_strides.size() * sizeof(int64_t), hipMemcpyHostToDevice);

  encoder.launch_kernel([&](hipStream_t stream) {
    // Dispatch based on dtype and number of indices
    #define LAUNCH_GATHER(T, IdxT, NIDX) \
      hipLaunchKernelGGL( \
          (rocm::gather_general_kernel<T, IdxT, NIDX>), \
          dim3(num_blocks), dim3(block_size), 0, stream, \
          src.data<T>(), out.data<T>(), total, \
          d_src_shape, d_src_strides, src.ndim(), \
          d_slice_sizes, slice_size, d_axes, \
          (const IdxT* const*)d_indices, d_indices_shape, d_indices_strides, idx_ndim)

    #define DISPATCH_NIDX(T, IdxT) \
      switch (nidx) { \
        case 0: LAUNCH_GATHER(T, IdxT, 1); break; \
        case 1: LAUNCH_GATHER(T, IdxT, 1); break; \
        case 2: LAUNCH_GATHER(T, IdxT, 2); break; \
        case 3: LAUNCH_GATHER(T, IdxT, 3); break; \
        case 4: LAUNCH_GATHER(T, IdxT, 4); break; \
        default: LAUNCH_GATHER(T, IdxT, 8); break; \
      }

    Dtype idx_dtype = nidx > 0 ? inputs[1].dtype() : int32;
    
    if (idx_dtype == int32 || idx_dtype == uint32) {
      switch (out.dtype()) {
        case float32: DISPATCH_NIDX(float, int32_t); break;
        case float16: DISPATCH_NIDX(__half, int32_t); break;
        case bfloat16: DISPATCH_NIDX(hip_bfloat16, int32_t); break;
        case int32: DISPATCH_NIDX(int32_t, int32_t); break;
        case int64: DISPATCH_NIDX(int64_t, int32_t); break;
        case uint32: DISPATCH_NIDX(uint32_t, int32_t); break;
        case uint64: DISPATCH_NIDX(uint64_t, int32_t); break;
        case int8: DISPATCH_NIDX(int8_t, int32_t); break;
        case uint8: DISPATCH_NIDX(uint8_t, int32_t); break;
        case bool_: DISPATCH_NIDX(bool, int32_t); break;
        default:
          throw std::runtime_error("Unsupported dtype for Gather");
      }
    } else {
      switch (out.dtype()) {
        case float32: DISPATCH_NIDX(float, int64_t); break;
        case float16: DISPATCH_NIDX(__half, int64_t); break;
        case bfloat16: DISPATCH_NIDX(hip_bfloat16, int64_t); break;
        case int32: DISPATCH_NIDX(int32_t, int64_t); break;
        case int64: DISPATCH_NIDX(int64_t, int64_t); break;
        default:
          throw std::runtime_error("Unsupported dtype for Gather");
      }
    }

    #undef DISPATCH_NIDX
    #undef LAUNCH_GATHER
  });

  // Schedule cleanup of device memory
  encoder.add_completed_handler([=]() {
    hipFree(d_src_shape);
    hipFree(d_src_strides);
    hipFree(d_slice_sizes);
    hipFree(d_axes);
    hipFree(d_indices);
    hipFree(d_indices_shape);
    hipFree(d_indices_strides);
  });
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

  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  int nidx = axes_.size();
  int32_t idx_ndim = nidx > 0 ? inputs[1].ndim() : 0;

  int32_t upd_post_idx_size = std::accumulate(
      upd.shape().begin() + idx_ndim,
      upd.shape().end(),
      1,
      std::multiplies<int32_t>());

  // Prepare device memory for parameters
  std::vector<int32_t> h_upd_shape(upd.shape().begin(), upd.shape().end());
  std::vector<int64_t> h_upd_strides(upd.strides().begin(), upd.strides().end());
  std::vector<int32_t> h_out_shape(out.shape().begin(), out.shape().end());
  std::vector<int64_t> h_out_strides(out.strides().begin(), out.strides().end());
  std::vector<int32_t> h_axes(axes_.begin(), axes_.end());

  // Prepare indices pointers and metadata
  std::vector<const void*> h_indices(nidx);
  std::vector<int32_t> h_indices_shape(nidx * std::max(idx_ndim, 1));
  std::vector<int64_t> h_indices_strides(nidx * std::max(idx_ndim, 1));

  for (int i = 0; i < nidx; ++i) {
    h_indices[i] = inputs[i + 1].data<void>();
    for (int j = 0; j < idx_ndim; ++j) {
      h_indices_shape[i * idx_ndim + j] = inputs[i + 1].shape(j);
      h_indices_strides[i * idx_ndim + j] = inputs[i + 1].strides(j);
    }
  }

  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  encoder.set_output_array(out);

  int64_t total = upd.size();
  int block_size = 256;
  int num_blocks = (total + block_size - 1) / block_size;

  // Allocate device memory
  int32_t* d_upd_shape;
  int64_t* d_upd_strides;
  int32_t* d_out_shape;
  int64_t* d_out_strides;
  int32_t* d_axes;
  const void** d_indices;
  int32_t* d_indices_shape;
  int64_t* d_indices_strides;

  hipMalloc(&d_upd_shape, h_upd_shape.size() * sizeof(int32_t));
  hipMalloc(&d_upd_strides, h_upd_strides.size() * sizeof(int64_t));
  hipMalloc(&d_out_shape, h_out_shape.size() * sizeof(int32_t));
  hipMalloc(&d_out_strides, h_out_strides.size() * sizeof(int64_t));
  hipMalloc(&d_axes, std::max((size_t)1, h_axes.size()) * sizeof(int32_t));
  hipMalloc(&d_indices, std::max((size_t)1, h_indices.size()) * sizeof(void*));
  hipMalloc(&d_indices_shape, std::max((size_t)1, h_indices_shape.size()) * sizeof(int32_t));
  hipMalloc(&d_indices_strides, std::max((size_t)1, h_indices_strides.size()) * sizeof(int64_t));

  hipMemcpy(d_upd_shape, h_upd_shape.data(), h_upd_shape.size() * sizeof(int32_t), hipMemcpyHostToDevice);
  hipMemcpy(d_upd_strides, h_upd_strides.data(), h_upd_strides.size() * sizeof(int64_t), hipMemcpyHostToDevice);
  hipMemcpy(d_out_shape, h_out_shape.data(), h_out_shape.size() * sizeof(int32_t), hipMemcpyHostToDevice);
  hipMemcpy(d_out_strides, h_out_strides.data(), h_out_strides.size() * sizeof(int64_t), hipMemcpyHostToDevice);
  if (!h_axes.empty()) {
    hipMemcpy(d_axes, h_axes.data(), h_axes.size() * sizeof(int32_t), hipMemcpyHostToDevice);
  }
  if (!h_indices.empty()) {
    hipMemcpy(d_indices, h_indices.data(), h_indices.size() * sizeof(void*), hipMemcpyHostToDevice);
    hipMemcpy(d_indices_shape, h_indices_shape.data(), h_indices_shape.size() * sizeof(int32_t), hipMemcpyHostToDevice);
    hipMemcpy(d_indices_strides, h_indices_strides.data(), h_indices_strides.size() * sizeof(int64_t), hipMemcpyHostToDevice);
  }

  int reduce_type = reduce_type_; // 0=Assign, 1=Sum, 2=Prod, 3=Max, 4=Min

  encoder.launch_kernel([&](hipStream_t stream) {
    #define LAUNCH_SCATTER(T, IdxT, NIDX, RT) \
      hipLaunchKernelGGL( \
          (rocm::scatter_general_kernel<T, IdxT, NIDX, RT>), \
          dim3(num_blocks), dim3(block_size), 0, stream, \
          upd.data<T>(), out.data<T>(), total, \
          d_upd_shape, d_upd_strides, upd.ndim(), upd_post_idx_size, \
          d_out_shape, d_out_strides, out.ndim(), \
          d_axes, (const IdxT* const*)d_indices, d_indices_shape, d_indices_strides, idx_ndim)

    #define DISPATCH_REDUCE(T, IdxT, NIDX) \
      switch (reduce_type) { \
        case 0: LAUNCH_SCATTER(T, IdxT, NIDX, 0); break; \
        case 1: LAUNCH_SCATTER(T, IdxT, NIDX, 1); break; \
        case 2: LAUNCH_SCATTER(T, IdxT, NIDX, 2); break; \
        case 3: LAUNCH_SCATTER(T, IdxT, NIDX, 3); break; \
        case 4: LAUNCH_SCATTER(T, IdxT, NIDX, 4); break; \
        default: LAUNCH_SCATTER(T, IdxT, NIDX, 0); break; \
      }

    #define DISPATCH_NIDX(T, IdxT) \
      switch (nidx) { \
        case 0: DISPATCH_REDUCE(T, IdxT, 1); break; \
        case 1: DISPATCH_REDUCE(T, IdxT, 1); break; \
        case 2: DISPATCH_REDUCE(T, IdxT, 2); break; \
        case 3: DISPATCH_REDUCE(T, IdxT, 3); break; \
        default: DISPATCH_REDUCE(T, IdxT, 4); break; \
      }

    Dtype idx_dtype = nidx > 0 ? inputs[1].dtype() : int32;
    
    if (idx_dtype == int32 || idx_dtype == uint32) {
      switch (out.dtype()) {
        case float32: DISPATCH_NIDX(float, int32_t); break;
        case float16: DISPATCH_NIDX(__half, int32_t); break;
        case int32: DISPATCH_NIDX(int32_t, int32_t); break;
        case int64: DISPATCH_NIDX(int64_t, int32_t); break;
        default:
          throw std::runtime_error("Unsupported dtype for Scatter");
      }
    } else {
      switch (out.dtype()) {
        case float32: DISPATCH_NIDX(float, int64_t); break;
        case int32: DISPATCH_NIDX(int32_t, int64_t); break;
        case int64: DISPATCH_NIDX(int64_t, int64_t); break;
        default:
          throw std::runtime_error("Unsupported dtype for Scatter");
      }
    }

    #undef DISPATCH_NIDX
    #undef DISPATCH_REDUCE
    #undef LAUNCH_SCATTER
  });

  // Schedule cleanup
  encoder.add_completed_handler([=]() {
    hipFree(d_upd_shape);
    hipFree(d_upd_strides);
    hipFree(d_out_shape);
    hipFree(d_out_strides);
    hipFree(d_axes);
    hipFree(d_indices);
    hipFree(d_indices_shape);
    hipFree(d_indices_strides);
  });
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
