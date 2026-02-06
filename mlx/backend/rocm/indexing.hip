// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/jit_module.h"
#include "mlx/backend/rocm/device/indexing.hpp"
#include "mlx/backend/rocm/device/utils.hpp"
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

// Simple gather kernel for axis-based gather (for contiguous arrays)
template <typename T, typename IdxT, int NDIM, bool SrcC, bool IdxC>
__global__ void gather_axis_kernel(
    const T* src,
    const IdxT* idx,
    T* out,
    int64_t idx_size_pre,
    int64_t idx_size_axis,
    int64_t idx_size_post,
    const hip_array<int32_t, MAX_NDIM> shape,
    const hip_array<int64_t, MAX_NDIM> src_strides,
    const hip_array<int64_t, MAX_NDIM> idx_strides,
    int32_t axis,
    int32_t axis_size,
    int64_t src_stride_axis,
    int64_t idx_stride_axis) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = idx_size_pre * idx_size_axis * idx_size_post;
  if (index >= total) return;
  
  // Decompose index into x (post), y (axis), z (pre) coordinates
  int64_t x, y, z;
  index_to_dims(index, idx_size_axis, idx_size_pre, x, y, z);
  
  int64_t elem_idx = z * idx_size_post;
  
  // Compute index location
  int64_t idx_loc = y * idx_stride_axis;
  if constexpr (IdxC) {
    idx_loc += elem_idx * idx_size_axis + x;
  } else {
    idx_loc += elem_to_loc_nd<NDIM>(elem_idx + x, shape.data_, idx_strides.data_);
  }
  
  // Get index value and handle negative indices
  IdxT idx_val = idx[idx_loc];
  if (idx_val < 0) {
    idx_val += axis_size;
  }
  
  // Compute source location
  int64_t src_loc = idx_val * src_stride_axis;
  if constexpr (SrcC) {
    src_loc += elem_idx * axis_size + x;
  } else {
    src_loc += elem_to_loc_nd<NDIM>(elem_idx + x, shape.data_, src_strides.data_);
  }
  
  // Output is always contiguous
  int64_t out_idx = y * idx_size_post + elem_idx * idx_size_axis + x;
  
  out[out_idx] = src[src_loc];
}

// Simple scatter kernel for axis-based scatter
template <typename T, typename IdxT, bool IS_SUM, int NDIM, bool UpdC, bool IdxC>
__global__ void scatter_axis_kernel(
    const T* upd,
    const IdxT* idx,
    T* out,
    int64_t idx_size_pre,
    int64_t idx_size_axis,
    int64_t idx_size_post,
    const hip_array<int32_t, MAX_NDIM> shape,
    const hip_array<int64_t, MAX_NDIM> upd_strides,
    const hip_array<int64_t, MAX_NDIM> idx_strides,
    const hip_array<int64_t, MAX_NDIM> out_strides,
    int32_t axis,
    int32_t axis_size,
    int64_t upd_stride_axis,
    int64_t idx_stride_axis,
    int64_t out_stride_axis) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = idx_size_pre * idx_size_axis * idx_size_post;
  if (index >= total) return;
  
  // Decompose index into x (post), y (axis), z (pre) coordinates
  int64_t x, y, z;
  index_to_dims(index, idx_size_axis, idx_size_pre, x, y, z);
  
  int64_t elem_idx = z * idx_size_post;
  
  // Compute index location
  int64_t idx_loc = y * idx_stride_axis;
  if constexpr (IdxC) {
    idx_loc += elem_idx * idx_size_axis + x;
  } else {
    idx_loc += elem_to_loc_nd<NDIM>(elem_idx + x, shape.data_, idx_strides.data_);
  }
  
  // Get index value and handle negative indices
  IdxT idx_val = idx[idx_loc];
  if (idx_val < 0) {
    idx_val += axis_size;
  }
  
  // Compute update location
  int64_t upd_loc = y * upd_stride_axis;
  if constexpr (UpdC) {
    upd_loc += elem_idx * idx_size_axis + x;
  } else {
    upd_loc += elem_to_loc_nd<NDIM>(elem_idx + x, shape.data_, upd_strides.data_);
  }
  
  // Compute output location
  int64_t out_loc = idx_val * out_stride_axis;
  out_loc += elem_to_loc_nd<NDIM>(elem_idx + x, shape.data_, out_strides.data_);
  
  if constexpr (IS_SUM) {
    atomicAdd(&out[out_loc], upd[upd_loc]);
  } else {
    out[out_loc] = upd[upd_loc];
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

  int64_t out_elem = gid % upd_post_idx_size;
  int64_t idx_elem = gid / upd_post_idx_size;

  // Compute output location from out_elem using upd_shape after idx_ndim dimensions
  // This matches the CUDA implementation: elem_to_loc(out_elem, upd_shape + IDX_NDIM, out_strides, out_ndim)
  int64_t out_loc = 0;
  int64_t tmp = out_elem;
  for (int i = out_ndim - 1; i >= 0; --i) {
    // Use upd_shape[idx_ndim + i] for the shape dimensions after the index dimensions
    int32_t dim_size = (idx_ndim + i < upd_ndim) ? upd_shape[idx_ndim + i] : 1;
    out_loc += (tmp % dim_size) * out_strides[i];
    tmp /= dim_size;
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

  // Compute update location
  int64_t upd_loc = 0;
  tmp = out_elem + idx_elem * upd_post_idx_size;
  for (int i = upd_ndim - 1; i >= 0; --i) {
    upd_loc += (tmp % upd_shape[i]) * upd_strides[i];
    tmp /= upd_shape[i];
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
      // Fallback for types without atomic support - use CAS loop
      T* addr = &out[out_loc];
      T old_val = *addr;
      T new_val;
      do {
        new_val = old_val + val;
      } while (!__hip_atomic_compare_exchange_strong(addr, &old_val, new_val, 
               __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
    }
  } else if constexpr (ReduceType == 2) { // Prod
    // Use CAS loop for atomic multiply
    if constexpr (std::is_same_v<T, float>) {
      float* addr = &out[out_loc];
      float old_val = *addr;
      float new_val;
      do {
        new_val = old_val * val;
      } while (!__hip_atomic_compare_exchange_strong(addr, &old_val, new_val,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
    } else if constexpr (std::is_same_v<T, int32_t>) {
      int32_t* addr = &out[out_loc];
      int32_t old_val = *addr;
      int32_t new_val;
      do {
        new_val = old_val * val;
      } while (!__hip_atomic_compare_exchange_strong(addr, &old_val, new_val,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
    } else {
      // Fallback for other types
      T* addr = &out[out_loc];
      T old_val = *addr;
      T new_val;
      do {
        new_val = old_val * val;
      } while (!__hip_atomic_compare_exchange_strong(addr, &old_val, new_val,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
    }
  } else if constexpr (ReduceType == 3) { // Max
    // Use CAS loop for atomic max
    if constexpr (std::is_same_v<T, int32_t>) {
      int32_t* addr = &out[out_loc];
      int32_t old_val = *addr;
      while (val > old_val) {
        if (__hip_atomic_compare_exchange_strong(addr, &old_val, val,
            __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) {
          break;
        }
      }
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      uint32_t* addr = &out[out_loc];
      uint32_t old_val = *addr;
      while (val > old_val) {
        if (__hip_atomic_compare_exchange_strong(addr, &old_val, val,
            __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) {
          break;
        }
      }
    } else if constexpr (std::is_same_v<T, float>) {
      // Use CAS loop for float max
      float* addr = &out[out_loc];
      float old_val = *addr;
      while (val > old_val) {
        if (__hip_atomic_compare_exchange_strong(addr, &old_val, val,
            __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) {
          break;
        }
      }
    } else {
      // Fallback for other types
      T* addr = &out[out_loc];
      T old_val = *addr;
      while (val > old_val) {
        if (__hip_atomic_compare_exchange_strong(addr, &old_val, val,
            __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) {
          break;
        }
      }
    }
  } else if constexpr (ReduceType == 4) { // Min
    // Use CAS loop for atomic min
    if constexpr (std::is_same_v<T, int32_t>) {
      int32_t* addr = &out[out_loc];
      int32_t old_val = *addr;
      while (val < old_val) {
        if (__hip_atomic_compare_exchange_strong(addr, &old_val, val,
            __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) {
          break;
        }
      }
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      uint32_t* addr = &out[out_loc];
      uint32_t old_val = *addr;
      while (val < old_val) {
        if (__hip_atomic_compare_exchange_strong(addr, &old_val, val,
            __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) {
          break;
        }
      }
    } else if constexpr (std::is_same_v<T, float>) {
      // Use CAS loop for float min
      float* addr = &out[out_loc];
      float old_val = *addr;
      while (val < old_val) {
        if (__hip_atomic_compare_exchange_strong(addr, &old_val, val,
            __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) {
          break;
        }
      }
    } else {
      // Fallback for other types
      T* addr = &out[out_loc];
      T old_val = *addr;
      while (val < old_val) {
        if (__hip_atomic_compare_exchange_strong(addr, &old_val, val,
            __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) {
          break;
        }
      }
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

  // Prepare host data for parameters
  std::vector<int32_t> h_src_shape(src.shape().begin(), src.shape().end());
  std::vector<int64_t> h_src_strides(src.strides().begin(), src.strides().end());
  std::vector<int32_t> h_slice_sizes(slice_sizes_.begin(), slice_sizes_.end());
  std::vector<int32_t> h_axes(axes_.begin(), axes_.end());

  // Prepare indices pointers and metadata
  std::vector<const void*> h_indices(std::max(nidx, 1));
  std::vector<int32_t> h_indices_shape(std::max(nidx, 1) * std::max(idx_ndim, 1));
  std::vector<int64_t> h_indices_strides(std::max(nidx, 1) * std::max(idx_ndim, 1));

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

  // Allocate device memory using allocator
  array src_shape_arr({static_cast<int>(h_src_shape.size())}, int32, nullptr, {});
  src_shape_arr.set_data(allocator::malloc(h_src_shape.size() * sizeof(int32_t)));
  
  array src_strides_arr({static_cast<int>(h_src_strides.size())}, int64, nullptr, {});
  src_strides_arr.set_data(allocator::malloc(h_src_strides.size() * sizeof(int64_t)));
  
  array slice_sizes_arr({static_cast<int>(h_slice_sizes.size())}, int32, nullptr, {});
  slice_sizes_arr.set_data(allocator::malloc(h_slice_sizes.size() * sizeof(int32_t)));
  
  array axes_arr({static_cast<int>(h_axes.size())}, int32, nullptr, {});
  axes_arr.set_data(allocator::malloc(std::max(h_axes.size(), (size_t)1) * sizeof(int32_t)));
  
  array indices_arr({static_cast<int>(h_indices.size())}, int64, nullptr, {});
  indices_arr.set_data(allocator::malloc(h_indices.size() * sizeof(void*)));
  
  array indices_shape_arr({static_cast<int>(h_indices_shape.size())}, int32, nullptr, {});
  indices_shape_arr.set_data(allocator::malloc(h_indices_shape.size() * sizeof(int32_t)));
  
  array indices_strides_arr({static_cast<int>(h_indices_strides.size())}, int64, nullptr, {});
  indices_strides_arr.set_data(allocator::malloc(h_indices_strides.size() * sizeof(int64_t)));

  encoder.launch_kernel([&, h_src_shape, h_src_strides, h_slice_sizes, h_axes, 
                         h_indices, h_indices_shape, h_indices_strides](hipStream_t stream) {
    // Copy data to device asynchronously
    (void)hipMemcpyAsync(src_shape_arr.data<int32_t>(), h_src_shape.data(), 
                         h_src_shape.size() * sizeof(int32_t), hipMemcpyHostToDevice, stream);
    (void)hipMemcpyAsync(src_strides_arr.data<int64_t>(), h_src_strides.data(), 
                         h_src_strides.size() * sizeof(int64_t), hipMemcpyHostToDevice, stream);
    (void)hipMemcpyAsync(slice_sizes_arr.data<int32_t>(), h_slice_sizes.data(), 
                         h_slice_sizes.size() * sizeof(int32_t), hipMemcpyHostToDevice, stream);
    if (!h_axes.empty()) {
      (void)hipMemcpyAsync(axes_arr.data<int32_t>(), h_axes.data(), 
                           h_axes.size() * sizeof(int32_t), hipMemcpyHostToDevice, stream);
    }
    (void)hipMemcpyAsync(indices_arr.data<void*>(), h_indices.data(), 
                         h_indices.size() * sizeof(void*), hipMemcpyHostToDevice, stream);
    (void)hipMemcpyAsync(indices_shape_arr.data<int32_t>(), h_indices_shape.data(), 
                         h_indices_shape.size() * sizeof(int32_t), hipMemcpyHostToDevice, stream);
    (void)hipMemcpyAsync(indices_strides_arr.data<int64_t>(), h_indices_strides.data(), 
                         h_indices_strides.size() * sizeof(int64_t), hipMemcpyHostToDevice, stream);

    // Dispatch based on dtype and number of indices
    #define LAUNCH_GATHER(T, IdxT, NIDX) \
      hipLaunchKernelGGL( \
          (rocm::gather_general_kernel<T, IdxT, NIDX>), \
          dim3(num_blocks), dim3(block_size), 0, stream, \
          src.data<T>(), out.data<T>(), total, \
          src_shape_arr.data<int32_t>(), src_strides_arr.data<int64_t>(), src.ndim(), \
          slice_sizes_arr.data<int32_t>(), slice_size, axes_arr.data<int32_t>(), \
          (const IdxT* const*)indices_arr.data<void*>(), indices_shape_arr.data<int32_t>(), \
          indices_strides_arr.data<int64_t>(), idx_ndim)

    #define DISPATCH_NIDX(T, IdxT) \
      switch (nidx) { \
        case 0: LAUNCH_GATHER(T, IdxT, 0); break; \
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
        case int16: DISPATCH_NIDX(int16_t, int32_t); break;
        case uint8: DISPATCH_NIDX(uint8_t, int32_t); break;
        case uint16: DISPATCH_NIDX(uint16_t, int32_t); break;
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
        case uint32: DISPATCH_NIDX(uint32_t, int64_t); break;
        case uint64: DISPATCH_NIDX(uint64_t, int64_t); break;
        case int8: DISPATCH_NIDX(int8_t, int64_t); break;
        case int16: DISPATCH_NIDX(int16_t, int64_t); break;
        case uint8: DISPATCH_NIDX(uint8_t, int64_t); break;
        case uint16: DISPATCH_NIDX(uint16_t, int64_t); break;
        case bool_: DISPATCH_NIDX(bool, int64_t); break;
        default:
          throw std::runtime_error("Unsupported dtype for Gather");
      }
    }

    #undef DISPATCH_NIDX
    #undef LAUNCH_GATHER
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

  // Prepare host data for parameters
  std::vector<int32_t> h_upd_shape(upd.shape().begin(), upd.shape().end());
  std::vector<int64_t> h_upd_strides(upd.strides().begin(), upd.strides().end());
  std::vector<int32_t> h_out_shape(out.shape().begin(), out.shape().end());
  std::vector<int64_t> h_out_strides(out.strides().begin(), out.strides().end());
  std::vector<int32_t> h_axes(axes_.begin(), axes_.end());

  // Prepare indices pointers and metadata
  std::vector<const void*> h_indices(std::max(nidx, 1));
  std::vector<int32_t> h_indices_shape(std::max(nidx, 1) * std::max(idx_ndim, 1));
  std::vector<int64_t> h_indices_strides(std::max(nidx, 1) * std::max(idx_ndim, 1));

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

  // Allocate device memory using allocator
  array upd_shape_arr({static_cast<int>(h_upd_shape.size())}, int32, nullptr, {});
  upd_shape_arr.set_data(allocator::malloc(h_upd_shape.size() * sizeof(int32_t)));
  
  array upd_strides_arr({static_cast<int>(h_upd_strides.size())}, int64, nullptr, {});
  upd_strides_arr.set_data(allocator::malloc(h_upd_strides.size() * sizeof(int64_t)));
  
  array out_shape_arr({static_cast<int>(h_out_shape.size())}, int32, nullptr, {});
  out_shape_arr.set_data(allocator::malloc(h_out_shape.size() * sizeof(int32_t)));
  
  array out_strides_arr({static_cast<int>(h_out_strides.size())}, int64, nullptr, {});
  out_strides_arr.set_data(allocator::malloc(h_out_strides.size() * sizeof(int64_t)));
  
  array axes_arr({static_cast<int>(std::max(h_axes.size(), (size_t)1))}, int32, nullptr, {});
  axes_arr.set_data(allocator::malloc(std::max(h_axes.size(), (size_t)1) * sizeof(int32_t)));
  
  array indices_arr({static_cast<int>(h_indices.size())}, int64, nullptr, {});
  indices_arr.set_data(allocator::malloc(h_indices.size() * sizeof(void*)));
  
  array indices_shape_arr({static_cast<int>(h_indices_shape.size())}, int32, nullptr, {});
  indices_shape_arr.set_data(allocator::malloc(h_indices_shape.size() * sizeof(int32_t)));
  
  array indices_strides_arr({static_cast<int>(h_indices_strides.size())}, int64, nullptr, {});
  indices_strides_arr.set_data(allocator::malloc(h_indices_strides.size() * sizeof(int64_t)));

  int reduce_type = reduce_type_; // Scatter::ReduceType: Max=0, Min=1, Sum=2, Prod=3, None=4
  // Map to kernel ReduceType: Assign=0, Sum=1, Prod=2, Max=3, Min=4
  int kernel_reduce_type;
  switch (reduce_type) {
    case 0: kernel_reduce_type = 3; break; // Max
    case 1: kernel_reduce_type = 4; break; // Min
    case 2: kernel_reduce_type = 1; break; // Sum
    case 3: kernel_reduce_type = 2; break; // Prod
    case 4: kernel_reduce_type = 0; break; // None -> Assign
    default: kernel_reduce_type = 0; break;
  }

  encoder.launch_kernel([&, h_upd_shape, h_upd_strides, h_out_shape, h_out_strides, 
                         h_axes, h_indices, h_indices_shape, h_indices_strides, kernel_reduce_type](hipStream_t stream) {
    // Copy data to device asynchronously
    (void)hipMemcpyAsync(upd_shape_arr.data<int32_t>(), h_upd_shape.data(), 
                         h_upd_shape.size() * sizeof(int32_t), hipMemcpyHostToDevice, stream);
    (void)hipMemcpyAsync(upd_strides_arr.data<int64_t>(), h_upd_strides.data(), 
                         h_upd_strides.size() * sizeof(int64_t), hipMemcpyHostToDevice, stream);
    (void)hipMemcpyAsync(out_shape_arr.data<int32_t>(), h_out_shape.data(), 
                         h_out_shape.size() * sizeof(int32_t), hipMemcpyHostToDevice, stream);
    (void)hipMemcpyAsync(out_strides_arr.data<int64_t>(), h_out_strides.data(), 
                         h_out_strides.size() * sizeof(int64_t), hipMemcpyHostToDevice, stream);
    if (!h_axes.empty()) {
      (void)hipMemcpyAsync(axes_arr.data<int32_t>(), h_axes.data(), 
                           h_axes.size() * sizeof(int32_t), hipMemcpyHostToDevice, stream);
    }
    if (nidx > 0) {
      (void)hipMemcpyAsync(indices_arr.data<void*>(), h_indices.data(), 
                           h_indices.size() * sizeof(void*), hipMemcpyHostToDevice, stream);
      (void)hipMemcpyAsync(indices_shape_arr.data<int32_t>(), h_indices_shape.data(), 
                           h_indices_shape.size() * sizeof(int32_t), hipMemcpyHostToDevice, stream);
      (void)hipMemcpyAsync(indices_strides_arr.data<int64_t>(), h_indices_strides.data(), 
                           h_indices_strides.size() * sizeof(int64_t), hipMemcpyHostToDevice, stream);
    }

    #define LAUNCH_SCATTER(T, IdxT, NIDX, RT) \
      hipLaunchKernelGGL( \
          (rocm::scatter_general_kernel<T, IdxT, NIDX, RT>), \
          dim3(num_blocks), dim3(block_size), 0, stream, \
          upd.data<T>(), out.data<T>(), total, \
          upd_shape_arr.data<int32_t>(), upd_strides_arr.data<int64_t>(), upd.ndim(), upd_post_idx_size, \
          out_shape_arr.data<int32_t>(), out_strides_arr.data<int64_t>(), out.ndim(), \
          axes_arr.data<int32_t>(), (const IdxT* const*)indices_arr.data<void*>(), \
          indices_shape_arr.data<int32_t>(), indices_strides_arr.data<int64_t>(), idx_ndim)

    #define DISPATCH_REDUCE(T, IdxT, NIDX) \
      switch (kernel_reduce_type) { \
        case 0: LAUNCH_SCATTER(T, IdxT, NIDX, 0); break; \
        case 1: LAUNCH_SCATTER(T, IdxT, NIDX, 1); break; \
        case 2: LAUNCH_SCATTER(T, IdxT, NIDX, 2); break; \
        case 3: LAUNCH_SCATTER(T, IdxT, NIDX, 3); break; \
        case 4: LAUNCH_SCATTER(T, IdxT, NIDX, 4); break; \
        default: LAUNCH_SCATTER(T, IdxT, NIDX, 0); break; \
      }

    #define DISPATCH_NIDX(T, IdxT) \
      switch (nidx) { \
        case 0: DISPATCH_REDUCE(T, IdxT, 0); break; \
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
        case bfloat16: DISPATCH_NIDX(hip_bfloat16, int32_t); break;
        case int32: DISPATCH_NIDX(int32_t, int32_t); break;
        case int64: DISPATCH_NIDX(int64_t, int32_t); break;
        case uint32: DISPATCH_NIDX(uint32_t, int32_t); break;
        case uint64: DISPATCH_NIDX(uint64_t, int32_t); break;
        case int8: DISPATCH_NIDX(int8_t, int32_t); break;
        case int16: DISPATCH_NIDX(int16_t, int32_t); break;
        case uint8: DISPATCH_NIDX(uint8_t, int32_t); break;
        case uint16: DISPATCH_NIDX(uint16_t, int32_t); break;
        case bool_: DISPATCH_NIDX(bool, int32_t); break;
        default:
          throw std::runtime_error("Unsupported dtype for Scatter");
      }
    } else {
      switch (out.dtype()) {
        case float32: DISPATCH_NIDX(float, int64_t); break;
        case float16: DISPATCH_NIDX(__half, int64_t); break;
        case bfloat16: DISPATCH_NIDX(hip_bfloat16, int64_t); break;
        case int32: DISPATCH_NIDX(int32_t, int64_t); break;
        case int64: DISPATCH_NIDX(int64_t, int64_t); break;
        case uint32: DISPATCH_NIDX(uint32_t, int64_t); break;
        case uint64: DISPATCH_NIDX(uint64_t, int64_t); break;
        case int8: DISPATCH_NIDX(int8_t, int64_t); break;
        case int16: DISPATCH_NIDX(int16_t, int64_t); break;
        case uint8: DISPATCH_NIDX(uint8_t, int64_t); break;
        case uint16: DISPATCH_NIDX(uint16_t, int64_t); break;
        case bool_: DISPATCH_NIDX(bool, int64_t); break;
        default:
          throw std::runtime_error("Unsupported dtype for Scatter");
      }
    }

    #undef DISPATCH_NIDX
    #undef DISPATCH_REDUCE
    #undef LAUNCH_SCATTER
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
  
  // Create shape and strides with axis dimension removed
  int ndim = src.ndim() - 1;
  if (ndim == 0) {
    ndim = 1;  // Ensure at least 1 dimension for elem_to_loc_nd
  }
  
  std::vector<int32_t> shape_vec(ndim, 1);
  std::vector<int64_t> src_strides_vec(ndim, 0);
  std::vector<int64_t> idx_strides_vec(ndim, 0);
  
  for (int i = 0, j = 0; i < src.ndim(); ++i) {
    if (i != axis_) {
      if (j < ndim) {
        shape_vec[j] = idx.shape(i);
        src_strides_vec[j] = src.strides(i);
        idx_strides_vec[j] = idx.strides(i);
      }
      ++j;
    }
  }
  
  // Use const_param to pass shape and strides by value (like CUDA)
  auto shape_param = const_param(shape_vec);
  auto src_strides_param = const_param(src_strides_vec);
  auto idx_strides_param = const_param(idx_strides_vec);
  
  int64_t src_stride_axis = src.strides(axis_);
  int64_t idx_stride_axis = idx.strides(axis_);
  int32_t axis_size = src.shape(axis_);
  
  bool src_contiguous = src.flags().row_contiguous;
  bool idx_contiguous = idx.flags().row_contiguous;
  
  int64_t total = idx_size_pre * idx_size_axis * idx_size_post;
  int block_size = 256;
  int num_blocks = (total + block_size - 1) / block_size;
  
  // Dispatch based on ndim, contiguity, and index type
  #define LAUNCH_GATHER_KERNEL(T, IdxT, NDIM, SrcC, IdxC) \
    hipLaunchKernelGGL( \
        (rocm::gather_axis_kernel<T, IdxT, NDIM, SrcC, IdxC>), \
        dim3(num_blocks), dim3(block_size), 0, stream, \
        src.data<T>(), idx.data<IdxT>(), out.data<T>(), \
        idx_size_pre, idx_size_axis, idx_size_post, \
        shape_param, \
        src_strides_param, \
        idx_strides_param, \
        axis_, axis_size, src_stride_axis, idx_stride_axis)
  
  #define DISPATCH_CONTIGUOUS(T, IdxT, NDIM) \
    if (src_contiguous && idx_contiguous) { \
      LAUNCH_GATHER_KERNEL(T, IdxT, NDIM, true, true); \
    } else if (src_contiguous) { \
      LAUNCH_GATHER_KERNEL(T, IdxT, NDIM, true, false); \
    } else if (idx_contiguous) { \
      LAUNCH_GATHER_KERNEL(T, IdxT, NDIM, false, true); \
    } else { \
      LAUNCH_GATHER_KERNEL(T, IdxT, NDIM, false, false); \
    }
  
  #define DISPATCH_NDIM(T, IdxT) \
    switch (ndim) { \
      case 0: DISPATCH_CONTIGUOUS(T, IdxT, 1); break; \
      case 1: DISPATCH_CONTIGUOUS(T, IdxT, 1); break; \
      case 2: DISPATCH_CONTIGUOUS(T, IdxT, 2); break; \
      case 3: DISPATCH_CONTIGUOUS(T, IdxT, 3); break; \
      case 4: DISPATCH_CONTIGUOUS(T, IdxT, 4); break; \
      case 5: DISPATCH_CONTIGUOUS(T, IdxT, 5); break; \
      case 6: DISPATCH_CONTIGUOUS(T, IdxT, 6); break; \
      case 7: DISPATCH_CONTIGUOUS(T, IdxT, 7); break; \
      default: DISPATCH_CONTIGUOUS(T, IdxT, 8); break; \
    }
  
  #define DISPATCH_IDX_TYPE(T) \
    if (idx.dtype() == int32 || idx.dtype() == uint32) { \
      DISPATCH_NDIM(T, int32_t); \
    } else { \
      DISPATCH_NDIM(T, int64_t); \
    }
  
  encoder.launch_kernel([&](hipStream_t stream) {
    switch (src.dtype()) {
      case float32: DISPATCH_IDX_TYPE(float); break;
      case int32: DISPATCH_IDX_TYPE(int32_t); break;
      case uint32: DISPATCH_IDX_TYPE(uint32_t); break;
      case int64: DISPATCH_IDX_TYPE(int64_t); break;
      case uint64: DISPATCH_IDX_TYPE(uint64_t); break;
      case float16: DISPATCH_IDX_TYPE(__half); break;
      case bfloat16: DISPATCH_IDX_TYPE(hip_bfloat16); break;
      case int8: DISPATCH_IDX_TYPE(int8_t); break;
      case uint8: DISPATCH_IDX_TYPE(uint8_t); break;
      case int16: DISPATCH_IDX_TYPE(int16_t); break;
      case uint16: DISPATCH_IDX_TYPE(uint16_t); break;
      case bool_: DISPATCH_IDX_TYPE(bool); break;
      default:
        throw std::runtime_error("Unsupported dtype for GatherAxis");
    }
  });
  
  #undef LAUNCH_GATHER_KERNEL
  #undef DISPATCH_CONTIGUOUS
  #undef DISPATCH_NDIM
  #undef DISPATCH_IDX_TYPE
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
  
  // Create shape and strides with axis dimension removed
  int ndim = idx.ndim() - 1;
  if (ndim == 0) {
    ndim = 1;  // Ensure at least 1 dimension for elem_to_loc_nd
  }
  
  std::vector<int32_t> shape_vec(ndim, 1);
  std::vector<int64_t> upd_strides_vec(ndim, 0);
  std::vector<int64_t> idx_strides_vec(ndim, 0);
  std::vector<int64_t> out_strides_vec(ndim, 0);
  
  for (int i = 0, j = 0; i < idx.ndim(); ++i) {
    if (i != axis_) {
      if (j < ndim) {
        shape_vec[j] = idx.shape(i);
        upd_strides_vec[j] = upd.strides(i);
        idx_strides_vec[j] = idx.strides(i);
        out_strides_vec[j] = out.strides(i);
      }
      ++j;
    }
  }
  
  // Use const_param to pass shape and strides by value
  auto shape_param = const_param(shape_vec);
  auto upd_strides_param = const_param(upd_strides_vec);
  auto idx_strides_param = const_param(idx_strides_vec);
  auto out_strides_param = const_param(out_strides_vec);
  
  int64_t upd_stride_axis = upd.strides(axis_);
  int64_t idx_stride_axis = idx.strides(axis_);
  int64_t out_stride_axis = out.strides(axis_);
  int32_t axis_size = out.shape(axis_);
  
  bool upd_contiguous = upd.flags().row_contiguous;
  bool idx_contiguous = idx.flags().row_contiguous;
  
  int64_t total = idx_size_pre * idx_size_axis * idx_size_post;
  int block_size = 256;
  int num_blocks = (total + block_size - 1) / block_size;
  
  bool is_sum = (reduce_type_ == ScatterAxis::Sum);
  
  #define LAUNCH_SCATTER_KERNEL(T, IdxT, IS_SUM, NDIM, UpdC, IdxC) \
    hipLaunchKernelGGL( \
        (rocm::scatter_axis_kernel<T, IdxT, IS_SUM, NDIM, UpdC, IdxC>), \
        dim3(num_blocks), dim3(block_size), 0, stream, \
        upd.data<T>(), idx.data<IdxT>(), out.data<T>(), \
        idx_size_pre, idx_size_axis, idx_size_post, \
        shape_param, \
        upd_strides_param, \
        idx_strides_param, \
        out_strides_param, \
        axis_, axis_size, upd_stride_axis, idx_stride_axis, out_stride_axis)
  
  #define DISPATCH_CONTIGUOUS(T, IdxT, IS_SUM, NDIM) \
    if (upd_contiguous && idx_contiguous) { \
      LAUNCH_SCATTER_KERNEL(T, IdxT, IS_SUM, NDIM, true, true); \
    } else if (upd_contiguous) { \
      LAUNCH_SCATTER_KERNEL(T, IdxT, IS_SUM, NDIM, true, false); \
    } else if (idx_contiguous) { \
      LAUNCH_SCATTER_KERNEL(T, IdxT, IS_SUM, NDIM, false, true); \
    } else { \
      LAUNCH_SCATTER_KERNEL(T, IdxT, IS_SUM, NDIM, false, false); \
    }
  
  #define DISPATCH_NDIM(T, IdxT, IS_SUM) \
    switch (ndim) { \
      case 0: DISPATCH_CONTIGUOUS(T, IdxT, IS_SUM, 1); break; \
      case 1: DISPATCH_CONTIGUOUS(T, IdxT, IS_SUM, 1); break; \
      case 2: DISPATCH_CONTIGUOUS(T, IdxT, IS_SUM, 2); break; \
      case 3: DISPATCH_CONTIGUOUS(T, IdxT, IS_SUM, 3); break; \
      case 4: DISPATCH_CONTIGUOUS(T, IdxT, IS_SUM, 4); break; \
      case 5: DISPATCH_CONTIGUOUS(T, IdxT, IS_SUM, 5); break; \
      case 6: DISPATCH_CONTIGUOUS(T, IdxT, IS_SUM, 6); break; \
      case 7: DISPATCH_CONTIGUOUS(T, IdxT, IS_SUM, 7); break; \
      default: DISPATCH_CONTIGUOUS(T, IdxT, IS_SUM, 8); break; \
    }
  
  #define DISPATCH_IDX_TYPE(T, IS_SUM) \
    if (idx.dtype() == int32 || idx.dtype() == uint32) { \
      DISPATCH_NDIM(T, int32_t, IS_SUM); \
    } else { \
      DISPATCH_NDIM(T, int64_t, IS_SUM); \
    }
  
  encoder.launch_kernel([&](hipStream_t stream) {
    if (is_sum) {
      // Note: atomicAdd only supports float32 and float64 on ROCm
      // float16/bfloat16 would need custom atomic implementations
      switch (upd.dtype()) {
        case float32: DISPATCH_IDX_TYPE(float, true); break;
        default:
          throw std::runtime_error("Unsupported dtype for ScatterAxis Sum (only float32 supported)");
      }
    } else {
      switch (upd.dtype()) {
        case float32: DISPATCH_IDX_TYPE(float, false); break;
        case float16: DISPATCH_IDX_TYPE(__half, false); break;
        case bfloat16: DISPATCH_IDX_TYPE(hip_bfloat16, false); break;
        case int32: DISPATCH_IDX_TYPE(int32_t, false); break;
        case int64: DISPATCH_IDX_TYPE(int64_t, false); break;
        case uint32: DISPATCH_IDX_TYPE(uint32_t, false); break;
        case uint64: DISPATCH_IDX_TYPE(uint64_t, false); break;
        case int8: DISPATCH_IDX_TYPE(int8_t, false); break;
        case int16: DISPATCH_IDX_TYPE(int16_t, false); break;
        case uint8: DISPATCH_IDX_TYPE(uint8_t, false); break;
        case uint16: DISPATCH_IDX_TYPE(uint16_t, false); break;
        case bool_: DISPATCH_IDX_TYPE(bool, false); break;
        default:
          throw std::runtime_error("Unsupported dtype for ScatterAxis Assign");
      }
    }
  });
  
  #undef LAUNCH_SCATTER_KERNEL
  #undef DISPATCH_CONTIGUOUS
  #undef DISPATCH_NDIM
  #undef DISPATCH_IDX_TYPE
}

} // namespace mlx::core
