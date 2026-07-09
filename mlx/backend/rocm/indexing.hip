// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/jit_module.h"
#include "mlx/backend/rocm/device/indexing.hpp"
#include "mlx/backend/rocm/device/binary_ops.hpp"
#include "mlx/backend/rocm/device/atomic_ops.hpp"
#include "mlx/backend/rocm/device/utils.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/common/slicing.h"
#include "mlx/backend/common/utils.h"
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
    // Metadata passed BY VALUE (hip_array in kernel args) rather than via device
    // pointers. The previous by-pointer form required uploading these to device
    // buffers via hipMemcpyAsync; under HIP graph capture those H2D nodes record
    // the (transient) host source pointer and read freed memory on replay,
    // producing a garbage source offset -> out-of-bounds read -> GPU queue hang
    // on RDNA4 (gfx1201). By-value metadata is captured correctly and replays.
    hip_array<int32_t, MAX_NDIM> src_shape,
    hip_array<int64_t, MAX_NDIM> src_strides,
    int32_t src_ndim,
    hip_array<int32_t, MAX_NDIM> slice_sizes,
    uint32_t slice_size,
    hip_array<int32_t, 8> axes,
    hip_array<const IdxT*, 8> indices,
    hip_array<int32_t, 8 * MAX_NDIM> indices_shape,
    hip_array<int64_t, 8 * MAX_NDIM> indices_strides,
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

// Fast contiguous row gather: out[row, :] = src[idx[row], :] where each row is a
// contiguous block of `row_size` elements. One block per output row, coalesced
// copy, with the source-row base computed once (no per-element index math).
// Covers the common axis-0 gather of a row-contiguous source (e.g. the MoE
// token reorder), which the general kernel does per-element with mod/div loops.
template <typename T, typename IdxT>
__global__ void gather_rows_kernel(
    const T* src,
    const IdxT* idx,
    T* out,
    int64_t n_rows,
    uint32_t row_size,
    int32_t src_dim0) {
  int64_t row = blockIdx.x;
  if (row >= n_rows) {
    return;
  }
  int64_t r = static_cast<int64_t>(idx[row]);
  if (r < 0) {
    r += src_dim0;
  }
  const T* srow = src + r * static_cast<int64_t>(row_size);
  T* orow = out + row * static_cast<int64_t>(row_size);
  for (uint32_t e = threadIdx.x; e < row_size; e += blockDim.x) {
    orow[e] = srow[e];
  }
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
    // atomic_add has CAS-based specializations for hip_bfloat16 / __half, so
    // the Sum reduction (embedding-lookup / MoE-gather gradient) works for the
    // half-precision dtypes too, not just float32 as a raw atomicAdd would.
    atomic_add(&out[out_loc], upd[upd_loc]);
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
    // Metadata passed BY VALUE (hip_array in kernel args) rather than via device
    // pointers. The previous by-pointer form required uploading these to device
    // buffers via hipMemcpyAsync; under HIP graph capture those H2D nodes record
    // the (transient) host source pointer and read freed memory on replay,
    // producing a garbage source offset -> out-of-bounds read -> GPU queue hang
    // on RDNA4 (gfx1201). By-value metadata is captured correctly and replays.
    hip_array<int32_t, MAX_NDIM> upd_shape,
    hip_array<int64_t, MAX_NDIM> upd_strides,
    int32_t upd_ndim,
    int64_t upd_post_idx_size,
    hip_array<int32_t, MAX_NDIM> out_shape,
    hip_array<int64_t, MAX_NDIM> out_strides,
    int32_t out_ndim,
    hip_array<int32_t, 8> axes,
    hip_array<const IdxT*, 8> indices,
    hip_array<int32_t, 8 * MAX_NDIM> indices_shape,
    hip_array<int64_t, 8 * MAX_NDIM> indices_strides,
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

// SliceUpdate kernel: applies Op to combine existing output values with
// update values at computed slice positions.
template <
    typename T,
    typename IdxT,
    typename Op,
    bool OUT_ROW_CONTIG,
    bool UPD_ROW_CONTIG,
    bool UPD_SCALAR,
    int NWORK>
__global__ void slice_update_op_kernel(
    const T* updates,
    T* out,
    int64_t update_size,
    hip_array<int32_t, MAX_NDIM> update_shape,
    hip_array<int64_t, MAX_NDIM> update_strides,
    int32_t update_ndim,
    hip_array<int64_t, MAX_NDIM> output_strides,
    int64_t output_offset) {
  Op op;

  IdxT idx = (IdxT(blockIdx.x) * IdxT(blockDim.x) + IdxT(threadIdx.x)) * NWORK;
  IdxT out_idx;
  IdxT update_idx;

  if constexpr (OUT_ROW_CONTIG) {
    out_idx = idx;
  } else {
    out_idx = elem_to_loc<IdxT>(
        idx, update_shape.data_, output_strides.data_, update_ndim);
  }

  if constexpr (!UPD_SCALAR) {
    if constexpr (UPD_ROW_CONTIG) {
      update_idx = idx;
    } else {
      update_idx = elem_to_loc<IdxT>(
          idx, update_shape.data_, update_strides.data_, update_ndim);
    }
  } else {
    update_idx = 0;
  }

  out += output_offset;

  for (int j = 0; j < NWORK && idx < update_size; j++) {
    out[out_idx] = op(out[out_idx], updates[update_idx]);
    idx++;

    if constexpr (OUT_ROW_CONTIG) {
      out_idx = idx;
    } else {
      out_idx += output_strides[update_ndim - 1];
    }

    if constexpr (UPD_ROW_CONTIG) {
      update_idx = idx;
    } else if constexpr (!UPD_SCALAR) {
      update_idx += update_strides[update_ndim - 1];
    }
  }
}

template <int BLOCK_SIZE>
__global__ void masked_scatter_offsets_kernel(
    const bool* mask,
    uint32_t* scatter_offsets,
    int64_t mask_batch_size) {
  const int64_t batch_idx = static_cast<int64_t>(blockIdx.x);
  const int tid = threadIdx.x;
  const int64_t batch_base = batch_idx * mask_batch_size;

  __shared__ uint32_t scan_vals[BLOCK_SIZE];
  uint32_t batch_prefix = 0;

  for (int64_t i = 0; i < mask_batch_size; i += BLOCK_SIZE) {
    const int64_t mask_idx = i + tid;
    const bool in_range = mask_idx < mask_batch_size;
    const uint32_t mask_value =
        (in_range && mask[batch_base + mask_idx]) ? 1u : 0u;

    scan_vals[tid] = mask_value;
    __syncthreads();

    // In-place inclusive scan for a fixed-size block.
    for (int offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
      uint32_t add = 0;
      if (tid >= offset) {
        add = scan_vals[tid - offset];
      }
      __syncthreads();
      scan_vals[tid] += add;
      __syncthreads();
    }

    if (in_range) {
      // Convert the in-block inclusive scan to an exclusive offset.
      scatter_offsets[batch_base + mask_idx] =
          batch_prefix + (scan_vals[tid] - mask_value);
    }

    __syncthreads();
    batch_prefix += scan_vals[BLOCK_SIZE - 1];
    __syncthreads();
  }
}

template <typename T, bool SrcContiguous>
__global__ void masked_scatter_assign_kernel(
    const bool* mask,
    const uint32_t* scatter_offsets,
    const T* src,
    T* out,
    int64_t total,
    const rocm::hip_array<int32_t, MAX_NDIM> src_shape,
    const rocm::hip_array<int64_t, MAX_NDIM> src_strides,
    int32_t src_ndim,
    int64_t src_batch_size,
    int64_t mask_batch_size) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  if (idx >= total || !mask[idx]) {
    return;
  }

  const uint32_t src_index = scatter_offsets[idx];
  if (static_cast<int64_t>(src_index) >= src_batch_size) {
    return;
  }

  const int64_t batch_idx = idx / mask_batch_size;
  const int64_t src_elem =
      batch_idx * src_batch_size + static_cast<int64_t>(src_index);

  if constexpr (SrcContiguous) {
    out[idx] = src[src_elem];
  } else {
    const int64_t src_loc = rocm::elem_to_loc<int64_t>(
        src_elem, src_shape.data_, src_strides.data_, src_ndim);
    out[idx] = src[src_loc];
  }
}

} // namespace rocm

void Gather::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() > 0);
  const auto& src = inputs[0];

  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  out.set_data(mlx::core::rocm::malloc_async(out.nbytes(), encoder));
  if (out.size() == 0) {
    return;
  }

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
    h_indices[i] = gpu_ptr<void>(inputs[i + 1]);
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

  // Fast path: axis-0 full-row gather of a row-contiguous source with a single
  // contiguous (flattened) index -> coalesced per-row copy (gather_rows_kernel).
  bool fast_rows = (nidx == 1) && (axes_.size() == 1) && (axes_[0] == 0) &&
      src.ndim() >= 2 && src.flags().row_contiguous &&
      inputs[1].flags().contiguous &&
      ((int)slice_sizes_.size() == (int)src.ndim()) && (slice_sizes_[0] == 1);
  if (fast_rows) {
    for (int d = 1; d < (int)src.ndim(); ++d) {
      if (slice_sizes_[d] != src.shape(d)) {
        fast_rows = false;
        break;
      }
    }
  }
  // Grid is one block per row; keep within the launch limit.
  fast_rows = fast_rows && (total / (int64_t)slice_size) <= 0x7fffffffLL;

  // Pass all metadata BY VALUE (see gather_general_kernel) — no device buffers,
  // no H2D uploads, so nothing reads stale host memory on HIP graph replay.
  auto p_src_shape = const_param(h_src_shape);
  auto p_src_strides = const_param(h_src_strides);
  auto p_slice_sizes = const_param(h_slice_sizes);
  auto p_axes = const_param<8>(h_axes);
  auto p_indices_shape = const_param<8 * MAX_NDIM>(h_indices_shape);
  auto p_indices_strides = const_param<8 * MAX_NDIM>(h_indices_strides);
  int32_t src_ndim_v = static_cast<int32_t>(src.ndim());

  {
    if (fast_rows) {
      int64_t n_rows = total / (int64_t)slice_size;
      dim3 grid((unsigned int)n_rows);
      dim3 blk(256);
      Dtype it = inputs[1].dtype();
      int32_t src_dim0 = (int32_t)src.shape(0);
      #define LAUNCH_ROWS(T, IdxT) \
        encoder.add_kernel_node((&rocm::gather_rows_kernel<T, IdxT>), grid, blk, 0, \
            gpu_ptr<T>(src), \
            reinterpret_cast<const IdxT*>(h_indices[0]), gpu_ptr<T>(out), \
            n_rows, slice_size, src_dim0)
      #define ROWS_BY_T(IdxT) \
        switch (out.dtype()) { \
          case float32: LAUNCH_ROWS(float, IdxT); break; \
          case float16: LAUNCH_ROWS(__half, IdxT); break; \
          case bfloat16: LAUNCH_ROWS(hip_bfloat16, IdxT); break; \
          case int32: LAUNCH_ROWS(int32_t, IdxT); break; \
          case int64: LAUNCH_ROWS(int64_t, IdxT); break; \
          case uint32: LAUNCH_ROWS(uint32_t, IdxT); break; \
          case uint64: LAUNCH_ROWS(uint64_t, IdxT); break; \
          case int8: LAUNCH_ROWS(int8_t, IdxT); break; \
          case int16: LAUNCH_ROWS(int16_t, IdxT); break; \
          case uint8: LAUNCH_ROWS(uint8_t, IdxT); break; \
          case uint16: LAUNCH_ROWS(uint16_t, IdxT); break; \
          case bool_: LAUNCH_ROWS(bool, IdxT); break; \
          default: throw std::runtime_error("Unsupported dtype for Gather"); \
        }
      if (it == int32 || it == uint32) {
        ROWS_BY_T(int32_t);
      } else {
        ROWS_BY_T(int64_t);
      }
      #undef ROWS_BY_T
      #undef LAUNCH_ROWS
      return;
    }
    // Dispatch based on dtype and number of indices
    #define LAUNCH_GATHER(T, IdxT, NIDX) \
      do { \
        rocm::hip_array<const IdxT*, 8> idx_ptrs; \
        for (int _i = 0; _i < (NIDX); ++_i) \
          idx_ptrs[_i] = reinterpret_cast<const IdxT*>(h_indices[_i]); \
        encoder.add_kernel_node( \
            (&rocm::gather_general_kernel<T, IdxT, NIDX>), \
            dim3(num_blocks), dim3(block_size), 0, \
            gpu_ptr<T>(src), gpu_ptr<T>(out), total, \
            p_src_shape, p_src_strides, src_ndim_v, \
            p_slice_sizes, slice_size, p_axes, \
            idx_ptrs, p_indices_shape, p_indices_strides, idx_ndim); \
      } while (0)

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
  }
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
    h_indices[i] = gpu_ptr<void>(inputs[i + 1]);
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

  // Pass all metadata BY VALUE (see scatter_general_kernel) — no device buffers,
  // no H2D uploads, so nothing reads stale host memory on HIP graph replay.
  auto p_upd_shape = const_param(h_upd_shape);
  auto p_upd_strides = const_param(h_upd_strides);
  auto p_out_shape = const_param(h_out_shape);
  auto p_out_strides = const_param(h_out_strides);
  auto p_axes = const_param<8>(h_axes);
  auto p_indices_shape = const_param<8 * MAX_NDIM>(h_indices_shape);
  auto p_indices_strides = const_param<8 * MAX_NDIM>(h_indices_strides);
  int32_t upd_ndim_v = static_cast<int32_t>(upd.ndim());
  int32_t out_ndim_v = static_cast<int32_t>(out.ndim());

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

  {
    #define LAUNCH_SCATTER(T, IdxT, NIDX, RT) \
      do { \
        rocm::hip_array<const IdxT*, 8> idx_ptrs; \
        for (int _i = 0; _i < (NIDX); ++_i) \
          idx_ptrs[_i] = reinterpret_cast<const IdxT*>(h_indices[_i]); \
        encoder.add_kernel_node( \
            (&rocm::scatter_general_kernel<T, IdxT, NIDX, RT>), \
            dim3(num_blocks), dim3(block_size), 0, \
            gpu_ptr<T>(upd), gpu_ptr<T>(out), total, \
            p_upd_shape, p_upd_strides, upd_ndim_v, upd_post_idx_size, \
            p_out_shape, p_out_strides, out_ndim_v, \
            p_axes, idx_ptrs, \
            p_indices_shape, p_indices_strides, idx_ndim); \
      } while (0)

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
  }
}

void GatherAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() > 1);
  const auto& src = inputs[0];
  const auto& idx = inputs[1];

  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  out.set_data(mlx::core::rocm::malloc_async(out.nbytes(), encoder));
  if (out.size() == 0) {
    return;
  }

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
    encoder.add_kernel_node( \
        (&rocm::gather_axis_kernel<T, IdxT, NDIM, SrcC, IdxC>), \
        dim3(num_blocks), dim3(block_size), 0, \
        gpu_ptr<T>(src), gpu_ptr<IdxT>(idx), gpu_ptr<T>(out), \
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
    encoder.add_kernel_node( \
        (&rocm::scatter_axis_kernel<T, IdxT, IS_SUM, NDIM, UpdC, IdxC>), \
        dim3(num_blocks), dim3(block_size), 0, \
        gpu_ptr<T>(upd), gpu_ptr<IdxT>(idx), gpu_ptr<T>(out), \
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
  
  if (is_sum) {
    // atomic_add (device/atomic_ops.hpp) provides native atomicAdd for float32
    // and CAS-loop specializations for float16/bfloat16, so the Sum reduction —
    // the gradient op for embedding lookups and MoE expert gathers — covers the
    // half-precision dtypes too.
    switch (upd.dtype()) {
      case float32: DISPATCH_IDX_TYPE(float, true); break;
      case float16: DISPATCH_IDX_TYPE(__half, true); break;
      case bfloat16: DISPATCH_IDX_TYPE(hip_bfloat16, true); break;
      default:
        throw std::runtime_error("Unsupported dtype for ScatterAxis Sum (only float32/float16/bfloat16 supported)");
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

  #undef LAUNCH_SCATTER_KERNEL
  #undef DISPATCH_CONTIGUOUS
  #undef DISPATCH_NDIM
  #undef DISPATCH_IDX_TYPE
}

void SliceUpdate::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  if (out.size() == 0) {
    return;
  }

  auto& in = inputs[0];
  auto& upd = inputs[1];

  if (upd.size() == 0) {
    out.copy_shared_buffer(in);
    return;
  }

  // Donate the input buffer when uniquely owned, else copy.
  bool can_donate = in.data_shared_ptr() != nullptr &&
      in.data_shared_ptr().use_count() == 1 && in.flags().contiguous &&
      in.data_size() == in.size();
  if (can_donate) {
    out.copy_shared_buffer(in);
  } else {
    auto ctype = in.flags().contiguous && in.size() == in.data_size()
        ? CopyType::Vector
        : CopyType::General;
    copy_gpu(in, out, in.data_size() == 1 ? CopyType::Scalar : ctype, stream());
  }

  // Calculate out strides, initial offset
  auto [data_offset, out_strides] =
      prepare_slice(out, start_indices_, strides_);

  // Do copy for None reduce type
  if (reduce_type_ == SliceUpdate::None) {
    copy_gpu_inplace(
        /* const array& src = */ upd,
        /* array& dst = */ out,
        /* const Shape& data_shape = */ upd.shape(),
        /* const Strides& i_strides = */ upd.strides(),
        /* const Strides& o_strides = */ out_strides,
        /* int64_t i_offset = */ 0,
        /* int64_t o_offset = */ data_offset,
        /* CopyType ctype = */ CopyType::GeneralGeneral,
        /* const Stream& s = */ stream());
    return;
  }

  // For reduce types (Sum/Prod/Max/Min), launch a kernel
  auto [shape, strides] =
      collapse_contiguous_dims(upd.shape(), {upd.strides(), out_strides});
  int nwork = 1;
  if (shape.back() % 4 == 0) {
    nwork = 4;
  } else if (shape.back() % 2 == 0) {
    nwork = 2;
  }

  auto [ds, rc, cc] = check_contiguity(shape, strides[1]);
  bool upd_contiguous = upd.flags().row_contiguous;
  bool upd_scalar = upd.data_size() == 1;
  bool out_contiguous = rc;

  int ndim = shape.size();

  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  encoder.set_input_array(upd);
  encoder.set_output_array(out);

  auto shape_param = const_param(shape);
  auto upd_strides_param = const_param(strides[0]);
  auto out_strides_param = const_param(strides[1]);

  int64_t update_size = upd.size();
  int block_size = 256;
  int64_t adjusted_size = (update_size + nwork - 1) / nwork;
  int num_blocks = static_cast<int>(
      std::min((adjusted_size + block_size - 1) / block_size, (int64_t)65535));

  // Plain local: a structured binding (data_offset) cannot be captured by the
  // kernel-launch macro under C++17.
  int64_t data_offset_v = data_offset;

  #define SLICE_UPDATE_LAUNCH(T, Op, OUT_C, UPD_C, UPD_S, NWORK_VAL) \
    encoder.add_kernel_node( \
        (&rocm::slice_update_op_kernel<T, int64_t, Op, OUT_C, UPD_C, UPD_S, NWORK_VAL>), \
        dim3(num_blocks), dim3(block_size), 0, \
        gpu_ptr<T>(upd), gpu_ptr<T>(out), update_size, \
        shape_param, upd_strides_param, ndim, \
        out_strides_param, data_offset_v)

  // Dispatch helper for NWORK
  #define DISPATCH_NWORK(T, Op, OUT_C, UPD_C, UPD_S) \
    switch (nwork) { \
      case 4: SLICE_UPDATE_LAUNCH(T, Op, OUT_C, UPD_C, UPD_S, 4); break; \
      case 2: SLICE_UPDATE_LAUNCH(T, Op, OUT_C, UPD_C, UPD_S, 2); break; \
      default: SLICE_UPDATE_LAUNCH(T, Op, OUT_C, UPD_C, UPD_S, 1); break; \
    }

  // Dispatch helper for contiguity flags
  #define DISPATCH_CONTIG(T, Op) \
    if (upd_scalar) { \
      if (out_contiguous) { \
        DISPATCH_NWORK(T, Op, true, false, true); \
      } else { \
        DISPATCH_NWORK(T, Op, false, false, true); \
      } \
    } else if (upd_contiguous && out_contiguous) { \
      DISPATCH_NWORK(T, Op, true, true, false); \
    } else if (upd_contiguous) { \
      DISPATCH_NWORK(T, Op, false, true, false); \
    } else if (out_contiguous) { \
      DISPATCH_NWORK(T, Op, true, false, false); \
    } else { \
      DISPATCH_NWORK(T, Op, false, false, false); \
    }

  // Dispatch helper for reduce type
  #define DISPATCH_SLICE_OP(T) \
    switch (reduce_type_) { \
      case SliceUpdate::Max: DISPATCH_CONTIG(T, rocm::Maximum); break; \
      case SliceUpdate::Min: DISPATCH_CONTIG(T, rocm::Minimum); break; \
      case SliceUpdate::Sum: DISPATCH_CONTIG(T, rocm::Add); break; \
      case SliceUpdate::Prod: DISPATCH_CONTIG(T, rocm::Multiply); break; \
      default: \
        throw std::runtime_error("SliceUpdate: unsupported reduce type"); \
    }

  switch (out.dtype()) {
    case float32: DISPATCH_SLICE_OP(float); break;
    case float16: DISPATCH_SLICE_OP(__half); break;
    case bfloat16: DISPATCH_SLICE_OP(hip_bfloat16); break;
    case int32: DISPATCH_SLICE_OP(int32_t); break;
    case int64: DISPATCH_SLICE_OP(int64_t); break;
    case uint32: DISPATCH_SLICE_OP(uint32_t); break;
    case uint64: DISPATCH_SLICE_OP(uint64_t); break;
    case int8: DISPATCH_SLICE_OP(int8_t); break;
    case int16: DISPATCH_SLICE_OP(int16_t); break;
    case uint8: DISPATCH_SLICE_OP(uint8_t); break;
    case uint16: DISPATCH_SLICE_OP(uint16_t); break;
    case bool_: DISPATCH_SLICE_OP(bool); break;
    default:
      throw std::runtime_error("Unsupported dtype for SliceUpdate");
  }

  #undef DISPATCH_SLICE_OP
  #undef DISPATCH_CONTIG
  #undef DISPATCH_NWORK
  #undef SLICE_UPDATE_LAUNCH
}

void MaskedScatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 3);

  const auto& dst = inputs[0];
  const auto& mask = inputs[1];
  const auto& src = inputs[2];

  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  const int64_t total = mask.size();
  const CopyType copy_type = (total == 1)
      ? CopyType::Scalar
      : (dst.flags().row_contiguous ? CopyType::Vector : CopyType::General);
  copy_gpu(dst, out, copy_type, s);
  if (total == 0) {
    return;
  }

  array mask_flat = flatten_in_eval(mask, 1, -1, s);
  if (mask_flat.data<void>() != mask.data<void>()) {
    encoder.add_temporary(mask_flat);
  }
  if (!mask_flat.flags().row_contiguous) {
    mask_flat = contiguous_copy_gpu(mask_flat, s);
    encoder.add_temporary(mask_flat);
  }

  array scatter_offsets(mask_flat.shape(), uint32, nullptr, {});
  scatter_offsets.set_data(mlx::core::rocm::malloc_async(scatter_offsets.nbytes(), encoder));
  encoder.add_temporary(scatter_offsets);

  const int64_t batch_count = mask_flat.shape(0);
  const int64_t mask_batch_size = total / batch_count;
  const int64_t src_batch_size = src.size() / batch_count;

  std::vector<int32_t> src_shape(src.shape().begin(), src.shape().end());
  std::vector<int64_t> src_strides(src.strides().begin(), src.strides().end());
  auto src_shape_param = const_param(src_shape);
  auto src_strides_param = const_param(src_strides);
  const bool src_contiguous = src.flags().row_contiguous;

  constexpr int block_size = 256;
  const auto offset_grid = dim3(static_cast<uint32_t>(batch_count));
  const auto offset_block = dim3(block_size);
  const int64_t num_blocks = (total + block_size - 1) / block_size;
  const int32_t src_ndim_v = static_cast<int32_t>(src.ndim());

  // Offsets kernel: writes scatter_offsets (registered as output so the
  // following assign kernel records a graph dependency on it).
  encoder.set_input_array(mask_flat);
  encoder.set_output_array(scatter_offsets);
  encoder.add_kernel_node(
      &rocm::masked_scatter_offsets_kernel<block_size>,
      offset_grid,
      offset_block,
      0,
      gpu_ptr<bool>(mask_flat),
      gpu_ptr<uint32_t>(scatter_offsets),
      mask_batch_size);

  // Assign kernel: reads mask_flat, scatter_offsets, src; writes out.
  encoder.set_input_array(mask_flat);
  encoder.set_input_array(scatter_offsets);
  encoder.set_input_array(src);
  encoder.set_output_array(out);

#define LAUNCH_MASKED_SCATTER(T, SrcC) \
  encoder.add_kernel_node( \
      (&rocm::masked_scatter_assign_kernel<T, SrcC>), \
      dim3(static_cast<uint32_t>(num_blocks)), \
      dim3(block_size), \
      0, \
      gpu_ptr<bool>(mask_flat), \
      gpu_ptr<uint32_t>(scatter_offsets), \
      gpu_ptr<T>(src), \
      gpu_ptr<T>(out), \
      total, \
      src_shape_param, \
      src_strides_param, \
      src_ndim_v, \
      src_batch_size, \
      mask_batch_size)

#define DISPATCH_MASKED_SCATTER(T)         \
  if (src_contiguous) {                    \
    LAUNCH_MASKED_SCATTER(T, true);        \
  } else {                                 \
    LAUNCH_MASKED_SCATTER(T, false);       \
  }

  switch (out.dtype()) {
    case bool_:
      DISPATCH_MASKED_SCATTER(bool);
      break;
    case uint8:
      DISPATCH_MASKED_SCATTER(uint8_t);
      break;
    case uint16:
      DISPATCH_MASKED_SCATTER(uint16_t);
      break;
    case uint32:
      DISPATCH_MASKED_SCATTER(uint32_t);
      break;
    case uint64:
      DISPATCH_MASKED_SCATTER(uint64_t);
      break;
    case int8:
      DISPATCH_MASKED_SCATTER(int8_t);
      break;
    case int16:
      DISPATCH_MASKED_SCATTER(int16_t);
      break;
    case int32:
      DISPATCH_MASKED_SCATTER(int32_t);
      break;
    case int64:
      DISPATCH_MASKED_SCATTER(int64_t);
      break;
    case float16:
      DISPATCH_MASKED_SCATTER(__half);
      break;
    case float32:
      DISPATCH_MASKED_SCATTER(float);
      break;
    case float64:
      DISPATCH_MASKED_SCATTER(double);
      break;
    case bfloat16:
      DISPATCH_MASKED_SCATTER(hip_bfloat16);
      break;
    case complex64:
      DISPATCH_MASKED_SCATTER(hipFloatComplex);
      break;
    default:
      throw std::runtime_error("Unsupported dtype for MaskedScatter");
  }

#undef DISPATCH_MASKED_SCATTER
#undef LAUNCH_MASKED_SCATTER
}

// --- Device-position scalar for capture-once HIP-graph decode ---
// The decode position lives in a fixed-address [1] int32 device buffer. The
// captured graph bakes that address; RoPE / causal-mask / KV-write read it
// device-side, so they advance every replay. The position is advanced IN PLACE
// (no realloc -> address stays valid) by these raw kernels, loop-owned (called
// between replays, never inside the captured graph — an in-graph pos++ races the
// readers as a write-after-read hazard).
__global__ void _kv_pos_inc(int* p, int delta) { p[0] += delta; }
void gpu_kv_pos_increment(array& pos, int delta) {
  auto& enc = rocm::get_command_encoder(default_stream(default_device()));
  int* p = gpu_ptr<int>(pos);
  enc.set_input_array(pos);
  enc.set_output_array(pos);
  enc.launch_kernel([p, delta](hipStream_t s) {
    hipLaunchKernelGGL(_kv_pos_inc, dim3(1), dim3(1), 0, s, p, delta);
  });
}

__global__ void _kv_pos_set(int* p, int v) { p[0] = v; }
void gpu_kv_pos_set(array& pos, int v) {
  auto& enc = rocm::get_command_encoder(default_stream(default_device()));
  int* p = gpu_ptr<int>(pos);
  enc.set_output_array(pos);
  enc.launch_kernel([p, v](hipStream_t s) {
    hipLaunchKernelGGL(_kv_pos_set, dim3(1), dim3(1), 0, s, p, v);
  });
}

// Device->device int32 scalar copy. Feeds the freshly-sampled token (on a
// transient arena address) into the fixed-address graph-decode input buffer so
// the recorded graph's embedding gather reads the new token without the buffer
// being reallocated. Runs on-stream, so it orders after the producing forward.
__global__ void _scalar_copy_i32(int* dst, const int* src) { dst[0] = src[0]; }
void gpu_scalar_copy_i32(array& dst, array& src) {
  auto& enc = rocm::get_command_encoder(default_stream(default_device()));
  int* d = gpu_ptr<int>(dst);
  int* s = gpu_ptr<int>(src);
  enc.set_input_array(src);
  enc.set_output_array(dst);
  enc.launch_kernel([d, s](hipStream_t st) {
    hipLaunchKernelGGL(_scalar_copy_i32, dim3(1), dim3(1), 0, st, d, s);
  });
}

// Immediate (graph-split) device buffer copy src->dst. Used by the decode loop
// to move GDN scratch state into the read buffer between relaunches.
void gpu_buffer_copy(array& dst, array& src) {
  auto& enc = rocm::get_command_encoder(default_stream(default_device()));
  char* d = gpu_ptr<char>(dst);
  char* s = gpu_ptr<char>(src);
  size_t bytes = src.nbytes();
  enc.set_input_array(src);
  enc.set_output_array(dst);
  enc.launch_kernel([d, s, bytes](hipStream_t st) {
    (void)hipMemcpyAsync(d, s, bytes, hipMemcpyDeviceToDevice, st);
  });
}

} // namespace mlx::core
