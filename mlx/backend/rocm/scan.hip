// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/binary_ops.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>

#include <cassert>

namespace mlx::core {

namespace rocm {

// Scan operations
struct ScanSum {
  template <typename T>
  __device__ T operator()(T a, T b) const { return a + b; }
};

struct ScanProd {
  template <typename T>
  __device__ T operator()(T a, T b) const { return a * b; }
};

struct ScanMax {
  template <typename T>
  __device__ T operator()(T a, T b) const { return a > b ? a : b; }
};

struct ScanMin {
  template <typename T>
  __device__ T operator()(T a, T b) const { return a < b ? a : b; }
};

// Get initial value for scan operation
template <typename Op, typename T>
__device__ T scan_init();

template <>
__device__ float scan_init<ScanSum, float>() { return 0.0f; }

template <>
__device__ float scan_init<ScanProd, float>() { return 1.0f; }

template <>
__device__ float scan_init<ScanMax, float>() { return -1e38f; }

template <>
__device__ float scan_init<ScanMin, float>() { return 1e38f; }

template <>
__device__ int32_t scan_init<ScanSum, int32_t>() { return 0; }

template <>
__device__ int32_t scan_init<ScanProd, int32_t>() { return 1; }

template <>
__device__ int32_t scan_init<ScanMax, int32_t>() { return INT32_MIN; }

template <>
__device__ int32_t scan_init<ScanMin, int32_t>() { return INT32_MAX; }

// Warp scan using shuffle
template <typename T, typename Op>
__device__ T warp_scan_inclusive(T val, Op op) {
  for (int offset = 1; offset < 64; offset *= 2) {
    T other = __shfl_up(val, offset);
    if (threadIdx.x % 64 >= offset) {
      val = op(val, other);
    }
  }
  return val;
}

template <typename T, typename Op>
__device__ T warp_scan_exclusive(T val, Op op, T init) {
  T inclusive = warp_scan_inclusive(val, op);
  T exclusive = __shfl_up(inclusive, 1);
  return (threadIdx.x % 64 == 0) ? init : exclusive;
}

// Simple contiguous scan kernel
template <typename T, typename Op, bool inclusive, bool reverse>
__global__ void contiguous_scan_kernel(
    const T* in,
    T* out,
    int32_t axis_size,
    T init) {
  int row = blockIdx.x;
  in += row * axis_size;
  out += row * axis_size;
  
  Op op;
  
  __shared__ T shared[1024];  // Shared memory for block scan
  
  T prefix = init;
  
  // Process in chunks
  for (int base = 0; base < axis_size; base += blockDim.x) {
    int idx = base + threadIdx.x;
    int actual_idx = reverse ? (axis_size - 1 - idx) : idx;
    
    T val = (idx < axis_size) ? in[actual_idx] : init;
    
    // Warp-level inclusive scan
    T scanned = warp_scan_inclusive(val, op);
    
    // Store warp results
    int lane = threadIdx.x % 64;
    int warp_id = threadIdx.x / 64;
    
    __shared__ T warp_sums[16];  // Max 16 warps
    
    if (lane == 63) {
      warp_sums[warp_id] = scanned;
    }
    __syncthreads();
    
    // Scan warp sums in first warp
    if (warp_id == 0 && lane < (blockDim.x + 63) / 64) {
      T warp_val = warp_sums[lane];
      T warp_scanned = warp_scan_exclusive(warp_val, op, init);
      warp_sums[lane] = warp_scanned;
    }
    __syncthreads();
    
    // Add warp prefix and global prefix
    T warp_prefix = warp_sums[warp_id];
    
    if (inclusive) {
      scanned = op(scanned, warp_prefix);
      scanned = op(scanned, prefix);
    } else {
      T excl = warp_scan_exclusive(val, op, init);
      excl = op(excl, warp_prefix);
      excl = op(excl, prefix);
      scanned = excl;
    }
    
    // Write output
    if (idx < axis_size) {
      out[actual_idx] = scanned;
    }
    
    // Update prefix for next chunk
    __syncthreads();
    if (threadIdx.x == blockDim.x - 1 || base + blockDim.x > axis_size) {
      int last_idx = min(base + (int)blockDim.x - 1, axis_size - 1) - base;
      if (threadIdx.x == last_idx) {
        if (inclusive) {
          warp_sums[0] = scanned;
        } else {
          warp_sums[0] = op(scanned, val);
        }
      }
    }
    __syncthreads();
    prefix = warp_sums[0];
  }
}

} // namespace rocm

void Scan::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto in = inputs[0];
  auto& s = stream();

  if (in.flags().contiguous && in.strides()[axis_] != 0) {
    if (in.is_donatable() && in.itemsize() == out.itemsize()) {
      out.copy_shared_buffer(in);
    } else {
      out.set_data(
          allocator::malloc(in.data_size() * out.itemsize()),
          in.data_size(),
          in.strides(),
          in.flags());
    }
  } else {
    in = contiguous_copy_gpu(in, s);
    out.copy_shared_buffer(in);
  }

  int32_t axis_size = in.shape(axis_);
  bool contiguous = in.strides()[axis_] == 1;
  
  if (!contiguous) {
    throw std::runtime_error("Non-contiguous scan not yet implemented for ROCm");
  }

  auto& encoder = rocm::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  
  int n_rows = in.data_size() / axis_size;
  int block_size = std::min(256, ((axis_size + 63) / 64) * 64);
  block_size = std::max(block_size, 64);
  
  encoder.launch_kernel([&](hipStream_t stream) {
    switch (in.dtype()) {
      case float32: {
        float init;
        switch (reduce_type_) {
          case Scan::Sum: init = 0.0f; break;
          case Scan::Prod: init = 1.0f; break;
          case Scan::Max: init = -1e38f; break;
          case Scan::Min: init = 1e38f; break;
          default: throw std::runtime_error("Unsupported scan op");
        }
        
        if (reduce_type_ == Scan::Sum) {
          if (inclusive_) {
            if (reverse_) {
              hipLaunchKernelGGL(
                  (rocm::contiguous_scan_kernel<float, rocm::ScanSum, true, true>),
                  dim3(n_rows), dim3(block_size), 0, stream,
                  in.data<float>(), out.data<float>(), axis_size, init);
            } else {
              hipLaunchKernelGGL(
                  (rocm::contiguous_scan_kernel<float, rocm::ScanSum, true, false>),
                  dim3(n_rows), dim3(block_size), 0, stream,
                  in.data<float>(), out.data<float>(), axis_size, init);
            }
          } else {
            if (reverse_) {
              hipLaunchKernelGGL(
                  (rocm::contiguous_scan_kernel<float, rocm::ScanSum, false, true>),
                  dim3(n_rows), dim3(block_size), 0, stream,
                  in.data<float>(), out.data<float>(), axis_size, init);
            } else {
              hipLaunchKernelGGL(
                  (rocm::contiguous_scan_kernel<float, rocm::ScanSum, false, false>),
                  dim3(n_rows), dim3(block_size), 0, stream,
                  in.data<float>(), out.data<float>(), axis_size, init);
            }
          }
        } else if (reduce_type_ == Scan::Max) {
          if (inclusive_ && !reverse_) {
            hipLaunchKernelGGL(
                (rocm::contiguous_scan_kernel<float, rocm::ScanMax, true, false>),
                dim3(n_rows), dim3(block_size), 0, stream,
                in.data<float>(), out.data<float>(), axis_size, init);
          } else {
            throw std::runtime_error("Max scan variant not implemented");
          }
        } else if (reduce_type_ == Scan::Min) {
          if (inclusive_ && !reverse_) {
            hipLaunchKernelGGL(
                (rocm::contiguous_scan_kernel<float, rocm::ScanMin, true, false>),
                dim3(n_rows), dim3(block_size), 0, stream,
                in.data<float>(), out.data<float>(), axis_size, init);
          } else {
            throw std::runtime_error("Min scan variant not implemented");
          }
        } else if (reduce_type_ == Scan::Prod) {
          if (inclusive_ && !reverse_) {
            hipLaunchKernelGGL(
                (rocm::contiguous_scan_kernel<float, rocm::ScanProd, true, false>),
                dim3(n_rows), dim3(block_size), 0, stream,
                in.data<float>(), out.data<float>(), axis_size, init);
          } else {
            throw std::runtime_error("Prod scan variant not implemented");
          }
        }
        break;
      }
      case int32: {
        int32_t init;
        switch (reduce_type_) {
          case Scan::Sum: init = 0; break;
          case Scan::Prod: init = 1; break;
          case Scan::Max: init = INT32_MIN; break;
          case Scan::Min: init = INT32_MAX; break;
          default: throw std::runtime_error("Unsupported scan op");
        }
        
        if (reduce_type_ == Scan::Sum && inclusive_ && !reverse_) {
          hipLaunchKernelGGL(
              (rocm::contiguous_scan_kernel<int32_t, rocm::ScanSum, true, false>),
              dim3(n_rows), dim3(block_size), 0, stream,
              in.data<int32_t>(), out.data<int32_t>(), axis_size, init);
        } else {
          throw std::runtime_error("Int32 scan variant not implemented");
        }
        break;
      }
      default:
        throw std::runtime_error("Unsupported type for scan");
    }
  });
}

} // namespace mlx::core
