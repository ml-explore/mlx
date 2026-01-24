// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/utils.hpp"
#include "mlx/backend/common/reduce.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

// Reduce operations
struct ReduceSum {
  template <typename T>
  __device__ T operator()(T a, T b) const { return a + b; }
  
  template <typename T>
  __device__ T init() const { return T(0); }
};

struct ReduceProd {
  template <typename T>
  __device__ T operator()(T a, T b) const { return a * b; }
  
  template <typename T>
  __device__ T init() const { return T(1); }
};

struct ReduceMax {
  template <typename T>
  __device__ T operator()(T a, T b) const { return a > b ? a : b; }
  
  template <typename T>
  __device__ T init() const { return numeric_limits<T>::lowest(); }
};

struct ReduceMin {
  template <typename T>
  __device__ T operator()(T a, T b) const { return a < b ? a : b; }
  
  template <typename T>
  __device__ T init() const { return numeric_limits<T>::max(); }
};

struct ReduceAnd {
  __device__ bool operator()(bool a, bool b) const { return a && b; }
  __device__ bool init() const { return true; }
};

struct ReduceOr {
  __device__ bool operator()(bool a, bool b) const { return a || b; }
  __device__ bool init() const { return false; }
};

// Warp-level reduction using shuffle
template <typename T, typename Op>
__device__ T warp_reduce(T val, Op op) {
  constexpr int warp_size = 64;  // AMD wavefront size
  for (int offset = warp_size / 2; offset > 0; offset /= 2) {
    val = op(val, __shfl_xor(val, offset));
  }
  return val;
}

// Block-level reduction
template <typename T, typename Op, int BLOCK_SIZE>
__device__ T block_reduce(T val, Op op) {
  __shared__ T shared[BLOCK_SIZE / 64];  // One slot per warp
  
  int lane = threadIdx.x % 64;
  int warp_id = threadIdx.x / 64;
  
  // Warp-level reduction
  val = warp_reduce(val, op);
  
  // Write reduced value to shared memory
  if (lane == 0) {
    shared[warp_id] = val;
  }
  __syncthreads();
  
  // Final reduction in first warp
  if (warp_id == 0) {
    val = (lane < BLOCK_SIZE / 64) ? shared[lane] : op.template init<T>();
    val = warp_reduce(val, op);
  }
  
  return val;
}

// All reduce kernel - reduces entire input to single value
template <typename T, typename Op, typename IdxT>
__global__ void all_reduce_kernel(
    const T* input,
    T* output,
    IdxT size,
    Op op) {
  constexpr int BLOCK_SIZE = 256;
  
  __shared__ T shared[BLOCK_SIZE / 64];
  
  T val = op.template init<T>();
  
  // Grid-stride loop
  IdxT idx = blockIdx.x * blockDim.x + threadIdx.x;
  IdxT stride = blockDim.x * gridDim.x;
  
  for (IdxT i = idx; i < size; i += stride) {
    val = op(val, input[i]);
  }
  
  // Block reduction
  int lane = threadIdx.x % 64;
  int warp_id = threadIdx.x / 64;
  
  val = warp_reduce(val, op);
  
  if (lane == 0) {
    shared[warp_id] = val;
  }
  __syncthreads();
  
  if (warp_id == 0) {
    val = (lane < BLOCK_SIZE / 64) ? shared[lane] : op.template init<T>();
    val = warp_reduce(val, op);
    
    if (lane == 0) {
      atomicAdd(output, val);  // Atomic accumulation across blocks
    }
  }
}

// Row reduce kernel - reduces along last dimension
template <typename T, typename Op, typename IdxT>
__global__ void row_reduce_kernel(
    const T* input,
    T* output,
    IdxT reduce_size,
    IdxT out_size,
    Op op) {
  IdxT out_idx = blockIdx.x;
  if (out_idx >= out_size) return;
  
  T val = op.template init<T>();
  
  // Each thread reduces multiple elements
  for (IdxT i = threadIdx.x; i < reduce_size; i += blockDim.x) {
    val = op(val, input[out_idx * reduce_size + i]);
  }
  
  // Block reduction
  constexpr int BLOCK_SIZE = 256;
  __shared__ T shared[BLOCK_SIZE / 64];
  
  int lane = threadIdx.x % 64;
  int warp_id = threadIdx.x / 64;
  
  val = warp_reduce(val, op);
  
  if (lane == 0) {
    shared[warp_id] = val;
  }
  __syncthreads();
  
  if (warp_id == 0) {
    val = (lane < BLOCK_SIZE / 64) ? shared[lane] : op.template init<T>();
    val = warp_reduce(val, op);
    
    if (lane == 0) {
      output[out_idx] = val;
    }
  }
}

// Col reduce kernel - reduces along non-contiguous dimension
template <typename T, typename Op, typename IdxT>
__global__ void col_reduce_kernel(
    const T* input,
    T* output,
    IdxT reduce_size,
    IdxT reduce_stride,
    IdxT out_size,
    Op op) {
  IdxT out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (out_idx >= out_size) return;
  
  T val = op.template init<T>();
  
  // Reduce along strided dimension
  for (IdxT i = 0; i < reduce_size; ++i) {
    val = op(val, input[out_idx + i * reduce_stride]);
  }
  
  output[out_idx] = val;
}

} // namespace rocm

// Forward declarations
void init_reduce(
    rocm::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type);

void all_reduce(
    rocm::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type);

void row_reduce(
    rocm::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan);

void col_reduce(
    rocm::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan);

} // namespace mlx::core
