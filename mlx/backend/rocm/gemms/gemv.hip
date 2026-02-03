// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/config.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/gemms/gemv.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

namespace mlx::core {

namespace rocm {

constexpr int GEMV_BLOCK_SIZE = 256;
constexpr int GEMV_TILE_SIZE = 4;

// WARP_SIZE is defined in device/config.h based on target architecture

template <typename T, bool TransA>
__global__ void gemv_kernel(
    const T* __restrict__ A,
    const T* __restrict__ x,
    T* __restrict__ y,
    int M,
    int N,
    int lda,
    T alpha,
    T beta) {
  __shared__ T shared_x[GEMV_BLOCK_SIZE];
  
  int row = blockIdx.x;
  if (row >= M) return;
  
  T acc = T(0);
  
  if constexpr (TransA) {
    // A is transposed: y = alpha * A^T * x + beta * y
    // Each block handles one column of A^T (one row of A)
    for (int tile = 0; tile < (N + GEMV_BLOCK_SIZE - 1) / GEMV_BLOCK_SIZE; ++tile) {
      int col = tile * GEMV_BLOCK_SIZE + threadIdx.x;
      if (col < N) {
        shared_x[threadIdx.x] = x[col];
      } else {
        shared_x[threadIdx.x] = T(0);
      }
      __syncthreads();
      
      #pragma unroll
      for (int i = 0; i < GEMV_BLOCK_SIZE && (tile * GEMV_BLOCK_SIZE + i) < N; ++i) {
        int col_idx = tile * GEMV_BLOCK_SIZE + i;
        acc += A[col_idx * lda + row] * shared_x[i];
      }
      __syncthreads();
    }
  } else {
    // A is not transposed: y = alpha * A * x + beta * y
    // Each block handles one row of A
    for (int tile = 0; tile < (N + GEMV_BLOCK_SIZE - 1) / GEMV_BLOCK_SIZE; ++tile) {
      int col = tile * GEMV_BLOCK_SIZE + threadIdx.x;
      if (col < N) {
        shared_x[threadIdx.x] = x[col];
      } else {
        shared_x[threadIdx.x] = T(0);
      }
      __syncthreads();
      
      #pragma unroll
      for (int i = 0; i < GEMV_BLOCK_SIZE && (tile * GEMV_BLOCK_SIZE + i) < N; ++i) {
        int col_idx = tile * GEMV_BLOCK_SIZE + i;
        acc += A[row * lda + col_idx] * shared_x[i];
      }
      __syncthreads();
    }
  }
  
  // Only first thread writes result
  if (threadIdx.x == 0) {
    if (beta == T(0)) {
      y[row] = alpha * acc;
    } else {
      y[row] = alpha * acc + beta * y[row];
    }
  }
}

// Optimized GEMV using warp reduction
template <typename T, bool TransA>
__global__ void gemv_warp_kernel(
    const T* __restrict__ A,
    const T* __restrict__ x,
    T* __restrict__ y,
    int M,
    int N,
    int lda,
    T alpha,
    T beta) {
  int row = blockIdx.x;
  if (row >= M) return;
  
  T acc = T(0);
  
  // Each thread processes multiple elements
  for (int col = threadIdx.x; col < N; col += blockDim.x) {
    T a_val;
    if constexpr (TransA) {
      a_val = A[col * lda + row];
    } else {
      a_val = A[row * lda + col];
    }
    acc += a_val * x[col];
  }
  
  // Warp reduction
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    acc += __shfl_down(acc, offset);
  }
  
  // Block reduction using shared memory
  __shared__ T shared_acc[32];
  int lane = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;
  
  if (lane == 0) {
    shared_acc[warp_id] = acc;
  }
  __syncthreads();
  
  // Final reduction by first warp
  int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
  if (warp_id == 0) {
    acc = (lane < num_warps) ? shared_acc[lane] : T(0);
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      acc += __shfl_down(acc, offset);
    }
    
    if (lane == 0) {
      if (beta == T(0)) {
        y[row] = alpha * acc;
      } else {
        y[row] = alpha * acc + beta * y[row];
      }
    }
  }
}

// Gather-based GEMV kernel
template <typename T>
__global__ void gemv_gather_kernel(
    const T* __restrict__ mat,
    const T* __restrict__ vec,
    const uint32_t* __restrict__ mat_indices,
    const uint32_t* __restrict__ vec_indices,
    T* __restrict__ out,
    int M,
    int K,
    int mat_ld,
    int batch_size) {
  int batch_idx = blockIdx.x;
  if (batch_idx >= batch_size) return;
  
  uint32_t mat_idx = mat_indices[batch_idx];
  uint32_t vec_idx = vec_indices[batch_idx];
  
  const T* mat_ptr = mat + mat_idx * M * K;
  const T* vec_ptr = vec + vec_idx * K;
  T* out_ptr = out + batch_idx * M;
  
  // Each block processes one batch, threads process M outputs
  for (int row = threadIdx.x; row < M; row += blockDim.x) {
    T acc = T(0);
    for (int k = 0; k < K; ++k) {
      acc += mat_ptr[row * mat_ld + k] * vec_ptr[k];
    }
    out_ptr[row] = acc;
  }
}

} // namespace rocm

bool can_use_gemv(int M, int N, int K, bool trans_a, bool trans_b) {
  // Simple heuristic for when to use GEMV
  return (M == 1 || N == 1) && K <= 8192;
}

void gather_mv(
    const array& mat,
    const array& vec,
    const array& mat_indices,
    const array& vec_indices,
    array& out,
    int M,
    int K,
    rocm::CommandEncoder& encoder) {
  
  int batch_size = mat_indices.size();
  int threads = std::min(256, M);
  
  encoder.set_input_array(mat);
  encoder.set_input_array(vec);
  encoder.set_input_array(mat_indices);
  encoder.set_input_array(vec_indices);
  encoder.set_output_array(out);
  
  encoder.launch_kernel([&](hipStream_t stream) {
    switch (mat.dtype()) {
      case float32:
        hipLaunchKernelGGL(
            (rocm::gemv_gather_kernel<float>),
            dim3(batch_size), dim3(threads), 0, stream,
            mat.data<float>(), vec.data<float>(),
            mat_indices.data<uint32_t>(), vec_indices.data<uint32_t>(),
            out.data<float>(), M, K, K, batch_size);
        break;
      case float16:
        hipLaunchKernelGGL(
            (rocm::gemv_gather_kernel<__half>),
            dim3(batch_size), dim3(threads), 0, stream,
            mat.data<__half>(), vec.data<__half>(),
            mat_indices.data<uint32_t>(), vec_indices.data<uint32_t>(),
            out.data<__half>(), M, K, K, batch_size);
        break;
      case bfloat16:
        hipLaunchKernelGGL(
            (rocm::gemv_gather_kernel<hip_bfloat16>),
            dim3(batch_size), dim3(threads), 0, stream,
            mat.data<hip_bfloat16>(), vec.data<hip_bfloat16>(),
            mat_indices.data<uint32_t>(), vec_indices.data<uint32_t>(),
            out.data<hip_bfloat16>(), M, K, K, batch_size);
        break;
      default:
        throw std::runtime_error("Unsupported dtype for gather_mv");
    }
  });
}

void gemv(
    rocm::CommandEncoder& encoder,
    bool transpose_a,
    int M,
    int N,
    float alpha,
    const array& a,
    int lda,
    const array& x,
    float beta,
    array& y,
    Dtype dtype) {
  
  int threads = std::min(256, ((N + 63) / 64) * 64);
  threads = std::max(threads, 64);
  
  encoder.launch_kernel([&](hipStream_t stream) {
    switch (dtype) {
      case float32:
        if (transpose_a) {
          hipLaunchKernelGGL(
              (rocm::gemv_warp_kernel<float, true>),
              dim3(M), dim3(threads), 0, stream,
              a.data<float>(), x.data<float>(), y.data<float>(),
              M, N, lda, alpha, beta);
        } else {
          hipLaunchKernelGGL(
              (rocm::gemv_warp_kernel<float, false>),
              dim3(M), dim3(threads), 0, stream,
              a.data<float>(), x.data<float>(), y.data<float>(),
              M, N, lda, alpha, beta);
        }
        break;
      case float16:
        if (transpose_a) {
          hipLaunchKernelGGL(
              (rocm::gemv_warp_kernel<__half, true>),
              dim3(M), dim3(threads), 0, stream,
              a.data<__half>(), x.data<__half>(), y.data<__half>(),
              M, N, lda, __float2half(alpha), __float2half(beta));
        } else {
          hipLaunchKernelGGL(
              (rocm::gemv_warp_kernel<__half, false>),
              dim3(M), dim3(threads), 0, stream,
              a.data<__half>(), x.data<__half>(), y.data<__half>(),
              M, N, lda, __float2half(alpha), __float2half(beta));
        }
        break;
      default:
        throw std::runtime_error("Unsupported dtype for GEMV");
    }
  });
}

} // namespace mlx::core
