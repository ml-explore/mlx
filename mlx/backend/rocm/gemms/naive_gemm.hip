// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/gemms/naive_gemm.h"
#include "mlx/backend/rocm/device/config.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/dtype_utils.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

namespace mlx::core::rocm {

// Tile sizes for the naive GEMM kernel
static constexpr int TILE_M = 16;
static constexpr int TILE_N = 16;
static constexpr int TILE_K = 16;

// Accumulator type selection
template <typename T>
struct GemmAccType {
  using type = T;
};

template <>
struct GemmAccType<__half> {
  using type = float;
};

template <>
struct GemmAccType<hip_bfloat16> {
  using type = float;
};

// Naive GEMM kernel: C = alpha * A * B + beta * C
// A is M x K, B is K x N, C is M x N
// All matrices are row-major
template <typename T, bool TransA, bool TransB>
__global__ void naive_gemm_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    float alpha,
    float beta) {
  using Acc = typename GemmAccType<T>::type;
  
  int row = blockIdx.y * TILE_M + threadIdx.y;
  int col = blockIdx.x * TILE_N + threadIdx.x;
  
  if (row < M && col < N) {
    Acc sum = Acc(0);
    
    for (int k = 0; k < K; ++k) {
      Acc a_val, b_val;
      
      if constexpr (TransA) {
        a_val = static_cast<Acc>(A[k * lda + row]);
      } else {
        a_val = static_cast<Acc>(A[row * lda + k]);
      }
      
      if constexpr (TransB) {
        b_val = static_cast<Acc>(B[col * ldb + k]);
      } else {
        b_val = static_cast<Acc>(B[k * ldb + col]);
      }
      
      sum += a_val * b_val;
    }
    
    if (beta != 0.0f) {
      C[row * ldc + col] = static_cast<T>(alpha * sum + beta * static_cast<Acc>(C[row * ldc + col]));
    } else {
      C[row * ldc + col] = static_cast<T>(alpha * sum);
    }
  }
}

// Tiled GEMM kernel with shared memory for better performance
template <typename T, bool TransA, bool TransB>
__global__ void tiled_gemm_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    float alpha,
    float beta) {
  using Acc = typename GemmAccType<T>::type;
  
  __shared__ Acc As[TILE_M][TILE_K];
  __shared__ Acc Bs[TILE_K][TILE_N];
  
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  int row = by * TILE_M + ty;
  int col = bx * TILE_N + tx;
  
  Acc sum = Acc(0);
  
  // Loop over tiles
  for (int t = 0; t < (K + TILE_K - 1) / TILE_K; ++t) {
    // Load A tile into shared memory
    int a_col = t * TILE_K + tx;
    if (row < M && a_col < K) {
      if constexpr (TransA) {
        As[ty][tx] = static_cast<Acc>(A[a_col * lda + row]);
      } else {
        As[ty][tx] = static_cast<Acc>(A[row * lda + a_col]);
      }
    } else {
      As[ty][tx] = Acc(0);
    }
    
    // Load B tile into shared memory
    int b_row = t * TILE_K + ty;
    if (b_row < K && col < N) {
      if constexpr (TransB) {
        Bs[ty][tx] = static_cast<Acc>(B[col * ldb + b_row]);
      } else {
        Bs[ty][tx] = static_cast<Acc>(B[b_row * ldb + col]);
      }
    } else {
      Bs[ty][tx] = Acc(0);
    }
    
    __syncthreads();
    
    // Compute partial dot product
    #pragma unroll
    for (int k = 0; k < TILE_K; ++k) {
      sum += As[ty][k] * Bs[k][tx];
    }
    
    __syncthreads();
  }
  
  // Write result
  if (row < M && col < N) {
    if (beta != 0.0f) {
      C[row * ldc + col] = static_cast<T>(alpha * sum + beta * static_cast<Acc>(C[row * ldc + col]));
    } else {
      C[row * ldc + col] = static_cast<T>(alpha * sum);
    }
  }
}

// Batched GEMM kernel
template <typename T, bool TransA, bool TransB>
__global__ void batched_gemm_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    int64_t stride_a,
    int64_t stride_b,
    int64_t stride_c,
    float alpha,
    float beta) {
  using Acc = typename GemmAccType<T>::type;
  
  int batch = blockIdx.z;
  int row = blockIdx.y * TILE_M + threadIdx.y;
  int col = blockIdx.x * TILE_N + threadIdx.x;
  
  const T* A_batch = A + batch * stride_a;
  const T* B_batch = B + batch * stride_b;
  T* C_batch = C + batch * stride_c;
  
  if (row < M && col < N) {
    Acc sum = Acc(0);
    
    for (int k = 0; k < K; ++k) {
      Acc a_val, b_val;
      
      if constexpr (TransA) {
        a_val = static_cast<Acc>(A_batch[k * lda + row]);
      } else {
        a_val = static_cast<Acc>(A_batch[row * lda + k]);
      }
      
      if constexpr (TransB) {
        b_val = static_cast<Acc>(B_batch[col * ldb + k]);
      } else {
        b_val = static_cast<Acc>(B_batch[k * ldb + col]);
      }
      
      sum += a_val * b_val;
    }
    
    if (beta != 0.0f) {
      C_batch[row * ldc + col] = static_cast<T>(alpha * sum + beta * static_cast<Acc>(C_batch[row * ldc + col]));
    } else {
      C_batch[row * ldc + col] = static_cast<T>(alpha * sum);
    }
  }
}

template <typename T>
void launch_naive_gemm(
    hipStream_t stream,
    const T* A,
    const T* B,
    T* C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    bool trans_a,
    bool trans_b,
    float alpha,
    float beta) {
  dim3 block(TILE_N, TILE_M);
  dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
  
  // Use tiled kernel for larger matrices, naive for smaller ones
  bool use_tiled = (M >= 32 && N >= 32 && K >= 32);
  
  if (trans_a && trans_b) {
    if (use_tiled) {
      hipLaunchKernelGGL((tiled_gemm_kernel<T, true, true>),
          grid, block, 0, stream,
          A, B, C, M, N, K, lda, ldb, ldc, alpha, beta);
    } else {
      hipLaunchKernelGGL((naive_gemm_kernel<T, true, true>),
          grid, block, 0, stream,
          A, B, C, M, N, K, lda, ldb, ldc, alpha, beta);
    }
  } else if (trans_a && !trans_b) {
    if (use_tiled) {
      hipLaunchKernelGGL((tiled_gemm_kernel<T, true, false>),
          grid, block, 0, stream,
          A, B, C, M, N, K, lda, ldb, ldc, alpha, beta);
    } else {
      hipLaunchKernelGGL((naive_gemm_kernel<T, true, false>),
          grid, block, 0, stream,
          A, B, C, M, N, K, lda, ldb, ldc, alpha, beta);
    }
  } else if (!trans_a && trans_b) {
    if (use_tiled) {
      hipLaunchKernelGGL((tiled_gemm_kernel<T, false, true>),
          grid, block, 0, stream,
          A, B, C, M, N, K, lda, ldb, ldc, alpha, beta);
    } else {
      hipLaunchKernelGGL((naive_gemm_kernel<T, false, true>),
          grid, block, 0, stream,
          A, B, C, M, N, K, lda, ldb, ldc, alpha, beta);
    }
  } else {
    if (use_tiled) {
      hipLaunchKernelGGL((tiled_gemm_kernel<T, false, false>),
          grid, block, 0, stream,
          A, B, C, M, N, K, lda, ldb, ldc, alpha, beta);
    } else {
      hipLaunchKernelGGL((naive_gemm_kernel<T, false, false>),
          grid, block, 0, stream,
          A, B, C, M, N, K, lda, ldb, ldc, alpha, beta);
    }
  }
}

template <typename T>
void launch_batched_gemm(
    hipStream_t stream,
    const T* A,
    const T* B,
    T* C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    int64_t stride_a,
    int64_t stride_b,
    int64_t stride_c,
    int batch_count,
    bool trans_a,
    bool trans_b,
    float alpha,
    float beta) {
  dim3 block(TILE_N, TILE_M);
  dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M, batch_count);
  
  if (trans_a && trans_b) {
    hipLaunchKernelGGL((batched_gemm_kernel<T, true, true>),
        grid, block, 0, stream,
        A, B, C, M, N, K, lda, ldb, ldc, stride_a, stride_b, stride_c, alpha, beta);
  } else if (trans_a && !trans_b) {
    hipLaunchKernelGGL((batched_gemm_kernel<T, true, false>),
        grid, block, 0, stream,
        A, B, C, M, N, K, lda, ldb, ldc, stride_a, stride_b, stride_c, alpha, beta);
  } else if (!trans_a && trans_b) {
    hipLaunchKernelGGL((batched_gemm_kernel<T, false, true>),
        grid, block, 0, stream,
        A, B, C, M, N, K, lda, ldb, ldc, stride_a, stride_b, stride_c, alpha, beta);
  } else {
    hipLaunchKernelGGL((batched_gemm_kernel<T, false, false>),
        grid, block, 0, stream,
        A, B, C, M, N, K, lda, ldb, ldc, stride_a, stride_b, stride_c, alpha, beta);
  }
}

void naive_gemm(
    CommandEncoder& encoder,
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    bool b_transposed,
    int64_t ldb,
    float alpha,
    float beta) {
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);
  
  int ldc = N;
  
  encoder.launch_kernel([&](hipStream_t stream) {
    switch (a.dtype()) {
      case float32:
        launch_naive_gemm<float>(
            stream,
            a.data<float>(), b.data<float>(), out.data<float>(),
            M, N, K, lda, ldb, ldc,
            a_transposed, b_transposed, alpha, beta);
        break;
      case float64:
        launch_naive_gemm<double>(
            stream,
            a.data<double>(), b.data<double>(), out.data<double>(),
            M, N, K, lda, ldb, ldc,
            a_transposed, b_transposed, alpha, beta);
        break;
      case float16:
        launch_naive_gemm<__half>(
            stream,
            a.data<__half>(), b.data<__half>(), out.data<__half>(),
            M, N, K, lda, ldb, ldc,
            a_transposed, b_transposed, alpha, beta);
        break;
      case bfloat16:
        launch_naive_gemm<hip_bfloat16>(
            stream,
            a.data<hip_bfloat16>(), b.data<hip_bfloat16>(), out.data<hip_bfloat16>(),
            M, N, K, lda, ldb, ldc,
            a_transposed, b_transposed, alpha, beta);
        break;
      default:
        throw std::runtime_error("Unsupported dtype for naive GEMM");
    }
  });
}

void naive_gemm_batched(
    CommandEncoder& encoder,
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    int64_t stride_a,
    bool b_transposed,
    int64_t ldb,
    int64_t stride_b,
    int64_t stride_c,
    int batch_count,
    float alpha,
    float beta) {
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);
  
  int ldc = N;
  
  encoder.launch_kernel([&](hipStream_t stream) {
    switch (a.dtype()) {
      case float32:
        launch_batched_gemm<float>(
            stream,
            a.data<float>(), b.data<float>(), out.data<float>(),
            M, N, K, lda, ldb, ldc,
            stride_a, stride_b, stride_c, batch_count,
            a_transposed, b_transposed, alpha, beta);
        break;
      case float64:
        launch_batched_gemm<double>(
            stream,
            a.data<double>(), b.data<double>(), out.data<double>(),
            M, N, K, lda, ldb, ldc,
            stride_a, stride_b, stride_c, batch_count,
            a_transposed, b_transposed, alpha, beta);
        break;
      case float16:
        launch_batched_gemm<__half>(
            stream,
            a.data<__half>(), b.data<__half>(), out.data<__half>(),
            M, N, K, lda, ldb, ldc,
            stride_a, stride_b, stride_c, batch_count,
            a_transposed, b_transposed, alpha, beta);
        break;
      case bfloat16:
        launch_batched_gemm<hip_bfloat16>(
            stream,
            a.data<hip_bfloat16>(), b.data<hip_bfloat16>(), out.data<hip_bfloat16>(),
            M, N, K, lda, ldb, ldc,
            stride_a, stride_b, stride_c, batch_count,
            a_transposed, b_transposed, alpha, beta);
        break;
      default:
        throw std::runtime_error("Unsupported dtype for batched naive GEMM");
    }
  });
}

void naive_gemm_with_offset(
    CommandEncoder& encoder,
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    int64_t a_offset,
    bool b_transposed,
    int64_t ldb,
    int64_t b_offset,
    int64_t out_offset,
    float alpha,
    float beta) {
  // Default ldc = N (contiguous output)
  naive_gemm_with_offset_ldc(
      encoder, a, b, out, M, N, K,
      a_transposed, lda, a_offset,
      b_transposed, ldb, b_offset,
      N, out_offset, alpha, beta);
}

void naive_gemm_with_offset_ldc(
    CommandEncoder& encoder,
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    int64_t a_offset,
    bool b_transposed,
    int64_t ldb,
    int64_t b_offset,
    int64_t ldc,
    int64_t out_offset,
    float alpha,
    float beta) {
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);
  
  encoder.launch_kernel([&](hipStream_t stream) {
    switch (a.dtype()) {
      case float32:
        launch_naive_gemm<float>(
            stream,
            a.data<float>() + a_offset, 
            b.data<float>() + b_offset, 
            out.data<float>() + out_offset,
            M, N, K, lda, ldb, ldc,
            a_transposed, b_transposed, alpha, beta);
        break;
      case float64:
        launch_naive_gemm<double>(
            stream,
            a.data<double>() + a_offset, 
            b.data<double>() + b_offset, 
            out.data<double>() + out_offset,
            M, N, K, lda, ldb, ldc,
            a_transposed, b_transposed, alpha, beta);
        break;
      case float16:
        launch_naive_gemm<__half>(
            stream,
            a.data<__half>() + a_offset, 
            b.data<__half>() + b_offset, 
            out.data<__half>() + out_offset,
            M, N, K, lda, ldb, ldc,
            a_transposed, b_transposed, alpha, beta);
        break;
      case bfloat16:
        launch_naive_gemm<hip_bfloat16>(
            stream,
            a.data<hip_bfloat16>() + a_offset, 
            b.data<hip_bfloat16>() + b_offset, 
            out.data<hip_bfloat16>() + out_offset,
            M, N, K, lda, ldb, ldc,
            a_transposed, b_transposed, alpha, beta);
        break;
      default:
        throw std::runtime_error("Unsupported dtype for naive GEMM with offset");
    }
  });
}

} // namespace mlx::core::rocm
