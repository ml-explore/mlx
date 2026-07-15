// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/gemms/naive_gemm.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device/config.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/dtype_utils.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

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

// Gathered batched GEMM kernel. Each output matrix chooses its lhs/rhs matrix
// from index arrays on device.
template <typename T, bool TransA, bool TransB>
__global__ void gather_batched_gemm_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    const uint32_t* __restrict__ lhs_indices,
    const uint32_t* __restrict__ rhs_indices,
    Shape idx_batch_shape,
    Strides lhs_idx_strides,
    Strides rhs_idx_strides,
    int idx_batch_ndim,
    Shape a_batch_shape,
    Strides a_batch_strides,
    int a_batch_ndim,
    Shape b_batch_shape,
    Strides b_batch_strides,
    int b_batch_ndim,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int64_t stride_c,
    float alpha,
    float beta) {
  using Acc = typename GemmAccType<T>::type;

  int batch = blockIdx.z;
  int row = blockIdx.y * TILE_M + threadIdx.y;
  int col = blockIdx.x * TILE_N + threadIdx.x;

  int64_t lhs_idx_loc = 0;
  int64_t rhs_idx_loc = 0;
  if (idx_batch_ndim == 1) {
    lhs_idx_loc = static_cast<int64_t>(batch) * lhs_idx_strides[0];
    rhs_idx_loc = static_cast<int64_t>(batch) * rhs_idx_strides[0];
  } else if (idx_batch_ndim > 1) {
    elem_to_loc<int64_t>(
        static_cast<int64_t>(batch),
        idx_batch_shape.data_,
        lhs_idx_strides.data_,
        rhs_idx_strides.data_,
        idx_batch_ndim,
        lhs_idx_loc,
        rhs_idx_loc);
  }

  uint32_t lhs_idx = lhs_indices[lhs_idx_loc];
  uint32_t rhs_idx = rhs_indices[rhs_idx_loc];

  int64_t a_offset = 0;
  int64_t b_offset = 0;
  if (a_batch_ndim == 1) {
    a_offset = static_cast<int64_t>(lhs_idx) * a_batch_strides[0];
  } else if (a_batch_ndim > 1) {
    a_offset = elem_to_loc<int64_t>(
        static_cast<int64_t>(lhs_idx),
        a_batch_shape.data_,
        a_batch_strides.data_,
        a_batch_ndim);
  }

  if (b_batch_ndim == 1) {
    b_offset = static_cast<int64_t>(rhs_idx) * b_batch_strides[0];
  } else if (b_batch_ndim > 1) {
    b_offset = elem_to_loc<int64_t>(
        static_cast<int64_t>(rhs_idx),
        b_batch_shape.data_,
        b_batch_strides.data_,
        b_batch_ndim);
  }

  const T* A_batch = A + a_offset;
  const T* B_batch = B + b_offset;
  T* C_batch = C + static_cast<int64_t>(batch) * stride_c;

  // LDS-tiled accumulate (was a naive O(K) global-load loop — dominated MoE
  // training time on gfx1151 at ~0.5 TFLOP/s). Same tile sizes as tiled_gemm.
  __shared__ Acc As[TILE_M][TILE_K];
  __shared__ Acc Bs[TILE_K][TILE_N];

  Acc sum = Acc(0);
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  for (int t = 0; t < (K + TILE_K - 1) / TILE_K; ++t) {
    int a_col = t * TILE_K + tx;
    if (row < M && a_col < K) {
      if constexpr (TransA) {
        As[ty][tx] = static_cast<Acc>(A_batch[a_col * lda + row]);
      } else {
        As[ty][tx] = static_cast<Acc>(A_batch[row * lda + a_col]);
      }
    } else {
      As[ty][tx] = Acc(0);
    }

    int b_row = t * TILE_K + ty;
    if (b_row < K && col < N) {
      if constexpr (TransB) {
        Bs[ty][tx] = static_cast<Acc>(B_batch[col * ldb + b_row]);
      } else {
        Bs[ty][tx] = static_cast<Acc>(B_batch[b_row * ldb + col]);
      }
    } else {
      Bs[ty][tx] = Acc(0);
    }

    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE_K; ++k) {
      sum += As[ty][k] * Bs[k][tx];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    if (beta != 0.0f) {
      C_batch[row * N + col] = static_cast<T>(
          alpha * sum + beta * static_cast<Acc>(C_batch[row * N + col]));
    } else {
      C_batch[row * N + col] = static_cast<T>(alpha * sum);
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

template <typename T>
void launch_gather_batched_gemm(
    hipStream_t stream,
    const T* A,
    const T* B,
    T* C,
    const uint32_t* lhs_indices,
    const uint32_t* rhs_indices,
    Shape idx_batch_shape,
    Strides lhs_idx_strides,
    Strides rhs_idx_strides,
    int idx_batch_ndim,
    Shape a_batch_shape,
    Strides a_batch_strides,
    int a_batch_ndim,
    Shape b_batch_shape,
    Strides b_batch_strides,
    int b_batch_ndim,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int64_t stride_c,
    int batch_count,
    bool trans_a,
    bool trans_b,
    float alpha,
    float beta) {
  dim3 block(TILE_N, TILE_M);
  dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M, batch_count);

  if (trans_a && trans_b) {
    hipLaunchKernelGGL(
        (gather_batched_gemm_kernel<T, true, true>),
        grid,
        block,
        0,
        stream,
        A,
        B,
        C,
        lhs_indices,
        rhs_indices,
        idx_batch_shape,
        lhs_idx_strides,
        rhs_idx_strides,
        idx_batch_ndim,
        a_batch_shape,
        a_batch_strides,
        a_batch_ndim,
        b_batch_shape,
        b_batch_strides,
        b_batch_ndim,
        M,
        N,
        K,
        lda,
        ldb,
        stride_c,
        alpha,
        beta);
  } else if (trans_a && !trans_b) {
    hipLaunchKernelGGL(
        (gather_batched_gemm_kernel<T, true, false>),
        grid,
        block,
        0,
        stream,
        A,
        B,
        C,
        lhs_indices,
        rhs_indices,
        idx_batch_shape,
        lhs_idx_strides,
        rhs_idx_strides,
        idx_batch_ndim,
        a_batch_shape,
        a_batch_strides,
        a_batch_ndim,
        b_batch_shape,
        b_batch_strides,
        b_batch_ndim,
        M,
        N,
        K,
        lda,
        ldb,
        stride_c,
        alpha,
        beta);
  } else if (!trans_a && trans_b) {
    hipLaunchKernelGGL(
        (gather_batched_gemm_kernel<T, false, true>),
        grid,
        block,
        0,
        stream,
        A,
        B,
        C,
        lhs_indices,
        rhs_indices,
        idx_batch_shape,
        lhs_idx_strides,
        rhs_idx_strides,
        idx_batch_ndim,
        a_batch_shape,
        a_batch_strides,
        a_batch_ndim,
        b_batch_shape,
        b_batch_strides,
        b_batch_ndim,
        M,
        N,
        K,
        lda,
        ldb,
        stride_c,
        alpha,
        beta);
  } else {
    hipLaunchKernelGGL(
        (gather_batched_gemm_kernel<T, false, false>),
        grid,
        block,
        0,
        stream,
        A,
        B,
        C,
        lhs_indices,
        rhs_indices,
        idx_batch_shape,
        lhs_idx_strides,
        rhs_idx_strides,
        idx_batch_ndim,
        a_batch_shape,
        a_batch_strides,
        a_batch_ndim,
        b_batch_shape,
        b_batch_strides,
        b_batch_ndim,
        M,
        N,
        K,
        lda,
        ldb,
        stride_c,
        alpha,
        beta);
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
  const void* a_ptr = gpu_ptr<void>(a);
  const void* b_ptr = gpu_ptr<void>(b);
  void* out_ptr = gpu_ptr<void>(out);
  
  encoder.launch_kernel([&, a_ptr, b_ptr, out_ptr](hipStream_t stream) {
    switch (a.dtype()) {
      case float32:
        launch_naive_gemm<float>(
            stream,
            static_cast<const float*>(a_ptr),
            static_cast<const float*>(b_ptr),
            static_cast<float*>(out_ptr),
            M, N, K, lda, ldb, ldc,
            a_transposed, b_transposed, alpha, beta);
        break;
      case float64:
        launch_naive_gemm<double>(
            stream,
            static_cast<const double*>(a_ptr),
            static_cast<const double*>(b_ptr),
            static_cast<double*>(out_ptr),
            M, N, K, lda, ldb, ldc,
            a_transposed, b_transposed, alpha, beta);
        break;
      case float16:
        launch_naive_gemm<__half>(
            stream,
            static_cast<const __half*>(a_ptr),
            static_cast<const __half*>(b_ptr),
            static_cast<__half*>(out_ptr),
            M, N, K, lda, ldb, ldc,
            a_transposed, b_transposed, alpha, beta);
        break;
      case bfloat16:
        launch_naive_gemm<hip_bfloat16>(
            stream,
            static_cast<const hip_bfloat16*>(a_ptr),
            static_cast<const hip_bfloat16*>(b_ptr),
            static_cast<hip_bfloat16*>(out_ptr),
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
  const void* a_ptr = gpu_ptr<void>(a);
  const void* b_ptr = gpu_ptr<void>(b);
  void* out_ptr = gpu_ptr<void>(out);
  
  encoder.launch_kernel([&, a_ptr, b_ptr, out_ptr](hipStream_t stream) {
    switch (a.dtype()) {
      case float32:
        launch_batched_gemm<float>(
            stream,
            static_cast<const float*>(a_ptr),
            static_cast<const float*>(b_ptr),
            static_cast<float*>(out_ptr),
            M, N, K, lda, ldb, ldc,
            stride_a, stride_b, stride_c, batch_count,
            a_transposed, b_transposed, alpha, beta);
        break;
      case float64:
        launch_batched_gemm<double>(
            stream,
            static_cast<const double*>(a_ptr),
            static_cast<const double*>(b_ptr),
            static_cast<double*>(out_ptr),
            M, N, K, lda, ldb, ldc,
            stride_a, stride_b, stride_c, batch_count,
            a_transposed, b_transposed, alpha, beta);
        break;
      case float16:
        launch_batched_gemm<__half>(
            stream,
            static_cast<const __half*>(a_ptr),
            static_cast<const __half*>(b_ptr),
            static_cast<__half*>(out_ptr),
            M, N, K, lda, ldb, ldc,
            stride_a, stride_b, stride_c, batch_count,
            a_transposed, b_transposed, alpha, beta);
        break;
      case bfloat16:
        launch_batched_gemm<hip_bfloat16>(
            stream,
            static_cast<const hip_bfloat16*>(a_ptr),
            static_cast<const hip_bfloat16*>(b_ptr),
            static_cast<hip_bfloat16*>(out_ptr),
            M, N, K, lda, ldb, ldc,
            stride_a, stride_b, stride_c, batch_count,
            a_transposed, b_transposed, alpha, beta);
        break;
      default:
        throw std::runtime_error("Unsupported dtype for batched naive GEMM");
    }
  });
}

void naive_gemm_gather(
    CommandEncoder& encoder,
    const array& a,
    const array& b,
    const array& lhs_indices,
    const array& rhs_indices,
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
  encoder.set_input_array(lhs_indices);
  encoder.set_input_array(rhs_indices);
  encoder.set_output_array(out);

  auto [idx_batch_shape, idx_batch_strides] = collapse_contiguous_dims(
      lhs_indices.shape(), {lhs_indices.strides(), rhs_indices.strides()});
  auto lhs_idx_strides = idx_batch_strides[0];
  auto rhs_idx_strides = idx_batch_strides[1];
  int idx_batch_ndim = idx_batch_shape.size();

  mlx::core::Shape a_batch_shape{a.shape().begin(), a.shape().end() - 2};
  mlx::core::Strides a_batch_strides{a.strides().begin(), a.strides().end() - 2};
  int a_batch_ndim = a_batch_shape.size();

  mlx::core::Shape b_batch_shape{b.shape().begin(), b.shape().end() - 2};
  mlx::core::Strides b_batch_strides{b.strides().begin(), b.strides().end() - 2};
  int b_batch_ndim = b_batch_shape.size();

  auto idx_batch_shape_param = const_param(idx_batch_shape);
  auto lhs_idx_strides_param = const_param<MAX_NDIM, int64_t>(lhs_idx_strides);
  auto rhs_idx_strides_param = const_param<MAX_NDIM, int64_t>(rhs_idx_strides);

  auto a_batch_shape_param = const_param(a_batch_shape);
  auto a_batch_strides_param = const_param<MAX_NDIM, int64_t>(a_batch_strides);
  auto b_batch_shape_param = const_param(b_batch_shape);
  auto b_batch_strides_param = const_param<MAX_NDIM, int64_t>(b_batch_strides);

  const int64_t stride_c = static_cast<int64_t>(M) * N;
  const int batch_count = out.size() / (M * N);

  const void* a_ptr = gpu_ptr<void>(a);
  const void* b_ptr = gpu_ptr<void>(b);
  void* out_ptr = gpu_ptr<void>(out);
  const uint32_t* lhs_indices_ptr = gpu_ptr<const uint32_t>(lhs_indices);
  const uint32_t* rhs_indices_ptr = gpu_ptr<const uint32_t>(rhs_indices);

  encoder.launch_kernel([&,
                         a_ptr,
                         b_ptr,
                         out_ptr,
                         lhs_indices_ptr,
                         rhs_indices_ptr](hipStream_t stream) {
    switch (a.dtype()) {
      case float32:
        launch_gather_batched_gemm<float>(
            stream,
            static_cast<const float*>(a_ptr),
            static_cast<const float*>(b_ptr),
            static_cast<float*>(out_ptr),
            lhs_indices_ptr,
            rhs_indices_ptr,
            idx_batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            idx_batch_ndim,
            a_batch_shape_param,
            a_batch_strides_param,
            a_batch_ndim,
            b_batch_shape_param,
            b_batch_strides_param,
            b_batch_ndim,
            M,
            N,
            K,
            lda,
            ldb,
            stride_c,
            batch_count,
            a_transposed,
            b_transposed,
            alpha,
            beta);
        break;
      case float64:
        launch_gather_batched_gemm<double>(
            stream,
            static_cast<const double*>(a_ptr),
            static_cast<const double*>(b_ptr),
            static_cast<double*>(out_ptr),
            lhs_indices_ptr,
            rhs_indices_ptr,
            idx_batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            idx_batch_ndim,
            a_batch_shape_param,
            a_batch_strides_param,
            a_batch_ndim,
            b_batch_shape_param,
            b_batch_strides_param,
            b_batch_ndim,
            M,
            N,
            K,
            lda,
            ldb,
            stride_c,
            batch_count,
            a_transposed,
            b_transposed,
            alpha,
            beta);
        break;
      case float16:
        launch_gather_batched_gemm<__half>(
            stream,
            static_cast<const __half*>(a_ptr),
            static_cast<const __half*>(b_ptr),
            static_cast<__half*>(out_ptr),
            lhs_indices_ptr,
            rhs_indices_ptr,
            idx_batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            idx_batch_ndim,
            a_batch_shape_param,
            a_batch_strides_param,
            a_batch_ndim,
            b_batch_shape_param,
            b_batch_strides_param,
            b_batch_ndim,
            M,
            N,
            K,
            lda,
            ldb,
            stride_c,
            batch_count,
            a_transposed,
            b_transposed,
            alpha,
            beta);
        break;
      case bfloat16:
        launch_gather_batched_gemm<hip_bfloat16>(
            stream,
            static_cast<const hip_bfloat16*>(a_ptr),
            static_cast<const hip_bfloat16*>(b_ptr),
            static_cast<hip_bfloat16*>(out_ptr),
            lhs_indices_ptr,
            rhs_indices_ptr,
            idx_batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            idx_batch_ndim,
            a_batch_shape_param,
            a_batch_strides_param,
            a_batch_ndim,
            b_batch_shape_param,
            b_batch_strides_param,
            b_batch_ndim,
            M,
            N,
            K,
            lda,
            ldb,
            stride_c,
            batch_count,
            a_transposed,
            b_transposed,
            alpha,
            beta);
        break;
      default:
        throw std::runtime_error("Unsupported dtype for gathered naive GEMM");
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
  const void* a_ptr = gpu_ptr<void>(a);
  const void* b_ptr = gpu_ptr<void>(b);
  void* out_ptr = gpu_ptr<void>(out);
  
  encoder.launch_kernel([&, a_ptr, b_ptr, out_ptr](hipStream_t stream) {
    switch (a.dtype()) {
      case float32:
        launch_naive_gemm<float>(
            stream,
            static_cast<const float*>(a_ptr) + a_offset,
            static_cast<const float*>(b_ptr) + b_offset,
            static_cast<float*>(out_ptr) + out_offset,
            M, N, K, lda, ldb, ldc,
            a_transposed, b_transposed, alpha, beta);
        break;
      case float64:
        launch_naive_gemm<double>(
            stream,
            static_cast<const double*>(a_ptr) + a_offset,
            static_cast<const double*>(b_ptr) + b_offset,
            static_cast<double*>(out_ptr) + out_offset,
            M, N, K, lda, ldb, ldc,
            a_transposed, b_transposed, alpha, beta);
        break;
      case float16:
        launch_naive_gemm<__half>(
            stream,
            static_cast<const __half*>(a_ptr) + a_offset,
            static_cast<const __half*>(b_ptr) + b_offset,
            static_cast<__half*>(out_ptr) + out_offset,
            M, N, K, lda, ldb, ldc,
            a_transposed, b_transposed, alpha, beta);
        break;
      case bfloat16:
        launch_naive_gemm<hip_bfloat16>(
            stream,
            static_cast<const hip_bfloat16*>(a_ptr) + a_offset,
            static_cast<const hip_bfloat16*>(b_ptr) + b_offset,
            static_cast<hip_bfloat16*>(out_ptr) + out_offset,
            M, N, K, lda, ldb, ldc,
            a_transposed, b_transposed, alpha, beta);
        break;
      default:
        throw std::runtime_error("Unsupported dtype for naive GEMM with offset");
    }
  });
}

// ---------------------------------------------------------------------------
// Device-side MoE sorted expert GEMM (no host sync).
// Token-tiled grid: (ceil(N/TN), ceil(batch/TM)) — O(T·N) blocks, NOT O(T·N·E).
// Sorted ids → most tiles are single-expert (shared B); boundary tiles fall
// back to per-row expert lookup.
// ---------------------------------------------------------------------------
namespace {

constexpr int MOE_TM = 32;
constexpr int MOE_TN = 32;
constexpr int MOE_TK = 16;

__device__ __forceinline__ int moe_lower_bound(
    const uint32_t* idx,
    int n,
    uint32_t e) {
  int lo = 0, hi = n;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (idx[mid] < e)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

__device__ __forceinline__ int moe_upper_bound(
    const uint32_t* idx,
    int n,
    uint32_t e) {
  int lo = 0, hi = n;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (idx[mid] <= e)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

template <typename T, bool TransB>
__global__ void moe_sorted_expert_gemm_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    const uint32_t* __restrict__ idx,
    int batch,
    int K,
    int N,
    int64_t b_expert_stride,
    int ldb) {
  using Acc = typename GemmAccType<T>::type;
  const int tx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  const int row0 = static_cast<int>(blockIdx.y) * MOE_TM;
  const int col0 = static_cast<int>(blockIdx.x) * MOE_TN;
  const int row = row0 + ty;
  const int col = col0 + tx;
  if (row0 >= batch) {
    return;
  }
  const int last = (row0 + MOE_TM <= batch) ? (row0 + MOE_TM - 1) : (batch - 1);
  const uint32_t e0 = idx[row0];
  const uint32_t e1 = idx[last];
  const bool single_expert = (e0 == e1);
  const bool active = (row < batch) && (col < N);

  Acc sum = Acc(0);
  __shared__ Acc As[MOE_TM][MOE_TK];
  __shared__ Acc Bs[MOE_TK][MOE_TN];

  if (single_expert) {
    // Fast path: whole tile is one expert run → shared A and B tiles.
    const T* Be = B + static_cast<int64_t>(e0) * b_expert_stride;
    for (int t = 0; t < (K + MOE_TK - 1) / MOE_TK; ++t) {
      const int a_col = t * MOE_TK + tx;
      if (ty < MOE_TM && tx < MOE_TK) {
        if (row < batch && a_col < K) {
          As[ty][tx] =
              static_cast<Acc>(A[static_cast<int64_t>(row) * K + a_col]);
        } else {
          As[ty][tx] = Acc(0);
        }
      }
      const int b_k = t * MOE_TK + ty;
      if (ty < MOE_TK && tx < MOE_TN) {
        if (b_k < K && col < N) {
          if constexpr (TransB) {
            Bs[ty][tx] = static_cast<Acc>(
                Be[static_cast<int64_t>(col) * ldb + b_k]);
          } else {
            Bs[ty][tx] = static_cast<Acc>(
                Be[static_cast<int64_t>(b_k) * ldb + col]);
          }
        } else {
          Bs[ty][tx] = Acc(0);
        }
      }
      __syncthreads();
      if (active) {
#pragma unroll
        for (int kk = 0; kk < MOE_TK; ++kk) {
          sum += As[ty][kk] * Bs[kk][tx];
        }
      }
      __syncthreads();
    }
  } else {
    // Boundary tile: rows may straddle expert runs. Shared A, per-row B.
    for (int t = 0; t < (K + MOE_TK - 1) / MOE_TK; ++t) {
      const int a_col = t * MOE_TK + tx;
      if (ty < MOE_TM && tx < MOE_TK) {
        if (row < batch && a_col < K) {
          As[ty][tx] =
              static_cast<Acc>(A[static_cast<int64_t>(row) * K + a_col]);
        } else {
          As[ty][tx] = Acc(0);
        }
      }
      __syncthreads();
      if (active) {
        const uint32_t e = idx[row];
        const T* Be = B + static_cast<int64_t>(e) * b_expert_stride;
#pragma unroll
        for (int kk = 0; kk < MOE_TK; ++kk) {
          const int b_k = t * MOE_TK + kk;
          Acc bv = Acc(0);
          if (b_k < K) {
            if constexpr (TransB) {
              bv = static_cast<Acc>(
                  Be[static_cast<int64_t>(col) * ldb + b_k]);
            } else {
              bv = static_cast<Acc>(
                  Be[static_cast<int64_t>(b_k) * ldb + col]);
            }
          }
          sum += As[ty][kk] * bv;
        }
      }
      __syncthreads();
    }
  }
  if (active) {
    C[static_cast<int64_t>(row) * N + col] = static_cast<T>(sum);
  }
}

__global__ void moe_sorted_segments_kernel(
    const uint32_t* __restrict__ idx,
    uint32_t* __restrict__ segments,
    int T,
    int E) {
  int e = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (e >= E)
    return;
  int lo = moe_lower_bound(idx, T, static_cast<uint32_t>(e));
  int hi = moe_upper_bound(idx, T, static_cast<uint32_t>(e));
  segments[2 * e] = static_cast<uint32_t>(lo);
  segments[2 * e + 1] = static_cast<uint32_t>(hi);
}

__global__ void moe_max_run_length_kernel(
    const uint32_t* __restrict__ idx,
    int* __restrict__ max_run,
    int T,
    int E) {
  int e = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (e >= E)
    return;
  int lo = moe_lower_bound(idx, T, static_cast<uint32_t>(e));
  int hi = moe_upper_bound(idx, T, static_cast<uint32_t>(e));
  int len = hi - lo;
  if (len > 0)
    atomicMax(max_run, len);
}

template <typename T, bool TransA, bool TransB>
__global__ void segmented_mm_device_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    const uint32_t* __restrict__ segments,
    int M,
    int N,
    int64_t a_k_stride,
    int64_t b_k_stride,
    int64_t out_stride,
    int lda,
    int ldb) {
  using Acc = typename GemmAccType<T>::type;
  const int seg = static_cast<int>(blockIdx.z);
  const uint32_t k0 = segments[2 * seg];
  const uint32_t k1 = segments[2 * seg + 1];
  const int Kseg = (k1 > k0) ? static_cast<int>(k1 - k0) : 0;

  const int tx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  const int row = static_cast<int>(blockIdx.y) * TILE_M + ty;
  const int col = static_cast<int>(blockIdx.x) * TILE_N + tx;

  T* Cseg = C + static_cast<int64_t>(seg) * out_stride;
  const bool in_tile = (row < M) && (col < N);
  if (Kseg <= 0) {
    if (in_tile) {
      Cseg[static_cast<int64_t>(row) * N + col] = static_cast<T>(0);
    }
    return;
  }

  const T* Aseg = A + static_cast<int64_t>(k0) * a_k_stride;
  const T* Bseg = B + static_cast<int64_t>(k0) * b_k_stride;

  Acc sum = Acc(0);
  __shared__ Acc As[TILE_M][TILE_K];
  __shared__ Acc Bs[TILE_K][TILE_N];

  for (int t = 0; t < (Kseg + TILE_K - 1) / TILE_K; ++t) {
    const int a_col = t * TILE_K + tx;
    if (row < M && a_col < Kseg) {
      if constexpr (TransA) {
        As[ty][tx] = static_cast<Acc>(Aseg[static_cast<int64_t>(a_col) * lda + row]);
      } else {
        As[ty][tx] = static_cast<Acc>(Aseg[static_cast<int64_t>(row) * lda + a_col]);
      }
    } else {
      As[ty][tx] = Acc(0);
    }
    const int b_row = t * TILE_K + ty;
    if (b_row < Kseg && col < N) {
      if constexpr (TransB) {
        Bs[ty][tx] = static_cast<Acc>(Bseg[static_cast<int64_t>(col) * ldb + b_row]);
      } else {
        Bs[ty][tx] = static_cast<Acc>(Bseg[static_cast<int64_t>(b_row) * ldb + col]);
      }
    } else {
      Bs[ty][tx] = Acc(0);
    }
    __syncthreads();
    if (in_tile) {
#pragma unroll
      for (int kk = 0; kk < TILE_K; ++kk) {
        sum += As[ty][kk] * Bs[kk][tx];
      }
    }
    __syncthreads();
  }
  if (in_tile) {
    Cseg[static_cast<int64_t>(row) * N + col] = static_cast<T>(sum);
  }
}

} // namespace

// ---- MoE pack / unpack for fixed-shape strided-batched hipBLASLt ------------
// One thread per token (atomic once), then vectorized copy of the K-vector.
__global__ void moe_pack_kernel(
    const hip_bfloat16* __restrict__ A,
    const uint32_t* __restrict__ idx,
    hip_bfloat16* __restrict__ packed,
    int* __restrict__ slot_map,
    int* __restrict__ counts,
    int batch,
    int K,
    int M_fixed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= batch)
    return;
  uint32_t e = idx[i];
  int slot = atomicAdd(counts + static_cast<int>(e), 1);
  if (slot >= M_fixed)
    return; // overflow — caller must size M_fixed large enough
  slot_map[static_cast<int64_t>(e) * M_fixed + slot] = i;
  const hip_bfloat16* src = A + static_cast<int64_t>(i) * K;
  hip_bfloat16* dst =
      packed + (static_cast<int64_t>(e) * M_fixed + slot) * K;
  // 4 bf16 = 8 bytes = one float2
  if ((K & 3) == 0) {
    const float2* s2 = reinterpret_cast<const float2*>(src);
    float2* d2 = reinterpret_cast<float2*>(dst);
    int n2 = K >> 2;
    for (int k = 0; k < n2; ++k)
      d2[k] = s2[k];
  } else {
    for (int k = 0; k < K; ++k)
      dst[k] = src[k];
  }
}

__global__ void moe_unpack_kernel(
    const hip_bfloat16* __restrict__ packed_c,
    const int* __restrict__ slot_map,
    hip_bfloat16* __restrict__ C,
    int n_experts,
    int M_fixed,
    int N) {
  int e = blockIdx.z;
  int s = blockIdx.y * blockDim.y + threadIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_experts || s >= M_fixed || n >= N)
    return;
  int src = slot_map[static_cast<int64_t>(e) * M_fixed + s];
  if (src < 0)
    return;
  C[static_cast<int64_t>(src) * N + n] =
      packed_c[(static_cast<int64_t>(e) * M_fixed + s) * N + n];
}

__global__ void moe_fill_neg1(int* slot_map, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    slot_map[i] = -1;
}

void moe_pack_tokens(
    CommandEncoder& encoder,
    const array& a,
    const array& rhs_indices,
    array& packed_a,
    array& slot_map,
    array& counts,
    int batch,
    int K,
    int n_experts,
    int M_fixed) {
  encoder.set_input_array(a);
  encoder.set_input_array(rhs_indices);
  encoder.set_output_array(packed_a);
  encoder.set_output_array(slot_map);
  encoder.set_output_array(counts);

  const void* a_ptr = gpu_ptr<void>(a);
  const uint32_t* idx_ptr = gpu_ptr<const uint32_t>(rhs_indices);
  void* packed_ptr = gpu_ptr<void>(packed_a);
  int* map_ptr = gpu_ptr<int>(slot_map);
  int* counts_ptr = gpu_ptr<int>(counts);
  const int map_n = n_experts * M_fixed;
  const size_t packed_bytes = packed_a.nbytes();
  const size_t counts_bytes = counts.nbytes();

  encoder.launch_kernel([=](hipStream_t stream) {
    (void)hipMemsetAsync(counts_ptr, 0, counts_bytes, stream);
    (void)hipMemsetAsync(packed_ptr, 0, packed_bytes, stream);
    dim3 bfill(256);
    dim3 gfill((map_n + 255) / 256);
    hipLaunchKernelGGL(moe_fill_neg1, gfill, bfill, 0, stream, map_ptr, map_n);

    dim3 block(256);
    dim3 grid((batch + 255) / 256);
    if (a.dtype() == bfloat16) {
      hipLaunchKernelGGL(
          moe_pack_kernel,
          grid,
          block,
          0,
          stream,
          static_cast<const hip_bfloat16*>(a_ptr),
          idx_ptr,
          static_cast<hip_bfloat16*>(packed_ptr),
          map_ptr,
          counts_ptr,
          batch,
          K,
          M_fixed);
    } else {
      throw std::runtime_error("moe_pack_tokens: only bfloat16 supported");
    }
  });
}

void moe_unpack_tokens(
    CommandEncoder& encoder,
    const array& packed_c,
    const array& slot_map,
    array& out,
    int n_experts,
    int M_fixed,
    int N) {
  encoder.set_input_array(packed_c);
  encoder.set_input_array(slot_map);
  encoder.set_output_array(out);
  const void* pc = gpu_ptr<void>(packed_c);
  const int* map_ptr = gpu_ptr<const int>(slot_map);
  void* out_ptr = gpu_ptr<void>(out);

  dim3 block(32, 8);
  dim3 grid((N + 31) / 32, (M_fixed + 7) / 8, n_experts);
  encoder.launch_kernel([=](hipStream_t stream) {
    if (out.dtype() != bfloat16) {
      throw std::runtime_error("moe_unpack_tokens: only bfloat16 supported");
    }
    hipLaunchKernelGGL(
        moe_unpack_kernel,
        grid,
        block,
        0,
        stream,
        static_cast<const hip_bfloat16*>(pc),
        map_ptr,
        static_cast<hip_bfloat16*>(out_ptr),
        n_experts,
        M_fixed,
        N);
  });
}

// Gather src rows into packed using an existing slot_map (shared packing).
__global__ void moe_pack_using_slot_map_kernel(
    const hip_bfloat16* __restrict__ src,
    const int* __restrict__ slot_map,
    hip_bfloat16* __restrict__ packed,
    int n_experts,
    int M_fixed,
    int N) {
  int e = blockIdx.z;
  int s = blockIdx.y * blockDim.y + threadIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_experts || s >= M_fixed || n >= N)
    return;
  int src_i = slot_map[static_cast<int64_t>(e) * M_fixed + s];
  hip_bfloat16* dst =
      packed + (static_cast<int64_t>(e) * M_fixed + s) * N + n;
  if (src_i < 0) {
    *dst = hip_bfloat16(0.0f);
    return;
  }
  *dst = src[static_cast<int64_t>(src_i) * N + n];
}

void moe_pack_using_slot_map(
    CommandEncoder& encoder,
    const array& src,
    const array& slot_map,
    array& packed,
    int n_experts,
    int M_fixed,
    int N) {
  encoder.set_input_array(src);
  encoder.set_input_array(slot_map);
  encoder.set_output_array(packed);
  if (src.dtype() != bfloat16) {
    throw std::runtime_error("moe_pack_using_slot_map: only bfloat16");
  }
  const void* sp = gpu_ptr<void>(src);
  const int* map_ptr = gpu_ptr<const int>(slot_map);
  void* pp = gpu_ptr<void>(packed);
  dim3 block(32, 8);
  dim3 grid((N + 31) / 32, (M_fixed + 7) / 8, n_experts);
  encoder.launch_kernel([=](hipStream_t stream) {
    hipLaunchKernelGGL(
        moe_pack_using_slot_map_kernel,
        grid,
        block,
        0,
        stream,
        static_cast<const hip_bfloat16*>(sp),
        map_ptr,
        static_cast<hip_bfloat16*>(pp),
        n_experts,
        M_fixed,
        N);
  });
}

__global__ void silu_mul_bf16_kernel(
    const hip_bfloat16* __restrict__ g,
    const hip_bfloat16* __restrict__ u,
    hip_bfloat16* __restrict__ h,
    int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float gf = static_cast<float>(g[i]);
  float uf = static_cast<float>(u[i]);
  float s = 1.0f / (1.0f + expf(-gf));
  h[i] = static_cast<hip_bfloat16>(gf * s * uf);
}

void silu_mul_bf16(
    CommandEncoder& encoder,
    const void* gate,
    const void* up,
    void* h,
    int n) {
  encoder.launch_kernel([=](hipStream_t stream) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    hipLaunchKernelGGL(
        silu_mul_bf16_kernel,
        grid,
        block,
        0,
        stream,
        static_cast<const hip_bfloat16*>(gate),
        static_cast<const hip_bfloat16*>(up),
        static_cast<hip_bfloat16*>(h),
        n);
  });
}

__global__ void swiglu_bwd_elem_bf16_kernel(
    const hip_bfloat16* __restrict__ g,
    const hip_bfloat16* __restrict__ u,
    const hip_bfloat16* __restrict__ dh,
    hip_bfloat16* __restrict__ h,
    hip_bfloat16* __restrict__ dg,
    hip_bfloat16* __restrict__ du,
    int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float gf = static_cast<float>(g[i]);
  float uf = static_cast<float>(u[i]);
  float dhf = static_cast<float>(dh[i]);
  float s = 1.0f / (1.0f + expf(-gf));
  float silu = gf * s;
  float dsilu = s * (1.0f + gf * (1.0f - s));
  h[i] = static_cast<hip_bfloat16>(silu * uf);
  du[i] = static_cast<hip_bfloat16>(dhf * silu);
  dg[i] = static_cast<hip_bfloat16>(dhf * uf * dsilu);
}

void swiglu_bwd_elem_bf16(
    CommandEncoder& encoder,
    const void* gate,
    const void* up,
    const void* dh,
    void* h,
    void* dg,
    void* du,
    int n) {
  encoder.launch_kernel([=](hipStream_t stream) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    hipLaunchKernelGGL(
        swiglu_bwd_elem_bf16_kernel,
        grid,
        block,
        0,
        stream,
        static_cast<const hip_bfloat16*>(gate),
        static_cast<const hip_bfloat16*>(up),
        static_cast<const hip_bfloat16*>(dh),
        static_cast<hip_bfloat16*>(h),
        static_cast<hip_bfloat16*>(dg),
        static_cast<hip_bfloat16*>(du),
        n);
  });
}

__global__ void bf16_add_inplace_kernel(
    const hip_bfloat16* __restrict__ src,
    hip_bfloat16* __restrict__ out,
    int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  out[i] = static_cast<hip_bfloat16>(
      static_cast<float>(out[i]) + static_cast<float>(src[i]));
}

void bf16_add_inplace(
    CommandEncoder& encoder,
    const void* src,
    void* out,
    int n) {
  encoder.launch_kernel([=](hipStream_t stream) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    hipLaunchKernelGGL(
        bf16_add_inplace_kernel,
        grid,
        block,
        0,
        stream,
        static_cast<const hip_bfloat16*>(src),
        static_cast<hip_bfloat16*>(out),
        n);
  });
}

__global__ void bf16_transpose_2d_kernel(
    const hip_bfloat16* __restrict__ in,
    hip_bfloat16* __restrict__ out,
    int rows,
    int cols) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows && j < cols) {
    out[static_cast<size_t>(j) * rows + i] =
        in[static_cast<size_t>(i) * cols + j];
  }
}

void bf16_transpose_2d(
    CommandEncoder& encoder,
    const void* in,
    void* out,
    int rows,
    int cols) {
  encoder.launch_kernel([=](hipStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);
    hipLaunchKernelGGL(
        bf16_transpose_2d_kernel,
        grid,
        block,
        0,
        stream,
        static_cast<const hip_bfloat16*>(in),
        static_cast<hip_bfloat16*>(out),
        rows,
        cols);
  });
}

__global__ void bf16_concat_rows_kernel(
    const hip_bfloat16* __restrict__ a,
    const hip_bfloat16* __restrict__ b,
    hip_bfloat16* __restrict__ out,
    int T,
    int K) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int n = T * K;
  if (i >= n)
    return;
  int row = i / K;
  int col = i % K;
  out[row * (2 * K) + col] = a[i];
  out[row * (2 * K) + K + col] = b[i];
}

void bf16_concat_rows(
    CommandEncoder& encoder,
    const void* a,
    const void* b,
    void* out,
    int T,
    int K) {
  encoder.launch_kernel([=](hipStream_t stream) {
    int n = T * K;
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    hipLaunchKernelGGL(
        bf16_concat_rows_kernel,
        grid,
        block,
        0,
        stream,
        static_cast<const hip_bfloat16*>(a),
        static_cast<const hip_bfloat16*>(b),
        static_cast<hip_bfloat16*>(out),
        T,
        K);
  });
}

__global__ void bf16_split_rows_kernel(
    const hip_bfloat16* __restrict__ ab,
    hip_bfloat16* __restrict__ a,
    hip_bfloat16* __restrict__ b,
    int rows,
    int K) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int n = rows * K;
  if (i >= n)
    return;
  int row = i / K;
  int col = i % K;
  a[i] = ab[row * (2 * K) + col];
  b[i] = ab[row * (2 * K) + K + col];
}

void bf16_split_rows(
    CommandEncoder& encoder,
    const void* ab,
    void* a,
    void* b,
    int rows,
    int K) {
  encoder.launch_kernel([=](hipStream_t stream) {
    int n = rows * K;
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    hipLaunchKernelGGL(
        bf16_split_rows_kernel,
        grid,
        block,
        0,
        stream,
        static_cast<const hip_bfloat16*>(ab),
        static_cast<hip_bfloat16*>(a),
        static_cast<hip_bfloat16*>(b),
        rows,
        K);
  });
}

void moe_sorted_expert_gemm(
    CommandEncoder& encoder,
    const array& a,
    const array& b,
    const array& rhs_indices,
    array& out,
    int batch,
    int N,
    int K,
    int n_experts,
    bool b_transposed,
    int64_t ldb,
    int64_t b_expert_stride) {
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(rhs_indices);
  encoder.set_output_array(out);

  const void* a_ptr = gpu_ptr<void>(a);
  const void* b_ptr = gpu_ptr<void>(b);
  void* out_ptr = gpu_ptr<void>(out);
  const uint32_t* idx_ptr = gpu_ptr<const uint32_t>(rhs_indices);

  // Token-tiled: no expert grid dim. n_experts kept in API for callers.
  (void)n_experts;
  dim3 block(MOE_TN, MOE_TM);
  dim3 grid(
      (N + MOE_TN - 1) / MOE_TN,
      (batch + MOE_TM - 1) / MOE_TM);

  encoder.launch_kernel([=](hipStream_t stream) {
    switch (a.dtype()) {
      case float32:
        if (b_transposed) {
          hipLaunchKernelGGL(
              (moe_sorted_expert_gemm_kernel<float, true>),
              grid,
              block,
              0,
              stream,
              static_cast<const float*>(a_ptr),
              static_cast<const float*>(b_ptr),
              static_cast<float*>(out_ptr),
              idx_ptr,
              batch,
              K,
              N,
              b_expert_stride,
              static_cast<int>(ldb));
        } else {
          hipLaunchKernelGGL(
              (moe_sorted_expert_gemm_kernel<float, false>),
              grid,
              block,
              0,
              stream,
              static_cast<const float*>(a_ptr),
              static_cast<const float*>(b_ptr),
              static_cast<float*>(out_ptr),
              idx_ptr,
              batch,
              K,
              N,
              b_expert_stride,
              static_cast<int>(ldb));
        }
        break;
      case bfloat16:
        if (b_transposed) {
          hipLaunchKernelGGL(
              (moe_sorted_expert_gemm_kernel<hip_bfloat16, true>),
              grid,
              block,
              0,
              stream,
              static_cast<const hip_bfloat16*>(a_ptr),
              static_cast<const hip_bfloat16*>(b_ptr),
              static_cast<hip_bfloat16*>(out_ptr),
              idx_ptr,
              batch,
              K,
              N,
              b_expert_stride,
              static_cast<int>(ldb));
        } else {
          hipLaunchKernelGGL(
              (moe_sorted_expert_gemm_kernel<hip_bfloat16, false>),
              grid,
              block,
              0,
              stream,
              static_cast<const hip_bfloat16*>(a_ptr),
              static_cast<const hip_bfloat16*>(b_ptr),
              static_cast<hip_bfloat16*>(out_ptr),
              idx_ptr,
              batch,
              K,
              N,
              b_expert_stride,
              static_cast<int>(ldb));
        }
        break;
      case float16:
        if (b_transposed) {
          hipLaunchKernelGGL(
              (moe_sorted_expert_gemm_kernel<__half, true>),
              grid,
              block,
              0,
              stream,
              static_cast<const __half*>(a_ptr),
              static_cast<const __half*>(b_ptr),
              static_cast<__half*>(out_ptr),
              idx_ptr,
              batch,
              K,
              N,
              b_expert_stride,
              static_cast<int>(ldb));
        } else {
          hipLaunchKernelGGL(
              (moe_sorted_expert_gemm_kernel<__half, false>),
              grid,
              block,
              0,
              stream,
              static_cast<const __half*>(a_ptr),
              static_cast<const __half*>(b_ptr),
              static_cast<__half*>(out_ptr),
              idx_ptr,
              batch,
              K,
              N,
              b_expert_stride,
              static_cast<int>(ldb));
        }
        break;
      default:
        throw std::runtime_error("Unsupported dtype for moe_sorted_expert_gemm");
    }
  });
}

void moe_sorted_segments(
    CommandEncoder& encoder,
    const array& expert_ids,
    array& segments,
    int T,
    int E) {
  encoder.set_input_array(expert_ids);
  encoder.set_output_array(segments);
  const uint32_t* idx = gpu_ptr<const uint32_t>(expert_ids);
  uint32_t* seg = gpu_ptr<uint32_t>(segments);
  encoder.launch_kernel([=](hipStream_t stream) {
    dim3 block(256);
    dim3 grid((E + 255) / 256);
    hipLaunchKernelGGL(
        moe_sorted_segments_kernel, grid, block, 0, stream, idx, seg, T, E);
  });
}

void moe_max_run_length(
    CommandEncoder& encoder,
    const array& expert_ids,
    array& max_run,
    int T,
    int E) {
  encoder.set_input_array(expert_ids);
  encoder.set_output_array(max_run);
  const uint32_t* idx = gpu_ptr<const uint32_t>(expert_ids);
  int* mxp = gpu_ptr<int>(max_run);
  encoder.launch_kernel([=](hipStream_t stream) {
    (void)hipMemsetAsync(mxp, 0, sizeof(int), stream);
    dim3 block(256);
    dim3 grid((E + 255) / 256);
    hipLaunchKernelGGL(
        moe_max_run_length_kernel, grid, block, 0, stream, idx, mxp, T, E);
  });
}

int moe_max_run_length_sync(
    CommandEncoder& encoder,
    const array& expert_ids,
    int T,
    int E) {
  if (T <= 0 || E <= 0)
    return 32;
  // Scalar int32 on device (shape {1} — empty Shape{} not always constructible).
  array max_run({1}, int32, nullptr, {});
  max_run.set_data(rocm::malloc_async(sizeof(int), encoder));
  encoder.add_temporary(max_run);
  moe_max_run_length(encoder, expert_ids, max_run, T, E);

  static thread_local int* pin = nullptr;
  if (!pin) {
    if (hipHostMalloc(
            reinterpret_cast<void**>(&pin), sizeof(int), hipHostMallocDefault) !=
        hipSuccess) {
      throw std::runtime_error("moe_max_run_length_sync: hipHostMalloc failed");
    }
  }
  hipStream_t hs = static_cast<hipStream_t>(encoder.stream());
  if (hipMemcpyAsync(
          pin,
          gpu_ptr<const int>(max_run),
          sizeof(int),
          hipMemcpyDeviceToHost,
          hs) != hipSuccess) {
    throw std::runtime_error("moe_max_run_length_sync: D2H failed");
  }
  if (hipStreamSynchronize(hs) != hipSuccess) {
    throw std::runtime_error("moe_max_run_length_sync: sync failed");
  }
  int m = *pin;
  if (m < 1)
    m = 1;
  if (m > T)
    m = T;
  m = (m + 31) & ~31;
  if (m < 32)
    m = 32;
  return m;
}

namespace {

int align32_clamp(int m, int T) {
  if (m < 1)
    m = 1;
  if (m > T)
    m = T;
  m = (m + 31) & ~31;
  if (m < 32)
    m = 32;
  if (m > T)
    m = (T + 31) & ~31;
  if (m < 32)
    m = 32;
  return m;
}

// Heuristic: 2× average tokens/expert (sorted MoE rarely worse). No GPU sync.
int moe_pack_m_pad_heuristic(int T, int E) {
  if (T <= 0)
    return 32;
  if (E <= 0)
    E = 1;
  int avg = (T + E - 1) / E;
  int m = avg * 2;
  if (m < 64)
    m = 64;
  return align32_clamp(m, T);
}

} // namespace

int moe_pack_m_pad(
    CommandEncoder& encoder,
    const array& expert_ids,
    int T,
    int E) {
  if (T <= 0 || E <= 0)
    return 32;

  static const int pad_mode = [] {
    // 0 = heuristic only (no sync)
    // 1 = cache max-run (default) — one sync per (T,E)
    // 2 = always sync
    const char* e = std::getenv("MLX_ROCM_MOE_PAD_SYNC");
    if (!e || !*e)
      return 1;
    if (e[0] == '0' || e[0] == 'n' || e[0] == 'N' || e[0] == 'f' || e[0] == 'F')
      return 0;
    if (e[0] == 'a' || e[0] == 'A' || e[0] == '2') // always
      return 2;
    return 1;
  }();

  if (pad_mode == 0)
    return moe_pack_m_pad_heuristic(T, E);
  if (pad_mode == 2)
    return moe_max_run_length_sync(encoder, expert_ids, T, E);

  // Cached exact max-run (+12.5% slack). First call for this (T,E) syncs once;
  // later MoE layers in the same step reuse without StreamSynchronize.
  // rocprof showed ~150 syncs/step from uncached max_run — this kills that.
  static std::mutex mu;
  static std::unordered_map<uint64_t, int> cache;
  const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(T)) << 32) |
      static_cast<uint32_t>(E);
  {
    std::lock_guard<std::mutex> g(mu);
    auto it = cache.find(key);
    if (it != cache.end())
      return it->second;
  }
  int measured = moe_max_run_length_sync(encoder, expert_ids, T, E);
  int floor_h = moe_pack_m_pad_heuristic(T, E);
  // Take max(measured, 2×avg heuristic) then +50% slack so later batches with
  // the same T but worse load never overflow pack (silent slot drop → IMA).
  int m = measured > floor_h ? measured : floor_h;
  m = align32_clamp(m + m / 2, T);
  {
    std::lock_guard<std::mutex> g(mu);
    auto it = cache.find(key);
    if (it != cache.end() && it->second > m)
      m = it->second; // grow-only
    cache[key] = m;
  }
  return m;
}

// ---------------------------------------------------------------------------
// MoE pack workspace (grow-only bump pool)
// ---------------------------------------------------------------------------
namespace {

struct MoeWs {
  std::mutex mu;
  void* base{nullptr};
  size_t capacity{0};
  size_t offset{0};
  int device{-1};
  // Previous bases kept mapped for process lifetime so in-flight kernels that
  // still reference an old slab after grow never see a free.
  std::vector<void*> retired;
};

MoeWs& moe_ws() {
  static MoeWs* w = new MoeWs;
  return *w;
}

size_t align_up_256(size_t n) {
  return (n + 255) & ~size_t(255);
}

} // namespace

void moe_ws_begin(CommandEncoder& encoder, size_t bytes_needed) {
  auto& w = moe_ws();
  std::lock_guard<std::mutex> g(w.mu);
  w.offset = 0;
  if (bytes_needed == 0)
    return;
  // Pad a bit so tiny under-estimates don't force a mid-primitive grow.
  size_t need = align_up_256(bytes_needed + (1 << 20));
  int dev = encoder.device().hip_device();
  if (need <= w.capacity && w.base != nullptr && w.device == dev)
    return;

  size_t new_cap = need + need / 2;
  // 2 MiB pages — reduces fragmentation and matches common HBM alloc granularity.
  constexpr size_t kPage = 2ull << 20;
  new_cap = (new_cap + kPage - 1) & ~(kPage - 1);
  if (new_cap < (64ull << 20))
    new_cap = 64ull << 20;

  void* p = nullptr;
  // Synchronous hipMalloc once per grow (rare). Avoids hipMallocAsync storm and
  // does not thrash the serialized mem pool under concurrent free.
  hipError_t err = hipMalloc(&p, new_cap);
  if (err != hipSuccess || !p) {
    (void)hipGetLastError();
    throw std::runtime_error(
        "moe_ws_begin: hipMalloc failed for MoE workspace (" +
        std::to_string(new_cap) + " bytes)");
  }
  if (w.base)
    w.retired.push_back(w.base);
  w.base = p;
  w.capacity = new_cap;
  w.device = dev;
  w.offset = 0;
}

void moe_ws_alloc(CommandEncoder& encoder, array& a) {
  size_t nbytes = a.nbytes();
  size_t aligned = align_up_256(nbytes == 0 ? 256 : nbytes);

  auto& w = moe_ws();
  std::lock_guard<std::mutex> g(w.mu);
  if (!w.base || w.offset + aligned > w.capacity) {
    // Grow while holding the lock. In-flight arrays from *this* primitive still
    // point into the old base (kept in retired) — only new allocs use new base.
    size_t need = w.offset + aligned + (1 << 20);
    size_t new_cap = need + need / 2;
    constexpr size_t kPage = 2ull << 20;
    new_cap = (new_cap + kPage - 1) & ~(kPage - 1);
    void* p = nullptr;
    hipError_t err = hipMalloc(&p, new_cap);
    if (err != hipSuccess || !p) {
      (void)hipGetLastError();
      throw std::runtime_error("moe_ws_alloc: hipMalloc grow failed");
    }
    if (w.base)
      w.retired.push_back(w.base);
    w.base = p;
    w.capacity = new_cap;
    w.device = encoder.device().hip_device();
    // New base starts empty; prior shells still reference retired bases.
    w.offset = 0;
  }

  void* ptr = static_cast<char*>(w.base) + w.offset;
  w.offset += aligned;

  // Shell RocmBuffer; device memory owned by the pool, never freed per-array.
  auto* rb = new RocmBuffer{
      ptr,
      nbytes,
      /*is_managed=*/false,
      w.device,
      /*host_shadow=*/nullptr,
      /*host_dirty=*/false,
      /*alloc_stream=*/encoder.stream()};
  a.set_data(allocator::Buffer{rb}, [](allocator::Buffer b) {
    // Free only the wrapper — HBM stays in the grow-only pool.
    delete static_cast<RocmBuffer*>(b.ptr());
  });
}

void segmented_mm_device(
    CommandEncoder& encoder,
    const array& a,
    const array& b,
    const array& segments,
    array& out,
    int M,
    int N,
    int num_segments,
    bool a_transposed,
    int64_t lda,
    int64_t a_k_stride,
    bool b_transposed,
    int64_t ldb,
    int64_t b_k_stride,
    int64_t out_stride) {
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(segments);
  encoder.set_output_array(out);

  const void* a_ptr = gpu_ptr<void>(a);
  const void* b_ptr = gpu_ptr<void>(b);
  void* out_ptr = gpu_ptr<void>(out);
  const uint32_t* seg_ptr = gpu_ptr<const uint32_t>(segments);

  dim3 block(TILE_N, TILE_M);
  dim3 grid(
      (N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M, num_segments);

  encoder.launch_kernel([=](hipStream_t stream) {
    auto launch = [&](auto* typed_a, auto* typed_b, auto* typed_c) {
      // typed_a is const T* — strip pointer AND const so Acc is not const float.
      using T = std::remove_const_t<std::remove_pointer_t<decltype(typed_a)>>;
      if (a_transposed && b_transposed) {
        hipLaunchKernelGGL(
            (segmented_mm_device_kernel<T, true, true>),
            grid,
            block,
            0,
            stream,
            typed_a,
            typed_b,
            typed_c,
            seg_ptr,
            M,
            N,
            a_k_stride,
            b_k_stride,
            out_stride,
            static_cast<int>(lda),
            static_cast<int>(ldb));
      } else if (a_transposed && !b_transposed) {
        hipLaunchKernelGGL(
            (segmented_mm_device_kernel<T, true, false>),
            grid,
            block,
            0,
            stream,
            typed_a,
            typed_b,
            typed_c,
            seg_ptr,
            M,
            N,
            a_k_stride,
            b_k_stride,
            out_stride,
            static_cast<int>(lda),
            static_cast<int>(ldb));
      } else if (!a_transposed && b_transposed) {
        hipLaunchKernelGGL(
            (segmented_mm_device_kernel<T, false, true>),
            grid,
            block,
            0,
            stream,
            typed_a,
            typed_b,
            typed_c,
            seg_ptr,
            M,
            N,
            a_k_stride,
            b_k_stride,
            out_stride,
            static_cast<int>(lda),
            static_cast<int>(ldb));
      } else {
        hipLaunchKernelGGL(
            (segmented_mm_device_kernel<T, false, false>),
            grid,
            block,
            0,
            stream,
            typed_a,
            typed_b,
            typed_c,
            seg_ptr,
            M,
            N,
            a_k_stride,
            b_k_stride,
            out_stride,
            static_cast<int>(lda),
            static_cast<int>(ldb));
      }
    };
    switch (a.dtype()) {
      case float32:
        launch(
            static_cast<const float*>(a_ptr),
            static_cast<const float*>(b_ptr),
            static_cast<float*>(out_ptr));
        break;
      case bfloat16:
        launch(
            static_cast<const hip_bfloat16*>(a_ptr),
            static_cast<const hip_bfloat16*>(b_ptr),
            static_cast<hip_bfloat16*>(out_ptr));
        break;
      case float16:
        launch(
            static_cast<const __half*>(a_ptr),
            static_cast<const __half*>(b_ptr),
            static_cast<__half*>(out_ptr));
        break;
      default:
        throw std::runtime_error("Unsupported dtype for segmented_mm_device");
    }
  });
}

} // namespace mlx::core::rocm
