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

  if (row < M && col < N) {
    Acc sum = Acc(0);

    for (int k = 0; k < K; ++k) {
      Acc a_val;
      Acc b_val;

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

} // namespace mlx::core::rocm
