// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/gemms/gemv.h"
#include "mlx/backend/rocm/device/config.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/dtype_utils.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

namespace mlx::core::rocm {

static constexpr int rows_per_block = 8;

// Accumulator type selection per input element type T.
template <typename T>
struct GemvAccType {
  using type = T;
};

template <>
struct GemvAccType<__half> {
  using type = float;
};

template <>
struct GemvAccType<hip_bfloat16> {
  using type = float;
};

template <>
struct GemvAccType<float> {
  using type = float;
};

template <>
struct GemvAccType<double> {
  using type = double;
};

// Warp reduction for sum
template <typename T>
__device__ __forceinline__ T warp_reduce_sum_gemv(T val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down(val, offset);
  }
  return val;
}

// Specialization for hip_bfloat16
template <>
__device__ __forceinline__ float warp_reduce_sum_gemv(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down(val, offset);
  }
  return val;
}

template <typename T, int rows_per_block, int n_per_thread>
__device__ void
gemv_impl(const T* mat, const T* vec, T* out, int rows, int cols) {
  int row = blockIdx.x * rows_per_block + threadIdx.y;

  if (row < rows) {
    using Acc = typename GemvAccType<T>::type;
    Acc sum = Acc(0);
    
    // Each thread processes multiple elements
    for (int col = n_per_thread * threadIdx.x; col < cols;
         col += (WARP_SIZE * n_per_thread)) {
      // Load and accumulate
#pragma unroll
      for (int j = 0; j < n_per_thread; ++j) {
        int idx = col + j;
        if (idx < cols) {
          sum += static_cast<Acc>(mat[row * cols + idx]) * static_cast<Acc>(vec[idx]);
        }
      }
    }

    // Warp reduction
    sum = warp_reduce_sum_gemv(sum);
    
    if (threadIdx.x == 0) {
      out[row] = static_cast<T>(sum);
    }
  }
}

template <typename T, int rows_per_block, int n_per_thread>
__global__ void
gemv_single(const T* mat, const T* vec, T* out, int rows, int cols) {
  gemv_impl<T, rows_per_block, n_per_thread>(mat, vec, out, rows, cols);
}

// Helper to compute batch offset
__device__ __forceinline__ int64_t elem_to_loc_1d(
    int64_t idx,
    const int64_t* shape,
    const int64_t* strides,
    int ndim) {
  int64_t offset = 0;
  for (int i = ndim - 1; i >= 0; --i) {
    offset += (idx % shape[i]) * strides[i];
    idx /= shape[i];
  }
  return offset;
}

template <typename T, int rows_per_block, int n_per_thread>
__global__ void gemv_batched(
    const T* mat,
    const T* vec,
    T* out,
    int rows,
    int cols,
    const int64_t* batch_shape,
    const int64_t* mat_batch_strides,
    const int64_t* vec_batch_strides,
    int batch_ndim) {
  int batch_idx = blockIdx.y;
  
  int64_t mat_offset = elem_to_loc_1d(batch_idx, batch_shape, mat_batch_strides, batch_ndim);
  int64_t vec_offset = elem_to_loc_1d(batch_idx, batch_shape, vec_batch_strides, batch_ndim);
  
  gemv_impl<T, rows_per_block, n_per_thread>(
      mat + mat_offset, vec + vec_offset, out + batch_idx * rows, rows, cols);
}

template <typename T, int rows_per_block, int n_per_thread>
__global__ void gemv_gather(
    const T* mat,
    const T* vec,
    T* out,
    const uint32_t* mat_indices,
    const uint32_t* vec_indices,
    int rows,
    int cols,
    int64_t mat_batch_stride,
    int64_t vec_batch_stride) {
  int indices_idx = blockIdx.y;
  
  uint32_t index_mat = mat_indices[indices_idx];
  uint32_t index_vec = vec_indices[indices_idx];

  int64_t mat_offset = index_mat * mat_batch_stride;
  int64_t vec_offset = index_vec * vec_batch_stride;

  gemv_impl<T, rows_per_block, n_per_thread>(
      mat + mat_offset, vec + vec_offset, out + indices_idx * rows, rows, cols);
}

bool can_use_gemv(int M, int N, int K, bool a_transposed, bool b_transposed) {
  return K % 32 == 0 && ((M == 1 && b_transposed) || (N == 1 && !a_transposed));
}

template <typename F>
void dispatch_n_per_thread(int n_per_thread, F&& f) {
  switch (n_per_thread) {
    case 1:
      f(std::integral_constant<int, 1>{});
      break;
    case 2:
      f(std::integral_constant<int, 2>{});
      break;
    case 4:
      f(std::integral_constant<int, 4>{});
      break;
  }
}

void gemv(
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    uint32_t batch_count,
    const mlx::core::Shape& batch_shape,
    const mlx::core::Strides& a_batch_strides,
    const mlx::core::Strides& b_batch_strides,
    CommandEncoder& encoder) {
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);
  
  dim3 block_dims{WARP_SIZE, rows_per_block};
  int rows;
  int cols = K;
  
  // Determine which array is the matrix and which is the vector
  const void* mat_ptr;
  const void* vec_ptr;
  const mlx::core::Strides* mat_strides_ptr;
  const mlx::core::Strides* vec_strides_ptr;
  
  if (M == 1) {
    mat_ptr = b.data<void>();
    vec_ptr = a.data<void>();
    rows = N;
    mat_strides_ptr = &b_batch_strides;
    vec_strides_ptr = &a_batch_strides;
  } else {
    mat_ptr = a.data<void>();
    vec_ptr = b.data<void>();
    rows = M;
    mat_strides_ptr = &a_batch_strides;
    vec_strides_ptr = &b_batch_strides;
  }
  
  uint32_t num_blocks_x = (rows + rows_per_block - 1) / rows_per_block;
  
  // Determine n_per_thread based on alignment
  int n_per_t = 1;
  if (K % 128 == 0) {
    n_per_t = 4;
  } else if (K % 64 == 0) {
    n_per_t = 2;
  }
  
  // For batched operations, allocate device memory for parameters
  int64_t* d_batch_shape = nullptr;
  int64_t* d_mat_strides = nullptr;
  int64_t* d_vec_strides = nullptr;
  
  if (batch_count > 1) {
    size_t batch_ndim = batch_shape.size();
    (void)hipMalloc(&d_batch_shape, batch_ndim * sizeof(int64_t));
    (void)hipMalloc(&d_mat_strides, batch_ndim * sizeof(int64_t));
    (void)hipMalloc(&d_vec_strides, batch_ndim * sizeof(int64_t));
    
    (void)hipMemcpy(d_batch_shape, batch_shape.data(), batch_ndim * sizeof(int64_t), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_mat_strides, mat_strides_ptr->data(), batch_ndim * sizeof(int64_t), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_vec_strides, vec_strides_ptr->data(), batch_ndim * sizeof(int64_t), hipMemcpyHostToDevice);
  }
  
  encoder.launch_kernel([&](hipStream_t stream) {
    auto launch_kernel = [&](auto type_tag, auto n_per_thread) {
      using T = typename decltype(type_tag)::type;
      const T* mat = static_cast<const T*>(mat_ptr);
      const T* vec = static_cast<const T*>(vec_ptr);
      T* out_ptr = out.data<T>();
      
      if (batch_count == 1) {
        hipLaunchKernelGGL(
            (gemv_single<T, rows_per_block, n_per_thread()>),
            dim3(num_blocks_x), block_dims, 0, stream,
            mat, vec, out_ptr, rows, cols);
      } else {
        hipLaunchKernelGGL(
            (gemv_batched<T, rows_per_block, n_per_thread()>),
            dim3(num_blocks_x, batch_count), block_dims, 0, stream,
            mat, vec, out_ptr, rows, cols,
            d_batch_shape,
            d_mat_strides,
            d_vec_strides,
            static_cast<int>(batch_shape.size()));
      }
    };
    
    dispatch_n_per_thread(n_per_t, [&](auto n_per_thread) {
      switch (out.dtype()) {
        case float32:
          launch_kernel(type_identity<float>{}, n_per_thread);
          break;
        case float16:
          launch_kernel(type_identity<__half>{}, n_per_thread);
          break;
        case bfloat16:
          launch_kernel(type_identity<hip_bfloat16>{}, n_per_thread);
          break;
        case float64:
          launch_kernel(type_identity<double>{}, n_per_thread);
          break;
        default:
          break;
      }
    });
  });
  
  // Free device memory after kernel completes
  if (batch_count > 1) {
    (void)hipFree(d_batch_shape);
    (void)hipFree(d_mat_strides);
    (void)hipFree(d_vec_strides);
  }
}

void gather_mv(
    const array& mat_,
    const array& vec_,
    const array& mat_indices,
    const array& vec_indices,
    array& out,
    int N,
    int K,
    CommandEncoder& encoder) {
  encoder.set_input_array(mat_);
  encoder.set_input_array(vec_);
  encoder.set_input_array(mat_indices);
  encoder.set_input_array(vec_indices);
  encoder.set_output_array(out);
  
  dim3 block_dims{WARP_SIZE, rows_per_block};
  int rows = N;
  int cols = K;
  uint32_t batch_size = static_cast<uint32_t>(out.size() / N);
  
  uint32_t num_blocks_x = (rows + rows_per_block - 1) / rows_per_block;
  
  int n_per_t = 1;
  if (K % 128 == 0) {
    n_per_t = 4;
  } else if (K % 64 == 0) {
    n_per_t = 2;
  }
  
  // Compute batch strides for simple case
  int64_t mat_batch_stride = N * K;
  int64_t vec_batch_stride = K;
  
  encoder.launch_kernel([&](hipStream_t stream) {
    auto launch_kernel = [&](auto type_tag, auto n_per_thread) {
      using T = typename decltype(type_tag)::type;
      
      hipLaunchKernelGGL(
          (gemv_gather<T, rows_per_block, n_per_thread()>),
          dim3(num_blocks_x, batch_size), block_dims, 0, stream,
          mat_.data<T>(), vec_.data<T>(), out.data<T>(),
          mat_indices.data<uint32_t>(), vec_indices.data<uint32_t>(),
          rows, cols,
          mat_batch_stride,
          vec_batch_stride);
    };
    
    dispatch_n_per_thread(n_per_t, [&](auto n_per_thread) {
      switch (out.dtype()) {
        case float32:
          launch_kernel(type_identity<float>{}, n_per_thread);
          break;
        case float16:
          launch_kernel(type_identity<__half>{}, n_per_thread);
          break;
        case bfloat16:
          launch_kernel(type_identity<hip_bfloat16>{}, n_per_thread);
          break;
        case float64:
          launch_kernel(type_identity<double>{}, n_per_thread);
          break;
        default:
          break;
      }
    });
  });
}

} // namespace mlx::core::rocm
