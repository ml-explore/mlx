// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device/config.h"
#include "mlx/backend/rocm/gemms/gemv.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/dtype_utils.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

static constexpr int rows_per_block = 16;
static constexpr int kMaxInlineBatchDims = 8;

struct GemvBatchParams {
  int batch_ndim;
  int32_t batch_shape[kMaxInlineBatchDims];
  int64_t mat_batch_strides[kMaxInlineBatchDims];
  int64_t vec_batch_strides[kMaxInlineBatchDims];
};

struct GemvGatherParams {
  int mat_batch_ndim;
  int vec_batch_ndim;
  int index_batch_ndim;
  int32_t mat_batch_shape[kMaxInlineBatchDims];
  int64_t mat_batch_strides[kMaxInlineBatchDims];
  int32_t vec_batch_shape[kMaxInlineBatchDims];
  int64_t vec_batch_strides[kMaxInlineBatchDims];
  int32_t index_shape[kMaxInlineBatchDims];
  int64_t mat_index_strides[kMaxInlineBatchDims];
  int64_t vec_index_strides[kMaxInlineBatchDims];
};

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
      // Load and accumulate using vectorized loads if possible
      auto mat_v = load_vector<n_per_thread>(mat + row * cols, col / n_per_thread, cols, T(0));
      auto vec_v = load_vector<n_per_thread>(vec, col / n_per_thread, cols, T(0));

#pragma unroll
      for (int j = 0; j < n_per_thread; ++j) {
        sum += static_cast<Acc>(mat_v[j]) * static_cast<Acc>(vec_v[j]);
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
template <typename ShapeT>
__device__ __forceinline__ int64_t elem_to_loc_1d(
    int64_t idx,
    const ShapeT* shape,
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
    const int32_t* batch_shape,
    const int64_t* mat_batch_strides,
    const int64_t* vec_batch_strides,
    int batch_ndim) {
  int batch_idx = blockIdx.y;

  int64_t mat_offset =
      elem_to_loc_1d(batch_idx, batch_shape, mat_batch_strides, batch_ndim);
  int64_t vec_offset =
      elem_to_loc_1d(batch_idx, batch_shape, vec_batch_strides, batch_ndim);

  gemv_impl<T, rows_per_block, n_per_thread>(
      mat + mat_offset, vec + vec_offset, out + batch_idx * rows, rows, cols);
}

template <typename T, int rows_per_block, int n_per_thread>
__global__ void gemv_batched_inline(
    const T* mat,
    const T* vec,
    T* out,
    int rows,
    int cols,
    GemvBatchParams params) {
  int batch_idx = blockIdx.y;

  int64_t mat_offset = elem_to_loc_1d(
      batch_idx,
      params.batch_shape,
      params.mat_batch_strides,
      params.batch_ndim);
  int64_t vec_offset = elem_to_loc_1d(
      batch_idx,
      params.batch_shape,
      params.vec_batch_strides,
      params.batch_ndim);

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
    const int32_t* mat_batch_shape,
    const int64_t* mat_batch_strides,
    int mat_batch_ndim,
    const int32_t* vec_batch_shape,
    const int64_t* vec_batch_strides,
    int vec_batch_ndim,
    const int32_t* index_shape,
    const int64_t* mat_index_strides,
    const int64_t* vec_index_strides,
    int index_batch_ndim);

__device__ __forceinline__ uint32_t gather_index(
    const uint32_t* indices,
    int64_t indices_idx,
    const int32_t* index_shape,
    const int64_t* index_strides,
    int index_batch_ndim) {
  if (index_batch_ndim > 1) {
    auto index_offset = elem_to_loc_1d(
        indices_idx, index_shape, index_strides, index_batch_ndim);
    return indices[index_offset];
  }
  if (index_batch_ndim == 1) {
    return indices[indices_idx * index_strides[0]];
  }
  return indices[0];
}

__device__ __forceinline__ int64_t gather_batch_offset(
    uint32_t index,
    const int32_t* batch_shape,
    const int64_t* batch_strides,
    int batch_ndim) {
  if (batch_ndim > 1) {
    return elem_to_loc_1d(index, batch_shape, batch_strides, batch_ndim);
  }
  if (batch_ndim == 1) {
    return index * batch_strides[0];
  }
  return 0;
}

template <typename T, int rows_per_block, int n_per_thread>
__device__ void gemv_gather_impl(
    const T* mat,
    const T* vec,
    T* out,
    const uint32_t* mat_indices,
    const uint32_t* vec_indices,
    int rows,
    int cols,
    int indices_idx,
    const int32_t* mat_batch_shape,
    const int64_t* mat_batch_strides,
    int mat_batch_ndim,
    const int32_t* vec_batch_shape,
    const int64_t* vec_batch_strides,
    int vec_batch_ndim,
    const int32_t* index_shape,
    const int64_t* mat_index_strides,
    const int64_t* vec_index_strides,
    int index_batch_ndim) {
  uint32_t index_mat = gather_index(
      mat_indices,
      indices_idx,
      index_shape,
      mat_index_strides,
      index_batch_ndim);
  uint32_t index_vec = gather_index(
      vec_indices,
      indices_idx,
      index_shape,
      vec_index_strides,
      index_batch_ndim);

  int64_t mat_offset = gather_batch_offset(
      index_mat, mat_batch_shape, mat_batch_strides, mat_batch_ndim);
  int64_t vec_offset = gather_batch_offset(
      index_vec, vec_batch_shape, vec_batch_strides, vec_batch_ndim);

  gemv_impl<T, rows_per_block, n_per_thread>(
      mat + mat_offset, vec + vec_offset, out + indices_idx * rows, rows, cols);
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
    const int32_t* mat_batch_shape,
    const int64_t* mat_batch_strides,
    int mat_batch_ndim,
    const int32_t* vec_batch_shape,
    const int64_t* vec_batch_strides,
    int vec_batch_ndim,
    const int32_t* index_shape,
    const int64_t* mat_index_strides,
    const int64_t* vec_index_strides,
    int index_batch_ndim) {
  int indices_idx = blockIdx.y;

  gemv_gather_impl<T, rows_per_block, n_per_thread>(
      mat,
      vec,
      out,
      mat_indices,
      vec_indices,
      rows,
      cols,
      indices_idx,
      mat_batch_shape,
      mat_batch_strides,
      mat_batch_ndim,
      vec_batch_shape,
      vec_batch_strides,
      vec_batch_ndim,
      index_shape,
      mat_index_strides,
      vec_index_strides,
      index_batch_ndim);
}

template <typename T, int rows_per_block, int n_per_thread>
__global__ void gemv_gather_inline(
    const T* mat,
    const T* vec,
    T* out,
    const uint32_t* mat_indices,
    const uint32_t* vec_indices,
    int rows,
    int cols,
    GemvGatherParams params) {
  int indices_idx = blockIdx.y;

  gemv_gather_impl<T, rows_per_block, n_per_thread>(
      mat,
      vec,
      out,
      mat_indices,
      vec_indices,
      rows,
      cols,
      indices_idx,
      params.mat_batch_shape,
      params.mat_batch_strides,
      params.mat_batch_ndim,
      params.vec_batch_shape,
      params.vec_batch_strides,
      params.vec_batch_ndim,
      params.index_shape,
      params.mat_index_strides,
      params.vec_index_strides,
      params.index_batch_ndim);
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
    case 8:
      f(std::integral_constant<int, 8>{});
      break;
    case 16:
      f(std::integral_constant<int, 16>{});
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
    mat_ptr = gpu_ptr<void>(b);
    vec_ptr = gpu_ptr<void>(a);
    rows = N;
    mat_strides_ptr = &b_batch_strides;
    vec_strides_ptr = &a_batch_strides;
  } else {
    mat_ptr = gpu_ptr<void>(a);
    vec_ptr = gpu_ptr<void>(b);
    rows = M;
    mat_strides_ptr = &a_batch_strides;
    vec_strides_ptr = &b_batch_strides;
  }
  void* out_base_ptr = gpu_ptr<void>(out);

  uint32_t num_blocks_x = (rows + rows_per_block - 1) / rows_per_block;

  // Determine n_per_thread based on alignment
  int n_per_t = 1;
  if (K % 512 == 0) {
    n_per_t = 16;
  } else if (K % 256 == 0) {
    n_per_t = 8;
  } else if (K % 128 == 0) {
    n_per_t = 4;
  } else if (K % 64 == 0) {
    n_per_t = 2;
  }

  // For batched operations, allocate device memory for parameters
  int32_t* d_batch_shape = nullptr;
  int64_t* d_mat_strides = nullptr;
  int64_t* d_vec_strides = nullptr;
  GemvBatchParams inline_batch_params{};
  bool use_inline_batch_params = false;

  if (batch_count > 1) {
    size_t batch_ndim = batch_shape.size();
    if (batch_ndim <= kMaxInlineBatchDims) {
      use_inline_batch_params = true;
      inline_batch_params.batch_ndim = static_cast<int>(batch_ndim);
      for (size_t i = 0; i < batch_ndim; ++i) {
        inline_batch_params.batch_shape[i] = batch_shape[i];
        inline_batch_params.mat_batch_strides[i] = (*mat_strides_ptr)[i];
        inline_batch_params.vec_batch_strides[i] = (*vec_strides_ptr)[i];
      }
    } else {
      (void)hipMalloc(&d_batch_shape, batch_ndim * sizeof(int32_t));
      (void)hipMalloc(&d_mat_strides, batch_ndim * sizeof(int64_t));
      (void)hipMalloc(&d_vec_strides, batch_ndim * sizeof(int64_t));

      (void)hipMemcpy(
          d_batch_shape,
          batch_shape.data(),
          batch_ndim * sizeof(int32_t),
          hipMemcpyHostToDevice);
      (void)hipMemcpy(
          d_mat_strides,
          mat_strides_ptr->data(),
          batch_ndim * sizeof(int64_t),
          hipMemcpyHostToDevice);
      (void)hipMemcpy(
          d_vec_strides,
          vec_strides_ptr->data(),
          batch_ndim * sizeof(int64_t),
          hipMemcpyHostToDevice);
    }
  }

  if (batch_count > 1 && !use_inline_batch_params) {
    // Rare fallback: batch_ndim > kMaxInlineBatchDims. Strides live in
    // stream-ordered device memory freed via hipFreeAsync, so keep this on the
    // legacy stream-capture path.
    encoder.launch_kernel([&,
                           mat_ptr,
                           vec_ptr,
                           out_base_ptr,
                           d_batch_shape,
                           d_mat_strides,
                           d_vec_strides](hipStream_t stream) {
      auto launch_kernel = [&](auto type_tag, auto n_per_thread) {
        using T = typename decltype(type_tag)::type;
        const T* mat = static_cast<const T*>(mat_ptr);
        const T* vec = static_cast<const T*>(vec_ptr);
        T* out_ptr = static_cast<T*>(out_base_ptr);
        hipLaunchKernelGGL(
            (gemv_batched<T, rows_per_block, n_per_thread()>),
            dim3(num_blocks_x, batch_count),
            block_dims,
            0,
            stream,
            mat,
            vec,
            out_ptr,
            rows,
            cols,
            d_batch_shape,
            d_mat_strides,
            d_vec_strides,
            static_cast<int>(batch_shape.size()));
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

      (void)hipFreeAsync(d_batch_shape, stream);
      (void)hipFreeAsync(d_mat_strides, stream);
      (void)hipFreeAsync(d_vec_strides, stream);
    });
    return;
  }

  auto add_node = [&](auto type_tag, auto n_per_thread) {
    using T = typename decltype(type_tag)::type;
    const T* mat = static_cast<const T*>(mat_ptr);
    const T* vec = static_cast<const T*>(vec_ptr);
    T* out_ptr = static_cast<T*>(out_base_ptr);

    if (batch_count == 1) {
      encoder.add_kernel_node(
          &gemv_single<T, rows_per_block, n_per_thread()>,
          dim3(num_blocks_x),
          block_dims,
          0u,
          mat,
          vec,
          out_ptr,
          rows,
          cols);
    } else {
      encoder.add_kernel_node(
          &gemv_batched_inline<T, rows_per_block, n_per_thread()>,
          dim3(num_blocks_x, batch_count),
          block_dims,
          0u,
          mat,
          vec,
          out_ptr,
          rows,
          cols,
          inline_batch_params);
    }
  };

  dispatch_n_per_thread(n_per_t, [&](auto n_per_thread) {
    switch (out.dtype()) {
      case float32:
        add_node(type_identity<float>{}, n_per_thread);
        break;
      case float16:
        add_node(type_identity<__half>{}, n_per_thread);
        break;
      case bfloat16:
        add_node(type_identity<hip_bfloat16>{}, n_per_thread);
        break;
      case float64:
        add_node(type_identity<double>{}, n_per_thread);
        break;
      default:
        break;
    }
  });
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

  auto [index_shape, index_strides] = collapse_contiguous_dims(
      mat_indices.shape(), {mat_indices.strides(), vec_indices.strides()});
  auto mat_index_strides = index_strides[0];
  auto vec_index_strides = index_strides[1];

  mlx::core::Shape mat_batch_shape{
      mat_.shape().begin(), mat_.shape().end() - 2};
  mlx::core::Strides mat_batch_strides{
      mat_.strides().begin(), mat_.strides().end() - 2};
  int mat_batch_ndim = mat_batch_shape.size();

  mlx::core::Shape vec_batch_shape{
      vec_.shape().begin(), vec_.shape().end() - 2};
  mlx::core::Strides vec_batch_strides{
      vec_.strides().begin(), vec_.strides().end() - 2};
  int vec_batch_ndim = vec_batch_shape.size();

  int index_batch_ndim = index_shape.size();

  int32_t* d_mat_batch_shape = nullptr;
  int64_t* d_mat_batch_strides = nullptr;
  int32_t* d_vec_batch_shape = nullptr;
  int64_t* d_vec_batch_strides = nullptr;
  int32_t* d_index_shape = nullptr;
  int64_t* d_mat_index_strides = nullptr;
  int64_t* d_vec_index_strides = nullptr;

  GemvGatherParams inline_gather_params{};
  bool use_inline_gather_params = mat_batch_ndim <= kMaxInlineBatchDims &&
      vec_batch_ndim <= kMaxInlineBatchDims &&
      index_batch_ndim <= kMaxInlineBatchDims;

  if (use_inline_gather_params) {
    inline_gather_params.mat_batch_ndim = mat_batch_ndim;
    inline_gather_params.vec_batch_ndim = vec_batch_ndim;
    inline_gather_params.index_batch_ndim = index_batch_ndim;
    for (int i = 0; i < mat_batch_ndim; ++i) {
      inline_gather_params.mat_batch_shape[i] = mat_batch_shape[i];
      inline_gather_params.mat_batch_strides[i] = mat_batch_strides[i];
    }
    for (int i = 0; i < vec_batch_ndim; ++i) {
      inline_gather_params.vec_batch_shape[i] = vec_batch_shape[i];
      inline_gather_params.vec_batch_strides[i] = vec_batch_strides[i];
    }
    for (int i = 0; i < index_batch_ndim; ++i) {
      inline_gather_params.index_shape[i] = index_shape[i];
      inline_gather_params.mat_index_strides[i] = mat_index_strides[i];
      inline_gather_params.vec_index_strides[i] = vec_index_strides[i];
    }
  } else {
    auto copy_shape_to_device = [](const mlx::core::Shape& shape,
                                   int32_t** dst_shape) {
      if (shape.empty()) {
        return;
      }
      (void)hipMalloc(dst_shape, shape.size() * sizeof(int32_t));
      (void)hipMemcpy(
          *dst_shape,
          shape.data(),
          shape.size() * sizeof(int32_t),
          hipMemcpyHostToDevice);
    };

    auto copy_strides_to_device = [](const mlx::core::Strides& strides,
                                     int64_t** dst_strides) {
      if (strides.empty()) {
        return;
      }
      (void)hipMalloc(dst_strides, strides.size() * sizeof(int64_t));
      (void)hipMemcpy(
          *dst_strides,
          strides.data(),
          strides.size() * sizeof(int64_t),
          hipMemcpyHostToDevice);
    };

    copy_shape_to_device(mat_batch_shape, &d_mat_batch_shape);
    copy_strides_to_device(mat_batch_strides, &d_mat_batch_strides);
    copy_shape_to_device(vec_batch_shape, &d_vec_batch_shape);
    copy_strides_to_device(vec_batch_strides, &d_vec_batch_strides);
    copy_shape_to_device(index_shape, &d_index_shape);
    copy_strides_to_device(mat_index_strides, &d_mat_index_strides);
    copy_strides_to_device(vec_index_strides, &d_vec_index_strides);
  }

  const void* mat_ptr = gpu_ptr<void>(mat_);
  const void* vec_ptr = gpu_ptr<void>(vec_);
  void* out_ptr = gpu_ptr<void>(out);
  const uint32_t* mat_indices_ptr = gpu_ptr<const uint32_t>(mat_indices);
  const uint32_t* vec_indices_ptr = gpu_ptr<const uint32_t>(vec_indices);

  if (!use_inline_gather_params) {
    // Rare fallback: batch dims exceed kMaxInlineBatchDims. Params live in
    // stream-ordered device memory freed via hipFreeAsync, so keep this on the
    // legacy stream-capture path.
    encoder.launch_kernel([&,
                           mat_ptr,
                           vec_ptr,
                           out_ptr,
                           mat_indices_ptr,
                           vec_indices_ptr,
                           d_mat_batch_shape,
                           d_mat_batch_strides,
                           d_vec_batch_shape,
                           d_vec_batch_strides,
                           d_index_shape,
                           d_mat_index_strides,
                           d_vec_index_strides](hipStream_t stream) {
      auto launch_kernel = [&](auto type_tag, auto n_per_thread) {
        using T = typename decltype(type_tag)::type;
        hipLaunchKernelGGL(
            (gemv_gather<T, rows_per_block, n_per_thread()>),
            dim3(num_blocks_x, batch_size),
            block_dims,
            0,
            stream,
            static_cast<const T*>(mat_ptr),
            static_cast<const T*>(vec_ptr),
            static_cast<T*>(out_ptr),
            mat_indices_ptr,
            vec_indices_ptr,
            rows,
            cols,
            d_mat_batch_shape,
            d_mat_batch_strides,
            mat_batch_ndim,
            d_vec_batch_shape,
            d_vec_batch_strides,
            vec_batch_ndim,
            d_index_shape,
            d_mat_index_strides,
            d_vec_index_strides,
            index_batch_ndim);
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

      if (d_mat_batch_shape != nullptr) {
        (void)hipFreeAsync(d_mat_batch_shape, stream);
      }
      if (d_mat_batch_strides != nullptr) {
        (void)hipFreeAsync(d_mat_batch_strides, stream);
      }
      if (d_vec_batch_shape != nullptr) {
        (void)hipFreeAsync(d_vec_batch_shape, stream);
      }
      if (d_vec_batch_strides != nullptr) {
        (void)hipFreeAsync(d_vec_batch_strides, stream);
      }
      if (d_index_shape != nullptr) {
        (void)hipFreeAsync(d_index_shape, stream);
      }
      if (d_mat_index_strides != nullptr) {
        (void)hipFreeAsync(d_mat_index_strides, stream);
      }
      if (d_vec_index_strides != nullptr) {
        (void)hipFreeAsync(d_vec_index_strides, stream);
      }
    });
    return;
  }

  auto add_node = [&](auto type_tag, auto n_per_thread) {
    using T = typename decltype(type_tag)::type;
    encoder.add_kernel_node(
        &gemv_gather_inline<T, rows_per_block, n_per_thread()>,
        dim3(num_blocks_x, batch_size),
        block_dims,
        0u,
        static_cast<const T*>(mat_ptr),
        static_cast<const T*>(vec_ptr),
        static_cast<T*>(out_ptr),
        mat_indices_ptr,
        vec_indices_ptr,
        rows,
        cols,
        inline_gather_params);
  };

  dispatch_n_per_thread(n_per_t, [&](auto n_per_thread) {
    switch (out.dtype()) {
      case float32:
        add_node(type_identity<float>{}, n_per_thread);
        break;
      case float16:
        add_node(type_identity<__half>{}, n_per_thread);
        break;
      case bfloat16:
        add_node(type_identity<hip_bfloat16>{}, n_per_thread);
        break;
      case float64:
        add_node(type_identity<double>{}, n_per_thread);
        break;
      default:
        break;
    }
  });
}

} // namespace mlx::core::rocm
