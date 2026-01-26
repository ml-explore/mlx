// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/gemms/gemv.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace mlx::core::cu {

namespace cg = cooperative_groups;

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
struct GemvAccType<__nv_bfloat16> {
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

template <>
struct GemvAccType<cu::complex64_t> {
  using type = cu::complex64_t;
};

template <typename T, int rows_per_block, int n_per_thread>
__device__ void
gemv_impl(const T* mat, const T* vec, T* out, int rows, int cols) {
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  auto g_idx = block.group_index();
  auto t_idx = block.thread_index();
  int row = g_idx.x * rows_per_block + t_idx.y;

  if (row < rows) {
    using Acc = typename GemvAccType<T>::type;
    Acc sum = Acc(0);
    for (int col = n_per_thread * warp.thread_rank(); col < cols;
         col += (WARP_SIZE * n_per_thread)) {
      auto local_mat =
          unsafe_load_vector<n_per_thread>(mat + row * cols + col, 0);
      auto local_vec = unsafe_load_vector<n_per_thread>(vec + col, 0);
#pragma unroll
      for (int j = 0; j < n_per_thread; ++j) {
        sum += static_cast<Acc>(local_mat[j]) * static_cast<Acc>(local_vec[j]);
      }
    }

    sum = cg::reduce(warp, sum, cg::plus<Acc>{});
    if (warp.thread_rank() == 0) {
      out[row] = static_cast<T>(sum);
    }
  }
}

template <typename T, int rows_per_block, int n_per_thread>
__global__ void
gemv_single(const T* mat, const T* vec, T* out, int rows, int cols) {
  gemv_impl<T, rows_per_block, n_per_thread>(mat, vec, out, rows, cols);
}

template <typename T, int rows_per_block, int n_per_thread>
__global__ void gemv_batched(
    const T* mat,
    const T* vec,
    T* out,
    int rows,
    int cols,
    const __grid_constant__ Shape batch_shape,
    const __grid_constant__ Strides mat_batch_strides,
    const __grid_constant__ Strides vec_batch_strides,
    int batch_ndim) {
  auto block = cg::this_thread_block();
  auto batch_idx = block.group_index().y;
  auto [vec_offset, mat_offset] = elem_to_loc(
      batch_idx,
      batch_shape.data(),
      vec_batch_strides.data(),
      mat_batch_strides.data(),
      batch_ndim);
  gemv_impl<T, rows_per_block, n_per_thread>(
      mat + mat_offset, vec + vec_offset, out + batch_idx * rows, rows, cols);
}

template <typename T, int rows_per_block, int n_per_thread>
__global__ void gemv_gather(
    const T* mat,
    const T* vec,
    T* out,
    uint32_t* mat_indices,
    uint32_t* vec_indices,
    int rows,
    int cols,
    const __grid_constant__ Shape mat_batch_shape,
    const __grid_constant__ Strides mat_batch_strides,
    int mat_batch_ndim,
    const __grid_constant__ Shape vec_batch_shape,
    const __grid_constant__ Strides vec_batch_strides,
    int vec_batch_ndim,
    const __grid_constant__ Shape index_shape,
    const __grid_constant__ Strides mat_index_strides,
    const __grid_constant__ Strides vec_index_strides,
    int index_batch_ndim) {
  auto block = cg::this_thread_block();
  auto indices_idx = block.group_index().y;
  uint32_t index_mat, index_vec;
  if (index_batch_ndim > 1) {
    auto [mat_idx_offset, vec_idx_offset] = elem_to_loc(
        indices_idx,
        index_shape.data(),
        mat_index_strides.data(),
        vec_index_strides.data(),
        index_batch_ndim);
    index_mat = mat_indices[mat_idx_offset];
    index_vec = vec_indices[vec_idx_offset];
  } else {
    index_mat = mat_indices[indices_idx * mat_index_strides[0]];
    index_vec = vec_indices[indices_idx * vec_index_strides[0]];
  }

  int64_t mat_offset;
  if (mat_batch_ndim > 1) {
    mat_offset = elem_to_loc(
        index_mat,
        mat_batch_shape.data(),
        mat_batch_strides.data(),
        mat_batch_ndim);
  } else {
    mat_offset = index_mat * mat_batch_strides[0];
  }

  int64_t vec_offset;
  if (vec_batch_ndim > 1) {
    vec_offset = elem_to_loc(
        index_vec,
        vec_batch_shape.data(),
        vec_batch_strides.data(),
        vec_batch_ndim);
  } else {
    vec_offset = index_vec * vec_batch_strides[0];
  }

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
  dispatch_inexact_types(out.dtype(), "gemv", [&](auto type_tag) {
    using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    dim3 block_dims{WARP_SIZE, rows_per_block};
    const DataType* mat;
    const DataType* vec;
    int rows;
    int cols = K;
    auto mat_strides = const_param(a_batch_strides);
    auto vec_strides = const_param(b_batch_strides);

    if (M == 1) {
      mat = gpu_ptr<DataType>(b);
      vec = gpu_ptr<DataType>(a);
      rows = N;
      std::swap(mat_strides, vec_strides);
    } else {
      mat = gpu_ptr<DataType>(a);
      vec = gpu_ptr<DataType>(b);
      rows = M;
    }
    uint32_t num_blocks_x = (rows + rows_per_block - 1) / rows_per_block;
    int n_per_t;
    if (K % 128 == 0 && is_aligned<4>(mat) && is_aligned<4>(vec)) {
      n_per_t = 4;
    } else if (K % 64 == 0 && is_aligned<2>(mat) && is_aligned<2>(vec)) {
      n_per_t = 2;
    } else {
      n_per_t = 1;
    }
    dispatch_n_per_thread(n_per_t, [&](auto n_per_thread) {
      if (batch_count == 1) {
        auto kernel = gemv_single<DataType, rows_per_block, n_per_thread()>;
        // Store params in variables to ensure they remain valid
        const DataType* mat_ptr = mat;
        const DataType* vec_ptr = vec;
        DataType* out_ptr = gpu_ptr<DataType>(out);
        int rows_val = rows;
        int cols_val = cols;
        void* params[] = {&mat_ptr, &vec_ptr, &out_ptr, &rows_val, &cols_val};
        encoder.add_kernel_node(
            reinterpret_cast<void*>(kernel),
            num_blocks_x,
            block_dims,
            0,
            params);
      } else {
        auto kernel = gemv_batched<DataType, rows_per_block, n_per_thread()>;
        // Store params in variables to ensure they remain valid
        const DataType* mat_ptr = mat;
        const DataType* vec_ptr = vec;
        DataType* out_ptr = gpu_ptr<DataType>(out);
        int rows_val = rows;
        int cols_val = cols;
        auto batch_shape_param = const_param(batch_shape);
        auto mat_strides_copy = mat_strides;
        auto vec_strides_copy = vec_strides;
        int batch_ndim = batch_shape.size();
        void* params[] = {
            &mat_ptr,
            &vec_ptr,
            &out_ptr,
            &rows_val,
            &cols_val,
            &batch_shape_param,
            &mat_strides_copy,
            &vec_strides_copy,
            &batch_ndim};
        encoder.add_kernel_node(
            reinterpret_cast<void*>(kernel),
            dim3{num_blocks_x, batch_count},
            block_dims,
            0,
            params);
      }
    });
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
  dispatch_inexact_types(out.dtype(), "gather_mv", [&](auto type_tag) {
    using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    dim3 block_dims{WARP_SIZE, rows_per_block};
    int rows = N;
    int cols = K;
    uint32_t batch_size = static_cast<uint32_t>(out.size() / N);
    const DataType* mat = gpu_ptr<DataType>(mat_);
    const DataType* vec = gpu_ptr<DataType>(vec_);

    uint32_t num_blocks_x = (rows + rows_per_block - 1) / rows_per_block;
    int n_per_t;
    if (K % 128 == 0 && is_aligned<4>(mat) && is_aligned<4>(vec)) {
      n_per_t = 4;
    } else if (K % 64 == 0 && is_aligned<2>(mat) && is_aligned<2>(vec)) {
      n_per_t = 2;
    } else {
      n_per_t = 1;
    }

    dispatch_n_per_thread(n_per_t, [&](auto n_per_thread) {
      auto kernel = gemv_gather<DataType, rows_per_block, n_per_thread()>;
      encoder.add_kernel_node(
          kernel,
          dim3{num_blocks_x, batch_size},
          block_dims,
          0,
          mat,
          vec,
          gpu_ptr<DataType>(out),
          gpu_ptr<uint32_t>(mat_indices),
          gpu_ptr<uint32_t>(vec_indices),
          rows,
          cols,
          const_param(mat_.shape()),
          const_param(mat_.strides()),
          mat_.ndim() - 2,
          const_param(vec_.shape()),
          const_param(vec_.strides()),
          vec_.ndim() - 2,
          const_param(mat_indices.shape()),
          const_param(mat_indices.strides()),
          const_param(vec_indices.strides()),
          mat_indices.ndim());
    });
  });
}

} // namespace mlx::core::cu
