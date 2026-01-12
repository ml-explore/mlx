// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device/utils.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/qmv.h"
#include "mlx/backend/cuda/quantized/quantized_utils.cuh"
#include "mlx/dtype_utils.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace mlx::core::cu {

namespace cg = cooperative_groups;

static constexpr int rows_per_block = 8;

template <typename T>
__device__ void adjust_matrix_offsets(
    const T*& x,
    const uint32_t*& w,
    const uint8_t*& scales,
    T*& y,
    int output_stride,
    const int& x_batch_ndims,
    const Shape x_shape,
    const Strides x_strides,
    const int& w_batch_ndims,
    const Shape w_shape,
    const Strides w_strides,
    const Strides s_strides) {
  uint32_t idx = cg::this_grid().block_index().z;
  if (x_batch_ndims == 1) {
    x += idx * x_strides[0];
  } else {
    x += elem_to_loc(idx, x_shape.data(), x_strides.data(), x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += idx * w_strides[0];
    scales += idx * s_strides[0];
  } else {
    auto [w_idx, s_idx] = elem_to_loc(
        idx, w_shape.data(), w_strides.data(), s_strides.data(), w_batch_ndims);
    w += w_idx;
    scales += s_idx;
  }
  y += idx * output_stride;
}

template <
    typename T,
    int rows_per_block,
    int n_per_thread,
    int bits,
    int group_size,
    bool use_mx_scale>
__device__ void fp_qmv_impl(
    const uint32_t* mat,
    const uint8_t* scales_,
    const T* vec,
    T* out,
    int rows,
    int cols) {
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  constexpr int vals_per_item = bits == 8 ? 4 : 8;
  constexpr int nv_per_thread = vals_per_item * n_per_thread;
  auto g_idx = block.group_index();
  auto t_idx = block.thread_index();
  int row = g_idx.y * rows_per_block + t_idx.y;

  vec += g_idx.x * cols;
  out += g_idx.x * rows;

  using ScaleType =
      std::conditional_t<use_mx_scale, __nv_fp8_e8m0, __nv_fp8_e4m3>;
  auto scales = (ScaleType*)(scales_);
  auto packed_cols = cols / vals_per_item;

  if (row < rows) {
    constexpr int scales_per_step = std::max(nv_per_thread / group_size, 1);
    constexpr int scale_step = (WARP_SIZE * nv_per_thread) / group_size;
    constexpr int n_per_step = n_per_thread / scales_per_step;
    // Offset scales to correct row
    scales += row * (cols / group_size) +
        (warp.thread_rank() * nv_per_thread) / group_size;
    float sum = 0.0f;
    for (int col = n_per_thread * warp.thread_rank(); col < packed_cols;
         col += (WARP_SIZE * n_per_thread)) {
      auto local_vec =
          unsafe_load_vector<nv_per_thread>(vec + vals_per_item * col, 0);
      auto local_mat =
          unsafe_load_vector<n_per_thread>(mat + row * packed_cols + col, 0);
#pragma unroll
      for (int i = 0; i < scales_per_step; ++i) {
        float2 local_sum = {0.0f, 0.0f};
#pragma unroll
        for (int j = 0; j < n_per_step; ++j) {
          int k = n_per_step * i + j;
          if constexpr (bits == 8) {
            auto v = dequant_fp8(local_mat[k]);
            local_sum.x +=
                v.x * static_cast<float>(local_vec[vals_per_item * k]);
            local_sum.x +=
                v.y * static_cast<float>(local_vec[vals_per_item * k + 1]);
            local_sum.y +=
                v.z * static_cast<float>(local_vec[vals_per_item * k + 2]);
            local_sum.y +=
                v.w * static_cast<float>(local_vec[vals_per_item * k + 3]);
          } else {
            auto v = dequant_fp4(local_mat[k]);
            local_sum.x +=
                v.x * static_cast<float>(local_vec[vals_per_item * k]);
            local_sum.y +=
                v.y * static_cast<float>(local_vec[vals_per_item * k + 1]);
            local_sum.x +=
                v.z * static_cast<float>(local_vec[vals_per_item * k + 2]);
            local_sum.y +=
                v.w * static_cast<float>(local_vec[vals_per_item * k + 3]);

            v = dequant_fp4(local_mat[k] >> 16);
            local_sum.x +=
                v.x * static_cast<float>(local_vec[vals_per_item * k + 4]);
            local_sum.y +=
                v.y * static_cast<float>(local_vec[vals_per_item * k + 5]);
            local_sum.x +=
                v.z * static_cast<float>(local_vec[vals_per_item * k + 6]);
            local_sum.y +=
                v.w * static_cast<float>(local_vec[vals_per_item * k + 7]);
          }
        }
        sum += (local_sum.x + local_sum.y) * float(scales[i]);
      }
      scales += scale_step;
    }

    sum = cg::reduce(warp, sum, cg::plus<float>{});
    if (warp.thread_rank() == 0) {
      out[row] = static_cast<T>(sum);
    }
  }
}

template <
    typename T,
    int rows_per_block,
    int n_per_thread,
    int bits,
    int group_size,
    bool use_mx_scale>
__global__ void fp_qmv_single(
    const uint32_t* mat,
    const uint8_t* scales,
    const T* vec,
    T* out,
    int rows,
    int cols) {
  fp_qmv_impl<T, rows_per_block, n_per_thread, bits, group_size, use_mx_scale>(
      mat, scales, vec, out, rows, cols);
}

template <
    typename T,
    int rows_per_block,
    int n_per_thread,
    int bits,
    int group_size,
    bool use_mx_scale>
__global__ void fp_qmv_batched(
    const uint32_t* mat,
    const uint8_t* scales,
    const T* vec,
    T* out,
    int rows,
    int cols,
    int vec_batch_ndims,
    const __grid_constant__ Shape vec_shape,
    const __grid_constant__ Strides vec_strides,
    int mat_batch_ndims,
    const __grid_constant__ Shape mat_shape,
    const __grid_constant__ Strides mat_strides,
    const __grid_constant__ Strides scales_strides) {
  adjust_matrix_offsets<T>(
      vec,
      mat,
      scales,
      out,
      rows * vec_shape[vec_batch_ndims],
      vec_batch_ndims,
      vec_shape,
      vec_strides,
      mat_batch_ndims,
      mat_shape,
      mat_strides,
      scales_strides);
  fp_qmv_impl<T, rows_per_block, n_per_thread, bits, group_size, use_mx_scale>(
      mat, scales, vec, out, rows, cols);
}

void fp_qmv(
    const array& mat,
    const array& scales,
    const array& vec,
    array& out,
    int bits,
    int group_size,
    int M,
    int N,
    int K,
    CommandEncoder& encoder) {
  encoder.set_input_array(mat);
  encoder.set_input_array(scales);
  encoder.set_input_array(vec);
  encoder.set_output_array(out);
  dispatch_float_types(out.dtype(), "qmv", [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    if constexpr (!std::is_same_v<T, double>) {
      dim3 block_dims{WARP_SIZE, rows_per_block};
      uint B = out.size() / (M * N);
      uint blocks_y = (N + rows_per_block - 1) / rows_per_block;
      const uint32_t* mat_ptr = gpu_ptr<uint32_t>(mat);
      const T* vec_ptr = gpu_ptr<T>(vec);
      // TODO deal with multiple of 16 but not 32
      bool aligned = cu::is_aligned<4>(mat_ptr);
      aligned &=
          ((bits == 4 && cu::is_aligned<8>(vec_ptr)) ||
           cu::is_aligned<4>(vec_ptr));
      dispatch_bool(aligned, [&](auto aligned) {
        dispatch_bool(B > 1, [&](auto batched) {
          constexpr int n = aligned() ? 4 : 1;
          if (!batched()) {
            auto kernel = fp_qmv_single<T, rows_per_block, n, 4, 32, true>;
            if (bits == 8) {
              kernel = fp_qmv_single<T, rows_per_block, n, 8, 32, true>;
            } else if (group_size == 16) {
              kernel = fp_qmv_single<T, rows_per_block, n, 4, 16, false>;
            }
            encoder.add_kernel_node(
                kernel,
                {static_cast<uint>(M), blocks_y},
                block_dims,
                0,
                mat_ptr,
                gpu_ptr<uint8_t>(scales),
                vec_ptr,
                gpu_ptr<T>(out),
                N,
                K);
          } else {
            auto kernel = fp_qmv_batched<T, rows_per_block, n, 4, 32, true>;
            if (bits == 8) {
              kernel = fp_qmv_batched<T, rows_per_block, n, 8, 32, true>;
            } else if (group_size == 16) {
              kernel = fp_qmv_batched<T, rows_per_block, n, 4, 16, false>;
            }
            encoder.add_kernel_node(
                kernel,
                {static_cast<uint>(M), blocks_y, B},
                block_dims,
                0,
                mat_ptr,
                gpu_ptr<uint8_t>(scales),
                vec_ptr,
                gpu_ptr<T>(out),
                N,
                K,
                vec.ndim() - 2,
                const_param(vec.shape()),
                const_param(vec.strides()),
                mat.ndim() - 2,
                const_param(mat.shape()),
                const_param(mat.strides()),
                const_param(scales.strides()));
          }
        });
      });
    }
  });
}

} // namespace mlx::core::cu
