// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/qmv.h"
#include "mlx/dtype_utils.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_fp4.h>

namespace mlx::core::cu {

namespace cg = cooperative_groups;

static constexpr int rows_per_block = 8;

template <int bits>
struct Dequantize {
  __device__ float operator()(uint8_t x) {
    if constexpr (bits == 8) {
      return float(*(__nv_fp8_e4m3*)(&x));
    } else {
      return float(*(__nv_fp4_e2m1*)(&x));
    }
  }
};

template <
    typename T,
    int rows_per_block,
    int n_per_thread,
    int bits,
    int group_size,
    bool use_mx_scale>
__device__ void qmv_impl(
    const uint8_t* mat,
    const uint8_t* scales_,
    const T* vec,
    T* out,
    int rows,
    int cols) {
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  constexpr int vals_per_byte = bits == 8 ? 1 : 2;
  constexpr int nv_per_thread = vals_per_byte * n_per_thread;
  auto g_idx = block.group_index();
  auto t_idx = block.thread_index();
  int row = g_idx.x * rows_per_block + t_idx.y;
  using ScaleType =
      std::conditional_t<use_mx_scale, __nv_fp8_e8m0, __nv_fp8_e4m3>;
  auto scales = (ScaleType*)(scales_);
  auto packed_cols = cols / vals_per_byte;

  if (row < rows) {
    // Offset scales to correct row
    int scale_step = (WARP_SIZE * nv_per_thread) / group_size;
    scales += row * (cols / group_size) +
        (warp.thread_rank() * nv_per_thread) / group_size;
    float sum = 0.0f;
    for (int col = n_per_thread * warp.thread_rank(); col < packed_cols;
         col += (WARP_SIZE * n_per_thread)) {
      auto local_mat =
          unsafe_load_vector<n_per_thread>(mat + row * packed_cols + col, 0);
      auto local_vec =
          unsafe_load_vector<nv_per_thread>(vec + vals_per_byte * col, 0);
      float local_sum = 0.0f;
#pragma unroll
      for (int j = 0; j < n_per_thread; ++j) {
        if constexpr (bits == 8) {
          local_sum += Dequantize<bits>{}(local_mat[j]) *
              static_cast<float>(local_vec[j]);
        } else {
          local_sum += Dequantize<bits>{}(local_mat[j]) *
              static_cast<float>(local_vec[2 * j]);
          local_sum += Dequantize<bits>{}(local_mat[j] >> 4) *
              static_cast<float>(local_vec[2 * j + 1]);
        }
      }
      sum += local_sum * float(scales[0]);
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
__global__ void qmv_single(
    const uint8_t* mat,
    const uint8_t* scales,
    const T* vec,
    T* out,
    int rows,
    int cols) {
  qmv_impl<T, rows_per_block, n_per_thread, bits, group_size, use_mx_scale>(
      mat, scales, vec, out, rows, cols);
}

template <
    typename T,
    int rows_per_block,
    int n_per_thread,
    int bits,
    int group_size,
    bool use_mx_scale>
__global__ void qmv_batched(
    const uint8_t* mat,
    const uint8_t* scales,
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
  // TODO not sure about the offset there
  auto [vec_offset, mat_offset] = elem_to_loc(
      batch_idx,
      batch_shape.data(),
      vec_batch_strides.data(),
      mat_batch_strides.data(),
      batch_ndim);
  qmv_impl<T, rows_per_block, n_per_thread, bits, group_size, use_mx_scale>(
      mat + mat_offset, vec + vec_offset, out + batch_idx * rows, rows, cols);
}

void fp_qmv(
    const array& mat,
    const array& scales,
    const array& vec,
    array& out,
    int bits,
    int group_size,
    CommandEncoder& encoder) {
  encoder.set_input_array(mat);
  encoder.set_input_array(scales);
  encoder.set_input_array(vec);
  encoder.set_output_array(out);
  dispatch_float_types(out.dtype(), "qmv", [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    if constexpr (!std::is_same_v<T, double>) {
      dim3 block_dims{WARP_SIZE, rows_per_block};
      int rows = out.shape(-1);
      int cols = vec.shape(-1);
      uint32_t num_blocks_x = (rows + rows_per_block - 1) / rows_per_block;
      constexpr int n_per_t = 32;
      int vals_per_byte = bits == 4 ? 2 : 1;
      // TODO check alignment as well and fall back if unaligned
      // TODO handle edges maybe with fallback kernel
      if (cols % (n_per_t * vals_per_byte) != 0) {
        throw std::runtime_error("NYI");
      }
      auto kernel = qmv_single<T, rows_per_block, n_per_t, 4, 32, true>;
      if (bits == 8) {
        kernel = qmv_single<T, rows_per_block, n_per_t, 8, 32, true>;
      } else if (group_size == 16) {
        kernel = qmv_single<T, rows_per_block, n_per_t, 4, 16, false>;
      }
      encoder.add_kernel_node(
          kernel,
          num_blocks_x,
          block_dims,
          0,
          gpu_ptr<uint8_t>(mat),
          gpu_ptr<uint8_t>(scales),
          gpu_ptr<T>(vec),
          gpu_ptr<T>(out),
          rows,
          cols);
    }
  });
}

} // namespace mlx::core::cu
