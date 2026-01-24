// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/conv/conv.h"
#include "mlx/backend/cuda/gemms/cublas_gemm.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"

#include <cooperative_groups.h>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename T, int NDIM>
__global__ void naive_unfold_nd(
    const T* in,
    T* out,
    int filter_size,
    int out_pixels,
    const __grid_constant__ ConvParams<NDIM> params) {
  auto block = cg::this_thread_block();
  auto tid = block.group_index();
  auto lid = block.thread_index();

  int index_batch = tid.z / out_pixels; // [0, N)
  int index_out_spatial = tid.z % out_pixels; // [0, H_out * W_out)
  int index_wt_spatial =
      tid.x * block.dim_threads().x + lid.x; // [0, H_wt * W_wt)

  if (index_wt_spatial >= filter_size / params.C) {
    return;
  }

  in += tid.y; // [0, C)
  out += tid.z * filter_size + index_wt_spatial * params.C + tid.y;

  bool valid = index_batch < params.N;

  // Get the coordinates in input.
  int index_in[NDIM] = {};
#pragma unroll
  for (int i = NDIM - 1; i >= 0; --i) {
    int index_out = index_out_spatial % params.out_spatial_dims[i];
    int index_wt = index_wt_spatial % params.wt_spatial_dims[i];

    if (params.flip) {
      index_wt = params.wt_spatial_dims[i] - index_wt - 1;
    }

    int index = index_out * params.strides[i] - params.padding[i] +
        index_wt * params.kernel_dilation[i];
    int index_max =
        1 + params.input_dilation[i] * (params.in_spatial_dims[i] - 1);

    valid &= (index >= 0) && (index < index_max) &&
        (index % params.input_dilation[i] == 0);

    index_in[i] = index / params.input_dilation[i];

    index_out_spatial /= params.out_spatial_dims[i];
    index_wt_spatial /= params.wt_spatial_dims[i];
  }

  if (valid) {
    int in_offset = index_batch * params.in_strides[0];
#pragma unroll
    for (int i = 0; i < NDIM; ++i) {
      in_offset += index_in[i] * params.in_strides[i + 1];
    }
    *out = in[in_offset];
  } else {
    *out = T{0};
  }
}

} // namespace cu

template <int NDIM>
array unfold_inputs_nd(
    cu::CommandEncoder& encoder,
    const array& in,
    int mat_M,
    int mat_K,
    int mat_N,
    ConvParams<NDIM>& params) {
  array unfolded({mat_M, mat_K}, in.dtype(), nullptr, {});
  unfolded.set_data(allocator::malloc(unfolded.nbytes()));
  encoder.add_temporary(unfolded);

  int filter_size = params.C;
#pragma unroll
  for (int i = 0; i < NDIM; ++i) {
    filter_size *= params.wt_spatial_dims[i];
  }

  int out_pixels = 1;
#pragma unroll
  for (int i = 0; i < NDIM; ++i) {
    out_pixels *= params.out_spatial_dims[i];
  }

  int wt_spatial_size = mat_K / params.C;
  dim3 block_dims;
  block_dims.x = std::min(std::max(wt_spatial_size, 32), 1024);
  dim3 num_blocks;
  num_blocks.x = cuda::ceil_div(wt_spatial_size, block_dims.x);
  num_blocks.y = params.C;
  num_blocks.z = mat_M;

  encoder.set_input_array(in);
  encoder.set_output_array(unfolded);
  dispatch_float_types(in.dtype(), "unfold", [&](auto type_tag) {
    using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    encoder.add_kernel_node(
        cu::naive_unfold_nd<DataType, NDIM>,
        num_blocks,
        block_dims,
        0,
        in.data<DataType>(),
        unfolded.data<DataType>(),
        filter_size,
        out_pixels,
        params);
  });

  return unfolded;
}

template <int NDIM>
void gemm_conv_nd(
    cu::CommandEncoder& encoder,
    const array& in,
    const array& wt,
    array& out,
    ConvParams<NDIM>& params,
    Stream s) {
  // Get gemm shapes.
  int mat_M = out.size() / params.O; // N * H_out * W_out
  int mat_K = wt.size() / params.O; // C * H_wt * W_wt
  int mat_N = params.O; // O

  // Unfold input to (N * H_out * W_out, C * H_wt * W_wt) for gemm.
  array in_unfolded =
      unfold_inputs_nd<NDIM>(encoder, in, mat_M, mat_K, mat_N, params);

  // Reshape weight to (C * H_wt * W_wt, O) for gemm.
  array wt_reshaped({mat_K, mat_N}, wt.dtype(), nullptr, {});
  wt_reshaped.copy_shared_buffer(
      wt,
      {1, mat_K},
      {false, false, /* col_contiguous */ true},
      wt.data_size());

  // Single batch.
  Shape batch_shape{1};
  Strides a_batch_strides{0};
  Strides b_batch_strides{0};

  // Run matmul.
  CublasGemm gemm(
      encoder.device(),
      in.dtype(),
      false, // a_transposed
      mat_M, // a_rows
      mat_K, // a_cols
      mat_K, // lda
      true, // b_transposed
      mat_K, // b_rows
      mat_N, // b_cols
      mat_K, // ldb
      batch_shape.back(),
      a_batch_strides.back(),
      b_batch_strides.back());
  gemm.run(
      encoder,
      out,
      in_unfolded,
      wt_reshaped,
      batch_shape,
      a_batch_strides,
      b_batch_strides);
}

void gemm_conv(
    cu::CommandEncoder& encoder,
    const array& in,
    const array& wt,
    array& out,
    const std::vector<int>& strides,
    const std::vector<int>& padding,
    const std::vector<int>& kernel_dilation,
    const std::vector<int>& input_dilation,
    bool flip,
    Stream s) {
  int conv_ndim = in.ndim() - 2;
  if (conv_ndim < 1 || conv_ndim > 3) {
    throw std::runtime_error(
        fmt::format("[conv] Unsupported gemm_conv for {}D conv.", conv_ndim));
  }
  dispatch_1_2_3(conv_ndim, [&](auto ndim_constant) {
    ConvParams<ndim_constant()> params(
        in,
        wt,
        out,
        strides,
        padding,
        kernel_dilation,
        input_dilation,
        1, // groups
        flip);
    gemm_conv_nd<ndim_constant()>(encoder, in, wt, out, params, s);
  });
}

} // namespace mlx::core
