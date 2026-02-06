// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/conv/conv.h"
#include "mlx/backend/rocm/gemms/naive_gemm.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace {

// N-dimensional grouped unfold kernel
template <typename T, int NDIM>
__global__ void naive_grouped_unfold_transpose_nd(
    const T* __restrict__ in,
    T* __restrict__ out,
    int filter_size,
    int out_pixels,
    ConvParams<NDIM> params) {
  
  int index_batch = blockIdx.z / out_pixels;
  int index_out_spatial = blockIdx.z % out_pixels;
  int index_wt_spatial = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (index_wt_spatial >= filter_size / params.C) {
    return;
  }
  
  in += blockIdx.y;  // Channel offset
  out += blockIdx.z * filter_size + blockIdx.y * (filter_size / params.C);
  
  bool valid = index_batch < params.N;
  
  // Get coordinates in input
  int index_in[NDIM] = {};
  int wt_stride = 1;
  int tmp_out_spatial = index_out_spatial;
  int tmp_wt_spatial = index_wt_spatial;
  
  for (int i = NDIM - 1; i >= 0; --i) {
    int index_out = tmp_out_spatial % params.out_spatial_dims[i];
    int index_wt = tmp_wt_spatial % params.wt_spatial_dims[i];
    out += index_wt * wt_stride;
    
    if (params.flip) {
      index_wt = params.wt_spatial_dims[i] - index_wt - 1;
    }
    
    int index = index_out * params.strides[i] - params.padding[i] +
        index_wt * params.kernel_dilation[i];
    int index_max = 1 + params.input_dilation[i] * (params.in_spatial_dims[i] - 1);
    
    valid &= (index >= 0) && (index < index_max) &&
        (index % params.input_dilation[i] == 0);
    
    index_in[i] = index / params.input_dilation[i];
    
    tmp_out_spatial /= params.out_spatial_dims[i];
    tmp_wt_spatial /= params.wt_spatial_dims[i];
    wt_stride *= params.wt_spatial_dims[i];
  }
  
  if (valid) {
    int64_t in_offset = index_batch * params.in_strides[0];
    for (int i = 0; i < NDIM; ++i) {
      in_offset += index_in[i] * params.in_strides[i + 1];
    }
    *out = in[in_offset];
  } else {
    *out = T{0};
  }
}

// Helper to launch unfold kernel for specific NDIM
template <int NDIM>
void launch_unfold_kernel(
    hipStream_t stream,
    const array& in,
    array& unfolded,
    dim3 num_blocks,
    dim3 block_dims,
    int filter_size,
    int out_pixels,
    const ConvParams<NDIM>& params) {
  
  switch (in.dtype()) {
    case float32:
      naive_grouped_unfold_transpose_nd<float, NDIM><<<num_blocks, block_dims, 0, stream>>>(
          in.data<float>(), unfolded.data<float>(),
          filter_size, out_pixels, params);
      break;
    case float16:
      naive_grouped_unfold_transpose_nd<__half, NDIM><<<num_blocks, block_dims, 0, stream>>>(
          in.data<__half>(), unfolded.data<__half>(),
          filter_size, out_pixels, params);
      break;
    case bfloat16:
      naive_grouped_unfold_transpose_nd<hip_bfloat16, NDIM><<<num_blocks, block_dims, 0, stream>>>(
          in.data<hip_bfloat16>(), unfolded.data<hip_bfloat16>(),
          filter_size, out_pixels, params);
      break;
    default:
      throw std::runtime_error("Unsupported dtype for conv unfold");
  }
}

// Implementation for specific NDIM
template <int NDIM>
void gemm_conv_nd(
    rocm::CommandEncoder& encoder,
    const array& in,
    const array& wt,
    array& out,
    const std::vector<int>& strides,
    const std::vector<int>& padding,
    const std::vector<int>& kernel_dilation,
    const std::vector<int>& input_dilation,
    bool flip,
    Stream s) {
  
  ConvParams<NDIM> params(
      in, wt, out, strides, padding, kernel_dilation, input_dilation, 1, flip);
  
  int mat_M = out.size() / params.O;
  int mat_K = wt.size() / params.O;
  int mat_N = params.O;
  
  int filter_size = params.C;
  for (int i = 0; i < NDIM; ++i) {
    filter_size *= params.wt_spatial_dims[i];
  }
  
  int out_pixels = 1;
  for (int i = 0; i < NDIM; ++i) {
    out_pixels *= params.out_spatial_dims[i];
  }
  
  array unfolded({mat_M, mat_K}, in.dtype(), nullptr, {});
  unfolded.set_data(allocator::malloc(unfolded.nbytes()));
  encoder.add_temporary(unfolded);
  
  int wt_spatial_size = mat_K / params.C;
  dim3 block_dims(std::min(std::max(wt_spatial_size, 32), 1024));
  dim3 num_blocks(
      (wt_spatial_size + block_dims.x - 1) / block_dims.x,
      params.C,
      mat_M);
  
  encoder.set_input_array(in);
  encoder.set_output_array(unfolded);
  
  encoder.launch_kernel([&](hipStream_t stream) {
    launch_unfold_kernel<NDIM>(
        stream, in, unfolded, num_blocks, block_dims,
        filter_size, out_pixels, params);
  });
  
  int wt_spatial_total = 1;
  for (int i = 0; i < NDIM; ++i) {
    wt_spatial_total *= params.wt_spatial_dims[i];
  }
  
  array wt_view({params.O, params.C, wt_spatial_total}, wt.dtype(), nullptr, {});
  wt_view.copy_shared_buffer(
      wt, {wt.strides(0), 1, params.C}, wt.flags(), wt.size());
  array wt_reshaped = contiguous_copy_gpu(wt_view, s);
  encoder.add_temporary(wt_reshaped);
  
  rocm::naive_gemm(
      encoder, unfolded, wt_reshaped, out,
      mat_M, mat_N, mat_K,
      false, mat_K, true, mat_K, 1.0f, 0.0f);
}

template <int NDIM>
void gemm_grouped_conv_nd(
    rocm::CommandEncoder& encoder,
    const array& in,
    const array& wt,
    array& out,
    const std::vector<int>& strides,
    const std::vector<int>& padding,
    const std::vector<int>& kernel_dilation,
    const std::vector<int>& input_dilation,
    int groups,
    bool flip,
    Stream s) {
  
  ConvParams<NDIM> params(
      in, wt, out, strides, padding, kernel_dilation, input_dilation, groups, flip);
  
  int C_per_group = params.C / params.groups;
  int O_per_group = params.O / params.groups;
  int mat_M = out.size() / params.O;
  int mat_K = wt.size() / params.O;
  int mat_N = O_per_group;
  
  int filter_size = params.C;
  for (int i = 0; i < NDIM; ++i) {
    filter_size *= params.wt_spatial_dims[i];
  }
  
  int out_pixels = 1;
  for (int i = 0; i < NDIM; ++i) {
    out_pixels *= params.out_spatial_dims[i];
  }
  
  array unfolded({mat_M, mat_K * params.groups}, in.dtype(), nullptr, {});
  unfolded.set_data(allocator::malloc(unfolded.nbytes()));
  encoder.add_temporary(unfolded);
  
  int wt_spatial_size = (mat_K * params.groups) / params.C;
  dim3 block_dims(std::min(std::max(wt_spatial_size, 32), 1024));
  dim3 num_blocks(
      (wt_spatial_size + block_dims.x - 1) / block_dims.x,
      params.C,
      mat_M);
  
  encoder.set_input_array(in);
  encoder.set_output_array(unfolded);
  
  encoder.launch_kernel([&](hipStream_t stream) {
    launch_unfold_kernel<NDIM>(
        stream, in, unfolded, num_blocks, block_dims,
        filter_size, out_pixels, params);
  });
  
  int wt_spatial_total = 1;
  for (int i = 0; i < NDIM; ++i) {
    wt_spatial_total *= params.wt_spatial_dims[i];
  }
  
  array wt_view({params.O, C_per_group, wt_spatial_total}, wt.dtype(), nullptr, {});
  wt_view.copy_shared_buffer(
      wt, {wt.strides(0), 1, C_per_group}, wt.flags(), wt.size());
  array wt_reshaped = contiguous_copy_gpu(wt_view, s);
  encoder.add_temporary(wt_reshaped);
  
  for (int g = 0; g < params.groups; ++g) {
    int64_t a_offset = g * mat_K;
    int64_t b_offset = g * O_per_group * mat_K;
    int64_t c_offset = g * O_per_group;
    
    rocm::naive_gemm_with_offset_ldc(
        encoder, unfolded, wt_reshaped, out,
        mat_M, mat_N, mat_K,
        false, mat_K * params.groups, a_offset,
        true, mat_K, b_offset,
        mat_N * params.groups, c_offset,  // ldc = full output row width
        1.0f, 0.0f);
  }
}

} // namespace

void gemm_conv(
    rocm::CommandEncoder& encoder,
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
  
  switch (conv_ndim) {
    case 1:
      gemm_conv_nd<1>(encoder, in, wt, out, strides, padding,
                      kernel_dilation, input_dilation, flip, s);
      break;
    case 2:
      gemm_conv_nd<2>(encoder, in, wt, out, strides, padding,
                      kernel_dilation, input_dilation, flip, s);
      break;
    case 3:
      gemm_conv_nd<3>(encoder, in, wt, out, strides, padding,
                      kernel_dilation, input_dilation, flip, s);
      break;
    default:
      throw std::runtime_error(
          "[conv] ROCm GEMM-based convolution only supports 1D, 2D, 3D.");
  }
}

void gemm_grouped_conv(
    rocm::CommandEncoder& encoder,
    const array& in,
    const array& wt,
    array& out,
    const std::vector<int>& strides,
    const std::vector<int>& padding,
    const std::vector<int>& kernel_dilation,
    const std::vector<int>& input_dilation,
    int groups,
    bool flip,
    Stream s) {
  
  int conv_ndim = in.ndim() - 2;
  
  switch (conv_ndim) {
    case 1:
      gemm_grouped_conv_nd<1>(encoder, in, wt, out, strides, padding,
                              kernel_dilation, input_dilation, groups, flip, s);
      break;
    case 2:
      gemm_grouped_conv_nd<2>(encoder, in, wt, out, strides, padding,
                              kernel_dilation, input_dilation, groups, flip, s);
      break;
    case 3:
      gemm_grouped_conv_nd<3>(encoder, in, wt, out, strides, padding,
                              kernel_dilation, input_dilation, groups, flip, s);
      break;
    default:
      throw std::runtime_error(
          "[conv] ROCm grouped convolution only supports 1D, 2D, 3D.");
  }
}

} // namespace mlx::core
