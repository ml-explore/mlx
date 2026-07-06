// Copyright © 2025 Apple Inc.

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/rocm/conv/conv.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/gemms/naive_gemm.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/dtype_utils.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace {

template <typename T>
__global__ void depthwise_conv1d_kernel(
    const T* __restrict__ in,
    const T* __restrict__ wt,
    T* __restrict__ out,
    ConvParams<1> params) {
  int out_channel = blockIdx.x * blockDim.x + threadIdx.x;
  int out_pos = blockIdx.y;
  int batch = blockIdx.z;

  if (out_channel >= params.O || out_pos >= params.out_spatial_dims[0] ||
      batch >= params.N) {
    return;
  }

  float acc = 0.0f;
  int kernel_size = params.wt_spatial_dims[0];
  int index_max =
      1 + params.input_dilation[0] * (params.in_spatial_dims[0] - 1);

  for (int k = 0; k < kernel_size; ++k) {
    int k_input = params.flip ? (kernel_size - 1 - k) : k;
    int in_index = out_pos * params.strides[0] - params.padding[0] +
        k_input * params.kernel_dilation[0];
    if (in_index >= 0 && in_index < index_max &&
        (in_index % params.input_dilation[0] == 0)) {
      int in_pos = in_index / params.input_dilation[0];
      int64_t in_offset = static_cast<int64_t>(batch) * params.in_strides[0] +
          static_cast<int64_t>(in_pos) * params.in_strides[1] +
          static_cast<int64_t>(out_channel) * params.in_strides[2];
      int64_t wt_offset = static_cast<int64_t>(out_channel) * kernel_size + k;
      acc +=
          static_cast<float>(in[in_offset]) * static_cast<float>(wt[wt_offset]);
    }
  }

  int64_t out_offset =
      (static_cast<int64_t>(batch) * params.out_spatial_dims[0] + out_pos) *
          params.O +
      out_channel;
  out[out_offset] = static_cast<T>(acc);
}

void depthwise_conv1d(
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
  (void)s;
  ConvParams<1> params(
      in,
      wt,
      out,
      strides,
      padding,
      kernel_dilation,
      input_dilation,
      groups,
      flip);

  int block_size = 256;
  dim3 block_dims(block_size);
  dim3 num_blocks(
      (params.O + block_size - 1) / block_size,
      params.out_spatial_dims[0],
      params.N);

  encoder.set_input_array(in);
  encoder.set_input_array(wt);
  encoder.set_output_array(out);

  encoder.launch_kernel([&](hipStream_t stream) {
    switch (in.dtype()) {
      case float32:
        depthwise_conv1d_kernel<float><<<num_blocks, block_dims, 0, stream>>>(
            gpu_ptr<float>(in), gpu_ptr<float>(wt), gpu_ptr<float>(out), params);
        break;
      case float16:
        depthwise_conv1d_kernel<__half><<<num_blocks, block_dims, 0, stream>>>(
            gpu_ptr<__half>(in), gpu_ptr<__half>(wt), gpu_ptr<__half>(out), params);
        break;
      case bfloat16:
        depthwise_conv1d_kernel<hip_bfloat16>
            <<<num_blocks, block_dims, 0, stream>>>(
                gpu_ptr<hip_bfloat16>(in),
                gpu_ptr<hip_bfloat16>(wt),
                gpu_ptr<hip_bfloat16>(out),
                params);
        break;
      default:
        throw std::runtime_error("Unsupported dtype for depthwise conv1d");
    }
  });
}

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

  in += blockIdx.y; // Channel offset
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
    int index_max =
        1 + params.input_dilation[i] * (params.in_spatial_dims[i] - 1);

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
      naive_grouped_unfold_transpose_nd<float, NDIM>
          <<<num_blocks, block_dims, 0, stream>>>(
              gpu_ptr<float>(in),
              gpu_ptr<float>(unfolded),
              filter_size,
              out_pixels,
              params);
      break;
    case float16:
      naive_grouped_unfold_transpose_nd<__half, NDIM>
          <<<num_blocks, block_dims, 0, stream>>>(
              gpu_ptr<__half>(in),
              gpu_ptr<__half>(unfolded),
              filter_size,
              out_pixels,
              params);
      break;
    case bfloat16:
      naive_grouped_unfold_transpose_nd<hip_bfloat16, NDIM>
          <<<num_blocks, block_dims, 0, stream>>>(
              gpu_ptr<hip_bfloat16>(in),
              gpu_ptr<hip_bfloat16>(unfolded),
              filter_size,
              out_pixels,
              params);
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

  bool is_pointwise = !flip;
  for (int i = 0; i < NDIM; ++i) {
    is_pointwise = is_pointwise && params.wt_spatial_dims[i] == 1 &&
        params.strides[i] == 1 && params.padding[i] == 0 &&
        params.kernel_dilation[i] == 1 && params.input_dilation[i] == 1;
  }

  if (is_pointwise) {
    array wt_2d({params.O, params.C}, wt.dtype(), nullptr, {});
    wt_2d.copy_shared_buffer(
        wt, {wt.strides(0), wt.strides(-1)}, wt.flags(), wt.size());
    array wt_contig = contiguous_copy_gpu(wt_2d, s);
    encoder.add_temporary(wt_contig);

    rocm::naive_gemm(
        encoder,
        in,
        wt_contig,
        out,
        mat_M,
        mat_N,
        mat_K,
        false,
        mat_K,
        true,
        mat_K,
        1.0f,
        0.0f);
    return;
  }

  int filter_size = params.C;
  for (int i = 0; i < NDIM; ++i) {
    filter_size *= params.wt_spatial_dims[i];
  }

  int out_pixels = 1;
  for (int i = 0; i < NDIM; ++i) {
    out_pixels *= params.out_spatial_dims[i];
  }

  array unfolded({mat_M, mat_K}, in.dtype(), nullptr, {});
  unfolded.set_data(mlx::core::rocm::malloc_async(unfolded.nbytes(), encoder));
  encoder.add_temporary(unfolded);

  int wt_spatial_size = mat_K / params.C;
  dim3 block_dims(std::min(std::max(wt_spatial_size, 32), 1024));
  dim3 num_blocks(
      (wt_spatial_size + block_dims.x - 1) / block_dims.x, params.C, mat_M);

  encoder.set_input_array(in);
  encoder.set_output_array(unfolded);

  encoder.launch_kernel([&](hipStream_t stream) {
    launch_unfold_kernel<NDIM>(
        stream,
        in,
        unfolded,
        num_blocks,
        block_dims,
        filter_size,
        out_pixels,
        params);
  });

  int wt_spatial_total = 1;
  for (int i = 0; i < NDIM; ++i) {
    wt_spatial_total *= params.wt_spatial_dims[i];
  }

  array wt_view(
      {params.O, params.C, wt_spatial_total}, wt.dtype(), nullptr, {});
  wt_view.copy_shared_buffer(
      wt, {wt.strides(0), 1, params.C}, wt.flags(), wt.size());
  array wt_reshaped = contiguous_copy_gpu(wt_view, s);
  encoder.add_temporary(wt_reshaped);

  rocm::naive_gemm(
      encoder,
      unfolded,
      wt_reshaped,
      out,
      mat_M,
      mat_N,
      mat_K,
      false,
      mat_K,
      true,
      mat_K,
      1.0f,
      0.0f);
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
      in,
      wt,
      out,
      strides,
      padding,
      kernel_dilation,
      input_dilation,
      groups,
      flip);

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
  unfolded.set_data(mlx::core::rocm::malloc_async(unfolded.nbytes(), encoder));
  encoder.add_temporary(unfolded);

  int wt_spatial_size = (mat_K * params.groups) / params.C;
  dim3 block_dims(std::min(std::max(wt_spatial_size, 32), 1024));
  dim3 num_blocks(
      (wt_spatial_size + block_dims.x - 1) / block_dims.x, params.C, mat_M);

  encoder.set_input_array(in);
  encoder.set_output_array(unfolded);

  encoder.launch_kernel([&](hipStream_t stream) {
    launch_unfold_kernel<NDIM>(
        stream,
        in,
        unfolded,
        num_blocks,
        block_dims,
        filter_size,
        out_pixels,
        params);
  });

  int wt_spatial_total = 1;
  for (int i = 0; i < NDIM; ++i) {
    wt_spatial_total *= params.wt_spatial_dims[i];
  }

  array wt_view(
      {params.O, C_per_group, wt_spatial_total}, wt.dtype(), nullptr, {});
  wt_view.copy_shared_buffer(
      wt, {wt.strides(0), 1, C_per_group}, wt.flags(), wt.size());
  array wt_reshaped = contiguous_copy_gpu(wt_view, s);
  encoder.add_temporary(wt_reshaped);

  for (int g = 0; g < params.groups; ++g) {
    int64_t a_offset = g * mat_K;
    int64_t b_offset = g * O_per_group * mat_K;
    int64_t c_offset = g * O_per_group;

    rocm::naive_gemm_with_offset_ldc(
        encoder,
        unfolded,
        wt_reshaped,
        out,
        mat_M,
        mat_N,
        mat_K,
        false,
        mat_K * params.groups,
        a_offset,
        true,
        mat_K,
        b_offset,
        mat_N * params.groups,
        c_offset, // ldc = full output row width
        1.0f,
        0.0f);
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
      gemm_conv_nd<1>(
          encoder,
          in,
          wt,
          out,
          strides,
          padding,
          kernel_dilation,
          input_dilation,
          flip,
          s);
      break;
    case 2:
      gemm_conv_nd<2>(
          encoder,
          in,
          wt,
          out,
          strides,
          padding,
          kernel_dilation,
          input_dilation,
          flip,
          s);
      break;
    case 3:
      gemm_conv_nd<3>(
          encoder,
          in,
          wt,
          out,
          strides,
          padding,
          kernel_dilation,
          input_dilation,
          flip,
          s);
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

  // Depthwise 1D convolution with channel multiplier 1 (C == O == groups)
  // is a common decode-time pattern (e.g. Qwen3-Next linear attention).
  // Running it through unfold + per-group GEMMs is very launch-heavy.
  // Use a direct kernel in this configuration.
  if (conv_ndim == 1 && in.shape(-1) == groups && wt.shape(0) == groups &&
      out.shape(-1) == groups && wt.shape(-1) == 1) {
    depthwise_conv1d(
        encoder,
        in,
        wt,
        out,
        strides,
        padding,
        kernel_dilation,
        input_dilation,
        groups,
        flip,
        s);
    return;
  }

  switch (conv_ndim) {
    case 1:
      gemm_grouped_conv_nd<1>(
          encoder,
          in,
          wt,
          out,
          strides,
          padding,
          kernel_dilation,
          input_dilation,
          groups,
          flip,
          s);
      break;
    case 2:
      gemm_grouped_conv_nd<2>(
          encoder,
          in,
          wt,
          out,
          strides,
          padding,
          kernel_dilation,
          input_dilation,
          groups,
          flip,
          s);
      break;
    case 3:
      gemm_grouped_conv_nd<3>(
          encoder,
          in,
          wt,
          out,
          strides,
          padding,
          kernel_dilation,
          input_dilation,
          groups,
          flip,
          s);
      break;
    default:
      throw std::runtime_error(
          "[conv] ROCm grouped convolution only supports 1D, 2D, 3D.");
  }
}

} // namespace mlx::core
