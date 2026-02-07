// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/fast_primitives.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

namespace mlx::core {

namespace rocm {

// Single position RoPE implementation (B=1, T=1)
template <typename T, bool traditional, bool forward>
__device__ void rope_single_impl(
    const T* in,
    T* out,
    int32_t offset,
    float inv_freq,
    float scale,
    int64_t stride,
    uint2 pos,
    uint2 dims) {
  float L = scale * static_cast<float>(offset);

  // Compute costheta, sintheta
  float theta = L * inv_freq;
  float costheta = cosf(theta);
  float sintheta = sinf(theta);

  // Compute the input and output indices
  uint32_t index_1, index_2;
  if (traditional) {
    index_1 = 2 * pos.x + pos.y * stride;
    index_2 = index_1 + 1;
  } else {
    index_1 = pos.x + pos.y * stride;
    index_2 = index_1 + dims.x;
  }

  // Read and write the output
  float x1 = static_cast<float>(in[index_1]);
  float x2 = static_cast<float>(in[index_2]);
  float rx1;
  float rx2;
  if (forward) {
    rx1 = x1 * costheta - x2 * sintheta;
    rx2 = x1 * sintheta + x2 * costheta;
  } else {
    rx1 = x2 * sintheta + x1 * costheta;
    rx2 = x2 * costheta - x1 * sintheta;
  }
  out[index_1] = static_cast<T>(rx1);
  out[index_2] = static_cast<T>(rx2);
}

template <typename T, bool traditional, bool forward>
__global__ void rope_single(
    const T* in,
    T* out,
    const int32_t* offset,
    float scale,
    float base,
    int64_t stride,
    uint2 dims) {
  uint2 pos = make_uint2(
      blockIdx.x * blockDim.x + threadIdx.x,
      blockIdx.y * blockDim.y + threadIdx.y);
  if (pos.x >= dims.x || pos.y >= dims.y) {
    return;
  }

  float d = static_cast<float>(pos.x) / static_cast<float>(dims.x);
  float inv_freq = exp2f(-d * base);
  rope_single_impl<T, traditional, forward>(
      in, out, *offset, inv_freq, scale, stride, pos, dims);
}

template <typename T, bool traditional, bool forward>
__global__ void rope_single_freqs(
    const T* in,
    T* out,
    const int32_t* offset,
    const float* freqs,
    float scale,
    int64_t stride,
    uint2 dims,
    int64_t freq_stride) {
  uint2 pos = make_uint2(
      blockIdx.x * blockDim.x + threadIdx.x,
      blockIdx.y * blockDim.y + threadIdx.y);
  if (pos.x >= dims.x || pos.y >= dims.y) {
    return;
  }

  float inv_freq = 1.0f / freqs[freq_stride * pos.x];
  rope_single_impl<T, traditional, forward>(
      in, out, *offset, inv_freq, scale, stride, pos, dims);
}

// General RoPE implementation with batching
template <typename T, bool traditional, bool forward, int N = 4>
__device__ void rope_impl(
    const T* in,
    T* out,
    const int* offset,
    float inv_freq,
    float scale,
    const hip_array<int64_t, 3> strides,
    const hip_array<int64_t, 3> out_strides,
    int64_t offset_stride,
    int n_head,
    uint3 pos,
    uint3 dims) {
  auto n_head_up = N * ((n_head + N - 1) / N);
  auto head_idx = static_cast<int>((pos.z * N) % n_head_up);
  auto batch_idx = (pos.z * N) / n_head_up;
  auto batch_offset = offset[batch_idx * offset_stride];
  float L = scale * static_cast<float>(pos.y + batch_offset);
  auto mat_idx = batch_idx * n_head + head_idx;

  // Compute costheta, sintheta
  float theta = L * inv_freq;
  float costheta = cosf(theta);
  float sintheta = sinf(theta);

  // Compute the input and output indices
  size_t in_index_1, in_index_2;
  size_t out_index_1, out_index_2;
  if (traditional) {
    out_index_1 = 2 * pos.x * out_strides[2] + pos.y * out_strides[1] +
        mat_idx * out_strides[0];
    out_index_2 = out_index_1 + 1;
    in_index_1 =
        2 * pos.x * strides[2] + pos.y * strides[1] + mat_idx * strides[0];
    in_index_2 = in_index_1 + strides[2];
  } else {
    out_index_1 = pos.x * out_strides[2] + pos.y * out_strides[1] +
        mat_idx * out_strides[0];
    out_index_2 = out_index_1 + dims.x * out_strides[2];
    in_index_1 = pos.x * strides[2] + pos.y * strides[1] + mat_idx * strides[0];
    in_index_2 = in_index_1 + dims.x * strides[2];
  }
  for (int i = 0; i < N && head_idx + i < n_head; ++i) {
    // Read and write the output
    float x1 = static_cast<float>(in[in_index_1]);
    float x2 = static_cast<float>(in[in_index_2]);
    float rx1;
    float rx2;
    if (forward) {
      rx1 = x1 * costheta - x2 * sintheta;
      rx2 = x1 * sintheta + x2 * costheta;
    } else {
      rx1 = x2 * sintheta + x1 * costheta;
      rx2 = x2 * costheta - x1 * sintheta;
    }
    out[out_index_1] = static_cast<T>(rx1);
    out[out_index_2] = static_cast<T>(rx2);
    in_index_1 += strides[0];
    in_index_2 += strides[0];
    out_index_1 += out_strides[0];
    out_index_2 += out_strides[0];
  }
}

template <typename T, bool traditional, bool forward>
__global__ void rope(
    const T* in,
    T* out,
    const int32_t* offset,
    float scale,
    float base,
    const hip_array<int64_t, 3> strides,
    const hip_array<int64_t, 3> out_strides,
    int64_t offset_stride,
    int n_head,
    uint3 dims) {
  uint3 pos = make_uint3(
      blockIdx.x * blockDim.x + threadIdx.x,
      blockIdx.y * blockDim.y + threadIdx.y,
      blockIdx.z * blockDim.z + threadIdx.z);
  if (pos.x >= dims.x || pos.y >= dims.y || pos.z >= dims.z) {
    return;
  }

  float d = static_cast<float>(pos.x) / static_cast<float>(dims.x);
  float inv_freq = exp2f(-d * base);
  rope_impl<T, traditional, forward>(
      in,
      out,
      offset,
      inv_freq,
      scale,
      strides,
      out_strides,
      offset_stride,
      n_head,
      pos,
      dims);
}

template <typename T, bool traditional, bool forward>
__global__ void rope_freqs(
    const T* in,
    T* out,
    const int32_t* offset,
    const float* freqs,
    float scale,
    float base,
    const hip_array<int64_t, 3> strides,
    const hip_array<int64_t, 3> out_strides,
    int64_t offset_stride,
    int n_head,
    uint3 dims,
    int64_t freq_stride) {
  uint3 pos = make_uint3(
      blockIdx.x * blockDim.x + threadIdx.x,
      blockIdx.y * blockDim.y + threadIdx.y,
      blockIdx.z * blockDim.z + threadIdx.z);
  if (pos.x >= dims.x || pos.y >= dims.y || pos.z >= dims.z) {
    return;
  }

  float inv_freq = 1.0f / freqs[freq_stride * pos.x];
  rope_impl<T, traditional, forward>(
      in,
      out,
      offset,
      inv_freq,
      scale,
      strides,
      out_strides,
      offset_stride,
      n_head,
      pos,
      dims);
}

// Helper to get grid and block dimensions
inline std::pair<dim3, dim3> get_grid_and_block(uint32_t x, uint32_t y, uint32_t z) {
  dim3 block(16, 16, 1);
  dim3 grid(
      (x + block.x - 1) / block.x,
      (y + block.y - 1) / block.y,
      z);
  return {grid, block};
}

} // namespace rocm

namespace fast {

bool RoPE::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void RoPE::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);
  auto& in = inputs[0];
  auto& offset = inputs[1];
  auto& out = outputs[0];

  rocm::hip_array<int64_t, 3> strides;
  rocm::hip_array<int64_t, 3> out_strides;
  bool donated = false;
  int ndim = in.ndim();

  int B = in.shape(0);
  int T = in.shape(-2);
  int D = in.shape(-1);
  size_t mat_size = T * D;
  int dispatch_ndim = ndim;
  while (in.shape(-dispatch_ndim) == 1 && dispatch_ndim > 3) {
    dispatch_ndim--;
  }

  int N = 1;
  for (int i = 1; i < (ndim - 2); ++i) {
    N *= in.shape(i);
  }

  // We apply rope to less than the whole vector so copy to output and then
  // apply in-place.
  if (dims_ < D) {
    donated = true;
    auto ctype =
        (in.flags().row_contiguous) ? CopyType::Vector : CopyType::General;
    copy_gpu(in, out, ctype, s);
    strides[0] = mat_size;
    strides[1] = out.strides()[ndim - 2];
    strides[2] = out.strides()[ndim - 1];
  }

  // Either copy or apply in-place
  else if (in.flags().row_contiguous) {
    if (in.is_donatable()) {
      donated = true;
      out.copy_shared_buffer(in);
    } else {
      out.set_data(allocator::malloc(out.nbytes()));
    }
    strides[0] = mat_size;
    strides[1] = in.strides()[ndim - 2];
    strides[2] = in.strides()[ndim - 1];
  } else if (dispatch_ndim == 3) {
    // Handle non-contiguous 3D inputs
    out.set_data(allocator::malloc(out.nbytes()));
    strides[0] = in.strides()[ndim - 3];
    strides[1] = in.strides()[ndim - 2];
    strides[2] = in.strides()[ndim - 1];
  } else {
    // Copy non-contiguous > 3D inputs into the output and treat
    // input as donated
    donated = true;
    copy_gpu(in, out, CopyType::General, s);
    strides[0] = mat_size;
    strides[1] = out.strides()[ndim - 2];
    strides[2] = out.strides()[ndim - 1];
  }
  out_strides[0] = mat_size;
  out_strides[1] = out.strides()[ndim - 2];
  out_strides[2] = out.strides()[ndim - 1];

  // Some flags to help us dispatch below
  bool single = in.flags().row_contiguous && B == 1 && T == 1;
  bool with_freqs = inputs.size() == 3;

  encoder.set_input_array(donated ? out : in);
  encoder.set_input_array(offset);
  if (with_freqs) {
    encoder.set_input_array(inputs[2]);
  }
  encoder.set_output_array(out);

  // Helper lambda to launch kernels - avoids structured binding capture issues
  auto launch_rope_single = [&](auto kernel, dim3 grid, dim3 block, uint2 dims) {
    encoder.launch_kernel([&, grid, block, dims](hipStream_t stream) {
      hipLaunchKernelGGL(
          kernel,
          grid, block, 0, stream,
          gpu_ptr<typename std::remove_pointer<decltype(kernel)>::type::first_argument_type>(donated ? out : in),
          gpu_ptr<typename std::remove_pointer<decltype(kernel)>::type::first_argument_type>(out),
          gpu_ptr<int32_t>(offset),
          scale_,
          std::log2(base_),
          mat_size,
          dims);
    });
  };

  // Dispatch based on dtype
  dispatch_float_types(out.dtype(), "rope", [&](auto type_tag) {
    using CTYPE = MLX_GET_TYPE(type_tag);
    using DataType = hip_type_t<CTYPE>;
    
    // Get grid/block dimensions outside the lambda to avoid C++20 structured binding capture
    if (single && !with_freqs) {
      uint2 dims2 = make_uint2(dims_ / 2, N);
      std::pair<dim3, dim3> gb = rocm::get_grid_and_block(dims2.x, dims2.y, 1);
      dim3 grid = gb.first;
      dim3 block = gb.second;
      
      encoder.launch_kernel([=, &encoder, &out, &in, &offset, this](hipStream_t stream) {
        if (traditional_ && forward_) {
          hipLaunchKernelGGL(
              (rocm::rope_single<DataType, true, true>),
              grid, block, 0, stream,
              gpu_ptr<DataType>(donated ? out : in),
              gpu_ptr<DataType>(out),
              gpu_ptr<int32_t>(offset),
              scale_,
              std::log2(base_),
              mat_size,
              dims2);
        } else if (traditional_ && !forward_) {
          hipLaunchKernelGGL(
              (rocm::rope_single<DataType, true, false>),
              grid, block, 0, stream,
              gpu_ptr<DataType>(donated ? out : in),
              gpu_ptr<DataType>(out),
              gpu_ptr<int32_t>(offset),
              scale_,
              std::log2(base_),
              mat_size,
              dims2);
        } else if (!traditional_ && forward_) {
          hipLaunchKernelGGL(
              (rocm::rope_single<DataType, false, true>),
              grid, block, 0, stream,
              gpu_ptr<DataType>(donated ? out : in),
              gpu_ptr<DataType>(out),
              gpu_ptr<int32_t>(offset),
              scale_,
              std::log2(base_),
              mat_size,
              dims2);
        } else {
          hipLaunchKernelGGL(
              (rocm::rope_single<DataType, false, false>),
              grid, block, 0, stream,
              gpu_ptr<DataType>(donated ? out : in),
              gpu_ptr<DataType>(out),
              gpu_ptr<int32_t>(offset),
              scale_,
              std::log2(base_),
              mat_size,
              dims2);
        }
      });
    } else if (single) {
      uint2 dims2 = make_uint2(dims_ / 2, N);
      std::pair<dim3, dim3> gb = rocm::get_grid_and_block(dims2.x, dims2.y, 1);
      dim3 grid = gb.first;
      dim3 block = gb.second;
      int64_t freq_stride = inputs[2].strides(0);
      
      encoder.launch_kernel([=, &encoder, &out, &in, &offset, &inputs, this](hipStream_t stream) {
        if (traditional_ && forward_) {
          hipLaunchKernelGGL(
              (rocm::rope_single_freqs<DataType, true, true>),
              grid, block, 0, stream,
              gpu_ptr<DataType>(donated ? out : in),
              gpu_ptr<DataType>(out),
              gpu_ptr<int32_t>(offset),
              gpu_ptr<float>(inputs[2]),
              scale_,
              mat_size,
              dims2,
              freq_stride);
        } else if (traditional_ && !forward_) {
          hipLaunchKernelGGL(
              (rocm::rope_single_freqs<DataType, true, false>),
              grid, block, 0, stream,
              gpu_ptr<DataType>(donated ? out : in),
              gpu_ptr<DataType>(out),
              gpu_ptr<int32_t>(offset),
              gpu_ptr<float>(inputs[2]),
              scale_,
              mat_size,
              dims2,
              freq_stride);
        } else if (!traditional_ && forward_) {
          hipLaunchKernelGGL(
              (rocm::rope_single_freqs<DataType, false, true>),
              grid, block, 0, stream,
              gpu_ptr<DataType>(donated ? out : in),
              gpu_ptr<DataType>(out),
              gpu_ptr<int32_t>(offset),
              gpu_ptr<float>(inputs[2]),
              scale_,
              mat_size,
              dims2,
              freq_stride);
        } else {
          hipLaunchKernelGGL(
              (rocm::rope_single_freqs<DataType, false, false>),
              grid, block, 0, stream,
              gpu_ptr<DataType>(donated ? out : in),
              gpu_ptr<DataType>(out),
              gpu_ptr<int32_t>(offset),
              gpu_ptr<float>(inputs[2]),
              scale_,
              mat_size,
              dims2,
              freq_stride);
        }
      });
    } else if (with_freqs) {
      int n_per_thread = 4;
      uint32_t dimz = B * ((N + n_per_thread - 1) / n_per_thread);
      uint3 dims3 = make_uint3(dims_ / 2, T, dimz);
      std::pair<dim3, dim3> gb = rocm::get_grid_and_block(dims3.x, dims3.y, dims3.z);
      dim3 grid = gb.first;
      dim3 block = gb.second;
      int64_t offset_stride = 0;
      if (inputs[1].ndim() > 0) {
        offset_stride = inputs[1].strides()[0];
      }
      int64_t freq_stride = inputs[2].strides(0);
      
      encoder.launch_kernel([=, &encoder, &out, &in, &offset, &inputs, this](hipStream_t stream) {
        if (traditional_ && forward_) {
          hipLaunchKernelGGL(
              (rocm::rope_freqs<DataType, true, true>),
              grid, block, 0, stream,
              gpu_ptr<DataType>(donated ? out : in),
              gpu_ptr<DataType>(out),
              gpu_ptr<int32_t>(offset),
              gpu_ptr<float>(inputs[2]),
              scale_,
              std::log2(base_),
              strides,
              out_strides,
              offset_stride,
              N,
              dims3,
              freq_stride);
        } else if (traditional_ && !forward_) {
          hipLaunchKernelGGL(
              (rocm::rope_freqs<DataType, true, false>),
              grid, block, 0, stream,
              gpu_ptr<DataType>(donated ? out : in),
              gpu_ptr<DataType>(out),
              gpu_ptr<int32_t>(offset),
              gpu_ptr<float>(inputs[2]),
              scale_,
              std::log2(base_),
              strides,
              out_strides,
              offset_stride,
              N,
              dims3,
              freq_stride);
        } else if (!traditional_ && forward_) {
          hipLaunchKernelGGL(
              (rocm::rope_freqs<DataType, false, true>),
              grid, block, 0, stream,
              gpu_ptr<DataType>(donated ? out : in),
              gpu_ptr<DataType>(out),
              gpu_ptr<int32_t>(offset),
              gpu_ptr<float>(inputs[2]),
              scale_,
              std::log2(base_),
              strides,
              out_strides,
              offset_stride,
              N,
              dims3,
              freq_stride);
        } else {
          hipLaunchKernelGGL(
              (rocm::rope_freqs<DataType, false, false>),
              grid, block, 0, stream,
              gpu_ptr<DataType>(donated ? out : in),
              gpu_ptr<DataType>(out),
              gpu_ptr<int32_t>(offset),
              gpu_ptr<float>(inputs[2]),
              scale_,
              std::log2(base_),
              strides,
              out_strides,
              offset_stride,
              N,
              dims3,
              freq_stride);
        }
      });
    } else {
      int n_per_thread = 4;
      uint32_t dimz = B * ((N + n_per_thread - 1) / n_per_thread);
      uint3 dims3 = make_uint3(dims_ / 2, T, dimz);
      std::pair<dim3, dim3> gb = rocm::get_grid_and_block(dims3.x, dims3.y, dims3.z);
      dim3 grid = gb.first;
      dim3 block = gb.second;
      int64_t offset_stride = 0;
      if (inputs[1].ndim() > 0) {
        offset_stride = inputs[1].strides()[0];
      }
      
      encoder.launch_kernel([=, &encoder, &out, &in, &offset, this](hipStream_t stream) {
        if (traditional_ && forward_) {
          hipLaunchKernelGGL(
              (rocm::rope<DataType, true, true>),
              grid, block, 0, stream,
              gpu_ptr<DataType>(donated ? out : in),
              gpu_ptr<DataType>(out),
              gpu_ptr<int32_t>(offset),
              scale_,
              std::log2(base_),
              strides,
              out_strides,
              offset_stride,
              N,
              dims3);
        } else if (traditional_ && !forward_) {
          hipLaunchKernelGGL(
              (rocm::rope<DataType, true, false>),
              grid, block, 0, stream,
              gpu_ptr<DataType>(donated ? out : in),
              gpu_ptr<DataType>(out),
              gpu_ptr<int32_t>(offset),
              scale_,
              std::log2(base_),
              strides,
              out_strides,
              offset_stride,
              N,
              dims3);
        } else if (!traditional_ && forward_) {
          hipLaunchKernelGGL(
              (rocm::rope<DataType, false, true>),
              grid, block, 0, stream,
              gpu_ptr<DataType>(donated ? out : in),
              gpu_ptr<DataType>(out),
              gpu_ptr<int32_t>(offset),
              scale_,
              std::log2(base_),
              strides,
              out_strides,
              offset_stride,
              N,
              dims3);
        } else {
          hipLaunchKernelGGL(
              (rocm::rope<DataType, false, false>),
              grid, block, 0, stream,
              gpu_ptr<DataType>(donated ? out : in),
              gpu_ptr<DataType>(out),
              gpu_ptr<int32_t>(offset),
              scale_,
              std::log2(base_),
              strides,
              out_strides,
              offset_stride,
              N,
              dims3);
        }
      });
    }
  });
}

} // namespace fast

} // namespace mlx::core
