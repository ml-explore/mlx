// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/allocator.h"
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

  // Compute costheta, sintheta using sincosf for better performance
  float theta = L * inv_freq;
  float sintheta, costheta;
  sincosf(theta, &sintheta, &costheta);

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

// Optimized 1D kernel for single-token decode case
// Uses flat indexing for better occupancy with small workloads
template <typename T, bool traditional, bool forward>
__global__ void rope_single_1d(
    const T* in,
    T* out,
    const int32_t* offset,
    float scale,
    float base,
    int64_t stride,
    uint32_t half_dims,  // dims.x = dims_ / 2
    uint32_t n_heads) {  // dims.y = N
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t total = half_dims * n_heads;
  if (tid >= total) {
    return;
  }

  // Convert flat index to 2D position
  uint32_t pos_x = tid % half_dims;  // position within dimension
  uint32_t pos_y = tid / half_dims;  // head index

  float d = static_cast<float>(pos_x) / static_cast<float>(half_dims);
  float inv_freq = exp2f(-d * base);

  // Inline the implementation for better performance
  float L = scale * static_cast<float>(*offset);
  float theta = L * inv_freq;
  float sintheta, costheta;
  sincosf(theta, &sintheta, &costheta);

  uint32_t index_1, index_2;
  if (traditional) {
    index_1 = 2 * pos_x + pos_y * stride;
    index_2 = index_1 + 1;
  } else {
    index_1 = pos_x + pos_y * stride;
    index_2 = index_1 + half_dims;
  }

  float x1 = static_cast<float>(in[index_1]);
  float x2 = static_cast<float>(in[index_2]);
  float rx1, rx2;
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

// Optimized 1D kernel for single-token decode with custom frequencies
template <typename T, bool traditional, bool forward>
__global__ void rope_single_freqs_1d(
    const T* in,
    T* out,
    const int32_t* offset,
    const float* freqs,
    float scale,
    int64_t stride,
    uint32_t half_dims,
    uint32_t n_heads,
    int64_t freq_stride) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t total = half_dims * n_heads;
  if (tid >= total) {
    return;
  }

  uint32_t pos_x = tid % half_dims;
  uint32_t pos_y = tid / half_dims;

  float inv_freq = 1.0f / freqs[freq_stride * pos_x];

  float L = scale * static_cast<float>(*offset);
  float theta = L * inv_freq;
  float sintheta, costheta;
  sincosf(theta, &sintheta, &costheta);

  uint32_t index_1, index_2;
  if (traditional) {
    index_1 = 2 * pos_x + pos_y * stride;
    index_2 = index_1 + 1;
  } else {
    index_1 = pos_x + pos_y * stride;
    index_2 = index_1 + half_dims;
  }

  float x1 = static_cast<float>(in[index_1]);
  float x2 = static_cast<float>(in[index_2]);
  float rx1, rx2;
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

  // Compute costheta, sintheta using sincosf for better performance
  float theta = L * inv_freq;
  float sintheta, costheta;
  sincosf(theta, &sintheta, &costheta);

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

// Optimized grid/block for single-token decode case
// Uses 1D blocks for better coalescing when y (n_heads) is small
inline std::pair<dim3, dim3> get_grid_and_block_single(uint32_t x, uint32_t y) {
  // For decode: x = dims/2 (e.g., 64), y = n_heads (e.g., 40)
  // Total elements = x * y (e.g., 2560)
  // Use 1D layout for better occupancy with small workloads
  constexpr uint32_t BLOCK_SIZE = 256;
  uint32_t total = x * y;
  dim3 block(BLOCK_SIZE, 1, 1);
  dim3 grid((total + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
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

  // We apply rope to less than the whole vector. Normally that needs a full
  // copy to `out` (so the untouched [dims_:D] tail is present) followed by an
  // in-place rotate. But if the input is donatable we can rotate the first
  // dims_ channels IN PLACE and leave the tail untouched — no copy at all, and
  // `out` simply adopts the input's (possibly strided) layout. Downstream ops
  // (SDPA) accept non-contiguous q/k, so this is safe. (Port of ml-explore/mlx
  // PR #3704, "RoPE without copy".)
  if (dims_ < D) {
    donated = true;
    if (in.is_donatable() && in.flags().row_contiguous) {
      out.copy_shared_buffer(in);
      strides[0] = mat_size;
      strides[1] = in.strides()[ndim - 2];
      strides[2] = in.strides()[ndim - 1];
    } else {
      auto ctype =
          (in.flags().row_contiguous) ? CopyType::Vector : CopyType::General;
      copy_gpu(in, out, ctype, s);
      strides[0] = mat_size;
      strides[1] = out.strides()[ndim - 2];
      strides[2] = out.strides()[ndim - 1];
    }
  }

  // Either copy or apply in-place
  else if (in.flags().row_contiguous) {
    if (in.is_donatable()) {
      donated = true;
      out.copy_shared_buffer(in);
    } else {
      out.set_data(mlx::core::rocm::malloc_async(out.nbytes(), encoder));
    }
    strides[0] = mat_size;
    strides[1] = in.strides()[ndim - 2];
    strides[2] = in.strides()[ndim - 1];
  } else if (dispatch_ndim == 3) {
    // Handle non-contiguous 3D inputs
    out.set_data(mlx::core::rocm::malloc_async(out.nbytes(), encoder));
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

  // Dispatch based on dtype
  dispatch_float_types(out.dtype(), "rope", [&](auto type_tag) {
    using CTYPE = MLX_GET_TYPE(type_tag);
    using DataType = hip_type_t<CTYPE>;
    
    // Get grid/block dimensions outside the lambda to avoid C++20 structured binding capture
    if (single && !with_freqs) {
      // Use optimized 1D kernel for single-token decode
      uint32_t half_dims = dims_ / 2;
      uint32_t n_heads = N;
      std::pair<dim3, dim3> gb = rocm::get_grid_and_block_single(half_dims, n_heads);
      dim3 grid = gb.first;
      dim3 block = gb.second;

      const DataType* in_ptr = gpu_ptr<DataType>(donated ? out : in);
      DataType* out_ptr = gpu_ptr<DataType>(out);
      const int32_t* off_ptr = gpu_ptr<int32_t>(offset);
      float base_log2 = std::log2(base_);
      int64_t stride_v = static_cast<int64_t>(mat_size);

      #define ADD_ROPE_SINGLE_1D(TRAD, FWD) \
        encoder.add_kernel_node( \
            &rocm::rope_single_1d<DataType, TRAD, FWD>, \
            grid, block, 0, \
            in_ptr, out_ptr, off_ptr, scale_, base_log2, stride_v, \
            half_dims, n_heads)
      if (traditional_ && forward_) {
        ADD_ROPE_SINGLE_1D(true, true);
      } else if (traditional_ && !forward_) {
        ADD_ROPE_SINGLE_1D(true, false);
      } else if (!traditional_ && forward_) {
        ADD_ROPE_SINGLE_1D(false, true);
      } else {
        ADD_ROPE_SINGLE_1D(false, false);
      }
      #undef ADD_ROPE_SINGLE_1D
    } else if (single) {
      // Use optimized 1D kernel for single-token decode with freqs
      uint32_t half_dims = dims_ / 2;
      uint32_t n_heads = N;
      std::pair<dim3, dim3> gb = rocm::get_grid_and_block_single(half_dims, n_heads);
      dim3 grid = gb.first;
      dim3 block = gb.second;
      int64_t freq_stride = inputs[2].strides(0);

      const DataType* in_ptr = gpu_ptr<DataType>(donated ? out : in);
      DataType* out_ptr = gpu_ptr<DataType>(out);
      const int32_t* off_ptr = gpu_ptr<int32_t>(offset);
      const float* freqs_ptr = gpu_ptr<float>(inputs[2]);
      int64_t stride_v = static_cast<int64_t>(mat_size);

      #define ADD_ROPE_SINGLE_FREQS_1D(TRAD, FWD) \
        encoder.add_kernel_node( \
            &rocm::rope_single_freqs_1d<DataType, TRAD, FWD>, \
            grid, block, 0, \
            in_ptr, out_ptr, off_ptr, freqs_ptr, scale_, stride_v, \
            half_dims, n_heads, freq_stride)
      if (traditional_ && forward_) {
        ADD_ROPE_SINGLE_FREQS_1D(true, true);
      } else if (traditional_ && !forward_) {
        ADD_ROPE_SINGLE_FREQS_1D(true, false);
      } else if (!traditional_ && forward_) {
        ADD_ROPE_SINGLE_FREQS_1D(false, true);
      } else {
        ADD_ROPE_SINGLE_FREQS_1D(false, false);
      }
      #undef ADD_ROPE_SINGLE_FREQS_1D
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

      const DataType* in_ptr = gpu_ptr<DataType>(donated ? out : in);
      DataType* out_ptr = gpu_ptr<DataType>(out);
      const int32_t* off_ptr = gpu_ptr<int32_t>(offset);
      const float* freqs_ptr = gpu_ptr<float>(inputs[2]);
      float base_log2 = std::log2(base_);

      #define ADD_ROPE_FREQS(TRAD, FWD) \
        encoder.add_kernel_node( \
            &rocm::rope_freqs<DataType, TRAD, FWD>, \
            grid, block, 0, \
            in_ptr, out_ptr, off_ptr, freqs_ptr, scale_, base_log2, \
            strides, out_strides, offset_stride, N, dims3, freq_stride)
      if (traditional_ && forward_) {
        ADD_ROPE_FREQS(true, true);
      } else if (traditional_ && !forward_) {
        ADD_ROPE_FREQS(true, false);
      } else if (!traditional_ && forward_) {
        ADD_ROPE_FREQS(false, true);
      } else {
        ADD_ROPE_FREQS(false, false);
      }
      #undef ADD_ROPE_FREQS
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

      const DataType* in_ptr = gpu_ptr<DataType>(donated ? out : in);
      DataType* out_ptr = gpu_ptr<DataType>(out);
      const int32_t* off_ptr = gpu_ptr<int32_t>(offset);
      float base_log2 = std::log2(base_);

      #define ADD_ROPE(TRAD, FWD) \
        encoder.add_kernel_node( \
            &rocm::rope<DataType, TRAD, FWD>, \
            grid, block, 0, \
            in_ptr, out_ptr, off_ptr, scale_, base_log2, \
            strides, out_strides, offset_stride, N, dims3)
      if (traditional_ && forward_) {
        ADD_ROPE(true, true);
      } else if (traditional_ && !forward_) {
        ADD_ROPE(true, false);
      } else if (!traditional_ && forward_) {
        ADD_ROPE(false, true);
      } else {
        ADD_ROPE(false, false);
      }
      #undef ADD_ROPE
    }
  });
}

} // namespace fast

} // namespace mlx::core
