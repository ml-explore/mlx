// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/reduce/reduce.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/fast_primitives.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

// Warp reduce for sum
__device__ float warp_reduce_sum_rms(float val) {
  for (int offset = 32; offset > 0; offset /= 2) {
    val += __shfl_xor(val, offset);
  }
  return val;
}

// Warp reduce for float2 (wg*x_sum, x^2_sum)
struct float2_sum {
  float x, y;
};

__device__ float2_sum warp_reduce_sum_f2(float2_sum val) {
  for (int offset = 32; offset > 0; offset /= 2) {
    val.x += __shfl_xor(val.x, offset);
    val.y += __shfl_xor(val.y, offset);
  }
  return val;
}

template <typename T, int BLOCK_DIM, int N_READS = 4>
__global__ void rms_norm_kernel(
    const T* x,
    const T* w,
    T* out,
    float eps,
    uint32_t axis_size,
    int64_t w_stride) {
  int row = blockIdx.x;
  
  x += row * axis_size;
  out += row * axis_size;

  // Compute sum of squares
  float normalizer = 0;
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      float t = static_cast<float>(x[i + j]);
      normalizer += t * t;
    }
  }

  // Block reduce for normalizer
  __shared__ float shared_sum[BLOCK_DIM / 64 + 1];
  
  float warp_sum = warp_reduce_sum_rms(normalizer);
  int lane = threadIdx.x % 64;
  int warp_id = threadIdx.x / 64;
  
  if (lane == 0) {
    shared_sum[warp_id] = warp_sum;
  }
  __syncthreads();
  
  if (warp_id == 0) {
    normalizer = (lane < (BLOCK_DIM + 63) / 64) ? shared_sum[lane] : 0;
    normalizer = warp_reduce_sum_rms(normalizer);
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
    shared_sum[0] = normalizer;
  }
  __syncthreads();
  normalizer = rsqrtf(shared_sum[0] / axis_size + eps);

  // Write output
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      int idx = i + j;
      float y = static_cast<float>(x[idx]) * normalizer;
      float wi = (w_stride == 0) ? static_cast<float>(w[0]) : static_cast<float>(w[idx * w_stride]);
      out[idx] = static_cast<T>(wi * y);
    }
  }
}

template <typename T, bool HAS_W, int BLOCK_DIM, int N_READS = 4>
__global__ void rms_norm_vjp_kernel(
    const T* x,
    const T* w,
    const T* g,
    T* gx,
    T* gw,
    float eps,
    int32_t axis_size,
    int64_t w_stride) {
  int row = blockIdx.x;
  
  x += row * axis_size;
  g += row * axis_size;
  gx += row * axis_size;
  gw += row * axis_size;

  // Compute factors: (wg*x_sum, x^2_sum)
  float2_sum factors = {0, 0};
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      int idx = i + j;
      float t = static_cast<float>(x[idx]);
      float wi = (w_stride == 0) ? static_cast<float>(w[0]) : static_cast<float>(w[idx * w_stride]);
      float gi = static_cast<float>(g[idx]);
      float wg = wi * gi;
      factors.x += wg * t;
      factors.y += t * t;
    }
  }

  // Block reduce for factors
  __shared__ float2_sum shared_f2[BLOCK_DIM / 64 + 1];
  
  float2_sum warp_f2 = warp_reduce_sum_f2(factors);
  int lane = threadIdx.x % 64;
  int warp_id = threadIdx.x / 64;
  
  if (lane == 0) {
    shared_f2[warp_id] = warp_f2;
  }
  __syncthreads();
  
  if (warp_id == 0) {
    factors = (lane < (BLOCK_DIM + 63) / 64) ? shared_f2[lane] : float2_sum{0, 0};
    factors = warp_reduce_sum_f2(factors);
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
    shared_f2[0] = factors;
  }
  __syncthreads();
  factors = shared_f2[0];
  
  float meangwx = factors.x / axis_size;
  float normalizer = rsqrtf(factors.y / axis_size + eps);
  float normalizer3 = normalizer * normalizer * normalizer;

  // Write outputs
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      int idx = i + j;
      float xi = static_cast<float>(x[idx]);
      float wi = (w_stride == 0) ? static_cast<float>(w[0]) : static_cast<float>(w[idx * w_stride]);
      float gi = static_cast<float>(g[idx]);
      
      // Gradient for x
      gx[idx] = static_cast<T>(normalizer * wi * gi - xi * meangwx * normalizer3);
      
      // Gradient for w (per-element, will be reduced later)
      if constexpr (HAS_W) {
        gw[idx] = static_cast<T>(gi * xi * normalizer);
      }
    }
  }
}

} // namespace rocm

namespace fast {

bool RMSNorm::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void RMSNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& out = outputs[0];

  // Make sure that the last dimension is contiguous.
  auto set_output = [&s, &out](const array& x) {
    bool no_copy = x.flags().contiguous && x.strides()[x.ndim() - 1] == 1;
    if (no_copy && x.ndim() > 1) {
      auto s = x.strides()[x.ndim() - 2];
      no_copy &= (s == 0 || s == x.shape().back());
    }
    if (no_copy) {
      if (x.is_donatable()) {
        out.copy_shared_buffer(x);
      } else {
        out.set_data(
            allocator::malloc(x.data_size() * x.itemsize()),
            x.data_size(),
            x.strides(),
            x.flags());
      }
      return x;
    } else {
      array x_copy = contiguous_copy_gpu(x, s);
      out.copy_shared_buffer(x_copy);
      return x_copy;
    }
  };

  const array x = set_output(inputs[0]);
  const array& w = inputs[1];

  int32_t axis_size = x.shape().back();
  int32_t n_rows = x.data_size() / axis_size;
  int64_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;

  auto& encoder = rocm::get_command_encoder(s);
  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_output_array(out);
  
  constexpr int BLOCK_DIM = 256;
  constexpr int N_READS = 4;
  
  encoder.launch_kernel([&](hipStream_t stream) {
    switch (out.dtype()) {
      case float32:
        hipLaunchKernelGGL(
            (rocm::rms_norm_kernel<float, BLOCK_DIM, N_READS>),
            dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
            x.data<float>(), w.data<float>(), out.data<float>(),
            eps_, axis_size, w_stride);
        break;
      case float16:
        hipLaunchKernelGGL(
            (rocm::rms_norm_kernel<__half, BLOCK_DIM, N_READS>),
            dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
            x.data<__half>(), w.data<__half>(), out.data<__half>(),
            eps_, axis_size, w_stride);
        break;
      case bfloat16:
        hipLaunchKernelGGL(
            (rocm::rms_norm_kernel<__hip_bfloat16, BLOCK_DIM, N_READS>),
            dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
            x.data<__hip_bfloat16>(), w.data<__hip_bfloat16>(), out.data<__hip_bfloat16>(),
            eps_, axis_size, w_stride);
        break;
      default:
        throw std::runtime_error("Unsupported type for rms_norm");
    }
  });
}

void RMSNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  // Ensure row contiguity
  auto check_input = [&s](const array& x, bool& copied) {
    if (x.flags().row_contiguous) {
      copied = false;
      return x;
    }
    copied = true;
    return contiguous_copy_gpu(x, s);
  };
  
  bool donate_x = inputs[0].is_donatable();
  bool donate_g = inputs[2].is_donatable();
  bool copied;
  auto x = check_input(inputs[0], copied);
  donate_x |= copied;
  const array& w = inputs[1];
  bool g_copied;
  auto g = check_input(inputs[2], g_copied);
  donate_g |= g_copied;
  array& gx = outputs[0];
  array& gw = outputs[1];

  // Check whether we had a weight
  bool has_w = w.ndim() != 0;

  // Allocate space for the outputs
  bool g_in_gx = false;
  if (donate_x) {
    gx.copy_shared_buffer(x);
  } else if (donate_g) {
    gx.copy_shared_buffer(g);
    g_in_gx = true;
  } else {
    gx.set_data(allocator::malloc(gx.nbytes()));
  }
  if (g_copied && !g_in_gx) {
    encoder.add_temporary(g);
  }

  int32_t axis_size = x.shape().back();
  int32_t n_rows = x.data_size() / axis_size;
  int64_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;

  // Allocate a temporary to store the gradients for w
  array gw_temp =
      (has_w) ? array({n_rows, x.shape().back()}, gw.dtype(), nullptr, {}) : w;
  if (has_w) {
    if (!g_in_gx && donate_g) {
      gw_temp.copy_shared_buffer(g);
    } else {
      gw_temp.set_data(allocator::malloc(gw_temp.nbytes()));
      encoder.add_temporary(gw_temp);
    }
  }

  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(g);
  encoder.set_output_array(gx);
  encoder.set_output_array(gw_temp);
  
  constexpr int BLOCK_DIM = 256;
  constexpr int N_READS = 4;
  
  encoder.launch_kernel([&](hipStream_t stream) {
    if (has_w) {
      switch (gx.dtype()) {
        case float32:
          hipLaunchKernelGGL(
              (rocm::rms_norm_vjp_kernel<float, true, BLOCK_DIM, N_READS>),
              dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
              x.data<float>(), w.data<float>(), g.data<float>(),
              gx.data<float>(), gw_temp.data<float>(),
              eps_, axis_size, w_stride);
          break;
        case float16:
          hipLaunchKernelGGL(
              (rocm::rms_norm_vjp_kernel<__half, true, BLOCK_DIM, N_READS>),
              dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
              x.data<__half>(), w.data<__half>(), g.data<__half>(),
              gx.data<__half>(), gw_temp.data<__half>(),
              eps_, axis_size, w_stride);
          break;
        case bfloat16:
          hipLaunchKernelGGL(
              (rocm::rms_norm_vjp_kernel<__hip_bfloat16, true, BLOCK_DIM, N_READS>),
              dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
              x.data<__hip_bfloat16>(), w.data<__hip_bfloat16>(), g.data<__hip_bfloat16>(),
              gx.data<__hip_bfloat16>(), gw_temp.data<__hip_bfloat16>(),
              eps_, axis_size, w_stride);
          break;
        default:
          throw std::runtime_error("Unsupported type for rms_norm_vjp");
      }
    } else {
      switch (gx.dtype()) {
        case float32:
          hipLaunchKernelGGL(
              (rocm::rms_norm_vjp_kernel<float, false, BLOCK_DIM, N_READS>),
              dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
              x.data<float>(), w.data<float>(), g.data<float>(),
              gx.data<float>(), nullptr,
              eps_, axis_size, w_stride);
          break;
        case float16:
          hipLaunchKernelGGL(
              (rocm::rms_norm_vjp_kernel<__half, false, BLOCK_DIM, N_READS>),
              dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
              x.data<__half>(), w.data<__half>(), g.data<__half>(),
              gx.data<__half>(), nullptr,
              eps_, axis_size, w_stride);
          break;
        case bfloat16:
          hipLaunchKernelGGL(
              (rocm::rms_norm_vjp_kernel<__hip_bfloat16, false, BLOCK_DIM, N_READS>),
              dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
              x.data<__hip_bfloat16>(), w.data<__hip_bfloat16>(), g.data<__hip_bfloat16>(),
              gx.data<__hip_bfloat16>(), nullptr,
              eps_, axis_size, w_stride);
          break;
        default:
          throw std::runtime_error("Unsupported type for rms_norm_vjp");
      }
    }
  });

  // Reduce gw_temp to gw if we have weights
  if (has_w) {
    ReductionPlan plan(
        ReductionOpType::ContiguousStridedReduce, {n_rows}, {axis_size});
    col_reduce(encoder, gw_temp, gw, Reduce::ReduceType::Sum, {0}, plan);
  }
}

} // namespace fast

} // namespace mlx::core
