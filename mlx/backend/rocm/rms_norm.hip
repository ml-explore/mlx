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
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_xor(val, offset);
  }
  return val;
}

// Warp reduce for float2 (wg*x_sum, x^2_sum)
struct float2_sum {
  float x, y;
};

__device__ float2_sum warp_reduce_sum_f2(float2_sum val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val.x += __shfl_xor(val.x, offset);
    val.y += __shfl_xor(val.y, offset);
  }
  return val;
}

// Per-row RMS norm body. `x`/`out` are already offset to this row's base.
template <typename T, int BLOCK_DIM, int N_READS = 4>
__device__ void rms_norm_row(
    const T* x,
    const T* w,
    T* out,
    float eps,
    uint32_t axis_size,
    int64_t w_stride) {
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
  __shared__ float shared_sum[BLOCK_DIM / WARP_SIZE + 1];
  
  float warp_sum = warp_reduce_sum_rms(normalizer);
  int lane = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;
  
  if (lane == 0) {
    shared_sum[warp_id] = warp_sum;
  }
  __syncthreads();
  
  if (warp_id == 0) {
    normalizer = (lane < (BLOCK_DIM + WARP_SIZE - 1) / WARP_SIZE) ? shared_sum[lane] : 0;
    normalizer = warp_reduce_sum_rms(normalizer);
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
    shared_sum[0] = normalizer;
  }
  __syncthreads();
  // Use 1/sqrt instead of rsqrtf for IEEE-compliant precision
  // (matches Metal's metal::precise::rsqrt behavior)
  normalizer = 1.0f / sqrtf(shared_sum[0] / axis_size + eps);

  // Write output
  // Match Metal's weight application order: w * T(x * normalizer)
  // Weight multiply in output type T after truncation, not in float32
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      int idx = i + j;
      T normalized = static_cast<T>(static_cast<float>(x[idx]) * normalizer);
      T wi = (w_stride == 0) ? w[0] : w[idx * w_stride];
      out[idx] = wi * normalized;
    }
  }
}

// Packed input: rows tightly packed at row * axis_size.
template <typename T, int BLOCK_DIM, int N_READS = 4>
__global__ void rms_norm_kernel(
    const T* x,
    const T* w,
    T* out,
    float eps,
    uint32_t axis_size,
    int64_t w_stride) {
  int row = blockIdx.x;
  rms_norm_row<T, BLOCK_DIM, N_READS>(
      x + (int64_t)row * axis_size, w, out + (int64_t)row * axis_size, eps,
      axis_size, w_stride);
}

// Strided input: each row's base offset is computed from the leading dims'
// shape/strides (last dim must be contiguous). Output is packed contiguous.
// Avoids the contiguous_copy_gpu the host would otherwise insert for a
// non-packed (sliced/transposed) input, e.g. per-head q/k norm.
template <typename T, int BLOCK_DIM, int N_READS = 4>
__global__ void rms_norm_strided_kernel(
    const T* x,
    const T* w,
    T* out,
    float eps,
    uint32_t axis_size,
    int64_t w_stride,
    int n_row_dims,
    hip_array<int, 4> row_shape,
    hip_array<int64_t, 4> row_strides) {
  int row = blockIdx.x;
  int64_t x_off = 0;
  int r = row;
  for (int d = n_row_dims - 1; d >= 0; --d) {
    x_off += (int64_t)(r % row_shape[d]) * row_strides[d];
    r /= row_shape[d];
  }
  rms_norm_row<T, BLOCK_DIM, N_READS>(
      x + x_off, w, out + (int64_t)row * axis_size, eps, axis_size, w_stride);
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
  __shared__ float2_sum shared_f2[BLOCK_DIM / WARP_SIZE + 1];
  
  float2_sum warp_f2 = warp_reduce_sum_f2(factors);
  int lane = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;
  
  if (lane == 0) {
    shared_f2[warp_id] = warp_f2;
  }
  __syncthreads();
  
  if (warp_id == 0) {
    factors = (lane < (BLOCK_DIM + WARP_SIZE - 1) / WARP_SIZE) ? shared_f2[lane] : float2_sum{0, 0};
    factors = warp_reduce_sum_f2(factors);
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
    shared_f2[0] = factors;
  }
  __syncthreads();
  factors = shared_f2[0];
  
  float meangwx = factors.x / axis_size;
  // Use 1/sqrt instead of rsqrtf for IEEE-compliant precision
  // (matches Metal's metal::precise::rsqrt behavior)
  float normalizer = 1.0f / sqrtf(factors.y / axis_size + eps);
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

  const array& xin = inputs[0];
  const array& w = inputs[1];
  int ndim = xin.ndim();
  int32_t axis_size = xin.shape().back();

  // Layout decision:
  //  - packed:  rows tightly packed -> fast kernel, output adopts input layout.
  //  - strided: last dim contiguous but rows not packed (e.g. sliced per-head
  //             q/k norm) -> strided kernel reads the input in place and writes
  //             a packed output. Avoids a contiguous_copy_gpu launch.
  //  - else:    fall back to a contiguous copy.
  bool last_contig = ndim >= 1 && xin.strides()[ndim - 1] == 1;
  bool packed = xin.flags().contiguous && last_contig;
  if (packed && ndim > 1) {
    auto s2 = xin.strides()[ndim - 2];
    packed &= (s2 == 0 || s2 == (int64_t)axis_size);
  }
  bool strided = !packed && last_contig && (ndim - 1) <= 4;

  array x = xin;
  if (packed) {
    if (xin.is_donatable()) {
      out.copy_shared_buffer(xin);
    } else {
      out.set_data(
          allocator::malloc(xin.data_size() * xin.itemsize()),
          xin.data_size(), xin.strides(), xin.flags());
    }
  } else if (strided) {
    out.set_data(allocator::malloc(out.nbytes()));  // packed contiguous output
  } else {
    x = contiguous_copy_gpu(xin, s);
    out.copy_shared_buffer(x);
  }

  const array& xk = strided ? xin : x;
  int32_t n_rows = (int32_t)(out.size() / axis_size);
  int64_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;

  int n_row_dims = ndim - 1;
  rocm::hip_array<int, 4> row_shape;
  rocm::hip_array<int64_t, 4> row_strides;
  if (strided) {
    for (int d = 0; d < n_row_dims; ++d) {
      row_shape[d] = (int)xin.shape()[d];
      row_strides[d] = xin.strides()[d];
    }
  }

  auto& encoder = rocm::get_command_encoder(s);
  encoder.set_input_array(xk);
  encoder.set_input_array(w);
  encoder.set_output_array(out);

  constexpr int BLOCK_DIM = 256;
  constexpr int N_READS = 4;

  encoder.launch_kernel([&](hipStream_t stream) {
    auto launch = [&](auto tag) {
      using DT = decltype(tag);
      if (strided) {
        hipLaunchKernelGGL(
            (rocm::rms_norm_strided_kernel<DT, BLOCK_DIM, N_READS>),
            dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
            gpu_ptr<DT>(xk), gpu_ptr<DT>(w), gpu_ptr<DT>(out),
            eps_, axis_size, w_stride, n_row_dims, row_shape, row_strides);
      } else {
        hipLaunchKernelGGL(
            (rocm::rms_norm_kernel<DT, BLOCK_DIM, N_READS>),
            dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
            gpu_ptr<DT>(xk), gpu_ptr<DT>(w), gpu_ptr<DT>(out),
            eps_, axis_size, w_stride);
      }
    };
    switch (out.dtype()) {
      case float32: launch(float{}); break;
      case float16: launch(__half{}); break;
      case bfloat16: launch(hip_bfloat16{}); break;
      default: throw std::runtime_error("Unsupported type for rms_norm");
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
              gpu_ptr<float>(x), gpu_ptr<float>(w), gpu_ptr<float>(g),
              gpu_ptr<float>(gx), gpu_ptr<float>(gw_temp),
              eps_, axis_size, w_stride);
          break;
        case float16:
          hipLaunchKernelGGL(
              (rocm::rms_norm_vjp_kernel<__half, true, BLOCK_DIM, N_READS>),
              dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
              gpu_ptr<__half>(x), gpu_ptr<__half>(w), gpu_ptr<__half>(g),
              gpu_ptr<__half>(gx), gpu_ptr<__half>(gw_temp),
              eps_, axis_size, w_stride);
          break;
        case bfloat16:
          hipLaunchKernelGGL(
              (rocm::rms_norm_vjp_kernel<hip_bfloat16, true, BLOCK_DIM, N_READS>),
              dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
              gpu_ptr<hip_bfloat16>(x), gpu_ptr<hip_bfloat16>(w), gpu_ptr<hip_bfloat16>(g),
              gpu_ptr<hip_bfloat16>(gx), gpu_ptr<hip_bfloat16>(gw_temp),
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
              gpu_ptr<float>(x), gpu_ptr<float>(w), gpu_ptr<float>(g),
              gpu_ptr<float>(gx), nullptr,
              eps_, axis_size, w_stride);
          break;
        case float16:
          hipLaunchKernelGGL(
              (rocm::rms_norm_vjp_kernel<__half, false, BLOCK_DIM, N_READS>),
              dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
              gpu_ptr<__half>(x), gpu_ptr<__half>(w), gpu_ptr<__half>(g),
              gpu_ptr<__half>(gx), nullptr,
              eps_, axis_size, w_stride);
          break;
        case bfloat16:
          hipLaunchKernelGGL(
              (rocm::rms_norm_vjp_kernel<hip_bfloat16, false, BLOCK_DIM, N_READS>),
              dim3(n_rows), dim3(BLOCK_DIM), 0, stream,
              gpu_ptr<hip_bfloat16>(x), gpu_ptr<hip_bfloat16>(w), gpu_ptr<hip_bfloat16>(g),
              gpu_ptr<hip_bfloat16>(gx), nullptr,
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
