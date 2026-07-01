// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/reduce/reduce.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/fast_primitives.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

// Warp reduce for sum
__device__ float warp_reduce_sum_f(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_xor(val, offset);
  }
  return val;
}

// Warp reduce for float3 (sum, sum*t, t*t)
struct float3_sum {
  float x, y, z;
};

__device__ float3_sum warp_reduce_sum_f3(float3_sum val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val.x += __shfl_xor(val.x, offset);
    val.y += __shfl_xor(val.y, offset);
    val.z += __shfl_xor(val.z, offset);
  }
  return val;
}

template <typename T, int BLOCK_DIM, int N_READS = 4>
__global__ void layer_norm_kernel(
    const T* x,
    const T* w,
    const T* b,
    T* out,
    float eps,
    int32_t axis_size,
    int64_t w_stride,
    int64_t b_stride) {
  int row = blockIdx.x;
  
  x += row * axis_size;
  out += row * axis_size;

  // Sum for mean
  float sum = 0;
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      sum += static_cast<float>(x[i + j]);
    }
  }

  // Block reduce for sum
  __shared__ float shared_sum[BLOCK_DIM / WARP_SIZE + 1];
  
  float warp_sum = warp_reduce_sum_f(sum);
  int lane = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;
  
  if (lane == 0) {
    shared_sum[warp_id] = warp_sum;
  }
  __syncthreads();
  
  if (warp_id == 0) {
    sum = (lane < (BLOCK_DIM + WARP_SIZE - 1) / WARP_SIZE) ? shared_sum[lane] : 0;
    sum = warp_reduce_sum_f(sum);
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
    shared_sum[0] = sum;
  }
  __syncthreads();
  float mean = shared_sum[0] / axis_size;

  // Compute variance
  float var_sum = 0;
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      float t = static_cast<float>(x[i + j]) - mean;
      var_sum += t * t;
    }
  }

  // Block reduce for variance
  warp_sum = warp_reduce_sum_f(var_sum);
  
  if (lane == 0) {
    shared_sum[warp_id] = warp_sum;
  }
  __syncthreads();
  
  if (warp_id == 0) {
    var_sum = (lane < (BLOCK_DIM + WARP_SIZE - 1) / WARP_SIZE) ? shared_sum[lane] : 0;
    var_sum = warp_reduce_sum_f(var_sum);
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
    shared_sum[0] = var_sum;
  }
  __syncthreads();
  // Use 1/sqrt instead of rsqrtf for IEEE-compliant precision
  // (matches Metal's metal::precise::rsqrt behavior)
  float normalizer = 1.0f / sqrtf(shared_sum[0] / axis_size + eps);

  // Write output
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      int idx = i + j;
      float norm = (static_cast<float>(x[idx]) - mean) * normalizer;
      float wi = (w_stride == 0) ? static_cast<float>(w[0]) : static_cast<float>(w[idx * w_stride]);
      float bi = (b_stride == 0) ? static_cast<float>(b[0]) : static_cast<float>(b[idx * b_stride]);
      out[idx] = static_cast<T>(wi * norm + bi);
    }
  }
}

template <typename T, bool HAS_W, int BLOCK_DIM, int N_READS = 4>
__global__ void layer_norm_vjp_kernel(
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

  // Sum for mean
  float sum = 0;
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      sum += static_cast<float>(x[i + j]);
    }
  }

  // Block reduce for sum
  __shared__ float shared_sum[BLOCK_DIM / WARP_SIZE + 1];
  __shared__ float3_sum shared_f3[BLOCK_DIM / WARP_SIZE + 1];
  
  float warp_sum = warp_reduce_sum_f(sum);
  int lane = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;
  
  if (lane == 0) {
    shared_sum[warp_id] = warp_sum;
  }
  __syncthreads();
  
  if (warp_id == 0) {
    sum = (lane < (BLOCK_DIM + WARP_SIZE - 1) / WARP_SIZE) ? shared_sum[lane] : 0;
    sum = warp_reduce_sum_f(sum);
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
    shared_sum[0] = sum;
  }
  __syncthreads();
  float mean = shared_sum[0] / axis_size;

  // Compute factors: (wg_sum, wg*xc_sum, xc^2_sum)
  float3_sum factors = {0, 0, 0};
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      int idx = i + j;
      float t = static_cast<float>(x[idx]) - mean;
      float wi = (w_stride == 0) ? static_cast<float>(w[0]) : static_cast<float>(w[idx * w_stride]);
      float gi = static_cast<float>(g[idx]);
      float wg = wi * gi;
      factors.x += wg;
      factors.y += wg * t;
      factors.z += t * t;
    }
  }

  // Block reduce for factors
  float3_sum warp_f3 = warp_reduce_sum_f3(factors);
  
  if (lane == 0) {
    shared_f3[warp_id] = warp_f3;
  }
  __syncthreads();
  
  if (warp_id == 0) {
    factors = (lane < (BLOCK_DIM + WARP_SIZE - 1) / WARP_SIZE) ? shared_f3[lane] : float3_sum{0, 0, 0};
    factors = warp_reduce_sum_f3(factors);
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
    shared_f3[0] = factors;
  }
  __syncthreads();
  factors = shared_f3[0];
  
  float meanwg = factors.x / axis_size;
  float meanwgxc = factors.y / axis_size;
  float normalizer2 = 1.0f / (factors.z / axis_size + eps);
  float normalizer = sqrtf(normalizer2);

  // Write outputs
  for (int i = threadIdx.x * N_READS; i < axis_size; i += BLOCK_DIM * N_READS) {
    #pragma unroll
    for (int j = 0; j < N_READS && i + j < axis_size; ++j) {
      int idx = i + j;
      float xi_centered = static_cast<float>(x[idx]) - mean;
      float xi_norm = xi_centered * normalizer;
      float wi = (w_stride == 0) ? static_cast<float>(w[0]) : static_cast<float>(w[idx * w_stride]);
      float gi = static_cast<float>(g[idx]);
      
      // Gradient for x
      gx[idx] = static_cast<T>(normalizer * (wi * gi - meanwg) - xi_norm * meanwgxc * normalizer2);
      
      // Gradient for w (per-element, will be reduced later)
      if constexpr (HAS_W) {
        gw[idx] = static_cast<T>(gi * xi_norm);
      }
    }
  }
}

} // namespace rocm

namespace fast {

bool LayerNorm::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void LayerNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& out = outputs[0];
  auto& encoder = rocm::get_command_encoder(s);

  // Make sure that the last dimension is contiguous.
  auto set_output = [&s, &out, &encoder](const array& x) {
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
            mlx::core::rocm::malloc_async(x.data_size() * x.itemsize(), encoder),
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
  const array& b = inputs[2];

  int32_t axis_size = x.shape().back();
  int32_t n_rows = x.data_size() / axis_size;
  int64_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;
  int64_t b_stride = (b.ndim() == 1) ? b.strides()[0] : 0;

  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(b);
  encoder.set_output_array(out);
  
  constexpr int BLOCK_DIM = 256;
  constexpr int N_READS = 4;
  
  switch (out.dtype()) {
    case float32:
      encoder.add_kernel_node(
          &rocm::layer_norm_kernel<float, BLOCK_DIM, N_READS>,
          dim3(n_rows), dim3(BLOCK_DIM), 0,
          gpu_ptr<float>(x), gpu_ptr<float>(w), gpu_ptr<float>(b), gpu_ptr<float>(out),
          eps_, axis_size, w_stride, b_stride);
      break;
    case float16:
      encoder.add_kernel_node(
          &rocm::layer_norm_kernel<__half, BLOCK_DIM, N_READS>,
          dim3(n_rows), dim3(BLOCK_DIM), 0,
          gpu_ptr<__half>(x), gpu_ptr<__half>(w), gpu_ptr<__half>(b), gpu_ptr<__half>(out),
          eps_, axis_size, w_stride, b_stride);
      break;
    case bfloat16:
      encoder.add_kernel_node(
          &rocm::layer_norm_kernel<hip_bfloat16, BLOCK_DIM, N_READS>,
          dim3(n_rows), dim3(BLOCK_DIM), 0,
          gpu_ptr<hip_bfloat16>(x), gpu_ptr<hip_bfloat16>(w), gpu_ptr<hip_bfloat16>(b), gpu_ptr<hip_bfloat16>(out),
          eps_, axis_size, w_stride, b_stride);
      break;
    default:
      throw std::runtime_error("Unsupported type for layer_norm");
  }
}

void LayerNormVJP::eval_gpu(
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
  bool donate_g = inputs[3].is_donatable();
  bool copied;
  auto x = check_input(inputs[0], copied);
  donate_x |= copied;
  const array& w = inputs[1];
  const array& b = inputs[2];
  bool g_copied;
  auto g = check_input(inputs[3], g_copied);
  donate_g |= g_copied;
  array& gx = outputs[0];
  array& gw = outputs[1];
  array& gb = outputs[2];

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
    gx.set_data(mlx::core::rocm::malloc_async(gx.nbytes(), encoder));
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
  bool g_in_gw = false;
  if (has_w) {
    if (!g_in_gx && donate_g) {
      g_in_gw = true;
      gw_temp.copy_shared_buffer(g);
    } else {
      gw_temp.set_data(mlx::core::rocm::malloc_async(gw_temp.nbytes(), encoder));
      encoder.add_temporary(gw_temp);
    }
  }

  // The gradient for b in case we had a b
  bool has_gb = (gb.ndim() == 1 && gb.size() == axis_size);
  if (has_gb) {
    // Sum reduction over rows for gb
    gb.set_data(mlx::core::rocm::malloc_async(gb.nbytes(), encoder));
    // TODO: Implement proper column reduction for gb
    // For now, we'll compute it in the kernel or use a simple reduction
  }

  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(g);
  encoder.set_output_array(gx);
  encoder.set_output_array(gw_temp);
  
  constexpr int BLOCK_DIM = 256;
  constexpr int N_READS = 4;
  
  if (has_w) {
    switch (gx.dtype()) {
      case float32:
        encoder.add_kernel_node(
            &rocm::layer_norm_vjp_kernel<float, true, BLOCK_DIM, N_READS>,
            dim3(n_rows), dim3(BLOCK_DIM), 0,
            gpu_ptr<float>(x), gpu_ptr<float>(w), gpu_ptr<float>(g),
            gpu_ptr<float>(gx), gpu_ptr<float>(gw_temp),
            eps_, axis_size, w_stride);
        break;
      case float16:
        encoder.add_kernel_node(
            &rocm::layer_norm_vjp_kernel<__half, true, BLOCK_DIM, N_READS>,
            dim3(n_rows), dim3(BLOCK_DIM), 0,
            gpu_ptr<__half>(x), gpu_ptr<__half>(w), gpu_ptr<__half>(g),
            gpu_ptr<__half>(gx), gpu_ptr<__half>(gw_temp),
            eps_, axis_size, w_stride);
        break;
      case bfloat16:
        encoder.add_kernel_node(
            &rocm::layer_norm_vjp_kernel<hip_bfloat16, true, BLOCK_DIM, N_READS>,
            dim3(n_rows), dim3(BLOCK_DIM), 0,
            gpu_ptr<hip_bfloat16>(x), gpu_ptr<hip_bfloat16>(w), gpu_ptr<hip_bfloat16>(g),
            gpu_ptr<hip_bfloat16>(gx), gpu_ptr<hip_bfloat16>(gw_temp),
            eps_, axis_size, w_stride);
        break;
      default:
        throw std::runtime_error("Unsupported type for layer_norm_vjp");
    }
  } else {
    switch (gx.dtype()) {
      case float32: {
        float* gw_null = nullptr;
        encoder.add_kernel_node(
            &rocm::layer_norm_vjp_kernel<float, false, BLOCK_DIM, N_READS>,
            dim3(n_rows), dim3(BLOCK_DIM), 0,
            gpu_ptr<float>(x), gpu_ptr<float>(w), gpu_ptr<float>(g),
            gpu_ptr<float>(gx), gw_null,
            eps_, axis_size, w_stride);
        break;
      }
      case float16: {
        __half* gw_null = nullptr;
        encoder.add_kernel_node(
            &rocm::layer_norm_vjp_kernel<__half, false, BLOCK_DIM, N_READS>,
            dim3(n_rows), dim3(BLOCK_DIM), 0,
            gpu_ptr<__half>(x), gpu_ptr<__half>(w), gpu_ptr<__half>(g),
            gpu_ptr<__half>(gx), gw_null,
            eps_, axis_size, w_stride);
        break;
      }
      case bfloat16: {
        hip_bfloat16* gw_null = nullptr;
        encoder.add_kernel_node(
            &rocm::layer_norm_vjp_kernel<hip_bfloat16, false, BLOCK_DIM, N_READS>,
            dim3(n_rows), dim3(BLOCK_DIM), 0,
            gpu_ptr<hip_bfloat16>(x), gpu_ptr<hip_bfloat16>(w), gpu_ptr<hip_bfloat16>(g),
            gpu_ptr<hip_bfloat16>(gx), gw_null,
            eps_, axis_size, w_stride);
        break;
      }
      default:
        throw std::runtime_error("Unsupported type for layer_norm_vjp");
    }
  }

  // Reduce gw_temp to gw if we have weights
  if (has_w) {
    ReductionPlan plan(
        ReductionOpType::ContiguousStridedReduce, {n_rows}, {axis_size});
    col_reduce(encoder, gw_temp, gw, Reduce::ReduceType::Sum, {0}, plan);
  }
}

} // namespace fast

} // namespace mlx::core
