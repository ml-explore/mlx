// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/cast_op.hpp"
#include "mlx/backend/rocm/device/fp16_math.hpp"
#include "mlx/backend/rocm/device/utils.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>

#include <cassert>

namespace mlx::core {

namespace rocm {

template <typename T>
inline __device__ T softmax_exp(T x) {
  // Softmax doesn't need high precision exponential cause x is gonna be in
  // (-oo, 0] anyway and subsequently it will be divided by sum(exp(x_i)).
  if constexpr (std::is_same_v<T, float>) {
    return __expf(x);
  } else if constexpr (std::is_same_v<T, double>) {
    return exp(x);
  } else {
    return T(__expf(static_cast<float>(x)));
  }
}

// Warp reduce for max using shuffle
template <typename T>
__device__ T warp_reduce_max(T val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    T other = __shfl_xor(val, offset);
    val = val > other ? val : other;
  }
  return val;
}

// Warp reduce for sum using shuffle
template <typename T>
__device__ T warp_reduce_sum(T val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    T other = __shfl_xor(val, offset);
    val = val + other;
  }
  return val;
}

// Optimized softmax kernel using online normalizer calculation
// Reference: https://github.com/NVIDIA/online-softmax
template <typename T, typename AccT, int BLOCK_DIM, int N_READS = 4>
__global__ void softmax_kernel(const T* in, T* out, int axis_size) {
  int row = blockIdx.x;
  int thread_rank = threadIdx.x;
  int lane = thread_rank % WARP_SIZE;
  int warp_id = thread_rank / WARP_SIZE;
  int num_warps = BLOCK_DIM / WARP_SIZE;
  
  in += row * axis_size;
  out += row * axis_size;

  // Online softmax: compute max and normalizer in a single pass
  AccT prevmax;
  AccT maxval = Limits<AccT>::finite_min();
  AccT normalizer = AccT(0);
  
  int num_iterations = (axis_size + BLOCK_DIM * N_READS - 1) / (BLOCK_DIM * N_READS);
  for (int r = 0; r < num_iterations; ++r) {
    int index = r * BLOCK_DIM + thread_rank;
    
    // Load values
    AccT vals[N_READS];
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      int idx = index * N_READS + i;
      vals[i] = (idx < axis_size) ? static_cast<AccT>(in[idx]) : Limits<AccT>::min();
    }
    
    // Update max
    prevmax = maxval;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      maxval = maxval > vals[i] ? maxval : vals[i];
    }
    
    // Online normalizer calculation
    normalizer = normalizer * softmax_exp(prevmax - maxval);
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      normalizer = normalizer + softmax_exp(vals[i] - maxval);
    }
  }

  // First warp reduce
  prevmax = maxval;
  maxval = warp_reduce_max(maxval);
  normalizer = normalizer * softmax_exp(prevmax - maxval);
  normalizer = warp_reduce_sum(normalizer);

  __shared__ AccT local_max[WARP_SIZE];
  __shared__ AccT local_normalizer[WARP_SIZE];

  // Write to shared memory and do second warp reduce
  prevmax = maxval;
  if (lane == 0) {
    local_max[warp_id] = maxval;
  }
  __syncthreads();
  
  maxval = (lane < num_warps) ? local_max[lane] : Limits<AccT>::min();
  maxval = warp_reduce_max(maxval);
  normalizer = normalizer * softmax_exp(prevmax - maxval);
  
  if (lane == 0) {
    local_normalizer[warp_id] = normalizer;
  }
  __syncthreads();
  
  normalizer = (lane < num_warps) ? local_normalizer[lane] : AccT(0);
  normalizer = warp_reduce_sum(normalizer);
  normalizer = AccT(1) / normalizer;

  // Write output
  for (int r = 0; r < num_iterations; ++r) {
    int index = r * BLOCK_DIM + thread_rank;
    
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      int idx = index * N_READS + i;
      if (idx < axis_size) {
        AccT val = static_cast<AccT>(in[idx]);
        out[idx] = static_cast<T>(softmax_exp(val - maxval) * normalizer);
      }
    }
  }
}

// Vectorized softmax kernel for better memory throughput
template <typename T, typename AccT, int BLOCK_DIM, int N_READS = 4>
__global__ void softmax_kernel_vectorized(const T* in, T* out, int axis_size) {
  int row = blockIdx.x;
  int thread_rank = threadIdx.x;
  int lane = thread_rank % WARP_SIZE;
  int warp_id = thread_rank / WARP_SIZE;
  int num_warps = BLOCK_DIM / WARP_SIZE;
  
  in += row * axis_size;
  out += row * axis_size;

  // Online softmax: compute max and normalizer in a single pass
  AccT prevmax;
  AccT maxval = Limits<AccT>::finite_min();
  AccT normalizer = AccT(0);
  
  int vec_size = axis_size / N_READS;
  int num_iterations = (vec_size + BLOCK_DIM - 1) / BLOCK_DIM;
  
  for (int r = 0; r < num_iterations; ++r) {
    int index = r * BLOCK_DIM + thread_rank;
    
    // Load values using vectorized load
    AccT vals[N_READS];
    if (index < vec_size) {
      auto vec = load_vector<N_READS>(in, index);
#pragma unroll
      for (int i = 0; i < N_READS; ++i) {
        vals[i] = static_cast<AccT>(vec[i]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < N_READS; ++i) {
        int idx = index * N_READS + i;
        vals[i] = (idx < axis_size) ? static_cast<AccT>(in[idx]) : Limits<AccT>::min();
      }
    }
    
    // Update max
    prevmax = maxval;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      maxval = maxval > vals[i] ? maxval : vals[i];
    }
    
    // Online normalizer calculation
    normalizer = normalizer * softmax_exp(prevmax - maxval);
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      normalizer = normalizer + softmax_exp(vals[i] - maxval);
    }
  }

  // Handle remaining elements
  int remaining_start = vec_size * N_READS;
  for (int idx = remaining_start + thread_rank; idx < axis_size; idx += BLOCK_DIM) {
    prevmax = maxval;
    AccT val = static_cast<AccT>(in[idx]);
    maxval = maxval > val ? maxval : val;
    normalizer = normalizer * softmax_exp(prevmax - maxval);
    normalizer = normalizer + softmax_exp(val - maxval);
  }

  // First warp reduce
  prevmax = maxval;
  maxval = warp_reduce_max(maxval);
  normalizer = normalizer * softmax_exp(prevmax - maxval);
  normalizer = warp_reduce_sum(normalizer);

  __shared__ AccT local_max[WARP_SIZE];
  __shared__ AccT local_normalizer[WARP_SIZE];

  // Write to shared memory and do second warp reduce
  prevmax = maxval;
  if (lane == 0) {
    local_max[warp_id] = maxval;
  }
  __syncthreads();
  
  maxval = (lane < num_warps) ? local_max[lane] : Limits<AccT>::min();
  maxval = warp_reduce_max(maxval);
  normalizer = normalizer * softmax_exp(prevmax - maxval);
  
  if (lane == 0) {
    local_normalizer[warp_id] = normalizer;
  }
  __syncthreads();
  
  normalizer = (lane < num_warps) ? local_normalizer[lane] : AccT(0);
  normalizer = warp_reduce_sum(normalizer);
  normalizer = AccT(1) / normalizer;

  // Write output using vectorized store
  for (int r = 0; r < num_iterations; ++r) {
    int index = r * BLOCK_DIM + thread_rank;
    
    if (index < vec_size) {
      auto vec = load_vector<N_READS>(in, index);
      AlignedVector<T, N_READS> out_vec;
#pragma unroll
      for (int i = 0; i < N_READS; ++i) {
        AccT val = static_cast<AccT>(vec[i]);
        out_vec[i] = static_cast<T>(softmax_exp(val - maxval) * normalizer);
      }
      store_vector<N_READS>(out, index, out_vec);
    } else {
#pragma unroll
      for (int i = 0; i < N_READS; ++i) {
        int idx = index * N_READS + i;
        if (idx < axis_size) {
          AccT val = static_cast<AccT>(in[idx]);
          out[idx] = static_cast<T>(softmax_exp(val - maxval) * normalizer);
        }
      }
    }
  }

  // Handle remaining elements
  for (int idx = remaining_start + thread_rank; idx < axis_size; idx += BLOCK_DIM) {
    AccT val = static_cast<AccT>(in[idx]);
    out[idx] = static_cast<T>(softmax_exp(val - maxval) * normalizer);
  }
}

} // namespace rocm

void Softmax::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& s = stream();

  // Make sure that the last dimension is contiguous.
  auto set_output = [&s, &out](const array& x) {
    if (x.flags().contiguous && x.strides()[x.ndim() - 1] == 1) {
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

  array in = set_output(inputs[0]);
  bool precise = in.dtype() != float32 && precise_;

  int axis_size = in.shape().back();
  int n_rows = in.data_size() / axis_size;

  auto& encoder = rocm::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  
  // Choose block size based on axis size
  auto launch_softmax = [&](auto type_tag, auto acc_type_tag) {
    using T = typename decltype(type_tag)::type;
    using AccT = typename decltype(acc_type_tag)::type;
    
    constexpr int N_READS = 4;
    
    encoder.launch_kernel([&](hipStream_t stream) {
      // Choose block size based on axis size for better occupancy
      if (axis_size <= 256 * N_READS) {
        hipLaunchKernelGGL(
            (rocm::softmax_kernel<T, AccT, 256, N_READS>),
            dim3(n_rows), dim3(256), 0, stream,
            in.data<T>(), out.data<T>(), axis_size);
      } else if (axis_size <= 512 * N_READS) {
        hipLaunchKernelGGL(
            (rocm::softmax_kernel<T, AccT, 512, N_READS>),
            dim3(n_rows), dim3(512), 0, stream,
            in.data<T>(), out.data<T>(), axis_size);
      } else {
        hipLaunchKernelGGL(
            (rocm::softmax_kernel<T, AccT, 1024, N_READS>),
            dim3(n_rows), dim3(1024), 0, stream,
            in.data<T>(), out.data<T>(), axis_size);
      }
    });
  };
  
  switch (out.dtype()) {
    case float32:
      launch_softmax(type_identity<float>{}, type_identity<float>{});
      break;
    case float16:
      if (precise) {
        launch_softmax(type_identity<__half>{}, type_identity<float>{});
      } else {
        launch_softmax(type_identity<__half>{}, type_identity<__half>{});
      }
      break;
    case bfloat16:
      if (precise) {
        launch_softmax(type_identity<hip_bfloat16>{}, type_identity<float>{});
      } else {
        launch_softmax(type_identity<hip_bfloat16>{}, type_identity<hip_bfloat16>{});
      }
      break;
    default:
      throw std::runtime_error("Unsupported type for softmax");
  }
}

} // namespace mlx::core
