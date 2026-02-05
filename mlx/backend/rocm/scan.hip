// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/binary_ops.hpp"
#include "mlx/backend/rocm/device/cast_op.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/reduce/reduce_ops.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>

#include <cassert>

namespace mlx::core {

namespace rocm {

// Scan result type trait - Sum on bool produces int32
template <typename Op, typename T>
struct ScanResult {
  using type = T;
};

template <>
struct ScanResult<Sum, bool> {
  using type = int32_t;
};

// ReduceInit specialization for LogAddExp
template <typename T>
struct ReduceInit<LogAddExp, T> {
  __device__ static T value() {
    return Limits<T>::min();
  }
};

// Load values helper - handles reverse and boundary conditions
template <bool reverse, typename T, typename U, int N_READS>
__device__ void
load_values(int index, const T* in, U (&values)[N_READS], int size, U init) {
  int remaining = size - index * N_READS;
  if constexpr (reverse) {
    in += remaining - N_READS;
    if (remaining < N_READS) {
#pragma unroll
      for (int i = 0; i < N_READS; ++i) {
        values[N_READS - i - 1] =
            (N_READS - i - 1 < remaining) ? cast_to<U>(in[i]) : init;
      }
    } else {
#pragma unroll
      for (int i = 0; i < N_READS; ++i) {
        values[N_READS - i - 1] = cast_to<U>(in[i]);
      }
    }
  } else {
    in += index * N_READS;
    if (remaining < N_READS) {
#pragma unroll
      for (int i = 0; i < N_READS; ++i) {
        values[i] = (i < remaining) ? cast_to<U>(in[i]) : init;
      }
    } else {
#pragma unroll
      for (int i = 0; i < N_READS; ++i) {
        values[i] = cast_to<U>(in[i]);
      }
    }
  }
}

// Store values helper - handles reverse, exclusive offset, and boundary conditions
template <bool reverse, int offset, typename T, int N_READS>
__device__ void
store_values(int index, T* out, T (&values)[N_READS], int size) {
  int start = index * N_READS + offset;
  int remaining = size - start;
  if constexpr (reverse) {
    out += remaining - N_READS;
    if (remaining < N_READS) {
#pragma unroll
      for (int i = 0; i < N_READS; ++i) {
        if (N_READS - i - 1 < remaining) {
          out[i] = values[N_READS - i - 1];
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < N_READS; ++i) {
        out[i] = values[N_READS - i - 1];
      }
    }
  } else {
    out += start;
    if (remaining < N_READS) {
#pragma unroll
      for (int i = 0; i < N_READS; ++i) {
        if (i < remaining) {
          out[i] = values[i];
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < N_READS; ++i) {
        out[i] = values[i];
      }
    }
  }
}

// Type-safe shuffle wrappers that handle bfloat16 and half types
// For most types, __shfl_up returns the same type
template <typename T>
__device__ __forceinline__ T shfl_up_safe(T val, unsigned int delta) {
  return __shfl_up(val, delta);
}

// Specialization for hip_bfloat16 - __shfl_up returns float
template <>
__device__ __forceinline__ hip_bfloat16 shfl_up_safe(hip_bfloat16 val, unsigned int delta) {
  return hip_bfloat16(__shfl_up(static_cast<float>(val), delta));
}

// Specialization for __half - __shfl_up returns float
template <>
__device__ __forceinline__ __half shfl_up_safe(__half val, unsigned int delta) {
  return __half(__shfl_up(__half2float(val), delta));
}

// Specialization for hipFloatComplex (complex type)
template <>
__device__ __forceinline__ hipFloatComplex shfl_up_safe(hipFloatComplex val, unsigned int delta) {
  return make_hipFloatComplex(
      __shfl_up(val.x, delta),
      __shfl_up(val.y, delta));
}

// Type-safe shfl wrapper
template <typename T>
__device__ __forceinline__ T shfl_safe(T val, int src_lane) {
  return __shfl(val, src_lane);
}

// Specialization for hip_bfloat16
template <>
__device__ __forceinline__ hip_bfloat16 shfl_safe(hip_bfloat16 val, int src_lane) {
  return hip_bfloat16(__shfl(static_cast<float>(val), src_lane));
}

// Specialization for __half
template <>
__device__ __forceinline__ __half shfl_safe(__half val, int src_lane) {
  return __half(__shfl(__half2float(val), src_lane));
}

// Specialization for hipFloatComplex (complex type)
template <>
__device__ __forceinline__ hipFloatComplex shfl_safe(hipFloatComplex val, int src_lane) {
  return make_hipFloatComplex(
      __shfl(val.x, src_lane),
      __shfl(val.y, src_lane));
}

// Warp-level inclusive scan using shuffle
template <typename T, typename Op>
__device__ T warp_inclusive_scan(T val, Op op) {
  int lane = threadIdx.x % WARP_SIZE;
#pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
    T other = shfl_up_safe(val, offset);
    if (lane >= offset) {
      val = op(val, other);
    }
  }
  return val;
}

// Warp-level exclusive scan using shuffle
template <typename T, typename Op>
__device__ T warp_exclusive_scan(T val, Op op, T init) {
  T inclusive = warp_inclusive_scan(val, op);
  T exclusive = shfl_up_safe(inclusive, 1);
  return ((threadIdx.x % WARP_SIZE) == 0) ? init : exclusive;
}

// Contiguous scan kernel - optimized for stride=1 arrays
template <
    typename T,
    typename U,
    typename Op,
    int N_READS,
    bool inclusive,
    bool reverse>
__global__ void contiguous_scan(const T* in, U* out, int32_t axis_size) {
  // Calculate block and thread indices
  int block_rank = blockIdx.x;
  int thread_rank = threadIdx.x;
  int block_size = blockDim.x;
  int warp_id = thread_rank / WARP_SIZE;
  int lane_id = thread_rank % WARP_SIZE;
  int num_warps = block_size / WARP_SIZE;

  in += block_rank * axis_size;
  out += block_rank * axis_size;

  __shared__ U warp_sums[WARP_SIZE];

  Op op;
  U init = ReduceInit<Op, T>::value();
  U prefix = init;

  // Scan per block
  int num_iterations = (axis_size + block_size * N_READS - 1) / (block_size * N_READS);
  for (int r = 0; r < num_iterations; ++r) {
    int32_t index = r * block_size + thread_rank;
    U values[N_READS];
    load_values<reverse>(index, in, values, axis_size, init);

    // Compute an inclusive scan per thread
#pragma unroll
    for (int i = 1; i < N_READS; ++i) {
      values[i] = op(values[i], values[i - 1]);
    }

    // Compute exclusive scan of thread sums within warp
    U thread_sum = values[N_READS - 1];
    U prev_thread_sum = warp_exclusive_scan(thread_sum, op, init);

    // Write warp's sum to shared memory
    if (lane_id == WARP_SIZE - 1) {
      warp_sums[warp_id] = op(prev_thread_sum, thread_sum);
    }
    __syncthreads();

    // Compute exclusive scan of warp sums (first warp only)
    if (warp_id == 0) {
      U warp_val = (lane_id < num_warps) ? warp_sums[lane_id] : init;
      U prev_warp_sum = warp_exclusive_scan(warp_val, op, init);
      if (lane_id < num_warps) {
        warp_sums[lane_id] = prev_warp_sum;
      }
    }
    __syncthreads();

    // Compute the output
    U warp_prefix = warp_sums[warp_id];
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      values[i] = op(values[i], prefix);
      values[i] = op(values[i], warp_prefix);
      values[i] = op(values[i], prev_thread_sum);
    }

    // Write the values
    if (inclusive) {
      store_values<reverse, 0>(index, out, values, axis_size);
    } else {
      store_values<reverse, 1>(index, out, values, axis_size);
      if (reverse) {
        if (thread_rank == 0 && index == 0) {
          out[axis_size - 1] = init;
        }
      } else {
        if (thread_rank == 0 && index == 0) {
          out[0] = init;
        }
      }
    }
    __syncthreads();

    // Share the prefix for next iteration
    if ((warp_id == num_warps - 1) && (lane_id == WARP_SIZE - 1)) {
      warp_sums[0] = values[N_READS - 1];
    }
    __syncthreads();
    prefix = warp_sums[0];
  }
}

// Strided scan kernel - for non-contiguous arrays (stride > 1)
template <
    typename T,
    typename U,
    typename Op,
    int N_READS,
    int BM,
    int BN,
    bool inclusive,
    bool reverse>
__global__ void strided_scan(
    const T* in,
    U* out,
    int32_t axis_size,
    int64_t stride,
    int64_t stride_blocks) {
  int block_rank = blockIdx.x;
  int thread_rank = threadIdx.x;
  int warp_id = thread_rank / WARP_SIZE;
  int lane_id = thread_rank % WARP_SIZE;

  constexpr int BN_pad = WARP_SIZE + 16 / sizeof(U);
  constexpr int n_warps = BN / N_READS;
  constexpr int n_scans = BN / n_warps;

  __shared__ U read_buffer[BM * BN_pad];

  Op op;
  U init = ReduceInit<Op, T>::value();
  U values[n_scans];
  U prefix[n_scans];
#pragma unroll
  for (int i = 0; i < n_scans; ++i) {
    prefix[i] = init;
  }

  // Compute offsets
  int64_t offset = (block_rank / stride_blocks) * axis_size * stride;
  int64_t global_index_x = (block_rank % stride_blocks) * BN;
  uint32_t read_offset_y = (thread_rank * N_READS) / BN;
  uint32_t read_offset_x = (thread_rank * N_READS) % BN;
  uint32_t scan_offset_y = lane_id;
  uint32_t scan_offset_x = warp_id * n_scans;

  uint32_t stride_limit = stride - global_index_x;
  in += offset + global_index_x + read_offset_x;
  out += offset + global_index_x + read_offset_x;
  U* read_into = read_buffer + read_offset_y * BN_pad + read_offset_x;
  U* read_from = read_buffer + scan_offset_y * BN_pad + scan_offset_x;

  for (uint32_t j = 0; j < axis_size; j += BM) {
    // Calculate the indices for the current thread
    uint32_t index_y = j + read_offset_y;
    uint32_t check_index_y = index_y;
    if (reverse) {
      index_y = axis_size - 1 - index_y;
    }

    // Read into shared memory
    if (check_index_y < axis_size && (read_offset_x + N_READS) < stride_limit) {
#pragma unroll
      for (int i = 0; i < N_READS; ++i) {
        read_into[i] = cast_to<U>(in[index_y * stride + i]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < N_READS; ++i) {
        if (check_index_y < axis_size && (read_offset_x + i) < stride_limit) {
          read_into[i] = cast_to<U>(in[index_y * stride + i]);
        } else {
          read_into[i] = init;
        }
      }
    }
    __syncthreads();

    // Read strided into registers
#pragma unroll
    for (int i = 0; i < n_scans; ++i) {
      values[i] = read_from[i];
    }

    // Perform the scan using warp shuffle
#pragma unroll
    for (int i = 0; i < n_scans; ++i) {
      values[i] = warp_inclusive_scan(values[i], op);
      values[i] = op(values[i], prefix[i]);
      prefix[i] = shfl_safe(values[i], WARP_SIZE - 1);
    }

    // Write to shared memory
#pragma unroll
    for (int i = 0; i < n_scans; ++i) {
      read_from[i] = values[i];
    }
    __syncthreads();

    // Write to device memory
    if (!inclusive) {
      if (check_index_y == 0) {
        if ((read_offset_x + N_READS) < stride_limit) {
#pragma unroll
          for (int i = 0; i < N_READS; ++i) {
            out[index_y * stride + i] = init;
          }
        } else {
#pragma unroll
          for (int i = 0; i < N_READS; ++i) {
            if ((read_offset_x + i) < stride_limit) {
              out[index_y * stride + i] = init;
            }
          }
        }
      }
      if (reverse) {
        index_y -= 1;
        check_index_y += 1;
      } else {
        index_y += 1;
        check_index_y += 1;
      }
    }
    if (check_index_y < axis_size && (read_offset_x + N_READS) < stride_limit) {
#pragma unroll
      for (int i = 0; i < N_READS; ++i) {
        out[index_y * stride + i] = read_into[i];
      }
    } else {
#pragma unroll
      for (int i = 0; i < N_READS; ++i) {
        if (check_index_y < axis_size && (read_offset_x + i) < stride_limit) {
          out[index_y * stride + i] = read_into[i];
        }
      }
    }
  }
}

} // namespace rocm

// Dispatch scan operations
template <typename F>
void dispatch_scan_ops(Scan::ReduceType scan_op, F&& f) {
  if (scan_op == Scan::ReduceType::Max) {
    f(type_identity<rocm::Max>{});
  } else if (scan_op == Scan::ReduceType::Min) {
    f(type_identity<rocm::Min>{});
  } else if (scan_op == Scan::ReduceType::Sum) {
    f(type_identity<rocm::Sum>{});
  } else if (scan_op == Scan::ReduceType::Prod) {
    f(type_identity<rocm::Prod>{});
  } else if (scan_op == Scan::ReduceType::LogAddExp) {
    f(type_identity<rocm::LogAddExp>{});
  } else {
    throw std::invalid_argument("Unknown reduce type.");
  }
}

// Get operation name for error messages
template <typename Op>
const char* op_to_string() {
  if constexpr (std::is_same_v<Op, rocm::Max>) {
    return "Max";
  } else if constexpr (std::is_same_v<Op, rocm::Min>) {
    return "Min";
  } else if constexpr (std::is_same_v<Op, rocm::Sum>) {
    return "Sum";
  } else if constexpr (std::is_same_v<Op, rocm::Prod>) {
    return "Prod";
  } else if constexpr (std::is_same_v<Op, rocm::LogAddExp>) {
    return "LogAddExp";
  } else {
    return "Unknown";
  }
}

// Check if operation is supported for type
template <typename Op, typename T>
constexpr bool supports_scan_op() {
  if constexpr (std::is_same_v<Op, rocm::LogAddExp>) {
    return is_inexact_v<T>;
  } else {
    return true;
  }
}

// Dispatch scan types - excludes complex types which don't support warp shuffle
template <typename F>
void dispatch_scan_types(Dtype dtype, F&& f) {
  switch (dtype) {
    case bool_:
      f(type_identity<bool>{});
      break;
    case uint8:
      f(type_identity<uint8_t>{});
      break;
    case uint16:
      f(type_identity<uint16_t>{});
      break;
    case uint32:
      f(type_identity<uint32_t>{});
      break;
    case uint64:
      f(type_identity<uint64_t>{});
      break;
    case int8:
      f(type_identity<int8_t>{});
      break;
    case int16:
      f(type_identity<int16_t>{});
      break;
    case int32:
      f(type_identity<int32_t>{});
      break;
    case int64:
      f(type_identity<int64_t>{});
      break;
    case float16:
      f(type_identity<float16_t>{});
      break;
    case float32:
      f(type_identity<float>{});
      break;
    case bfloat16:
      f(type_identity<bfloat16_t>{});
      break;
    default:
      throw std::runtime_error(
          "Scan operations are not supported for complex types on ROCm.");
  }
}

void Scan::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto in = inputs[0];
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  // Check for complex types early
  if (in.dtype() == complex64) {
    throw std::runtime_error(
        "Scan operations are not supported for complex types on ROCm.");
  }

  if (in.flags().contiguous && in.strides()[axis_] != 0) {
    if (in.is_donatable() && in.itemsize() == out.itemsize()) {
      out.copy_shared_buffer(in);
    } else {
      out.set_data(
          allocator::malloc(in.data_size() * out.itemsize()),
          in.data_size(),
          in.strides(),
          in.flags());
    }
  } else {
    in = contiguous_copy_gpu(in, s);
    out.copy_shared_buffer(in);
  }

  constexpr int N_READS = 4;
  int32_t axis_size = in.shape(axis_);
  bool contiguous = in.strides()[axis_] == 1;

  encoder.set_input_array(in);
  encoder.set_output_array(out);

  dispatch_scan_types(in.dtype(), [&](auto type_tag) {
    using T = hip_type_t<MLX_GET_TYPE(type_tag)>;
    dispatch_scan_ops(reduce_type_, [&](auto scan_op_tag) {
      using Op = MLX_GET_TYPE(scan_op_tag);
      if constexpr (supports_scan_op<Op, T>()) {
        using U = typename rocm::ScanResult<Op, T>::type;
        dispatch_bool(inclusive_, [&](auto inclusive) {
          dispatch_bool(reverse_, [&](auto reverse) {
            encoder.launch_kernel([&](hipStream_t stream) {
              if (contiguous) {
                int block_dim = ceildiv(axis_size, N_READS);
                block_dim = ceildiv(block_dim, WARP_SIZE) * WARP_SIZE;
                block_dim = std::min(block_dim, WARP_SIZE * WARP_SIZE);
                int num_blocks = in.data_size() / axis_size;
                hipLaunchKernelGGL(
                    (rocm::contiguous_scan<
                        T,
                        U,
                        Op,
                        N_READS,
                        inclusive.value,
                        reverse.value>),
                    dim3(num_blocks),
                    dim3(block_dim),
                    0,
                    stream,
                    in.data<T>(),
                    out.data<U>(),
                    axis_size);
              } else {
                constexpr int BM = WARP_SIZE;
                constexpr int BN = WARP_SIZE;
                int64_t stride = in.strides()[axis_];
                int64_t stride_blocks = ceildiv(stride, (int64_t)BN);
                dim3 num_blocks = get_2d_grid_dims(
                    in.shape(), in.strides(), axis_size * stride);
                if (num_blocks.x * stride_blocks <= UINT32_MAX) {
                  num_blocks.x *= stride_blocks;
                } else {
                  num_blocks.y *= stride_blocks;
                }
                int block_dim = (BN / N_READS) * WARP_SIZE;
                hipLaunchKernelGGL(
                    (rocm::strided_scan<
                        T,
                        U,
                        Op,
                        N_READS,
                        BM,
                        BN,
                        inclusive.value,
                        reverse.value>),
                    num_blocks,
                    dim3(block_dim),
                    0,
                    stream,
                    in.data<T>(),
                    out.data<U>(),
                    axis_size,
                    stride,
                    stride_blocks);
              }
            });
          });
        });
      } else {
        throw std::runtime_error(
            std::string("Can not do scan op ") + op_to_string<Op>() +
            " on inputs of " + dtype_to_string(in.dtype()) +
            " with result of " + dtype_to_string(out.dtype()) + ".");
      }
    });
  });
}

} // namespace mlx::core
