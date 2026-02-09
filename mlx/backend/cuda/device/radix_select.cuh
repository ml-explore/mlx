// Copyright © 2025 Apple Inc.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include "mlx/backend/cuda/device/utils.cuh"

namespace mlx::core::cu {

///////////////////////////////////////////////////////////////////////////////
// Radix Select Implementation for CUDA
//
// Multi-pass radix-based selection algorithm for partition operations.
// Uses IEEE 754 bit manipulation for correct floating-point ordering.
///////////////////////////////////////////////////////////////////////////////

// Radix configuration
constexpr int RADIX_BITS = 8;
constexpr int RADIX_SIZE = 1 << RADIX_BITS; // 256 bins

///////////////////////////////////////////////////////////////////////////////
// Bit manipulation for radix sorting
///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct RadixTraits;

template <>
struct RadixTraits<float> {
  using UnsignedT = uint32_t;
  static constexpr int BITS = 32;

  __device__ __forceinline__ static UnsignedT to_radix(float val) {
    uint32_t bits = __float_as_uint(val);
    uint32_t mask = -int32_t(bits >> 31) | 0x80000000u;
    return bits ^ mask;
  }

  __device__ __forceinline__ static float from_radix(UnsignedT bits) {
    uint32_t mask = ((bits >> 31) - 1) | 0x80000000u;
    return __uint_as_float(bits ^ mask);
  }
};

template <>
struct RadixTraits<double> {
  using UnsignedT = uint64_t;
  static constexpr int BITS = 64;

  __device__ __forceinline__ static UnsignedT to_radix(double val) {
    uint64_t bits = __double_as_longlong(val);
    uint64_t mask = -int64_t(bits >> 63) | 0x8000000000000000ull;
    return bits ^ mask;
  }

  __device__ __forceinline__ static double from_radix(UnsignedT bits) {
    uint64_t mask = ((bits >> 63) - 1) | 0x8000000000000000ull;
    return __longlong_as_double(bits ^ mask);
  }
};

template <>
struct RadixTraits<__half> {
  using UnsignedT = uint16_t;
  static constexpr int BITS = 16;

  __device__ __forceinline__ static UnsignedT to_radix(__half val) {
    uint16_t bits = __half_as_ushort(val);
    uint16_t mask = -int16_t(bits >> 15) | 0x8000u;
    return bits ^ mask;
  }

  __device__ __forceinline__ static __half from_radix(UnsignedT bits) {
    uint16_t mask = ((bits >> 15) - 1) | 0x8000u;
    return __ushort_as_half(bits ^ mask);
  }
};

template <>
struct RadixTraits<__nv_bfloat16> {
  using UnsignedT = uint16_t;
  static constexpr int BITS = 16;

  __device__ __forceinline__ static UnsignedT to_radix(__nv_bfloat16 val) {
    uint16_t bits = __bfloat16_as_ushort(val);
    uint16_t mask = -int16_t(bits >> 15) | 0x8000u;
    return bits ^ mask;
  }

  __device__ __forceinline__ static __nv_bfloat16 from_radix(UnsignedT bits) {
    uint16_t mask = ((bits >> 15) - 1) | 0x8000u;
    return __ushort_as_bfloat16(bits ^ mask);
  }
};

template <>
struct RadixTraits<int8_t> {
  using UnsignedT = uint8_t;
  static constexpr int BITS = 8;

  __device__ __forceinline__ static UnsignedT to_radix(int8_t val) {
    return static_cast<UnsignedT>(val) ^ 0x80u;
  }

  __device__ __forceinline__ static int8_t from_radix(UnsignedT bits) {
    return static_cast<int8_t>(bits ^ 0x80u);
  }
};

template <>
struct RadixTraits<int16_t> {
  using UnsignedT = uint16_t;
  static constexpr int BITS = 16;

  __device__ __forceinline__ static UnsignedT to_radix(int16_t val) {
    return static_cast<UnsignedT>(val) ^ 0x8000u;
  }

  __device__ __forceinline__ static int16_t from_radix(UnsignedT bits) {
    return static_cast<int16_t>(bits ^ 0x8000u);
  }
};

template <>
struct RadixTraits<int32_t> {
  using UnsignedT = uint32_t;
  static constexpr int BITS = 32;

  __device__ __forceinline__ static UnsignedT to_radix(int32_t val) {
    return static_cast<UnsignedT>(val) ^ 0x80000000u;
  }

  __device__ __forceinline__ static int32_t from_radix(UnsignedT bits) {
    return static_cast<int32_t>(bits ^ 0x80000000u);
  }
};

template <>
struct RadixTraits<int64_t> {
  using UnsignedT = uint64_t;
  static constexpr int BITS = 64;

  __device__ __forceinline__ static UnsignedT to_radix(int64_t val) {
    return static_cast<UnsignedT>(val) ^ 0x8000000000000000ull;
  }

  __device__ __forceinline__ static int64_t from_radix(UnsignedT bits) {
    return static_cast<int64_t>(bits ^ 0x8000000000000000ull);
  }
};

template <>
struct RadixTraits<bool> {
  using UnsignedT = uint8_t;
  static constexpr int BITS = 8;

  __device__ __forceinline__ static UnsignedT to_radix(bool val) {
    return static_cast<uint8_t>(val);
  }

  __device__ __forceinline__ static bool from_radix(UnsignedT bits) {
    return bits != 0;
  }
};

template <>
struct RadixTraits<uint8_t> {
  using UnsignedT = uint8_t;
  static constexpr int BITS = 8;

  __device__ __forceinline__ static UnsignedT to_radix(uint8_t val) {
    return val;
  }

  __device__ __forceinline__ static uint8_t from_radix(UnsignedT bits) {
    return bits;
  }
};

template <>
struct RadixTraits<uint16_t> {
  using UnsignedT = uint16_t;
  static constexpr int BITS = 16;

  __device__ __forceinline__ static UnsignedT to_radix(uint16_t val) {
    return val;
  }

  __device__ __forceinline__ static uint16_t from_radix(UnsignedT bits) {
    return bits;
  }
};

template <>
struct RadixTraits<uint32_t> {
  using UnsignedT = uint32_t;
  static constexpr int BITS = 32;

  __device__ __forceinline__ static UnsignedT to_radix(uint32_t val) {
    return val;
  }

  __device__ __forceinline__ static uint32_t from_radix(UnsignedT bits) {
    return bits;
  }
};

template <>
struct RadixTraits<uint64_t> {
  using UnsignedT = uint64_t;
  static constexpr int BITS = 64;

  __device__ __forceinline__ static UnsignedT to_radix(uint64_t val) {
    return val;
  }

  __device__ __forceinline__ static uint64_t from_radix(UnsignedT bits) {
    return bits;
  }
};

template <typename UnsignedT>
__device__ __forceinline__ int
extract_digit(UnsignedT val, int start_bit, int num_bits) {
  return (val >> start_bit) & ((1 << num_bits) - 1);
}

template <typename T>
__device__ __forceinline__ bool is_nan_value(T val) {
  if constexpr (cuda::std::is_floating_point_v<T>) {
    return cuda::std::isnan(val);
  } else if constexpr (cuda::std::is_same_v<T, __half>) {
    return __hisnan(val);
  } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) {
    return __hisnan(val);
  } else {
    return false;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Warp-level utilities
///////////////////////////////////////////////////////////////////////////////

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

///////////////////////////////////////////////////////////////////////////////
// Single-pass Radix Select for small arrays (fits in shared memory)
///////////////////////////////////////////////////////////////////////////////

template <
    typename ValT,
    typename OutT,
    bool ARG_PARTITION,
    bool USE_SIMPLE_STRIDE,
    int BLOCK_THREADS,
    int ITEMS_PER_THREAD>
__global__ void radix_select_small_kernel(
    const ValT* input,
    OutT* output,
    int kth,
    int n,
    int64_t in_stride,
    int64_t out_stride,
    int64_t in_segment_stride,
    int64_t out_segment_stride,
    const int32_t* nc_shape,
    const int64_t* in_nc_strides,
    const int64_t* out_nc_strides,
    int nc_dim) {
  using Traits = RadixTraits<ValT>;
  using UnsignedT = typename Traits::UnsignedT;

  constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
  constexpr int NUM_PASSES = (Traits::BITS + RADIX_BITS - 1) / RADIX_BITS;

  // Shared memory
  __shared__ UnsignedT shared_keys[TILE_SIZE];
  __shared__ uint32_t shared_idxs[TILE_SIZE];
  __shared__ int shared_hist[RADIX_SIZE];
  __shared__ int shared_count[2];

  int row = blockIdx.y;

  // Compute row pointers based on addressing mode
  const ValT* row_input;
  OutT* row_output;
  if constexpr (USE_SIMPLE_STRIDE) {
    row_input = input + row * in_segment_stride;
    row_output = output + row * out_segment_stride;
  } else {
    int64_t in_block_idx =
        elem_to_loc(int64_t(row), nc_shape, in_nc_strides, nc_dim);
    int64_t out_block_idx =
        elem_to_loc(int64_t(row), nc_shape, out_nc_strides, nc_dim);
    row_input = input + in_block_idx;
    row_output = output + out_block_idx;
  }

  int tile_n = min(n, TILE_SIZE);

  // Load data into shared memory
  for (int i = threadIdx.x; i < TILE_SIZE; i += BLOCK_THREADS) {
    if (i < tile_n) {
      ValT val = row_input[i * in_stride];
      UnsignedT key = Traits::to_radix(val);
      if (is_nan_value(val)) {
        key = ~UnsignedT(0);
      }
      shared_keys[i] = key;
      shared_idxs[i] = i;
    } else {
      shared_keys[i] = ~UnsignedT(0);
      shared_idxs[i] = i;
    }
  }
  __syncthreads();

  // Radix select to find pivot
  int k = kth + 1;
  UnsignedT target_prefix = 0;
  UnsignedT prefix_mask = 0;

  for (int pass = NUM_PASSES - 1; pass >= 0; pass--) {
    int start_bit = pass * RADIX_BITS;

    // Clear histogram
    for (int i = threadIdx.x; i < RADIX_SIZE; i += BLOCK_THREADS) {
      shared_hist[i] = 0;
    }
    __syncthreads();

    // Build histogram
    for (int i = threadIdx.x; i < tile_n; i += BLOCK_THREADS) {
      UnsignedT key = shared_keys[i];
      if ((key & prefix_mask) == target_prefix) {
        int digit = extract_digit(key, start_bit, RADIX_BITS);
        atomicAdd(&shared_hist[digit], 1);
      }
    }
    __syncthreads();

    // Find target bin (single thread)
    if (threadIdx.x == 0) {
      int cumsum = 0;
      int target_bin = 0;
      for (int bin = 0; bin < RADIX_SIZE; bin++) {
        int count = shared_hist[bin];
        if (cumsum + count >= k) {
          target_bin = bin;
          k = k - cumsum;
          break;
        }
        cumsum += count;
      }
      shared_count[0] = target_bin;
      shared_count[1] = k;
    }
    __syncthreads();

    int target_bin = shared_count[0];
    k = shared_count[1];

    UnsignedT digit_mask = UnsignedT((1 << RADIX_BITS) - 1) << start_bit;
    target_prefix |= UnsignedT(target_bin) << start_bit;
    prefix_mask |= digit_mask;

    __syncthreads();
  }

  // Output partitioned array
  if (threadIdx.x == 0) {
    shared_count[0] = 0;
  }
  __syncthreads();

  // Phase 1: output elements less than pivot
  for (int i = threadIdx.x; i < tile_n; i += BLOCK_THREADS) {
    UnsignedT key = shared_keys[i];
    if (key < target_prefix) {
      int pos = atomicAdd(&shared_count[0], 1);
      if (ARG_PARTITION) {
        row_output[pos * out_stride] = shared_idxs[i];
      } else {
        row_output[pos * out_stride] = row_input[shared_idxs[i] * in_stride];
      }
    }
  }
  __syncthreads();

  // Phase 2: output elements equal to pivot
  for (int i = threadIdx.x; i < tile_n; i += BLOCK_THREADS) {
    UnsignedT key = shared_keys[i];
    if (key == target_prefix) {
      int pos = atomicAdd(&shared_count[0], 1);
      if (ARG_PARTITION) {
        row_output[pos * out_stride] = shared_idxs[i];
      } else {
        row_output[pos * out_stride] = row_input[shared_idxs[i] * in_stride];
      }
    }
  }
  __syncthreads();

  // Phase 3: output elements greater than pivot
  for (int i = threadIdx.x; i < tile_n; i += BLOCK_THREADS) {
    UnsignedT key = shared_keys[i];
    if (key > target_prefix) {
      int pos = atomicAdd(&shared_count[0], 1);
      if (ARG_PARTITION) {
        row_output[pos * out_stride] = shared_idxs[i];
      } else {
        row_output[pos * out_stride] = row_input[shared_idxs[i] * in_stride];
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Large array streaming kernel (multi-pass, in-place)
///////////////////////////////////////////////////////////////////////////////

template <typename ValT, typename OutT, bool ARG_PARTITION, int BLOCK_THREADS>
__global__ void radix_select_large_streaming_kernel(
    const ValT* input,
    OutT* output,
    int n,
    int kth,
    int64_t in_stride,
    int64_t out_stride,
    const int32_t* nc_shape,
    const int64_t* in_nc_strides,
    const int64_t* out_nc_strides,
    int nc_dim) {
  using Traits = RadixTraits<ValT>;
  using UnsignedT = typename Traits::UnsignedT;
  constexpr int NUM_PASSES = (Traits::BITS + RADIX_BITS - 1) / RADIX_BITS;

  int row = blockIdx.y;
  int64_t in_block_idx =
      elem_to_loc(int64_t(row), nc_shape, in_nc_strides, nc_dim);
  int64_t out_block_idx =
      elem_to_loc(int64_t(row), nc_shape, out_nc_strides, nc_dim);
  const ValT* row_input = input + in_block_idx;
  OutT* row_output = output + out_block_idx;

  // Shared memory
  __shared__ int shared_hist[RADIX_SIZE];
  __shared__ int shared_pivot_info[2];
  __shared__ int shared_counts[2];
  __shared__ int shared_output_counters[3];

  int k = kth + 1;
  UnsignedT target_prefix = 0;
  UnsignedT prefix_mask = 0;

  // Multi-pass to find pivot
  for (int pass = NUM_PASSES - 1; pass >= 0; pass--) {
    int start_bit = pass * RADIX_BITS;

    // Clear histogram
    for (int i = threadIdx.x; i < RADIX_SIZE; i += BLOCK_THREADS) {
      shared_hist[i] = 0;
    }
    __syncthreads();

    // Build histogram
    bool is_contiguous = (in_stride == 1);
    if (is_contiguous) {
      for (int i = threadIdx.x; i < n; i += BLOCK_THREADS) {
        ValT val = row_input[i];
        UnsignedT key = Traits::to_radix(val);
        if (is_nan_value(val)) {
          key = ~UnsignedT(0);
        }

        if ((key & prefix_mask) == target_prefix) {
          int digit = extract_digit(key, start_bit, RADIX_BITS);
          atomicAdd(&shared_hist[digit], 1);
        }
      }
    } else {
      for (int i = threadIdx.x; i < n; i += BLOCK_THREADS) {
        ValT val = row_input[i * in_stride];
        UnsignedT key = Traits::to_radix(val);
        if (is_nan_value(val)) {
          key = ~UnsignedT(0);
        }

        if ((key & prefix_mask) == target_prefix) {
          int digit = extract_digit(key, start_bit, RADIX_BITS);
          atomicAdd(&shared_hist[digit], 1);
        }
      }
    }
    __syncthreads();

    // Find target bin
    if (threadIdx.x == 0) {
      int cumsum = 0;
      int target_bin = 0;
      for (int bin = 0; bin < RADIX_SIZE; bin++) {
        int count = shared_hist[bin];
        if (cumsum + count >= k) {
          target_bin = bin;
          k = k - cumsum;
          break;
        }
        cumsum += count;
      }
      shared_pivot_info[0] = target_bin;
      shared_pivot_info[1] = k;
    }
    __syncthreads();

    int target_bin = shared_pivot_info[0];
    k = shared_pivot_info[1];

    UnsignedT digit_mask = UnsignedT((1 << RADIX_BITS) - 1) << start_bit;
    target_prefix |= UnsignedT(target_bin) << start_bit;
    prefix_mask |= digit_mask;

    // Initialize counters for next phase
    if (threadIdx.x == 0) {
      shared_counts[0] = 0;
      shared_counts[1] = 0;
    }
    __syncthreads();
  }

  // Count partition sizes with warp reduction
  int local_less = 0, local_equal = 0;
  bool is_contiguous = (in_stride == 1);

  if (is_contiguous) {
    for (int i = threadIdx.x; i < n; i += BLOCK_THREADS) {
      ValT val = row_input[i];
      UnsignedT key = Traits::to_radix(val);
      if (is_nan_value(val)) {
        key = ~UnsignedT(0);
      }
      if (key < target_prefix)
        local_less++;
      else if (key == target_prefix)
        local_equal++;
    }
  } else {
    for (int i = threadIdx.x; i < n; i += BLOCK_THREADS) {
      ValT val = row_input[i * in_stride];
      UnsignedT key = Traits::to_radix(val);
      if (is_nan_value(val)) {
        key = ~UnsignedT(0);
      }
      if (key < target_prefix)
        local_less++;
      else if (key == target_prefix)
        local_equal++;
    }
  }

  // Warp reduction
  local_less = warp_reduce_sum(local_less);
  local_equal = warp_reduce_sum(local_equal);

  // First lane of each warp aggregates to shared memory
  int lane = threadIdx.x % WARP_SIZE;
  if (lane == 0) {
    atomicAdd(&shared_counts[0], local_less);
    atomicAdd(&shared_counts[1], local_equal);
  }
  __syncthreads();

  // Read final counts
  int less_count = shared_counts[0];
  int equal_count = shared_counts[1];

  // Initialize output counters
  if (threadIdx.x == 0) {
    shared_output_counters[0] = 0;
    shared_output_counters[1] = 0;
    shared_output_counters[2] = 0;
  }
  __syncthreads();

  // Output partitioned elements
  if (is_contiguous && out_stride == 1) {
    // Fast path: both input and output are contiguous
    for (int i = threadIdx.x; i < n; i += BLOCK_THREADS) {
      ValT val = row_input[i];
      UnsignedT key = Traits::to_radix(val);
      if (is_nan_value(val)) {
        key = ~UnsignedT(0);
      }

      int pos;
      if (key < target_prefix) {
        pos = atomicAdd(&shared_output_counters[0], 1);
      } else if (key == target_prefix) {
        pos = less_count + atomicAdd(&shared_output_counters[1], 1);
      } else {
        pos =
            less_count + equal_count + atomicAdd(&shared_output_counters[2], 1);
      }

      if (ARG_PARTITION) {
        row_output[pos] = i;
      } else {
        row_output[pos] = val;
      }
    }
  } else {
    for (int i = threadIdx.x; i < n; i += BLOCK_THREADS) {
      ValT val = row_input[i * in_stride];
      UnsignedT key = Traits::to_radix(val);
      if (is_nan_value(val)) {
        key = ~UnsignedT(0);
      }

      int pos;
      if (key < target_prefix) {
        pos = atomicAdd(&shared_output_counters[0], 1);
      } else if (key == target_prefix) {
        pos = less_count + atomicAdd(&shared_output_counters[1], 1);
      } else {
        pos =
            less_count + equal_count + atomicAdd(&shared_output_counters[2], 1);
      }

      if (ARG_PARTITION) {
        row_output[pos * out_stride] = i;
      } else {
        row_output[pos * out_stride] = val;
      }
    }
  }
}

} // namespace mlx::core::cu
