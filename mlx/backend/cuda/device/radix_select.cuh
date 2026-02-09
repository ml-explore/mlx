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

template <typename ValT>
__device__ __forceinline__ typename RadixTraits<ValT>::UnsignedT
radix_key_with_nan_last(ValT val) {
  using UnsignedT = typename RadixTraits<ValT>::UnsignedT;
  UnsignedT key = RadixTraits<ValT>::to_radix(val);
  if (is_nan_value(val)) {
    key = ~UnsignedT(0);
  }
  return key;
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

template <int BLOCK_THREADS>
__device__ __forceinline__ int block_exclusive_scan(
    int val,
    int* shared_warp_sums,
    int* block_total = nullptr) {
  static_assert(BLOCK_THREADS % WARP_SIZE == 0);
  constexpr int NUM_WARPS = BLOCK_THREADS / WARP_SIZE;

  int lane = threadIdx.x & (WARP_SIZE - 1);
  int warp = threadIdx.x / WARP_SIZE;

  int inclusive = val;
#pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    int n = __shfl_up_sync(0xFFFFFFFF, inclusive, offset);
    if (lane >= offset) {
      inclusive += n;
    }
  }

  if (lane == WARP_SIZE - 1) {
    shared_warp_sums[warp] = inclusive;
  }
  __syncthreads();

  if (warp == 0) {
    int warp_scan = (lane < NUM_WARPS) ? shared_warp_sums[lane] : 0;
#pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
      int n = __shfl_up_sync(0xFFFFFFFF, warp_scan, offset);
      if (lane >= offset) {
        warp_scan += n;
      }
    }

    if (lane < NUM_WARPS) {
      shared_warp_sums[lane] = warp_scan - shared_warp_sums[lane];
    }
    if (block_total != nullptr && lane == NUM_WARPS - 1) {
      *block_total = warp_scan;
    }
  }
  __syncthreads();

  return shared_warp_sums[warp] + inclusive - val;
}

///////////////////////////////////////////////////////////////////////////////
// Single-pass Radix Select for small arrays (fits in shared memory)
///////////////////////////////////////////////////////////////////////////////

// Helper to calculate required shared memory size for small kernel
template <typename UnsignedT, int TILE_SIZE>
constexpr size_t radix_select_small_shared_mem_size() {
  return TILE_SIZE * sizeof(UnsignedT) + // shared_keys
      TILE_SIZE * sizeof(uint32_t) + // shared_idxs
      RADIX_SIZE * sizeof(int) + // shared_hist
      2 * sizeof(int); // shared_count
}

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

  // Dynamic shared memory layout
  extern __shared__ char shared_mem[];

  // Calculate offsets for different arrays in shared memory
  UnsignedT* shared_keys = reinterpret_cast<UnsignedT*>(shared_mem);
  uint32_t* shared_idxs = reinterpret_cast<uint32_t*>(shared_keys + TILE_SIZE);
  int* shared_hist = reinterpret_cast<int*>(shared_idxs + TILE_SIZE);
  int* shared_count = shared_hist + RADIX_SIZE;

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

  // Count per-thread bucket sizes once, then scatter in a single pass with
  // deterministic per-thread offsets.
  int local_less = 0;
  int local_equal = 0;
  for (int i = threadIdx.x; i < tile_n; i += BLOCK_THREADS) {
    UnsignedT key = shared_keys[i];
    if (key < target_prefix) {
      local_less++;
    } else if (key == target_prefix) {
      local_equal++;
    }
  }

  int less_thread_offset = block_exclusive_scan<BLOCK_THREADS>(
      local_less, shared_hist, &shared_count[0]);
  int equal_thread_offset = block_exclusive_scan<BLOCK_THREADS>(
      local_equal, shared_hist, &shared_count[1]);

  int q = tile_n / BLOCK_THREADS;
  int r = tile_n - q * BLOCK_THREADS;
  int prefix_total = int(threadIdx.x) * q + min(int(threadIdx.x), r);
  int greater_thread_offset =
      prefix_total - less_thread_offset - equal_thread_offset;

  int less_count = shared_count[0];
  int equal_count = shared_count[1];

  for (int i = threadIdx.x; i < tile_n; i += BLOCK_THREADS) {
    UnsignedT key = shared_keys[i];
    int pos;
    if (key < target_prefix) {
      pos = less_thread_offset++;
    } else if (key == target_prefix) {
      pos = less_count + equal_thread_offset++;
    } else {
      pos = less_count + equal_count + greater_thread_offset++;
    }

    if (ARG_PARTITION) {
      row_output[pos * out_stride] = shared_idxs[i];
    } else {
      row_output[pos * out_stride] = row_input[shared_idxs[i] * in_stride];
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
  }

  // Count partition sizes.
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

  local_less = warp_reduce_sum(local_less);
  local_equal = warp_reduce_sum(local_equal);

  if (threadIdx.x == 0) {
    shared_counts[0] = 0;
    shared_counts[1] = 0;
  }
  __syncthreads();

  if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
    atomicAdd(&shared_counts[0], local_less);
    atomicAdd(&shared_counts[1], local_equal);
  }
  __syncthreads();

  int less_count = shared_counts[0];
  int equal_count = shared_counts[1];

  // Deterministic scatter in iteration order (0..n): this keeps output stable
  // without thread-contention atomics in the hot scatter path.
  constexpr int NUM_WARPS = BLOCK_THREADS / WARP_SIZE;
  int lane = threadIdx.x & (WARP_SIZE - 1);
  int warp = threadIdx.x / WARP_SIZE;

  int* warp_less = shared_hist;
  int* warp_equal = shared_hist + NUM_WARPS;
  int* warp_greater = shared_hist + 2 * NUM_WARPS;
  int* iter_counts = shared_hist + 3 * NUM_WARPS;
  int* running_bases = iter_counts + 3;

  if (threadIdx.x == 0) {
    running_bases[0] = 0;
    running_bases[1] = less_count;
    running_bases[2] = less_count + equal_count;
  }
  __syncthreads();

  for (int base_i = 0; base_i < n; base_i += BLOCK_THREADS) {
    int i = base_i + threadIdx.x;
    bool active = i < n;

    ValT val{};
    UnsignedT key = 0;
    if (active) {
      val = is_contiguous ? row_input[i] : row_input[i * in_stride];
      key = Traits::to_radix(val);
      if (is_nan_value(val)) {
        key = ~UnsignedT(0);
      }
    }

    bool is_less = active && (key < target_prefix);
    bool is_equal = active && (key == target_prefix);
    bool is_greater = active && !is_less && !is_equal;

    unsigned less_mask = __ballot_sync(0xFFFFFFFF, is_less);
    unsigned equal_mask = __ballot_sync(0xFFFFFFFF, is_equal);
    unsigned greater_mask = __ballot_sync(0xFFFFFFFF, is_greater);

    unsigned lane_mask = (1u << lane) - 1u;
    int less_rank = __popc(less_mask & lane_mask);
    int equal_rank = __popc(equal_mask & lane_mask);
    int greater_rank = __popc(greater_mask & lane_mask);

    if (lane == 0) {
      warp_less[warp] = __popc(less_mask);
      warp_equal[warp] = __popc(equal_mask);
      warp_greater[warp] = __popc(greater_mask);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      int run = 0;
      for (int w = 0; w < NUM_WARPS; ++w) {
        int c = warp_less[w];
        warp_less[w] = run;
        run += c;
      }
      iter_counts[0] = run;

      run = 0;
      for (int w = 0; w < NUM_WARPS; ++w) {
        int c = warp_equal[w];
        warp_equal[w] = run;
        run += c;
      }
      iter_counts[1] = run;

      run = 0;
      for (int w = 0; w < NUM_WARPS; ++w) {
        int c = warp_greater[w];
        warp_greater[w] = run;
        run += c;
      }
      iter_counts[2] = run;
    }
    __syncthreads();

    if (active) {
      int pos;
      if (is_less) {
        pos = running_bases[0] + warp_less[warp] + less_rank;
      } else if (is_equal) {
        pos = running_bases[1] + warp_equal[warp] + equal_rank;
      } else {
        pos = running_bases[2] + warp_greater[warp] + greater_rank;
      }

      if (ARG_PARTITION) {
        if (out_stride == 1) {
          row_output[pos] = i;
        } else {
          row_output[pos * out_stride] = i;
        }
      } else {
        if (out_stride == 1) {
          row_output[pos] = val;
        } else {
          row_output[pos * out_stride] = val;
        }
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      running_bases[0] += iter_counts[0];
      running_bases[1] += iter_counts[1];
      running_bases[2] += iter_counts[2];
    }
    __syncthreads();
  }
}

///////////////////////////////////////////////////////////////////////////////
// Tiled large-array kernels
//
// These kernels run with a 2D launch:
// - x-dimension tiles one row across multiple blocks (multi-block-per-row)
// - y-dimension packs multiple rows into one block group (multi-row-per-block)
///////////////////////////////////////////////////////////////////////////////

template <typename UnsignedT>
__global__ void radix_select_tiled_init_state_kernel(
    UnsignedT* target_prefix,
    UnsignedT* prefix_mask,
    int* k_values,
    int* row_hist,
    int kth,
    int n_rows,
    int rows_per_block) {
  int row_start = blockIdx.y * rows_per_block;
  int row_end = min(n_rows, row_start + rows_per_block);
  for (int row = row_start; row < row_end; ++row) {
    if (threadIdx.x == 0) {
      target_prefix[row] = UnsignedT(0);
      prefix_mask[row] = UnsignedT(0);
      k_values[row] = kth + 1;
    }
    int* hist = row_hist + row * RADIX_SIZE;
    for (int i = threadIdx.x; i < RADIX_SIZE; i += blockDim.x) {
      hist[i] = 0;
    }
  }
}

template <typename ValT, int BLOCK_THREADS>
__global__ void radix_select_tiled_histogram_kernel(
    const ValT* input,
    int n,
    int64_t in_stride,
    int64_t in_segment_stride,
    const typename RadixTraits<ValT>::UnsignedT* target_prefix,
    const typename RadixTraits<ValT>::UnsignedT* prefix_mask,
    int start_bit,
    int blocks_per_row,
    int n_rows,
    int rows_per_block,
    int* row_hist) {
  using Traits = RadixTraits<ValT>;
  using UnsignedT = typename Traits::UnsignedT;

  int block_in_row = blockIdx.x;
  int row_start = blockIdx.y * rows_per_block;
  int row_end = min(n_rows, row_start + rows_per_block);

  int chunk = (n + blocks_per_row - 1) / blocks_per_row;
  int start = block_in_row * chunk;
  int end = min(n, start + chunk);
  if (start >= n || row_start >= row_end) {
    return;
  }

  __shared__ int shared_hist[RADIX_SIZE];
  for (int row = row_start; row < row_end; ++row) {
    const ValT* row_input = input + row * in_segment_stride;
    UnsignedT row_prefix = target_prefix[row];
    UnsignedT row_mask = prefix_mask[row];

    for (int i = threadIdx.x; i < RADIX_SIZE; i += BLOCK_THREADS) {
      shared_hist[i] = 0;
    }
    __syncthreads();

    for (int i = start + threadIdx.x; i < end; i += BLOCK_THREADS) {
      ValT val = row_input[i * in_stride];
      UnsignedT key = radix_key_with_nan_last(val);
      if ((key & row_mask) == row_prefix) {
        int digit = extract_digit(key, start_bit, RADIX_BITS);
        atomicAdd(&shared_hist[digit], 1);
      }
    }
    __syncthreads();

    int* hist = row_hist + row * RADIX_SIZE;
    for (int i = threadIdx.x; i < RADIX_SIZE; i += BLOCK_THREADS) {
      atomicAdd(&hist[i], shared_hist[i]);
    }
    __syncthreads();
  }
}

template <typename UnsignedT>
__global__ void radix_select_tiled_select_bin_kernel(
    int* row_hist,
    UnsignedT* target_prefix,
    UnsignedT* prefix_mask,
    int* k_values,
    int clear_hist_for_next_pass,
    int start_bit,
    int n_rows,
    int rows_per_block) {
  int row_start = blockIdx.y * rows_per_block;
  int row_end = min(n_rows, row_start + rows_per_block);
  for (int row = row_start; row < row_end; ++row) {
    int* hist = row_hist + row * RADIX_SIZE;

    if (threadIdx.x == 0) {
      int k = k_values[row];
      int cumsum = 0;
      int target_bin = 0;
      for (int bin = 0; bin < RADIX_SIZE; bin++) {
        int count = hist[bin];
        if (cumsum + count >= k) {
          target_bin = bin;
          k -= cumsum;
          break;
        }
        cumsum += count;
      }
      k_values[row] = k;

      UnsignedT digit_mask =
          (UnsignedT((UnsignedT(1) << RADIX_BITS) - UnsignedT(1)) << start_bit);
      target_prefix[row] |= UnsignedT(target_bin) << start_bit;
      prefix_mask[row] |= digit_mask;
    }
    __syncthreads();

    if (clear_hist_for_next_pass) {
      for (int i = threadIdx.x; i < RADIX_SIZE; i += blockDim.x) {
        hist[i] = 0;
      }
    }
    __syncthreads();
  }
}

template <typename ValT, int BLOCK_THREADS>
__global__ void radix_select_tiled_count_kernel(
    const ValT* input,
    int n,
    int64_t in_stride,
    int64_t in_segment_stride,
    const typename RadixTraits<ValT>::UnsignedT* target_prefix,
    int blocks_per_row,
    int n_rows,
    int rows_per_block,
    int* block_less,
    int* block_equal) {
  using Traits = RadixTraits<ValT>;
  using UnsignedT = typename Traits::UnsignedT;

  int block_in_row = blockIdx.x;
  int row_start = blockIdx.y * rows_per_block;
  int row_end = min(n_rows, row_start + rows_per_block);

  int chunk = (n + blocks_per_row - 1) / blocks_per_row;
  int start = block_in_row * chunk;
  int end = min(n, start + chunk);
  if (row_start >= row_end) {
    return;
  }

  __shared__ int shared_counts[2];
  for (int row = row_start; row < row_end; ++row) {
    int block_idx = row * blocks_per_row + block_in_row;
    const ValT* row_input = input + row * in_segment_stride;
    UnsignedT row_prefix = target_prefix[row];

    int local_less = 0;
    int local_equal = 0;
    for (int i = start + threadIdx.x; i < end; i += BLOCK_THREADS) {
      ValT val = row_input[i * in_stride];
      UnsignedT key = radix_key_with_nan_last(val);
      if (key < row_prefix) {
        local_less++;
      } else if (key == row_prefix) {
        local_equal++;
      }
    }

    local_less = warp_reduce_sum(local_less);
    local_equal = warp_reduce_sum(local_equal);

    if (threadIdx.x == 0) {
      shared_counts[0] = 0;
      shared_counts[1] = 0;
    }
    __syncthreads();

    if ((threadIdx.x % WARP_SIZE) == 0) {
      atomicAdd(&shared_counts[0], local_less);
      atomicAdd(&shared_counts[1], local_equal);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      block_less[block_idx] = shared_counts[0];
      block_equal[block_idx] = shared_counts[1];
    }
    __syncthreads();
  }
}

__global__ void radix_select_tiled_prefix_kernel(
    int n,
    int blocks_per_row,
    int n_rows,
    int rows_per_block,
    const int* block_less,
    const int* block_equal,
    int* less_base,
    int* equal_base,
    int* greater_base) {
  if (threadIdx.x != 0) {
    return;
  }

  int row_start = blockIdx.y * rows_per_block;
  int row_end = min(n_rows, row_start + rows_per_block);
  int chunk = (n + blocks_per_row - 1) / blocks_per_row;

  for (int row = row_start; row < row_end; ++row) {
    int row_off = row * blocks_per_row;
    int total_less = 0;
    int total_equal = 0;
    for (int b = 0; b < blocks_per_row; b++) {
      int idx = row_off + b;
      total_less += block_less[idx];
      total_equal += block_equal[idx];
    }

    int run_less = 0;
    int run_equal = 0;
    int run_greater = 0;
    for (int b = 0; b < blocks_per_row; b++) {
      int idx = row_off + b;
      less_base[idx] = run_less;
      equal_base[idx] = total_less + run_equal;
      greater_base[idx] = total_less + total_equal + run_greater;

      int start = b * chunk;
      int end = min(n, start + chunk);
      int chunk_size = max(0, end - start);
      int greater_count = chunk_size - block_less[idx] - block_equal[idx];

      run_less += block_less[idx];
      run_equal += block_equal[idx];
      run_greater += greater_count;
    }
  }
}

template <typename ValT, typename OutT, bool ARG_PARTITION, int BLOCK_THREADS>
__global__ void radix_select_tiled_scatter_kernel(
    const ValT* input,
    OutT* output,
    int n,
    int64_t in_stride,
    int64_t out_stride,
    int64_t in_segment_stride,
    int64_t out_segment_stride,
    const typename RadixTraits<ValT>::UnsignedT* target_prefix,
    int blocks_per_row,
    int n_rows,
    int rows_per_block,
    const int* less_base,
    const int* equal_base,
    const int* greater_base) {
  using Traits = RadixTraits<ValT>;
  using UnsignedT = typename Traits::UnsignedT;

  int block_in_row = blockIdx.x;
  int row_start = blockIdx.y * rows_per_block;
  int row_end = min(n_rows, row_start + rows_per_block);

  int chunk = (n + blocks_per_row - 1) / blocks_per_row;
  int start = block_in_row * chunk;
  int end = min(n, start + chunk);
  if (start >= n || row_start >= row_end) {
    return;
  }

  constexpr int NUM_WARPS = BLOCK_THREADS / WARP_SIZE;
  __shared__ int shared_warp_offsets[3 * NUM_WARPS];
  __shared__ int shared_iter_counts[3];
  __shared__ int shared_running_bases[3];

  int lane = threadIdx.x & (WARP_SIZE - 1);
  int warp = threadIdx.x / WARP_SIZE;

  int* warp_less = shared_warp_offsets;
  int* warp_equal = shared_warp_offsets + NUM_WARPS;
  int* warp_greater = shared_warp_offsets + 2 * NUM_WARPS;

  for (int row = row_start; row < row_end; ++row) {
    int block_idx = row * blocks_per_row + block_in_row;
    const ValT* row_input = input + row * in_segment_stride;
    OutT* row_output = output + row * out_segment_stride;
    UnsignedT row_prefix = target_prefix[row];

    if (threadIdx.x == 0) {
      shared_running_bases[0] = less_base[block_idx];
      shared_running_bases[1] = equal_base[block_idx];
      shared_running_bases[2] = greater_base[block_idx];
    }
    __syncthreads();

    for (int base_i = start; base_i < end; base_i += BLOCK_THREADS) {
      int i = base_i + threadIdx.x;
      bool active = i < end;

      ValT val{};
      UnsignedT key = 0;
      if (active) {
        val = row_input[i * in_stride];
        key = radix_key_with_nan_last(val);
      }

      bool is_less = active && (key < row_prefix);
      bool is_equal = active && (key == row_prefix);
      bool is_greater = active && !is_less && !is_equal;

      unsigned less_mask = __ballot_sync(0xFFFFFFFF, is_less);
      unsigned equal_mask = __ballot_sync(0xFFFFFFFF, is_equal);
      unsigned greater_mask = __ballot_sync(0xFFFFFFFF, is_greater);

      unsigned lane_mask = (1u << lane) - 1u;
      int less_rank = __popc(less_mask & lane_mask);
      int equal_rank = __popc(equal_mask & lane_mask);
      int greater_rank = __popc(greater_mask & lane_mask);

      if (lane == 0) {
        warp_less[warp] = __popc(less_mask);
        warp_equal[warp] = __popc(equal_mask);
        warp_greater[warp] = __popc(greater_mask);
      }
      __syncthreads();

      if (threadIdx.x == 0) {
        int run = 0;
        for (int w = 0; w < NUM_WARPS; ++w) {
          int c = warp_less[w];
          warp_less[w] = run;
          run += c;
        }
        shared_iter_counts[0] = run;

        run = 0;
        for (int w = 0; w < NUM_WARPS; ++w) {
          int c = warp_equal[w];
          warp_equal[w] = run;
          run += c;
        }
        shared_iter_counts[1] = run;

        run = 0;
        for (int w = 0; w < NUM_WARPS; ++w) {
          int c = warp_greater[w];
          warp_greater[w] = run;
          run += c;
        }
        shared_iter_counts[2] = run;
      }
      __syncthreads();

      if (active) {
        int pos;
        if (is_less) {
          pos = shared_running_bases[0] + warp_less[warp] + less_rank;
        } else if (is_equal) {
          pos = shared_running_bases[1] + warp_equal[warp] + equal_rank;
        } else {
          pos = shared_running_bases[2] + warp_greater[warp] + greater_rank;
        }
        if (ARG_PARTITION) {
          row_output[pos * out_stride] = static_cast<OutT>(i);
        } else {
          row_output[pos * out_stride] = static_cast<OutT>(val);
        }
      }
      __syncthreads();

      if (threadIdx.x == 0) {
        shared_running_bases[0] += shared_iter_counts[0];
        shared_running_bases[1] += shared_iter_counts[1];
        shared_running_bases[2] += shared_iter_counts[2];
      }
      __syncthreads();
    }
    __syncthreads();
  }
}

} // namespace mlx::core::cu
