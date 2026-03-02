// Copyright © 2025 Apple Inc.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda/std/type_traits>
#include "mlx/backend/cuda/device/utils.cuh"

namespace mlx::core::cu {

///////////////////////////////////////////////////////////////////////////////
// Radix Select Implementation for CUDA
//
// Multi-pass radix-based selection algorithm for partition operations.
// Uses IEEE 754 bit manipulation for correct floating-point ordering.
///////////////////////////////////////////////////////////////////////////////

// Radix configuration used by the small shared-memory kernel.
constexpr int RADIX_BITS = 5;
constexpr int RADIX_SIZE = 1 << RADIX_BITS; // 32 bins

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
    if ((bits << 1) == 0) {
      bits = 0; // Canonicalize +/-0.0 to +0.0 for stable equal-value ties.
    }
    uint32_t mask = -int32_t(bits >> 31) | 0x80000000u;
    return bits ^ mask;
  }
};

template <>
struct RadixTraits<double> {
  using UnsignedT = uint64_t;
  static constexpr int BITS = 64;

  __device__ __forceinline__ static UnsignedT to_radix(double val) {
    uint64_t bits = __double_as_longlong(val);
    if ((bits << 1) == 0) {
      bits = 0; // Canonicalize +/-0.0 to +0.0 for stable equal-value ties.
    }
    uint64_t mask = -int64_t(bits >> 63) | 0x8000000000000000ull;
    return bits ^ mask;
  }
};

template <>
struct RadixTraits<__half> {
  using UnsignedT = uint16_t;
  static constexpr int BITS = 16;

  __device__ __forceinline__ static UnsignedT to_radix(__half val) {
    uint16_t bits = __half_as_ushort(val);
    if ((bits & 0x7FFFu) == 0) {
      bits = 0; // Canonicalize +/-0 to +0 for stable equal-value ties.
    }
    uint16_t mask = -int16_t(bits >> 15) | 0x8000u;
    return bits ^ mask;
  }
};

template <>
struct RadixTraits<__nv_bfloat16> {
  using UnsignedT = uint16_t;
  static constexpr int BITS = 16;

  __device__ __forceinline__ static UnsignedT to_radix(__nv_bfloat16 val) {
    uint16_t bits = __bfloat16_as_ushort(val);
    if ((bits & 0x7FFFu) == 0) {
      bits = 0; // Canonicalize +/-0 to +0 for stable equal-value ties.
    }
    uint16_t mask = -int16_t(bits >> 15) | 0x8000u;
    return bits ^ mask;
  }
};

template <>
struct RadixTraits<int8_t> {
  using UnsignedT = uint8_t;
  static constexpr int BITS = 8;

  __device__ __forceinline__ static UnsignedT to_radix(int8_t val) {
    return static_cast<UnsignedT>(val) ^ 0x80u;
  }
};

template <>
struct RadixTraits<int16_t> {
  using UnsignedT = uint16_t;
  static constexpr int BITS = 16;

  __device__ __forceinline__ static UnsignedT to_radix(int16_t val) {
    return static_cast<UnsignedT>(val) ^ 0x8000u;
  }
};

template <>
struct RadixTraits<int32_t> {
  using UnsignedT = uint32_t;
  static constexpr int BITS = 32;

  __device__ __forceinline__ static UnsignedT to_radix(int32_t val) {
    return static_cast<UnsignedT>(val) ^ 0x80000000u;
  }
};

template <>
struct RadixTraits<int64_t> {
  using UnsignedT = uint64_t;
  static constexpr int BITS = 64;

  __device__ __forceinline__ static UnsignedT to_radix(int64_t val) {
    return static_cast<UnsignedT>(val) ^ 0x8000000000000000ull;
  }
};

template <>
struct RadixTraits<bool> {
  using UnsignedT = uint8_t;
  static constexpr int BITS = 8;

  __device__ __forceinline__ static UnsignedT to_radix(bool val) {
    return static_cast<uint8_t>(val);
  }
};

template <>
struct RadixTraits<uint8_t> {
  using UnsignedT = uint8_t;
  static constexpr int BITS = 8;

  __device__ __forceinline__ static UnsignedT to_radix(uint8_t val) {
    return val;
  }
};

template <>
struct RadixTraits<uint16_t> {
  using UnsignedT = uint16_t;
  static constexpr int BITS = 16;

  __device__ __forceinline__ static UnsignedT to_radix(uint16_t val) {
    return val;
  }
};

template <>
struct RadixTraits<uint32_t> {
  using UnsignedT = uint32_t;
  static constexpr int BITS = 32;

  __device__ __forceinline__ static UnsignedT to_radix(uint32_t val) {
    return val;
  }
};

template <>
struct RadixTraits<uint64_t> {
  using UnsignedT = uint64_t;
  static constexpr int BITS = 64;

  __device__ __forceinline__ static UnsignedT to_radix(uint64_t val) {
    return val;
  }
};

///////////////////////////////////////////////////////////////////////////////
// Warp-level utilities
///////////////////////////////////////////////////////////////////////////////

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
// Radix Select for small arrays (fits in shared memory)
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
    const __grid_constant__ Shape nc_shape,
    const __grid_constant__ Strides in_nc_strides,
    const __grid_constant__ Strides out_nc_strides,
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
  constexpr int NUM_WARPS = BLOCK_THREADS / WARP_SIZE;
  int* scatter_scratch = shared_count + 2;
  int* warp_less = scatter_scratch;
  int* warp_equal = warp_less + NUM_WARPS;
  int* warp_greater = warp_equal + NUM_WARPS;
  int* iter_counts = warp_greater + NUM_WARPS;
  int* running_bases = iter_counts + 3;

  int row = blockIdx.y;

  // Compute row pointers based on addressing mode
  const ValT* row_input;
  OutT* row_output;
  if constexpr (USE_SIMPLE_STRIDE) {
    row_input = input + row * in_segment_stride;
    row_output = output + row * out_segment_stride;
  } else {
    int64_t in_block_idx = elem_to_loc(
        int64_t(row), nc_shape.data(), in_nc_strides.data(), nc_dim);
    int64_t out_block_idx = elem_to_loc(
        int64_t(row), nc_shape.data(), out_nc_strides.data(), nc_dim);
    row_input = input + in_block_idx;
    row_output = output + out_block_idx;
  }

  int tile_n = min(n, TILE_SIZE);

  // Load data into shared memory
  for (int i = threadIdx.x; i < TILE_SIZE; i += BLOCK_THREADS) {
    if (i < tile_n) {
      ValT val = row_input[i * in_stride];
      UnsignedT key = Traits::to_radix(val);
      if constexpr (cuda::std::is_floating_point_v<ValT>) {
        if (cuda::std::isnan(val)) {
          key = ~UnsignedT(0);
        }
      } else if constexpr (cuda::std::is_same_v<ValT, __half>) {
        if (__hisnan(val)) {
          key = ~UnsignedT(0);
        }
      } else if constexpr (cuda::std::is_same_v<ValT, __nv_bfloat16>) {
        if (__hisnan(val)) {
          key = ~UnsignedT(0);
        }
      } else {
        // Non-floating types cannot produce NaN keys.
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
        int digit = (key >> start_bit) & ((1 << RADIX_BITS) - 1);
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

  (void)block_exclusive_scan<BLOCK_THREADS>(
      local_less, shared_hist, &shared_count[0]);
  (void)block_exclusive_scan<BLOCK_THREADS>(
      local_equal, shared_hist, &shared_count[1]);

  int less_count = shared_count[0];
  int equal_count = shared_count[1];

  // Scatter in increasing i order to keep tie behavior aligned with merge sort.
  int lane = threadIdx.x & (WARP_SIZE - 1);
  int warp = threadIdx.x / WARP_SIZE;

  if (threadIdx.x == 0) {
    running_bases[0] = 0;
    running_bases[1] = less_count;
    running_bases[2] = less_count + equal_count;
  }
  __syncthreads();

  for (int base_i = 0; base_i < tile_n; base_i += BLOCK_THREADS) {
    int i = base_i + threadIdx.x;
    bool active = i < tile_n;

    UnsignedT key = 0;
    if (active) {
      key = shared_keys[i];
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
        row_output[pos * out_stride] = shared_idxs[i];
      } else {
        row_output[pos * out_stride] = row_input[shared_idxs[i] * in_stride];
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

} // namespace mlx::core::cu
