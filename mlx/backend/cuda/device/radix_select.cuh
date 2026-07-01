// Copyright © 2025 Apple Inc.

#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
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
    if (cuda::std::isnan(val)) {
      return ~UnsignedT(0);
    }
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
    if (cuda::std::isnan(val)) {
      return ~UnsignedT(0);
    }
    uint64_t bits = __double_as_longlong(val);
    if ((bits << 1) == 0) {
      bits = 0; // Canonicalize +/-0.0 to +0.0 for stable equal-value ties.
    }
    uint64_t mask = -int64_t(bits >> 63) | 0x8000000000000000ull;
    return bits ^ mask;
  }
};

template <>
struct RadixTraits<complex64_t> {
  using UnsignedT = uint64_t;
  static constexpr int BITS = 64;

  __device__ __forceinline__ static UnsignedT to_radix(complex64_t val) {
    float real = val.real();
    float imag = val.imag();
    if (cuda::std::isnan(real) || cuda::std::isnan(imag)) {
      return ~UnsignedT(0);
    }

    auto real_key = RadixTraits<float>::to_radix(real);
    auto imag_key = RadixTraits<float>::to_radix(imag);
    return (static_cast<UnsignedT>(real_key) << 32) |
        static_cast<UnsignedT>(imag_key);
  }
};

template <>
struct RadixTraits<__half> {
  using UnsignedT = uint16_t;
  static constexpr int BITS = 16;

  __device__ __forceinline__ static UnsignedT to_radix(__half val) {
    if (cuda::std::isnan(val)) {
      return ~UnsignedT(0);
    }
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
    if (cuda::std::isnan(val)) {
      return ~UnsignedT(0);
    }
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
// Block-level utilities
///////////////////////////////////////////////////////////////////////////////

namespace cg = cooperative_groups;

template <int BLOCK_THREADS>
__device__ __forceinline__ int block_exclusive_scan(
    cg::thread_block& block,
    int val,
    int* shared_warp_sums,
    int* block_total = nullptr) {
  static_assert(BLOCK_THREADS % WARP_SIZE == 0);
  constexpr int NUM_WARPS = BLOCK_THREADS / WARP_SIZE;

  auto warp = cg::tiled_partition<WARP_SIZE>(block);
  int inclusive = cg::inclusive_scan(warp, val, cg::plus<int>());

  if (warp.thread_rank() == WARP_SIZE - 1) {
    shared_warp_sums[warp.meta_group_rank()] = inclusive;
  }
  block.sync();

  if (warp.meta_group_rank() == 0) {
    int warp_val = warp.thread_rank() < NUM_WARPS
        ? shared_warp_sums[warp.thread_rank()]
        : 0;
    int warp_scan = cg::inclusive_scan(warp, warp_val, cg::plus<int>());

    if (warp.thread_rank() < NUM_WARPS) {
      shared_warp_sums[warp.thread_rank()] =
          warp_scan - shared_warp_sums[warp.thread_rank()];
    }
    if (block_total != nullptr && warp.thread_rank() == NUM_WARPS - 1) {
      *block_total = warp_scan;
    }
  }
  block.sync();

  return shared_warp_sums[warp.meta_group_rank()] + inclusive - val;
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
  constexpr int NUM_WARPS = BLOCK_THREADS / WARP_SIZE;

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  __shared__ UnsignedT shared_keys[TILE_SIZE];
  __shared__ int shared_hist[RADIX_SIZE];
  __shared__ int shared_count[2];
  __shared__ int warp_less[NUM_WARPS];
  __shared__ int warp_equal[NUM_WARPS];
  __shared__ int warp_greater[NUM_WARPS];
  __shared__ int iter_counts[3];
  __shared__ int running_bases[3];

  int row = blockIdx.y;

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

  for (int i = block.thread_rank(); i < TILE_SIZE; i += BLOCK_THREADS) {
    shared_keys[i] = (i < tile_n) ? Traits::to_radix(row_input[i * in_stride])
                                  : ~UnsignedT(0);
  }
  block.sync();

  int k = kth + 1;
  UnsignedT target_prefix = 0;
  UnsignedT prefix_mask = 0;

  for (int pass = NUM_PASSES - 1; pass >= 0; pass--) {
    int start_bit = pass * RADIX_BITS;

    for (int i = block.thread_rank(); i < RADIX_SIZE; i += BLOCK_THREADS) {
      shared_hist[i] = 0;
    }
    block.sync();

    for (int i = block.thread_rank(); i < tile_n; i += BLOCK_THREADS) {
      UnsignedT key = shared_keys[i];
      if ((key & prefix_mask) == target_prefix) {
        int digit = (key >> start_bit) & ((1 << RADIX_BITS) - 1);
        atomicAdd(&shared_hist[digit], 1);
      }
    }
    block.sync();

    // Find target bin (single thread)
    if (block.thread_rank() == 0) {
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
    block.sync();

    int target_bin = shared_count[0];
    k = shared_count[1];

    UnsignedT digit_mask = UnsignedT((1 << RADIX_BITS) - 1) << start_bit;
    target_prefix |= UnsignedT(target_bin) << start_bit;
    prefix_mask |= digit_mask;

    block.sync();
  }

  // Count per-thread bucket sizes once, then scatter in a single pass with
  // deterministic per-thread offsets.
  int local_less = 0;
  int local_equal = 0;
  for (int i = block.thread_rank(); i < tile_n; i += BLOCK_THREADS) {
    UnsignedT key = shared_keys[i];
    if (key < target_prefix) {
      local_less++;
    } else if (key == target_prefix) {
      local_equal++;
    }
  }

  (void)block_exclusive_scan<BLOCK_THREADS>(
      block, local_less, shared_hist, &shared_count[0]);
  (void)block_exclusive_scan<BLOCK_THREADS>(
      block, local_equal, shared_hist, &shared_count[1]);

  int less_count = shared_count[0];
  int equal_count = shared_count[1];

  // Scatter in increasing i order to keep tie behavior aligned with merge sort.
  if (block.thread_rank() == 0) {
    running_bases[0] = 0;
    running_bases[1] = less_count;
    running_bases[2] = less_count + equal_count;
  }
  block.sync();

  for (int base_i = 0; base_i < tile_n; base_i += BLOCK_THREADS) {
    int i = base_i + block.thread_rank();
    bool active = i < tile_n;

    UnsignedT key = 0;
    if (active) {
      key = shared_keys[i];
    }

    bool is_less = active && (key < target_prefix);
    bool is_equal = active && (key == target_prefix);
    bool is_greater = active && !is_less && !is_equal;

    unsigned less_ballot = warp.ballot(is_less);
    unsigned equal_ballot = warp.ballot(is_equal);
    unsigned greater_ballot = warp.ballot(is_greater);

    unsigned lane_mask = (1u << warp.thread_rank()) - 1u;
    int less_rank = __popc(less_ballot & lane_mask);
    int equal_rank = __popc(equal_ballot & lane_mask);
    int greater_rank = __popc(greater_ballot & lane_mask);

    if (warp.thread_rank() == 0) {
      warp_less[warp.meta_group_rank()] = __popc(less_ballot);
      warp_equal[warp.meta_group_rank()] = __popc(equal_ballot);
      warp_greater[warp.meta_group_rank()] = __popc(greater_ballot);
    }
    block.sync();

    if (block.thread_rank() == 0) {
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
    block.sync();

    if (active) {
      int pos;
      if (is_less) {
        pos = running_bases[0] + warp_less[warp.meta_group_rank()] + less_rank;
      } else if (is_equal) {
        pos =
            running_bases[1] + warp_equal[warp.meta_group_rank()] + equal_rank;
      } else {
        pos = running_bases[2] + warp_greater[warp.meta_group_rank()] +
            greater_rank;
      }

      if constexpr (ARG_PARTITION) {
        row_output[pos * out_stride] = i;
      } else {
        row_output[pos * out_stride] = row_input[i * in_stride];
      }
    }
    block.sync();

    if (block.thread_rank() == 0) {
      running_bases[0] += iter_counts[0];
      running_bases[1] += iter_counts[1];
      running_bases[2] += iter_counts[2];
    }
    block.sync();
  }
}

} // namespace mlx::core::cu
