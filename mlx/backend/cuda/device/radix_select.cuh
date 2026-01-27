// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device/config.h"
#include "mlx/backend/cuda/device/utils.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

namespace mlx::core::cu {

///////////////////////////////////////////////////////////////////////////////
// Radix Select Utilities
//
// This implements an optimized radix-based top-k selection algorithm based on
// the RadiK paper (Li et al., ICS'24). Key optimizations include:
// - Hierarchical atomics (warp -> block -> global)
// - Flush-efficient write buffers
// - IEEE 754 bit manipulation for correct floating-point ordering
///////////////////////////////////////////////////////////////////////////////

// Radix configuration
constexpr int RADIX_BITS = 8;
constexpr int RADIX_SIZE = 1 << RADIX_BITS; // 256 bins

///////////////////////////////////////////////////////////////////////////////
// Bit manipulation for radix sorting
//
// For floating-point types, we need to convert to unsigned integers that
// preserve the sorting order. IEEE 754 floats have the property that positive
// floats sort correctly when interpreted as unsigned integers. For negative
// floats, we need to flip all bits.
///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct RadixTraits;

// Float32: 32-bit unsigned representation
template <>
struct RadixTraits<float> {
  using UnsignedT = uint32_t;
  static constexpr int BITS = 32;

  __device__ __forceinline__ static UnsignedT to_radix(float val) {
    UnsignedT bits = __float_as_uint(val);
    // If sign bit is set (negative), flip all bits
    // Otherwise, flip only the sign bit
    UnsignedT mask = -int32_t(bits >> 31) | 0x80000000u;
    return bits ^ mask;
  }

  __device__ __forceinline__ static float from_radix(UnsignedT bits) {
    // Reverse the transformation
    UnsignedT mask = ((bits >> 31) - 1) | 0x80000000u;
    return __uint_as_float(bits ^ mask);
  }
};

// Float64: 64-bit unsigned representation
template <>
struct RadixTraits<double> {
  using UnsignedT = uint64_t;
  static constexpr int BITS = 64;

  __device__ __forceinline__ static UnsignedT to_radix(double val) {
    UnsignedT bits = __double_as_longlong(val);
    UnsignedT mask = -int64_t(bits >> 63) | 0x8000000000000000ull;
    return bits ^ mask;
  }

  __device__ __forceinline__ static double from_radix(UnsignedT bits) {
    UnsignedT mask = ((bits >> 63) - 1) | 0x8000000000000000ull;
    return __longlong_as_double(bits ^ mask);
  }
};

// Float16: 16-bit unsigned representation
template <>
struct RadixTraits<__half> {
  using UnsignedT = uint16_t;
  static constexpr int BITS = 16;

  __device__ __forceinline__ static UnsignedT to_radix(__half val) {
    UnsignedT bits = __half_as_ushort(val);
    UnsignedT mask = -int16_t(bits >> 15) | 0x8000u;
    return bits ^ mask;
  }

  __device__ __forceinline__ static __half from_radix(UnsignedT bits) {
    UnsignedT mask = ((bits >> 15) - 1) | 0x8000u;
    return __ushort_as_half(bits ^ mask);
  }
};

// BFloat16: 16-bit unsigned representation
template <>
struct RadixTraits<__nv_bfloat16> {
  using UnsignedT = uint16_t;
  static constexpr int BITS = 16;

  __device__ __forceinline__ static UnsignedT to_radix(__nv_bfloat16 val) {
    UnsignedT bits = __bfloat16_as_ushort(val);
    UnsignedT mask = -int16_t(bits >> 15) | 0x8000u;
    return bits ^ mask;
  }

  __device__ __forceinline__ static __nv_bfloat16 from_radix(UnsignedT bits) {
    UnsignedT mask = ((bits >> 15) - 1) | 0x8000u;
    return __ushort_as_bfloat16(bits ^ mask);
  }
};

// Integer types: direct mapping (with sign bit flip for signed types)
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

// Unsigned types: direct mapping
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

template <>
struct RadixTraits<bool> {
  using UnsignedT = uint8_t;
  static constexpr int BITS = 8;

  __device__ __forceinline__ static UnsignedT to_radix(bool val) {
    return val ? 1 : 0;
  }

  __device__ __forceinline__ static bool from_radix(UnsignedT bits) {
    return bits != 0;
  }
};

///////////////////////////////////////////////////////////////////////////////
// Extract digit from radix representation
///////////////////////////////////////////////////////////////////////////////

template <typename UnsignedT>
__device__ __forceinline__ int
extract_digit(UnsignedT val, int start_bit, int num_bits) {
  return (val >> start_bit) & ((1 << num_bits) - 1);
}

///////////////////////////////////////////////////////////////////////////////
// Warp-level primitives for histogram aggregation
///////////////////////////////////////////////////////////////////////////////

// Warp-level ballot to count how many threads have the same bin
__device__ __forceinline__ int warp_histogram_increment(int bin, int target_bin) {
  unsigned int mask = __ballot_sync(0xFFFFFFFF, bin == target_bin);
  return __popc(mask);
}

///////////////////////////////////////////////////////////////////////////////
// Block-level histogram with hierarchical atomics
///////////////////////////////////////////////////////////////////////////////

template <int BLOCK_THREADS>
__device__ __forceinline__ void block_histogram_atomic(
    int* shared_hist,
    int bin,
    int count = 1) {
  // Use warp-aggregated atomics for better performance
  // First, aggregate within warp using ballot
  unsigned int warp_mask = __ballot_sync(0xFFFFFFFF, true);
  int lane_id = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;

  // Find threads with same bin in this warp
  for (int b = 0; b < RADIX_SIZE; b++) {
    unsigned int same_bin_mask = __ballot_sync(warp_mask, bin == b);
    int same_count = __popc(same_bin_mask);
    // First thread with this bin does the atomic add
    if (same_count > 0 && bin == b && (lane_id == __ffs(same_bin_mask) - 1)) {
      atomicAdd(&shared_hist[b], same_count * count);
    }
  }
}

// Simpler version: direct atomic add (works well with modern GPUs)
__device__ __forceinline__ void histogram_atomic_add(int* shared_hist, int bin) {
  atomicAdd(&shared_hist[bin], 1);
}

///////////////////////////////////////////////////////////////////////////////
// NaN handling for floating-point types
///////////////////////////////////////////////////////////////////////////////

template <typename T>
__device__ __forceinline__ bool is_nan_value(T val) {
  if constexpr (
      cuda::std::is_same_v<T, float> || cuda::std::is_same_v<T, double>) {
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
// Comparison operators for top-k selection
// For top-k largest: we want elements > pivot
// For top-k smallest: we want elements < pivot
///////////////////////////////////////////////////////////////////////////////

template <typename T, bool SELECT_LARGEST = true>
struct RadixCompare {
  using Traits = RadixTraits<T>;
  using UnsignedT = typename Traits::UnsignedT;

  // Returns true if 'a' should come before 'b' in the selection
  __device__ __forceinline__ static bool compare(T a, T b) {
    if constexpr (SELECT_LARGEST) {
      // For largest: we want descending order
      return Traits::to_radix(a) > Traits::to_radix(b);
    } else {
      // For smallest: we want ascending order
      return Traits::to_radix(a) < Traits::to_radix(b);
    }
  }

  // Returns true if 'val' should be included in top-k (compared to pivot)
  __device__ __forceinline__ static bool should_select(T val, T pivot) {
    if constexpr (SELECT_LARGEST) {
      return Traits::to_radix(val) > Traits::to_radix(pivot);
    } else {
      return Traits::to_radix(val) < Traits::to_radix(pivot);
    }
  }
};

} // namespace mlx::core::cu
