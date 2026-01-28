// Copyright © 2025 Apple Inc.

#pragma once

#include <metal_stdlib>
#include "mlx/backend/metal/kernels/bf16.h"

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// Radix Select Implementation for Metal
//
// Multi-pass radix-based selection algorithm for partition operations.
// Uses IEEE 754 bit manipulation for correct floating-point ordering.
///////////////////////////////////////////////////////////////////////////////

// Radix configuration
constant constexpr int RADIX_BITS = 8;
constant constexpr int RADIX_SIZE = 1 << RADIX_BITS; // 256 bins
constant constexpr int SIMD_SIZE = 32;

///////////////////////////////////////////////////////////////////////////////
// Bit manipulation for radix sorting
///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct RadixTraits;

template <>
struct RadixTraits<float> {
  using UnsignedT = uint32_t;
  static constexpr constant int BITS = 32;

  static METAL_FUNC UnsignedT to_radix(float val) {
    UnsignedT bits = as_type<UnsignedT>(val);
    UnsignedT mask = -int32_t(bits >> 31) | 0x80000000u;
    return bits ^ mask;
  }

  static METAL_FUNC float from_radix(UnsignedT bits) {
    UnsignedT mask = ((bits >> 31) - 1) | 0x80000000u;
    return as_type<float>(bits ^ mask);
  }
};

template <>
struct RadixTraits<half> {
  using UnsignedT = uint16_t;
  static constexpr constant int BITS = 16;

  static METAL_FUNC UnsignedT to_radix(half val) {
    UnsignedT bits = as_type<UnsignedT>(val);
    UnsignedT mask = -int16_t(bits >> 15) | 0x8000u;
    return bits ^ mask;
  }

  static METAL_FUNC half from_radix(UnsignedT bits) {
    UnsignedT mask = ((bits >> 15) - 1) | 0x8000u;
    UnsignedT result = bits ^ mask;
    return as_type<half>(result);
  }
};

template <>
struct RadixTraits<bfloat16_t> {
  using UnsignedT = uint16_t;
  static constexpr constant int BITS = 16;

  static METAL_FUNC UnsignedT to_radix(bfloat16_t val) {
    UnsignedT bits = as_type<UnsignedT>(val);
    UnsignedT mask = -int16_t(bits >> 15) | 0x8000u;
    return bits ^ mask;
  }

  static METAL_FUNC bfloat16_t from_radix(UnsignedT bits) {
    UnsignedT mask = ((bits >> 15) - 1) | 0x8000u;
    UnsignedT result = bits ^ mask;
    return as_type<bfloat16_t>(result);
  }
};

template <>
struct RadixTraits<int8_t> {
  using UnsignedT = uint8_t;
  static constexpr constant int BITS = 8;
  static METAL_FUNC UnsignedT to_radix(int8_t val) {
    return static_cast<UnsignedT>(val) ^ 0x80u;
  }
  static METAL_FUNC int8_t from_radix(UnsignedT bits) {
    return static_cast<int8_t>(bits ^ 0x80u);
  }
};

template <>
struct RadixTraits<int16_t> {
  using UnsignedT = uint16_t;
  static constexpr constant int BITS = 16;
  static METAL_FUNC UnsignedT to_radix(int16_t val) {
    return static_cast<UnsignedT>(val) ^ 0x8000u;
  }
  static METAL_FUNC int16_t from_radix(UnsignedT bits) {
    return static_cast<int16_t>(bits ^ 0x8000u);
  }
};

template <>
struct RadixTraits<int32_t> {
  using UnsignedT = uint32_t;
  static constexpr constant int BITS = 32;
  static METAL_FUNC UnsignedT to_radix(int32_t val) {
    return static_cast<UnsignedT>(val) ^ 0x80000000u;
  }
  static METAL_FUNC int32_t from_radix(UnsignedT bits) {
    return static_cast<int32_t>(bits ^ 0x80000000u);
  }
};

template <>
struct RadixTraits<int64_t> {
  using UnsignedT = uint64_t;
  static constexpr constant int BITS = 64;
  static METAL_FUNC UnsignedT to_radix(int64_t val) {
    return static_cast<UnsignedT>(val) ^ 0x8000000000000000ull;
  }
  static METAL_FUNC int64_t from_radix(UnsignedT bits) {
    return static_cast<int64_t>(bits ^ 0x8000000000000000ull);
  }
};

template <>
struct RadixTraits<uint8_t> {
  using UnsignedT = uint8_t;
  static constexpr constant int BITS = 8;
  static METAL_FUNC UnsignedT to_radix(uint8_t val) {
    return val;
  }
  static METAL_FUNC uint8_t from_radix(UnsignedT bits) {
    return bits;
  }
};

template <>
struct RadixTraits<uint16_t> {
  using UnsignedT = uint16_t;
  static constexpr constant int BITS = 16;
  static METAL_FUNC UnsignedT to_radix(uint16_t val) {
    return val;
  }
  static METAL_FUNC uint16_t from_radix(UnsignedT bits) {
    return bits;
  }
};

template <>
struct RadixTraits<uint32_t> {
  using UnsignedT = uint32_t;
  static constexpr constant int BITS = 32;
  static METAL_FUNC UnsignedT to_radix(uint32_t val) {
    return val;
  }
  static METAL_FUNC uint32_t from_radix(UnsignedT bits) {
    return bits;
  }
};

template <>
struct RadixTraits<uint64_t> {
  using UnsignedT = uint64_t;
  static constexpr constant int BITS = 64;
  static METAL_FUNC UnsignedT to_radix(uint64_t val) {
    return val;
  }
  static METAL_FUNC uint64_t from_radix(UnsignedT bits) {
    return bits;
  }
};

template <typename UnsignedT>
METAL_FUNC int extract_digit(UnsignedT val, int start_bit, int num_bits) {
  return (val >> start_bit) & ((1 << num_bits) - 1);
}

template <typename T>
METAL_FUNC bool is_nan_value(T val) {
  if constexpr (is_floating_point_v<T>) {
    return isnan(val);
  } else {
    return false;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Multi-pass Radix Select Kernels
///////////////////////////////////////////////////////////////////////////////

// Build histogram across all elements
template <typename ValT, short BLOCK_THREADS>
[[kernel, max_total_threads_per_threadgroup(BLOCK_THREADS)]] void
radix_histogram_kernel(
    const device ValT* input [[buffer(0)]],
    device atomic_int* histogram [[buffer(1)]],
    const constant int& n [[buffer(2)]],
    const constant int& stride [[buffer(3)]],
    const constant int& start_bit [[buffer(4)]],
    const constant int& segment_stride [[buffer(5)]],
    const constant uint64_t& prefix_mask [[buffer(6)]],
    const constant uint64_t& target_prefix [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 grid_dims [[threadgroups_per_grid]]) {
  using Traits = RadixTraits<ValT>;
  using UnsignedT = typename Traits::UnsignedT;

  threadgroup int shared_hist[RADIX_SIZE];

  for (int i = lid.x; i < RADIX_SIZE; i += BLOCK_THREADS) {
    shared_hist[i] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  int row = tid.y;
  const device ValT* row_input = input + row * segment_stride;

  // Each threadgroup processes a chunk of the array
  int total_threads = grid_dims.x * BLOCK_THREADS;
  int global_tid = tid.x * BLOCK_THREADS + lid.x;

  for (int i = global_tid; i < n; i += total_threads) {
    ValT val = row_input[i * stride];
    UnsignedT key = Traits::to_radix(val);
    if (is_nan_value(val)) {
      key = ~UnsignedT(0);
    }
    // Only count if matches current prefix
    if ((key & UnsignedT(prefix_mask)) == UnsignedT(target_prefix)) {
      int digit = extract_digit(key, start_bit, RADIX_BITS);
      atomic_fetch_add_explicit(
          (threadgroup atomic_int*)&shared_hist[digit],
          1,
          memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Reduce to global histogram
  device atomic_int* row_hist = histogram + row * RADIX_SIZE;
  for (int i = lid.x; i < RADIX_SIZE; i += BLOCK_THREADS) {
    if (shared_hist[i] > 0) {
      atomic_fetch_add_explicit(
          &row_hist[i], shared_hist[i], memory_order_relaxed);
    }
  }
}

// Find target bin from histogram
template <typename ValT>
[[kernel]] void radix_find_bin_kernel(
    const device int* histogram [[buffer(0)]],
    device int* target_bin [[buffer(1)]],
    device int* new_k [[buffer(2)]],
    const constant int& k [[buffer(3)]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  int row = tid.y;
  const device int* row_hist = histogram + row * RADIX_SIZE;

  int cumsum = 0;
  int bin = 0;
  int remaining_k = k;

  for (int i = 0; i < RADIX_SIZE; i++) {
    int count = row_hist[i];
    if (cumsum + count >= k) {
      bin = i;
      remaining_k = k - cumsum;
      break;
    }
    cumsum += count;
  }

  target_bin[row] = bin;
  new_k[row] = remaining_k;
}

// Partition output with known pivot
template <typename ValT, typename OutT, bool ARG_PARTITION, short BLOCK_THREADS>
[[kernel, max_total_threads_per_threadgroup(BLOCK_THREADS)]] void
radix_partition_output_kernel(
    const device ValT* input [[buffer(0)]],
    device OutT* output [[buffer(1)]],
    device atomic_int* counters [[buffer(2)]],
    const constant int& n [[buffer(3)]],
    const constant int& in_stride [[buffer(4)]],
    const constant int& out_stride [[buffer(5)]],
    const constant int& segment_stride [[buffer(6)]],
    const constant int& out_segment_stride [[buffer(7)]],
    const constant uint64_t& pivot_key [[buffer(8)]],
    const constant int& kth [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 grid_dims [[threadgroups_per_grid]]) {
  using Traits = RadixTraits<ValT>;
  using UnsignedT = typename Traits::UnsignedT;

  int row = tid.y;
  const device ValT* row_input = input + row * segment_stride;
  device OutT* row_output = output + row * out_segment_stride;

  // Counters: [0] = less_count, [1] = equal_count, [2] = greater_count
  device atomic_int* row_counters = counters + row * 3;

  int total_threads = grid_dims.x * BLOCK_THREADS;
  int global_tid = tid.x * BLOCK_THREADS + lid.x;

  UnsignedT pivot = UnsignedT(pivot_key);

  // Phase 1: Count and output elements less than pivot
  for (int i = global_tid; i < n; i += total_threads) {
    ValT val = row_input[i * in_stride];
    UnsignedT key = Traits::to_radix(val);
    if (is_nan_value(val)) {
      key = ~UnsignedT(0);
    }

    if (key < pivot) {
      int pos =
          atomic_fetch_add_explicit(&row_counters[0], 1, memory_order_relaxed);
      if (ARG_PARTITION) {
        row_output[pos * out_stride] = i;
      } else {
        row_output[pos * out_stride] = val;
      }
    }
  }
}

// Output equal elements
template <typename ValT, typename OutT, bool ARG_PARTITION, short BLOCK_THREADS>
[[kernel, max_total_threads_per_threadgroup(BLOCK_THREADS)]] void
radix_partition_equal_kernel(
    const device ValT* input [[buffer(0)]],
    device OutT* output [[buffer(1)]],
    device atomic_int* counters [[buffer(2)]],
    const constant int& n [[buffer(3)]],
    const constant int& in_stride [[buffer(4)]],
    const constant int& out_stride [[buffer(5)]],
    const constant int& segment_stride [[buffer(6)]],
    const constant int& out_segment_stride [[buffer(7)]],
    const constant uint64_t& pivot_key [[buffer(8)]],
    const constant int& less_count [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 grid_dims [[threadgroups_per_grid]]) {
  using Traits = RadixTraits<ValT>;
  using UnsignedT = typename Traits::UnsignedT;

  int row = tid.y;
  const device ValT* row_input = input + row * segment_stride;
  device OutT* row_output = output + row * out_segment_stride;
  device atomic_int* row_counters = counters + row * 3;

  int total_threads = grid_dims.x * BLOCK_THREADS;
  int global_tid = tid.x * BLOCK_THREADS + lid.x;

  UnsignedT pivot = UnsignedT(pivot_key);

  for (int i = global_tid; i < n; i += total_threads) {
    ValT val = row_input[i * in_stride];
    UnsignedT key = Traits::to_radix(val);
    if (is_nan_value(val)) {
      key = ~UnsignedT(0);
    }

    if (key == pivot) {
      int pos = less_count +
          atomic_fetch_add_explicit(&row_counters[1], 1, memory_order_relaxed);
      if (ARG_PARTITION) {
        row_output[pos * out_stride] = i;
      } else {
        row_output[pos * out_stride] = val;
      }
    }
  }
}

// Output greater elements
template <typename ValT, typename OutT, bool ARG_PARTITION, short BLOCK_THREADS>
[[kernel, max_total_threads_per_threadgroup(BLOCK_THREADS)]] void
radix_partition_greater_kernel(
    const device ValT* input [[buffer(0)]],
    device OutT* output [[buffer(1)]],
    device atomic_int* counters [[buffer(2)]],
    const constant int& n [[buffer(3)]],
    const constant int& in_stride [[buffer(4)]],
    const constant int& out_stride [[buffer(5)]],
    const constant int& segment_stride [[buffer(6)]],
    const constant int& out_segment_stride [[buffer(7)]],
    const constant uint64_t& pivot_key [[buffer(8)]],
    const constant int& less_equal_count [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 grid_dims [[threadgroups_per_grid]]) {
  using Traits = RadixTraits<ValT>;
  using UnsignedT = typename Traits::UnsignedT;

  int row = tid.y;
  const device ValT* row_input = input + row * segment_stride;
  device OutT* row_output = output + row * out_segment_stride;
  device atomic_int* row_counters = counters + row * 3;

  int total_threads = grid_dims.x * BLOCK_THREADS;
  int global_tid = tid.x * BLOCK_THREADS + lid.x;

  UnsignedT pivot = UnsignedT(pivot_key);

  for (int i = global_tid; i < n; i += total_threads) {
    ValT val = row_input[i * in_stride];
    UnsignedT key = Traits::to_radix(val);
    if (is_nan_value(val)) {
      key = ~UnsignedT(0);
    }

    if (key > pivot) {
      int pos = less_equal_count +
          atomic_fetch_add_explicit(&row_counters[2], 1, memory_order_relaxed);
      if (ARG_PARTITION) {
        row_output[pos * out_stride] = i;
      } else {
        row_output[pos * out_stride] = val;
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Fused Multi-pass Radix Select for Large Arrays
//
// Performs the complete radix select in a single dispatch:
// 1. Build histograms in parallel across threadgroups
// 2. Reduce histograms and find pivot
// 3. Output partitioned results
///////////////////////////////////////////////////////////////////////////////

template <typename ValT, typename OutT, bool ARG_PARTITION, short BLOCK_THREADS>
[[kernel, max_total_threads_per_threadgroup(BLOCK_THREADS)]] void
radix_select_large_fused(
    const device ValT* input [[buffer(0)]],
    device OutT* output [[buffer(1)]],
    device int* global_histogram [[buffer(2)]],
    device atomic_int* global_counters [[buffer(3)]],
    device int* pivot_info [[buffer(4)]],
    const constant int& n [[buffer(5)]],
    const constant int& kth [[buffer(6)]],
    const constant int& in_stride [[buffer(7)]],
    const constant int& out_stride [[buffer(8)]],
    const constant int& segment_stride [[buffer(9)]],
    const constant int& out_segment_stride [[buffer(10)]],
    const constant int& num_blocks [[buffer(11)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  using Traits = RadixTraits<ValT>;
  using UnsignedT = typename Traits::UnsignedT;
  constexpr int NUM_PASSES = (Traits::BITS + RADIX_BITS - 1) / RADIX_BITS;
  constexpr int NUM_SIMD_GROUPS = BLOCK_THREADS / SIMD_SIZE;

  int row = tid.y;
  int block_id = tid.x;
  const device ValT* row_input = input + row * segment_stride;
  device OutT* row_output = output + row * out_segment_stride;

  // Shared memory for histogram and reduction
  threadgroup int shared_hist[RADIX_SIZE];
  threadgroup int simd_hist[NUM_SIMD_GROUPS][RADIX_SIZE];
  threadgroup int
      shared_pivot[4]; // [target_bin, new_k, less_count, equal_count]
  threadgroup UnsignedT shared_pivot_key[1];

  // Per-row global state
  device int* row_histogram = global_histogram + row * RADIX_SIZE;
  device atomic_int* row_counters = global_counters + row * 4;
  device int* row_pivot = pivot_info + row * 4;

  int k = kth + 1;
  UnsignedT target_prefix = 0;
  UnsignedT prefix_mask = 0;

  // Multi-pass radix select
  for (int pass = NUM_PASSES - 1; pass >= 0; pass--) {
    int start_bit = pass * RADIX_BITS;

    // Clear shared histogram
    for (int i = lid.x; i < RADIX_SIZE; i += BLOCK_THREADS) {
      shared_hist[i] = 0;
    }
    // Clear SIMD histograms
    if (simd_lane < RADIX_SIZE / SIMD_SIZE) {
      for (int s = 0; s < NUM_SIMD_GROUPS; s++) {
        simd_hist[s][simd_lane * SIMD_SIZE + simd_group] = 0;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 1: Build local histogram with SIMD optimization
    // Each thread maintains private histogram bins
    int private_hist[4] = {0, 0, 0, 0}; // Process 4 bins at a time

    int elements_per_block = (n + num_blocks - 1) / num_blocks;
    int start_idx = block_id * elements_per_block;
    int end_idx = min(start_idx + elements_per_block, n);

    for (int i = start_idx + lid.x; i < end_idx; i += BLOCK_THREADS) {
      ValT val = row_input[i * in_stride];
      UnsignedT key = Traits::to_radix(val);
      if (is_nan_value(val)) {
        key = ~UnsignedT(0);
      }

      // Only count elements matching current prefix
      if ((key & prefix_mask) == target_prefix) {
        int digit = extract_digit(key, start_bit, RADIX_BITS);
        // Use SIMD shuffle to aggregate within SIMD group
        atomic_fetch_add_explicit(
            (threadgroup atomic_int*)&simd_hist[simd_group][digit],
            1,
            memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce SIMD histograms to shared histogram
    for (int i = lid.x; i < RADIX_SIZE; i += BLOCK_THREADS) {
      int sum = 0;
      for (int s = 0; s < NUM_SIMD_GROUPS; s++) {
        sum += simd_hist[s][i];
      }
      shared_hist[i] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Reduce to global histogram (only first block does final
    // reduction)
    if (block_id == 0) {
      // Clear global histogram first
      for (int i = lid.x; i < RADIX_SIZE; i += BLOCK_THREADS) {
        row_histogram[i] = 0;
      }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // All blocks contribute to global histogram
    for (int i = lid.x; i < RADIX_SIZE; i += BLOCK_THREADS) {
      if (shared_hist[i] > 0) {
        atomic_fetch_add_explicit(
            (device atomic_int*)&row_histogram[i],
            shared_hist[i],
            memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Phase 3: Find target bin (only block 0, thread 0)
    if (block_id == 0 && lid.x == 0) {
      int cumsum = 0;
      int target_bin = 0;
      int remaining_k = k;

      for (int bin = 0; bin < RADIX_SIZE; bin++) {
        int count = row_histogram[bin];
        if (cumsum + count >= k) {
          target_bin = bin;
          remaining_k = k - cumsum;
          break;
        }
        cumsum += count;
      }

      shared_pivot[0] = target_bin;
      shared_pivot[1] = remaining_k;
      row_pivot[pass * 2] = target_bin;
      row_pivot[pass * 2 + 1] = remaining_k;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All threads read the pivot info
    int target_bin = shared_pivot[0];
    k = shared_pivot[1];

    // Update prefix for next pass
    UnsignedT digit_mask = UnsignedT((1 << RADIX_BITS) - 1) << start_bit;
    target_prefix |= UnsignedT(target_bin) << start_bit;
    prefix_mask |= digit_mask;

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Store final pivot key
  if (block_id == 0 && lid.x == 0) {
    shared_pivot_key[0] = target_prefix;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  UnsignedT pivot_key = shared_pivot_key[0];

  // Phase 4: Output partitioned array
  // Reset counters
  if (block_id == 0 && lid.x == 0) {
    atomic_store_explicit(&row_counters[0], 0, memory_order_relaxed); // less
    atomic_store_explicit(&row_counters[1], 0, memory_order_relaxed); // equal
    atomic_store_explicit(&row_counters[2], 0, memory_order_relaxed); // greater
  }
  threadgroup_barrier(mem_flags::mem_device);

  // Count elements in each partition
  int local_less = 0, local_equal = 0, local_greater = 0;
  int elements_per_block = (n + num_blocks - 1) / num_blocks;
  int start_idx = block_id * elements_per_block;
  int end_idx = min(start_idx + elements_per_block, n);

  for (int i = start_idx + lid.x; i < end_idx; i += BLOCK_THREADS) {
    ValT val = row_input[i * in_stride];
    UnsignedT key = Traits::to_radix(val);
    if (is_nan_value(val)) {
      key = ~UnsignedT(0);
    }

    if (key < pivot_key)
      local_less++;
    else if (key == pivot_key)
      local_equal++;
    else
      local_greater++;
  }

  // Reduce within SIMD group
  local_less = simd_sum(local_less);
  local_equal = simd_sum(local_equal);
  local_greater = simd_sum(local_greater);

  // First lane of each SIMD group contributes to global count
  if (simd_lane == 0) {
    atomic_fetch_add_explicit(
        &row_counters[0], local_less, memory_order_relaxed);
    atomic_fetch_add_explicit(
        &row_counters[1], local_equal, memory_order_relaxed);
    atomic_fetch_add_explicit(
        &row_counters[2], local_greater, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_device);

  // Read final counts
  if (lid.x == 0) {
    shared_pivot[2] =
        atomic_load_explicit(&row_counters[0], memory_order_relaxed);
    shared_pivot[3] =
        atomic_load_explicit(&row_counters[1], memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  int less_count = shared_pivot[2];
  int equal_count = shared_pivot[3];

  // Reset counters for output phase
  if (block_id == 0 && lid.x == 0) {
    atomic_store_explicit(&row_counters[0], 0, memory_order_relaxed);
    atomic_store_explicit(&row_counters[1], 0, memory_order_relaxed);
    atomic_store_explicit(&row_counters[2], 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_device);

  // Output elements
  for (int i = start_idx + lid.x; i < end_idx; i += BLOCK_THREADS) {
    ValT val = row_input[i * in_stride];
    UnsignedT key = Traits::to_radix(val);
    if (is_nan_value(val)) {
      key = ~UnsignedT(0);
    }

    int pos;
    if (key < pivot_key) {
      pos =
          atomic_fetch_add_explicit(&row_counters[0], 1, memory_order_relaxed);
    } else if (key == pivot_key) {
      pos = less_count +
          atomic_fetch_add_explicit(&row_counters[1], 1, memory_order_relaxed);
    } else {
      pos = less_count + equal_count +
          atomic_fetch_add_explicit(&row_counters[2], 1, memory_order_relaxed);
    }

    if (ARG_PARTITION) {
      row_output[pos * out_stride] = i;
    } else {
      row_output[pos * out_stride] = val;
    }
  }
}

// Large array streaming kernel
template <typename ValT, typename OutT, bool ARG_PARTITION, short BLOCK_THREADS>
[[kernel, max_total_threads_per_threadgroup(BLOCK_THREADS)]] void
radix_select_large_streaming(
    const device ValT* input [[buffer(0)]],
    device OutT* output [[buffer(1)]],
    device atomic_int* counters [[buffer(2)]],
    const constant int& n [[buffer(3)]],
    const constant int& kth [[buffer(4)]],
    const constant int& in_stride [[buffer(5)]],
    const constant int& out_stride [[buffer(6)]],
    const constant int& segment_stride [[buffer(7)]],
    const constant int& out_segment_stride [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  using Traits = RadixTraits<ValT>;
  using UnsignedT = typename Traits::UnsignedT;
  constexpr int NUM_PASSES = (Traits::BITS + RADIX_BITS - 1) / RADIX_BITS;

  int row = tid.y;
  const device ValT* row_input = input + row * segment_stride;
  device OutT* row_output = output + row * out_segment_stride;

  // Shared memory
  threadgroup int shared_hist[RADIX_SIZE];
  threadgroup int shared_pivot_info[2];
  threadgroup int shared_counts[2];
  threadgroup int shared_output_counters[3];

  int k = kth + 1;
  UnsignedT target_prefix = 0;
  UnsignedT prefix_mask = 0;

  // Multi-pass to find pivot
  for (int pass = NUM_PASSES - 1; pass >= 0; pass--) {
    int start_bit = pass * RADIX_BITS;

    // Clear histogram
    for (int i = lid.x; i < RADIX_SIZE; i += BLOCK_THREADS) {
      shared_hist[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Build histogram
    for (int i = lid.x; i < n; i += BLOCK_THREADS) {
      ValT val = row_input[i * in_stride];
      UnsignedT key = Traits::to_radix(val);
      if (is_nan_value(val)) {
        key = ~UnsignedT(0);
      }

      if ((key & prefix_mask) == target_prefix) {
        int digit = extract_digit(key, start_bit, RADIX_BITS);
        atomic_fetch_add_explicit(
            (threadgroup atomic_int*)&shared_hist[digit],
            1,
            memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Find target bin
    if (lid.x == 0) {
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
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int target_bin = shared_pivot_info[0];
    k = shared_pivot_info[1];

    UnsignedT digit_mask = UnsignedT((1 << RADIX_BITS) - 1) << start_bit;
    target_prefix |= UnsignedT(target_bin) << start_bit;
    prefix_mask |= digit_mask;

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Initialize counters for partition size counting
  if (lid.x == 0) {
    shared_counts[0] = 0; // less_count
    shared_counts[1] = 0; // equal_count
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Count partition sizes with SIMD reduction
  int local_less = 0, local_equal = 0;
  for (int i = lid.x; i < n; i += BLOCK_THREADS) {
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

  // SIMD reduction
  local_less = simd_sum(local_less);
  local_equal = simd_sum(local_equal);

  // Aggregate across SIMD groups (only first lane of each SIMD group)
  if (simd_lane == 0) {
    atomic_fetch_add_explicit(
        (threadgroup atomic_int*)&shared_counts[0],
        local_less,
        memory_order_relaxed);
    atomic_fetch_add_explicit(
        (threadgroup atomic_int*)&shared_counts[1],
        local_equal,
        memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Read final counts - all threads read the same values
  int less_count = shared_counts[0];
  int equal_count = shared_counts[1];

  // Initialize output counters
  if (lid.x == 0) {
    shared_output_counters[0] = 0; // less output counter
    shared_output_counters[1] = 0; // equal output counter
    shared_output_counters[2] = 0; // greater output counter
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Output partitioned elements
  for (int i = lid.x; i < n; i += BLOCK_THREADS) {
    ValT val = row_input[i * in_stride];
    UnsignedT key = Traits::to_radix(val);
    if (is_nan_value(val)) {
      key = ~UnsignedT(0);
    }

    int pos;
    if (key < target_prefix) {
      pos = atomic_fetch_add_explicit(
          (threadgroup atomic_int*)&shared_output_counters[0],
          1,
          memory_order_relaxed);
    } else if (key == target_prefix) {
      pos = less_count +
          atomic_fetch_add_explicit(
                (threadgroup atomic_int*)&shared_output_counters[1],
                1,
                memory_order_relaxed);
    } else {
      pos = less_count + equal_count +
          atomic_fetch_add_explicit(
                (threadgroup atomic_int*)&shared_output_counters[2],
                1,
                memory_order_relaxed);
    }

    if (ARG_PARTITION) {
      row_output[pos * out_stride] = i;
    } else {
      row_output[pos * out_stride] = val;
    }
  }
}

// Large array streaming kernel for non-contiguous arrays
// Uses elem_to_loc for proper multi-dimensional indexing
template <typename ValT, typename OutT, bool ARG_PARTITION, short BLOCK_THREADS>
[[kernel, max_total_threads_per_threadgroup(BLOCK_THREADS)]] void
radix_select_large_streaming_nc(
    const device ValT* input [[buffer(0)]],
    device OutT* output [[buffer(1)]],
    device atomic_int* counters [[buffer(2)]],
    const constant int& n [[buffer(3)]],
    const constant int& kth [[buffer(4)]],
    const constant int& in_stride [[buffer(5)]],
    const constant int& out_stride [[buffer(6)]],
    const constant int& nc_dim [[buffer(7)]],
    const constant int* nc_shape [[buffer(8)]],
    const constant int64_t* in_nc_strides [[buffer(9)]],
    const constant int64_t* out_nc_strides [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  using Traits = RadixTraits<ValT>;
  using UnsignedT = typename Traits::UnsignedT;
  constexpr int NUM_PASSES = (Traits::BITS + RADIX_BITS - 1) / RADIX_BITS;

  // Compute row offsets using elem_to_loc for non-contiguous arrays
  int row = tid.y;
  auto in_offset = elem_to_loc(row, nc_shape, in_nc_strides, nc_dim);
  auto out_offset = elem_to_loc(row, nc_shape, out_nc_strides, nc_dim);

  const device ValT* row_input = input + in_offset;
  device OutT* row_output = output + out_offset;

  // Shared memory
  threadgroup int shared_hist[RADIX_SIZE];
  threadgroup int shared_pivot_info[2];
  threadgroup int shared_counts[2];
  threadgroup int shared_output_counters[3];

  int k = kth + 1;
  UnsignedT target_prefix = 0;
  UnsignedT prefix_mask = 0;

  // Multi-pass to find pivot
  for (int pass = NUM_PASSES - 1; pass >= 0; pass--) {
    int start_bit = pass * RADIX_BITS;

    // Clear histogram
    for (int i = lid.x; i < RADIX_SIZE; i += BLOCK_THREADS) {
      shared_hist[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Build histogram
    for (int i = lid.x; i < n; i += BLOCK_THREADS) {
      ValT val = row_input[i * in_stride];
      UnsignedT key = Traits::to_radix(val);
      if (is_nan_value(val)) {
        key = ~UnsignedT(0);
      }

      if ((key & prefix_mask) == target_prefix) {
        int digit = extract_digit(key, start_bit, RADIX_BITS);
        atomic_fetch_add_explicit(
            (threadgroup atomic_int*)&shared_hist[digit],
            1,
            memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Find target bin
    if (lid.x == 0) {
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
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int target_bin = shared_pivot_info[0];
    k = shared_pivot_info[1];

    UnsignedT digit_mask = UnsignedT((1 << RADIX_BITS) - 1) << start_bit;
    target_prefix |= UnsignedT(target_bin) << start_bit;
    prefix_mask |= digit_mask;

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Initialize counters for partition size counting
  if (lid.x == 0) {
    shared_counts[0] = 0; // less_count
    shared_counts[1] = 0; // equal_count
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Count partition sizes with SIMD reduction
  int local_less = 0, local_equal = 0;
  for (int i = lid.x; i < n; i += BLOCK_THREADS) {
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

  // SIMD reduction
  local_less = simd_sum(local_less);
  local_equal = simd_sum(local_equal);

  // Aggregate across SIMD groups (only first lane of each SIMD group)
  if (simd_lane == 0) {
    atomic_fetch_add_explicit(
        (threadgroup atomic_int*)&shared_counts[0],
        local_less,
        memory_order_relaxed);
    atomic_fetch_add_explicit(
        (threadgroup atomic_int*)&shared_counts[1],
        local_equal,
        memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Read final counts - all threads read the same values
  int less_count = shared_counts[0];
  int equal_count = shared_counts[1];

  // Initialize output counters
  if (lid.x == 0) {
    shared_output_counters[0] = 0; // less output counter
    shared_output_counters[1] = 0; // equal output counter
    shared_output_counters[2] = 0; // greater output counter
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Output partitioned elements
  for (int i = lid.x; i < n; i += BLOCK_THREADS) {
    ValT val = row_input[i * in_stride];
    UnsignedT key = Traits::to_radix(val);
    if (is_nan_value(val)) {
      key = ~UnsignedT(0);
    }

    int pos;
    if (key < target_prefix) {
      pos = atomic_fetch_add_explicit(
          (threadgroup atomic_int*)&shared_output_counters[0],
          1,
          memory_order_relaxed);
    } else if (key == target_prefix) {
      pos = less_count +
          atomic_fetch_add_explicit(
                (threadgroup atomic_int*)&shared_output_counters[1],
                1,
                memory_order_relaxed);
    } else {
      pos = less_count + equal_count +
          atomic_fetch_add_explicit(
                (threadgroup atomic_int*)&shared_output_counters[2],
                1,
                memory_order_relaxed);
    }

    if (ARG_PARTITION) {
      row_output[pos * out_stride] = i;
    } else {
      row_output[pos * out_stride] = val;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Single-pass Radix Select for small arrays (fits in threadgroup memory)
///////////////////////////////////////////////////////////////////////////////

template <
    typename ValT,
    typename OutT,
    bool ARG_PARTITION,
    short BLOCK_THREADS,
    short ITEMS_PER_THREAD>
struct RadixSelectSmall {
  using Traits = RadixTraits<ValT>;
  using UnsignedT = typename Traits::UnsignedT;

  static constexpr constant short TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

  static METAL_FUNC void partition(
      const device ValT* input,
      device OutT* output,
      int kth,
      int size_sorted_axis,
      int in_stride_sorted_axis,
      int out_stride_sorted_axis,
      int in_stride_segment_axis,
      int out_stride_segment_axis,
      threadgroup UnsignedT* shared_keys,
      threadgroup uint32_t* shared_idxs,
      threadgroup int* shared_hist,
      threadgroup int* shared_count,
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]]) {
    int row = tid.y;
    const device ValT* row_input = input + row * in_stride_segment_axis;
    device OutT* row_output = output + row * out_stride_segment_axis;

    int n = min(size_sorted_axis, int(TILE_SIZE));

    // Load data into threadgroup memory
    for (int i = lid.x; i < TILE_SIZE; i += BLOCK_THREADS) {
      if (i < n) {
        ValT val = row_input[i * in_stride_sorted_axis];
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
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Radix select
    int k = kth + 1;
    constexpr int NUM_PASSES = (Traits::BITS + RADIX_BITS - 1) / RADIX_BITS;

    UnsignedT target_prefix = 0;
    UnsignedT prefix_mask = 0;

    for (int pass = NUM_PASSES - 1; pass >= 0; pass--) {
      int start_bit = pass * RADIX_BITS;

      for (int i = lid.x; i < RADIX_SIZE; i += BLOCK_THREADS) {
        shared_hist[i] = 0;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      for (int i = lid.x; i < n; i += BLOCK_THREADS) {
        UnsignedT key = shared_keys[i];
        if ((key & prefix_mask) == target_prefix) {
          int digit = extract_digit(key, start_bit, RADIX_BITS);
          atomic_fetch_add_explicit(
              (threadgroup atomic_int*)&shared_hist[digit],
              1,
              memory_order_relaxed);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (lid.x == 0) {
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
      threadgroup_barrier(mem_flags::mem_threadgroup);

      int target_bin = shared_count[0];
      k = shared_count[1];

      UnsignedT digit_mask = UnsignedT((1 << RADIX_BITS) - 1) << start_bit;
      target_prefix |= UnsignedT(target_bin) << start_bit;
      prefix_mask |= digit_mask;

      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Output partitioned array
    if (lid.x == 0) {
      shared_count[0] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 1: less than pivot
    for (int i = lid.x; i < n; i += BLOCK_THREADS) {
      UnsignedT key = shared_keys[i];
      if (key < target_prefix) {
        int pos = atomic_fetch_add_explicit(
            (threadgroup atomic_int*)&shared_count[0], 1, memory_order_relaxed);
        if (ARG_PARTITION) {
          row_output[pos * out_stride_sorted_axis] = shared_idxs[i];
        } else {
          row_output[pos * out_stride_sorted_axis] =
              row_input[shared_idxs[i] * in_stride_sorted_axis];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: equal to pivot
    for (int i = lid.x; i < n; i += BLOCK_THREADS) {
      UnsignedT key = shared_keys[i];
      if (key == target_prefix) {
        int pos = atomic_fetch_add_explicit(
            (threadgroup atomic_int*)&shared_count[0], 1, memory_order_relaxed);
        if (ARG_PARTITION) {
          row_output[pos * out_stride_sorted_axis] = shared_idxs[i];
        } else {
          row_output[pos * out_stride_sorted_axis] =
              row_input[shared_idxs[i] * in_stride_sorted_axis];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: greater than pivot
    for (int i = lid.x; i < n; i += BLOCK_THREADS) {
      UnsignedT key = shared_keys[i];
      if (key > target_prefix) {
        int pos = atomic_fetch_add_explicit(
            (threadgroup atomic_int*)&shared_count[0], 1, memory_order_relaxed);
        if (ARG_PARTITION) {
          row_output[pos * out_stride_sorted_axis] = shared_idxs[i];
        } else {
          row_output[pos * out_stride_sorted_axis] =
              row_input[shared_idxs[i] * in_stride_sorted_axis];
        }
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
// Kernel entry points
///////////////////////////////////////////////////////////////////////////////

template <
    typename ValT,
    typename OutT,
    bool ARG_PARTITION,
    short BLOCK_THREADS,
    short ITEMS_PER_THREAD>
[[kernel, max_total_threads_per_threadgroup(BLOCK_THREADS)]] void
radix_select_partition(
    const device ValT* input [[buffer(0)]],
    device OutT* output [[buffer(1)]],
    const constant int& kth [[buffer(2)]],
    const constant int& size_sorted_axis [[buffer(3)]],
    const constant int& in_stride_sorted_axis [[buffer(4)]],
    const constant int& out_stride_sorted_axis [[buffer(5)]],
    const constant int& in_stride_segment_axis [[buffer(6)]],
    const constant int& out_stride_segment_axis [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  using SelectKernel = RadixSelectSmall<
      ValT,
      OutT,
      ARG_PARTITION,
      BLOCK_THREADS,
      ITEMS_PER_THREAD>;
  using UnsignedT = typename SelectKernel::UnsignedT;

  threadgroup UnsignedT shared_keys[SelectKernel::TILE_SIZE];
  threadgroup uint32_t shared_idxs[SelectKernel::TILE_SIZE];
  threadgroup int shared_hist[RADIX_SIZE];
  threadgroup int shared_count[2];

  SelectKernel::partition(
      input,
      output,
      kth,
      size_sorted_axis,
      in_stride_sorted_axis,
      out_stride_sorted_axis,
      in_stride_segment_axis,
      out_stride_segment_axis,
      shared_keys,
      shared_idxs,
      shared_hist,
      shared_count,
      tid,
      lid);
}

template <
    typename ValT,
    typename OutT,
    bool ARG_PARTITION,
    short BLOCK_THREADS,
    short ITEMS_PER_THREAD>
[[kernel, max_total_threads_per_threadgroup(BLOCK_THREADS)]] void
radix_select_partition_nc(
    const device ValT* input [[buffer(0)]],
    device OutT* output [[buffer(1)]],
    const constant int& kth [[buffer(2)]],
    const constant int& size_sorted_axis [[buffer(3)]],
    const constant int& in_stride_sorted_axis [[buffer(4)]],
    const constant int& out_stride_sorted_axis [[buffer(5)]],
    const constant int& nc_dim [[buffer(6)]],
    const constant int* nc_shape [[buffer(7)]],
    const constant int64_t* in_nc_strides [[buffer(8)]],
    const constant int64_t* out_nc_strides [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  using SelectKernel = RadixSelectSmall<
      ValT,
      OutT,
      ARG_PARTITION,
      BLOCK_THREADS,
      ITEMS_PER_THREAD>;
  using UnsignedT = typename SelectKernel::UnsignedT;

  auto in_offset = elem_to_loc(tid.y, nc_shape, in_nc_strides, nc_dim);
  auto out_offset = elem_to_loc(tid.y, nc_shape, out_nc_strides, nc_dim);

  threadgroup UnsignedT shared_keys[SelectKernel::TILE_SIZE];
  threadgroup uint32_t shared_idxs[SelectKernel::TILE_SIZE];
  threadgroup int shared_hist[RADIX_SIZE];
  threadgroup int shared_count[2];

  SelectKernel::partition(
      input + in_offset,
      output + out_offset,
      kth,
      size_sorted_axis,
      in_stride_sorted_axis,
      out_stride_sorted_axis,
      0,
      0,
      shared_keys,
      shared_idxs,
      shared_hist,
      shared_count,
      tid,
      lid);
}
