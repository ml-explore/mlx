// Copyright © 2025 Apple Inc.

#pragma once

#include <metal_stdlib>
#include "mlx/backend/metal/kernels/bf16.h"

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// Radix Select Implementation for Metal
//
// This implements an optimized radix-based top-k selection algorithm based on
// the RadiK paper (Li et al., ICS'24). Key optimizations include:
// - Threadgroup-level histogram building with hierarchical atomics
// - IEEE 754 bit manipulation for correct floating-point ordering
// - Efficient candidate filtering with coalesced memory access
// - Multi-pass support for large arrays
///////////////////////////////////////////////////////////////////////////////

// Radix configuration
constant constexpr int RADIX_BITS = 8;
constant constexpr int RADIX_SIZE = 1 << RADIX_BITS; // 256 bins

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
  static METAL_FUNC UnsignedT to_radix(uint8_t val) { return val; }
  static METAL_FUNC uint8_t from_radix(UnsignedT bits) { return bits; }
};

template <>
struct RadixTraits<uint16_t> {
  using UnsignedT = uint16_t;
  static constexpr constant int BITS = 16;
  static METAL_FUNC UnsignedT to_radix(uint16_t val) { return val; }
  static METAL_FUNC uint16_t from_radix(UnsignedT bits) { return bits; }
};

template <>
struct RadixTraits<uint32_t> {
  using UnsignedT = uint32_t;
  static constexpr constant int BITS = 32;
  static METAL_FUNC UnsignedT to_radix(uint32_t val) { return val; }
  static METAL_FUNC uint32_t from_radix(UnsignedT bits) { return bits; }
};

template <>
struct RadixTraits<uint64_t> {
  using UnsignedT = uint64_t;
  static constexpr constant int BITS = 64;
  static METAL_FUNC UnsignedT to_radix(uint64_t val) { return val; }
  static METAL_FUNC uint64_t from_radix(UnsignedT bits) { return bits; }
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

// Kernel 1: Build histogram across all elements
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
          (threadgroup atomic_int*)&shared_hist[digit], 1, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Reduce to global histogram
  device atomic_int* row_hist = histogram + row * RADIX_SIZE;
  for (int i = lid.x; i < RADIX_SIZE; i += BLOCK_THREADS) {
    if (shared_hist[i] > 0) {
      atomic_fetch_add_explicit(&row_hist[i], shared_hist[i], memory_order_relaxed);
    }
  }
}

// Kernel 2: Find target bin from histogram
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

// Kernel 3: Final partition output with known pivot
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
      int pos = atomic_fetch_add_explicit(&row_counters[0], 1, memory_order_relaxed);
      if (ARG_PARTITION) {
        row_output[pos * out_stride] = i;
      } else {
        row_output[pos * out_stride] = val;
      }
    }
  }
}

// Kernel 4: Output equal elements (second phase)
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
      int pos = less_count + atomic_fetch_add_explicit(&row_counters[1], 1, memory_order_relaxed);
      if (ARG_PARTITION) {
        row_output[pos * out_stride] = i;
      } else {
        row_output[pos * out_stride] = val;
      }
    }
  }
}

// Kernel 5: Output greater elements (third phase)
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
      int pos = less_equal_count + atomic_fetch_add_explicit(&row_counters[2], 1, memory_order_relaxed);
      if (ARG_PARTITION) {
        row_output[pos * out_stride] = i;
      } else {
        row_output[pos * out_stride] = val;
      }
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
              (threadgroup atomic_int*)&shared_hist[digit], 1, memory_order_relaxed);
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
  using SelectKernel =
      RadixSelectSmall<ValT, OutT, ARG_PARTITION, BLOCK_THREADS, ITEMS_PER_THREAD>;
  using UnsignedT = typename SelectKernel::UnsignedT;

  threadgroup UnsignedT shared_keys[SelectKernel::TILE_SIZE];
  threadgroup uint32_t shared_idxs[SelectKernel::TILE_SIZE];
  threadgroup int shared_hist[RADIX_SIZE];
  threadgroup int shared_count[2];

  SelectKernel::partition(
      input, output, kth, size_sorted_axis,
      in_stride_sorted_axis, out_stride_sorted_axis,
      in_stride_segment_axis, out_stride_segment_axis,
      shared_keys, shared_idxs, shared_hist, shared_count, tid, lid);
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
  using SelectKernel =
      RadixSelectSmall<ValT, OutT, ARG_PARTITION, BLOCK_THREADS, ITEMS_PER_THREAD>;
  using UnsignedT = typename SelectKernel::UnsignedT;

  auto in_offset = elem_to_loc(tid.y, nc_shape, in_nc_strides, nc_dim);
  auto out_offset = elem_to_loc(tid.y, nc_shape, out_nc_strides, nc_dim);

  threadgroup UnsignedT shared_keys[SelectKernel::TILE_SIZE];
  threadgroup uint32_t shared_idxs[SelectKernel::TILE_SIZE];
  threadgroup int shared_hist[RADIX_SIZE];
  threadgroup int shared_count[2];

  SelectKernel::partition(
      input + in_offset, output + out_offset, kth, size_sorted_axis,
      in_stride_sorted_axis, out_stride_sorted_axis, 0, 0,
      shared_keys, shared_idxs, shared_hist, shared_count, tid, lid);
}
