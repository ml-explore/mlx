// Copyright © 2025 Apple Inc.

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/fp16_math.cuh"
#include "mlx/backend/cuda/device/radix_select.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

namespace mlx::core {

constexpr int N_PER_THREAD = 8;

namespace cu {

template <typename T>
__device__ __forceinline__ T nan_value();

template <>
__device__ __forceinline__ float nan_value<float>() {
  return cuda::std::numeric_limits<float>::quiet_NaN();
}

template <>
__device__ __forceinline__ double nan_value<double>() {
  return cuda::std::numeric_limits<double>::quiet_NaN();
}

template <>
__device__ __forceinline__ __half nan_value<__half>() {
  return __float2half(cuda::std::numeric_limits<float>::quiet_NaN());
}

template <>
__device__ __forceinline__ __nv_bfloat16 nan_value<__nv_bfloat16>() {
  return __float2bfloat16(cuda::std::numeric_limits<float>::quiet_NaN());
}

template <typename T, typename = void>
struct InitValue {
  __device__ __forceinline__ static T value() {
    return Limits<T>::max();
  }
};

template <typename T>
struct InitValue<T, cuda::std::enable_if_t<std::is_floating_point_v<T>>> {
  __device__ __forceinline__ static T value() {
    return nan_value<T>();
  }
};

template <typename T>
__device__ __forceinline__ void thread_swap(T& a, T& b) {
  T w = a;
  a = b;
  b = w;
}

template <typename T>
struct LessThan {
  __device__ __forceinline__ static T init() {
    return InitValue<T>::value();
  }

  __device__ __forceinline__ bool operator()(T a, T b) const {
    if constexpr (std::is_floating_point_v<T>) {
      bool an = cuda::std::isnan(a);
      bool bn = cuda::std::isnan(b);
      if (an | bn) {
        return (!an) & bn;
      }
    }
    return a < b;
  }
};

template <
    typename ValT,
    typename IdxT,
    bool ARG_SORT,
    int N_PER_THREAD,
    typename CompareOp>
struct ThreadSort {
  __device__ __forceinline__ static void sort(
      ValT (&vals)[N_PER_THREAD],
      IdxT (&idxs)[N_PER_THREAD]) {
    CompareOp op;
#pragma unroll
    for (int i = 0; i < N_PER_THREAD; ++i) {
#pragma unroll
      for (int j = i & 1; j < N_PER_THREAD - 1; j += 2) {
        if (op(vals[j + 1], vals[j])) {
          thread_swap(vals[j + 1], vals[j]);
          if constexpr (ARG_SORT) {
            thread_swap(idxs[j + 1], idxs[j]);
          }
        }
      }
    }
  }
};

template <
    typename ValT,
    typename IdxT,
    bool ARG_SORT,
    int BLOCK_THREADS,
    int N_PER_THREAD,
    typename CompareOp>
struct BlockMergeSort {
  using thread_sort_t =
      ThreadSort<ValT, IdxT, ARG_SORT, N_PER_THREAD, CompareOp>;

  __device__ __forceinline__ static int merge_partition(
      const ValT* As,
      const ValT* Bs,
      int A_sz,
      int B_sz,
      int sort_md) {
    CompareOp op;

    int A_st = max(0, sort_md - B_sz);
    int A_ed = min(sort_md, A_sz);

    while (A_st < A_ed) {
      int md = A_st + (A_ed - A_st) / 2;
      auto a = As[md];
      auto b = Bs[sort_md - 1 - md];

      if (op(b, a)) {
        A_ed = md;
      } else {
        A_st = md + 1;
      }
    }

    return A_ed;
  }

  __device__ __forceinline__ static void merge_step(
      const ValT* As,
      const ValT* Bs,
      const IdxT* As_idx,
      const IdxT* Bs_idx,
      int A_sz,
      int B_sz,
      ValT (&vals)[N_PER_THREAD],
      IdxT (&idxs)[N_PER_THREAD]) {
    CompareOp op;
    int a_idx = 0;
    int b_idx = 0;

#pragma unroll
    for (int i = 0; i < N_PER_THREAD; ++i) {
      auto a = (a_idx < A_sz) ? As[a_idx] : ValT(CompareOp::init());
      auto b = (b_idx < B_sz) ? Bs[b_idx] : ValT(CompareOp::init());
      bool pred = (b_idx < B_sz) && (a_idx >= A_sz || op(b, a));

      vals[i] = pred ? b : a;
      if constexpr (ARG_SORT) {
        if (pred) {
          idxs[i] = Bs_idx[b_idx];
        } else {
          idxs[i] = (a_idx < A_sz) ? As_idx[a_idx] : IdxT(0);
        }
      }

      b_idx += int(pred);
      a_idx += int(!pred);
    }
  }

  __device__ __forceinline__ static void
  sort(ValT* tgp_vals, IdxT* tgp_idxs, int size_sorted_axis) {
    int idx = threadIdx.x * N_PER_THREAD;

    ValT thread_vals[N_PER_THREAD];
    IdxT thread_idxs[N_PER_THREAD];
#pragma unroll
    for (int i = 0; i < N_PER_THREAD; ++i) {
      thread_vals[i] = tgp_vals[idx + i];
      if constexpr (ARG_SORT) {
        thread_idxs[i] = tgp_idxs[idx + i];
      }
    }

    if (idx < size_sorted_axis) {
      thread_sort_t::sort(thread_vals, thread_idxs);
    }

    for (int merge_threads = 2; merge_threads <= BLOCK_THREADS;
         merge_threads *= 2) {
      __syncthreads();
#pragma unroll
      for (int i = 0; i < N_PER_THREAD; ++i) {
        tgp_vals[idx + i] = thread_vals[i];
        if constexpr (ARG_SORT) {
          tgp_idxs[idx + i] = thread_idxs[i];
        }
      }
      __syncthreads();

      int merge_group = threadIdx.x / merge_threads;
      int merge_lane = threadIdx.x % merge_threads;

      int sort_sz = N_PER_THREAD * merge_threads;
      int sort_st = N_PER_THREAD * merge_threads * merge_group;

      int A_st = sort_st;
      int A_ed = sort_st + sort_sz / 2;
      int B_st = sort_st + sort_sz / 2;
      int B_ed = sort_st + sort_sz;

      const ValT* As = tgp_vals + A_st;
      const ValT* Bs = tgp_vals + B_st;
      int A_sz = A_ed - A_st;
      int B_sz = B_ed - B_st;

      int sort_md = N_PER_THREAD * merge_lane;
      int partition = merge_partition(As, Bs, A_sz, B_sz, sort_md);

      As += partition;
      Bs += sort_md - partition;

      A_sz -= partition;
      B_sz -= sort_md - partition;

      const IdxT* As_idx = ARG_SORT ? tgp_idxs + A_st + partition : nullptr;
      const IdxT* Bs_idx =
          ARG_SORT ? tgp_idxs + B_st + sort_md - partition : nullptr;

      merge_step(As, Bs, As_idx, Bs_idx, A_sz, B_sz, thread_vals, thread_idxs);
    }

    __syncthreads();
#pragma unroll
    for (int i = 0; i < N_PER_THREAD; ++i) {
      tgp_vals[idx + i] = thread_vals[i];
      if constexpr (ARG_SORT) {
        tgp_idxs[idx + i] = thread_idxs[i];
      }
    }
  }
};

template <
    typename T,
    typename U,
    bool ARG_SORT,
    int BLOCK_THREADS,
    int N_PER_THREAD,
    typename CompareOp = LessThan<T>>
struct KernelMergeSort {
  using ValT = T;
  using IdxT = uint32_t;
  using block_merge_sort_t = BlockMergeSort<
      ValT,
      IdxT,
      ARG_SORT,
      BLOCK_THREADS,
      N_PER_THREAD,
      CompareOp>;

  static constexpr int N_PER_BLOCK = BLOCK_THREADS * N_PER_THREAD;

  __device__ __forceinline__ static void block_sort(
      const T* inp,
      U* out,
      int size_sorted_axis,
      int64_t in_stride_sorted_axis,
      int64_t out_stride_sorted_axis,
      int64_t in_stride_segment_axis,
      int64_t out_stride_segment_axis,
      ValT* tgp_vals,
      IdxT* tgp_idxs) {
    inp += blockIdx.y * in_stride_segment_axis;
    out += blockIdx.y * out_stride_segment_axis;

    for (int i = threadIdx.x; i < N_PER_BLOCK; i += BLOCK_THREADS) {
      tgp_vals[i] = i < size_sorted_axis ? inp[i * in_stride_sorted_axis]
                                         : ValT(CompareOp::init());
      if constexpr (ARG_SORT) {
        tgp_idxs[i] = i;
      }
    }

    __syncthreads();
    block_merge_sort_t::sort(tgp_vals, tgp_idxs, size_sorted_axis);
    __syncthreads();

    for (int i = threadIdx.x; i < size_sorted_axis; i += BLOCK_THREADS) {
      if constexpr (ARG_SORT) {
        out[i * out_stride_sorted_axis] = tgp_idxs[i];
      } else {
        out[i * out_stride_sorted_axis] = tgp_vals[i];
      }
    }
  }
};

template <
    typename T,
    typename U,
    bool ARG_SORT,
    int BLOCK_THREADS,
    int N_PER_THREAD>
__global__ void block_sort_kernel(
    const T* inp,
    U* out,
    int size_sorted_axis,
    int64_t in_stride_sorted_axis,
    int64_t out_stride_sorted_axis,
    int64_t in_stride_segment_axis,
    int64_t out_stride_segment_axis) {
  using sort_kernel =
      KernelMergeSort<T, U, ARG_SORT, BLOCK_THREADS, N_PER_THREAD>;
  using ValT = typename sort_kernel::ValT;
  using IdxT = typename sort_kernel::IdxT;

  if constexpr (ARG_SORT) {
    __shared__ ValT tgp_vals[sort_kernel::N_PER_BLOCK];
    __shared__ IdxT tgp_idxs[sort_kernel::N_PER_BLOCK];
    sort_kernel::block_sort(
        inp,
        out,
        size_sorted_axis,
        in_stride_sorted_axis,
        out_stride_sorted_axis,
        in_stride_segment_axis,
        out_stride_segment_axis,
        tgp_vals,
        tgp_idxs);
  } else {
    __shared__ ValT tgp_vals[sort_kernel::N_PER_BLOCK];
    sort_kernel::block_sort(
        inp,
        out,
        size_sorted_axis,
        in_stride_sorted_axis,
        out_stride_sorted_axis,
        in_stride_segment_axis,
        out_stride_segment_axis,
        tgp_vals,
        nullptr);
  }
}

template <
    typename T,
    typename U,
    bool ARG_SORT,
    int BLOCK_THREADS,
    int N_PER_THREAD>
__global__ void block_sort_nc_kernel(
    const T* inp,
    U* out,
    int size_sorted_axis,
    int64_t in_stride_sorted_axis,
    int64_t out_stride_sorted_axis,
    const __grid_constant__ Shape nc_shape,
    const __grid_constant__ Strides in_nc_strides,
    const __grid_constant__ Strides out_nc_strides,
    int nc_dim) {
  using sort_kernel =
      KernelMergeSort<T, U, ARG_SORT, BLOCK_THREADS, N_PER_THREAD>;
  using ValT = typename sort_kernel::ValT;
  using IdxT = typename sort_kernel::IdxT;

  int64_t in_block_idx = elem_to_loc(
      int64_t(blockIdx.y), nc_shape.data(), in_nc_strides.data(), nc_dim);
  int64_t out_block_idx = elem_to_loc(
      int64_t(blockIdx.y), nc_shape.data(), out_nc_strides.data(), nc_dim);

  inp += in_block_idx;
  out += out_block_idx;

  if constexpr (ARG_SORT) {
    __shared__ ValT tgp_vals[sort_kernel::N_PER_BLOCK];
    __shared__ IdxT tgp_idxs[sort_kernel::N_PER_BLOCK];
    sort_kernel::block_sort(
        inp,
        out,
        size_sorted_axis,
        in_stride_sorted_axis,
        out_stride_sorted_axis,
        0,
        0,
        tgp_vals,
        tgp_idxs);
  } else {
    __shared__ ValT tgp_vals[sort_kernel::N_PER_BLOCK];
    sort_kernel::block_sort(
        inp,
        out,
        size_sorted_axis,
        in_stride_sorted_axis,
        out_stride_sorted_axis,
        0,
        0,
        tgp_vals,
        nullptr);
  }
}

template <
    typename ValT,
    typename IdxT,
    bool ARG_SORT,
    int BLOCK_THREADS,
    int N_PER_THREAD,
    typename CompareOp = LessThan<ValT>>
struct KernelMultiBlockMergeSort {
  using block_merge_sort_t = BlockMergeSort<
      ValT,
      IdxT,
      ARG_SORT,
      BLOCK_THREADS,
      N_PER_THREAD,
      CompareOp>;

  static constexpr int N_PER_BLOCK = BLOCK_THREADS * N_PER_THREAD;

  __device__ __forceinline__ static void block_sort(
      const ValT* inp,
      ValT* out_vals,
      IdxT* out_idxs,
      int size_sorted_axis,
      int64_t stride_sorted_axis,
      ValT* tgp_vals,
      IdxT* tgp_idxs) {
    int base_idx = blockIdx.x * N_PER_BLOCK;

    for (int i = threadIdx.x; i < N_PER_BLOCK; i += BLOCK_THREADS) {
      int idx = base_idx + i;
      tgp_vals[i] = idx < size_sorted_axis ? inp[idx * stride_sorted_axis]
                                           : ValT(CompareOp::init());
      tgp_idxs[i] = idx;
    }

    __syncthreads();
    block_merge_sort_t::sort(tgp_vals, tgp_idxs, size_sorted_axis);
    __syncthreads();

    for (int i = threadIdx.x; i < N_PER_BLOCK; i += BLOCK_THREADS) {
      int idx = base_idx + i;
      if (idx < size_sorted_axis) {
        out_vals[idx] = tgp_vals[i];
        out_idxs[idx] = tgp_idxs[i];
      }
    }
  }

  __device__ __forceinline__ static int merge_partition(
      const ValT* As,
      const ValT* Bs,
      int A_sz,
      int B_sz,
      int sort_md) {
    CompareOp op;

    int A_st = max(0, sort_md - B_sz);
    int A_ed = min(sort_md, A_sz);

    while (A_st < A_ed) {
      int md = A_st + (A_ed - A_st) / 2;
      auto a = As[md];
      auto b = Bs[sort_md - 1 - md];

      if (op(b, a)) {
        A_ed = md;
      } else {
        A_st = md + 1;
      }
    }

    return A_ed;
  }
};

template <
    typename ValT,
    typename IdxT,
    bool ARG_SORT,
    int BLOCK_THREADS,
    int N_PER_THREAD>
__global__ void mb_block_sort_kernel(
    const ValT* inp,
    ValT* out_vals,
    IdxT* out_idxs,
    int size_sorted_axis,
    int64_t stride_sorted_axis,
    const __grid_constant__ Shape nc_shape,
    const __grid_constant__ Strides nc_strides,
    int nc_dim) {
  using sort_kernel = KernelMultiBlockMergeSort<
      ValT,
      IdxT,
      ARG_SORT,
      BLOCK_THREADS,
      N_PER_THREAD>;

  int64_t block_idx = elem_to_loc(
      int64_t(blockIdx.y), nc_shape.data(), nc_strides.data(), nc_dim);

  inp += block_idx;
  out_vals += blockIdx.y * size_sorted_axis;
  out_idxs += blockIdx.y * size_sorted_axis;

  __shared__ ValT tgp_vals[sort_kernel::N_PER_BLOCK];
  __shared__ IdxT tgp_idxs[sort_kernel::N_PER_BLOCK];

  sort_kernel::block_sort(
      inp,
      out_vals,
      out_idxs,
      size_sorted_axis,
      stride_sorted_axis,
      tgp_vals,
      tgp_idxs);
}

template <
    typename ValT,
    typename IdxT,
    bool ARG_SORT,
    int BLOCK_THREADS,
    int N_PER_THREAD>
__global__ void mb_block_partition_kernel(
    IdxT* block_partitions,
    const ValT* dev_vals,
    const IdxT* dev_idxs,
    int size_sorted_axis,
    int merge_tiles,
    int n_blocks) {
  using sort_kernel = KernelMultiBlockMergeSort<
      ValT,
      IdxT,
      ARG_SORT,
      BLOCK_THREADS,
      N_PER_THREAD>;

  (void)dev_idxs;

  block_partitions += blockIdx.y * blockDim.x;
  dev_vals += blockIdx.y * size_sorted_axis;
  dev_idxs += blockIdx.y * size_sorted_axis;

  for (int i = threadIdx.x; i <= n_blocks; i += blockDim.x) {
    int merge_group = i / merge_tiles;
    int merge_lane = i % merge_tiles;

    int sort_sz = sort_kernel::N_PER_BLOCK * merge_tiles;
    int sort_st = sort_kernel::N_PER_BLOCK * merge_tiles * merge_group;

    int A_st = min(size_sorted_axis, sort_st);
    int A_ed = min(size_sorted_axis, sort_st + sort_sz / 2);
    int B_st = A_ed;
    int B_ed = min(size_sorted_axis, B_st + sort_sz / 2);

    int partition_at = min(B_ed - A_st, sort_kernel::N_PER_BLOCK * merge_lane);
    int partition = sort_kernel::merge_partition(
        dev_vals + A_st,
        dev_vals + B_st,
        A_ed - A_st,
        B_ed - B_st,
        partition_at);

    block_partitions[i] = A_st + partition;
  }
}

template <
    typename ValT,
    typename IdxT,
    bool ARG_SORT,
    int BLOCK_THREADS,
    int N_PER_THREAD,
    typename CompareOp = LessThan<ValT>>
__global__ void mb_block_merge_kernel(
    const IdxT* block_partitions,
    const ValT* dev_vals_in,
    const IdxT* dev_idxs_in,
    ValT* dev_vals_out,
    IdxT* dev_idxs_out,
    int size_sorted_axis,
    int merge_tiles,
    int num_tiles) {
  using sort_kernel = KernelMultiBlockMergeSort<
      ValT,
      IdxT,
      ARG_SORT,
      BLOCK_THREADS,
      N_PER_THREAD,
      CompareOp>;

  using block_sort_t = typename sort_kernel::block_merge_sort_t;

  block_partitions += blockIdx.y * (num_tiles + 1);
  dev_vals_in += blockIdx.y * size_sorted_axis;
  dev_idxs_in += blockIdx.y * size_sorted_axis;
  dev_vals_out += blockIdx.y * size_sorted_axis;
  dev_idxs_out += blockIdx.y * size_sorted_axis;

  int block_idx = blockIdx.x;
  int merge_group = block_idx / merge_tiles;
  int sort_st = sort_kernel::N_PER_BLOCK * merge_tiles * merge_group;
  int sort_sz = sort_kernel::N_PER_BLOCK * merge_tiles;
  int sort_md = sort_kernel::N_PER_BLOCK * block_idx - sort_st;

  int A_st = block_partitions[block_idx + 0];
  int A_ed = block_partitions[block_idx + 1];
  int B_st = min(size_sorted_axis, 2 * sort_st + sort_sz / 2 + sort_md - A_st);
  int B_ed = min(
      size_sorted_axis,
      2 * sort_st + sort_sz / 2 + sort_md + sort_kernel::N_PER_BLOCK - A_ed);

  if ((block_idx % merge_tiles) == merge_tiles - 1) {
    A_ed = min(size_sorted_axis, sort_st + sort_sz / 2);
    B_ed = min(size_sorted_axis, sort_st + sort_sz);
  }

  int A_sz = A_ed - A_st;
  int B_sz = B_ed - B_st;

  ValT thread_vals[N_PER_THREAD];
  IdxT thread_idxs[N_PER_THREAD];
#pragma unroll
  for (int i = 0; i < N_PER_THREAD; i++) {
    int idx = BLOCK_THREADS * i + threadIdx.x;
    if (idx < (A_sz + B_sz)) {
      thread_vals[i] = (idx < A_sz) ? dev_vals_in[A_st + idx]
                                    : dev_vals_in[B_st + idx - A_sz];
      thread_idxs[i] = (idx < A_sz) ? dev_idxs_in[A_st + idx]
                                    : dev_idxs_in[B_st + idx - A_sz];
    } else {
      thread_vals[i] = CompareOp::init();
      thread_idxs[i] = 0;
    }
  }

  __shared__ ValT tgp_vals[sort_kernel::N_PER_BLOCK];
  __shared__ IdxT tgp_idxs[sort_kernel::N_PER_BLOCK];
  __syncthreads();
#pragma unroll
  for (int i = 0; i < N_PER_THREAD; i++) {
    int idx = BLOCK_THREADS * i + threadIdx.x;
    tgp_vals[idx] = thread_vals[i];
    tgp_idxs[idx] = thread_idxs[i];
  }
  __syncthreads();

  int sort_md_local = min(A_sz + B_sz, N_PER_THREAD * int(threadIdx.x));

  int A_st_local = block_sort_t::merge_partition(
      tgp_vals, tgp_vals + A_sz, A_sz, B_sz, sort_md_local);
  int A_ed_local = A_sz;

  int B_st_local = sort_md_local - A_st_local;
  int B_ed_local = B_sz;

  int A_sz_local = A_ed_local - A_st_local;
  int B_sz_local = B_ed_local - B_st_local;

  block_sort_t::merge_step(
      tgp_vals + A_st_local,
      tgp_vals + A_ed_local + B_st_local,
      tgp_idxs + A_st_local,
      tgp_idxs + A_ed_local + B_st_local,
      A_sz_local,
      B_sz_local,
      thread_vals,
      thread_idxs);

  __syncthreads();
#pragma unroll
  for (int i = 0; i < N_PER_THREAD; ++i) {
    int idx = threadIdx.x * N_PER_THREAD;
    tgp_vals[idx + i] = thread_vals[i];
    tgp_idxs[idx + i] = thread_idxs[i];
  }

  __syncthreads();
  int base_idx = blockIdx.x * sort_kernel::N_PER_BLOCK;
  for (int i = threadIdx.x; i < sort_kernel::N_PER_BLOCK; i += BLOCK_THREADS) {
    int idx = base_idx + i;
    if (idx < size_sorted_axis) {
      dev_vals_out[idx] = tgp_vals[i];
      dev_idxs_out[idx] = tgp_idxs[i];
    }
  }
}

} // namespace cu

///////////////////////////////////////////////////////////////////////////////
// Radix Select Implementation
//
// This implements an optimized radix-based selection algorithm for partition
// operations. Instead of fully sorting, we find the kth element using radix
// selection which is O(n) expected time.
//
// Algorithm:
// 1. Build histogram of current digit (8 bits at a time)
// 2. Find which bin contains the kth element via prefix sum
// 3. Filter candidates to only those in the target bin
// 4. Repeat until pivot is found
// 5. Final pass: collect all elements that should be in top-k
///////////////////////////////////////////////////////////////////////////////

namespace cu {

// Configuration for radix select
constexpr int RADIX_BLOCK_THREADS = 256;
constexpr int RADIX_ITEMS_PER_THREAD = 8;

///////////////////////////////////////////////////////////////////////////////
// Kernel 1: Count histogram for each row
// Each block processes one row, builds histogram in shared memory
///////////////////////////////////////////////////////////////////////////////

template <typename ValT, bool SELECT_LARGEST, int BLOCK_THREADS>
__global__ void radix_histogram_kernel(
    const ValT* __restrict__ input,
    int* __restrict__ histograms, // [n_rows, RADIX_SIZE]
    const int* __restrict__ candidate_counts, // [n_rows] - number of candidates per row
    const int* __restrict__ candidate_offsets, // [n_rows] - offset into candidates array
    const ValT* __restrict__ candidates, // candidates array (or nullptr for first pass)
    const uint32_t* __restrict__ candidate_indices, // indices of candidates
    int size_sorted_axis,
    int64_t stride_sorted_axis,
    int start_bit) {
  using Traits = RadixTraits<ValT>;
  using UnsignedT = typename Traits::UnsignedT;

  __shared__ int shared_hist[RADIX_SIZE];

  // Initialize shared histogram
  for (int i = threadIdx.x; i < RADIX_SIZE; i += BLOCK_THREADS) {
    shared_hist[i] = 0;
  }
  __syncthreads();

  int row = blockIdx.y;
  int n_candidates =
      (candidates == nullptr) ? size_sorted_axis : candidate_counts[row];
  int offset = (candidates == nullptr) ? 0 : candidate_offsets[row];

  // Each thread processes multiple elements
  for (int i = threadIdx.x; i < n_candidates; i += BLOCK_THREADS) {
    ValT val;
    if (candidates == nullptr) {
      // First pass: read from input
      val = input[row * stride_sorted_axis * size_sorted_axis +
                  i * stride_sorted_axis];
    } else {
      // Subsequent passes: read from candidates
      val = candidates[offset + i];
    }

    // Handle NaN: place at end for ascending, beginning for descending
    if (!is_nan_value(val)) {
      UnsignedT radix_val = Traits::to_radix(val);
      // For SELECT_LARGEST, we want descending order, so flip bits
      if constexpr (SELECT_LARGEST) {
        radix_val = ~radix_val;
      }
      int digit = extract_digit(radix_val, start_bit, RADIX_BITS);
      atomicAdd(&shared_hist[digit], 1);
    }
  }

  __syncthreads();

  // Write histogram to global memory
  int* row_hist = histograms + row * RADIX_SIZE;
  for (int i = threadIdx.x; i < RADIX_SIZE; i += BLOCK_THREADS) {
    row_hist[i] = shared_hist[i];
  }
}

///////////////////////////////////////////////////////////////////////////////
// Kernel 2: Find target bin and update k for each row
// Single block processes all rows
///////////////////////////////////////////////////////////////////////////////

template <int BLOCK_THREADS>
__global__ void radix_find_bin_kernel(
    const int* __restrict__ histograms, // [n_rows, RADIX_SIZE]
    int* __restrict__ target_bins, // [n_rows] - output: which bin contains kth
    int* __restrict__ new_ks, // [n_rows] - output: new k within target bin
    const int* __restrict__ ks, // [n_rows] - current k values
    int n_rows) {
  for (int row = blockIdx.x * BLOCK_THREADS + threadIdx.x; row < n_rows;
       row += gridDim.x * BLOCK_THREADS) {
    const int* row_hist = histograms + row * RADIX_SIZE;
    int k = ks[row];

    // Prefix sum to find target bin
    int cumsum = 0;
    int target_bin = 0;
    for (int bin = 0; bin < RADIX_SIZE; bin++) {
      int count = row_hist[bin];
      if (cumsum + count >= k) {
        target_bin = bin;
        break;
      }
      cumsum += count;
    }

    target_bins[row] = target_bin;
    new_ks[row] = k - cumsum; // k within the target bin
  }
}

///////////////////////////////////////////////////////////////////////////////
// Kernel 3: Filter candidates to target bin
///////////////////////////////////////////////////////////////////////////////

template <typename ValT, bool SELECT_LARGEST, int BLOCK_THREADS>
__global__ void radix_filter_kernel(
    const ValT* __restrict__ input,
    const uint32_t* __restrict__ input_indices,
    ValT* __restrict__ output,
    uint32_t* __restrict__ output_indices,
    int* __restrict__ output_counts, // [n_rows] - atomic counter
    const int* __restrict__ candidate_counts,
    const int* __restrict__ candidate_offsets,
    const ValT* __restrict__ candidates,
    const uint32_t* __restrict__ candidate_indices_in,
    const int* __restrict__ target_bins,
    int size_sorted_axis,
    int64_t stride_sorted_axis,
    int start_bit,
    int max_output_per_row) {
  using Traits = RadixTraits<ValT>;
  using UnsignedT = typename Traits::UnsignedT;

  // Shared memory write buffer for coalesced writes
  __shared__ ValT shared_vals[BLOCK_THREADS * 2];
  __shared__ uint32_t shared_idxs[BLOCK_THREADS * 2];
  __shared__ int shared_count;
  __shared__ int shared_base;

  int row = blockIdx.y;
  int target_bin = target_bins[row];
  int n_candidates =
      (candidates == nullptr) ? size_sorted_axis : candidate_counts[row];
  int in_offset = (candidates == nullptr) ? 0 : candidate_offsets[row];
  int out_offset = row * max_output_per_row;

  if (threadIdx.x == 0) {
    shared_count = 0;
  }
  __syncthreads();

  // Process elements
  for (int base = 0; base < n_candidates; base += BLOCK_THREADS) {
    int i = base + threadIdx.x;
    bool valid = i < n_candidates;

    ValT val;
    uint32_t idx;
    int digit = -1;

    if (valid) {
      if (candidates == nullptr) {
        val = input[row * stride_sorted_axis * size_sorted_axis +
                    i * stride_sorted_axis];
        idx = i;
      } else {
        val = candidates[in_offset + i];
        idx = candidate_indices_in[in_offset + i];
      }

      if (!is_nan_value(val)) {
        UnsignedT radix_val = Traits::to_radix(val);
        if constexpr (SELECT_LARGEST) {
          radix_val = ~radix_val;
        }
        digit = extract_digit(radix_val, start_bit, RADIX_BITS);
      }
    }

    // Check if this element belongs to target bin
    bool in_target = valid && (digit == target_bin);

    // Count elements going to buffer
    int local_pos = -1;
    if (in_target) {
      local_pos = atomicAdd(&shared_count, 1);
    }
    __syncthreads();

    // Write to shared buffer
    if (in_target && local_pos < BLOCK_THREADS * 2) {
      shared_vals[local_pos] = val;
      shared_idxs[local_pos] = idx;
    }
    __syncthreads();

    // Flush buffer if needed
    int count = shared_count;
    if (count >= BLOCK_THREADS) {
      // Get global position
      if (threadIdx.x == 0) {
        shared_base = atomicAdd(&output_counts[row], count);
        shared_count = 0;
      }
      __syncthreads();

      int global_base = shared_base;
      // Write out
      for (int j = threadIdx.x; j < count; j += BLOCK_THREADS) {
        int out_idx = out_offset + global_base + j;
        output[out_idx] = shared_vals[j];
        output_indices[out_idx] = shared_idxs[j];
      }
      __syncthreads();
    }
  }

  // Final flush
  __syncthreads();
  int count = shared_count;
  if (count > 0) {
    if (threadIdx.x == 0) {
      shared_base = atomicAdd(&output_counts[row], count);
    }
    __syncthreads();

    int global_base = shared_base;
    for (int j = threadIdx.x; j < count; j += BLOCK_THREADS) {
      int out_idx = out_offset + global_base + j;
      output[out_idx] = shared_vals[j];
      output_indices[out_idx] = shared_idxs[j];
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Kernel 4: Final collection - gather top-k elements
// After finding pivot, collect all elements that belong in top-k
///////////////////////////////////////////////////////////////////////////////

template <typename ValT, typename OutT, bool ARG_PARTITION, bool SELECT_LARGEST, int BLOCK_THREADS>
__global__ void radix_collect_topk_kernel(
    const ValT* __restrict__ input,
    OutT* __restrict__ output,
    const ValT* __restrict__ pivots, // [n_rows] - the kth element for each row
    const int* __restrict__ ks, // [n_rows] - k values
    int* __restrict__ output_counts, // [n_rows] - atomic counters
    int size_sorted_axis,
    int64_t in_stride_sorted_axis,
    int64_t out_stride_sorted_axis,
    int64_t in_stride_segment_axis,
    int64_t out_stride_segment_axis) {
  using Traits = RadixTraits<ValT>;
  using UnsignedT = typename Traits::UnsignedT;

  __shared__ int shared_count;
  __shared__ ValT shared_vals[BLOCK_THREADS];
  __shared__ uint32_t shared_idxs[BLOCK_THREADS];

  int row = blockIdx.y;
  ValT pivot = pivots[row];
  int k = ks[row];
  UnsignedT pivot_radix = Traits::to_radix(pivot);

  const ValT* row_input = input + row * in_stride_segment_axis;
  OutT* row_output = output + row * out_stride_segment_axis;

  if (threadIdx.x == 0) {
    shared_count = 0;
  }
  __syncthreads();

  // First pass: collect elements strictly greater/less than pivot
  for (int base = 0; base < size_sorted_axis; base += BLOCK_THREADS) {
    int i = base + threadIdx.x;
    bool valid = i < size_sorted_axis;

    ValT val;
    bool should_output = false;

    if (valid) {
      val = row_input[i * in_stride_sorted_axis];
      if (!is_nan_value(val)) {
        UnsignedT val_radix = Traits::to_radix(val);
        if constexpr (SELECT_LARGEST) {
          should_output = val_radix > pivot_radix;
        } else {
          should_output = val_radix < pivot_radix;
        }
      }
    }

    // Warp-level aggregation
    unsigned int mask = __ballot_sync(0xFFFFFFFF, should_output);
    int warp_count = __popc(mask);
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Get position within warp
    unsigned int lower_mask = (1u << lane_id) - 1;
    int pos_in_warp = __popc(mask & lower_mask);

    // First lane of each warp reserves space
    int warp_base = 0;
    if (lane_id == 0 && warp_count > 0) {
      warp_base = atomicAdd(&shared_count, warp_count);
    }
    warp_base = __shfl_sync(0xFFFFFFFF, warp_base, 0);

    // Write to shared buffer
    if (should_output) {
      int local_pos = warp_base + pos_in_warp;
      if (local_pos < BLOCK_THREADS) {
        shared_vals[local_pos] = val;
        shared_idxs[local_pos] = i;
      }
    }
    __syncthreads();

    // Flush if buffer is getting full
    int count = shared_count;
    if (count >= BLOCK_THREADS / 2) {
      // Write to output
      for (int j = threadIdx.x; j < count; j += BLOCK_THREADS) {
        int out_pos = atomicAdd(&output_counts[row], 1);
        if (out_pos < k) {
          if constexpr (ARG_PARTITION) {
            row_output[out_pos * out_stride_sorted_axis] = shared_idxs[j];
          } else {
            row_output[out_pos * out_stride_sorted_axis] = shared_vals[j];
          }
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        shared_count = 0;
      }
      __syncthreads();
    }
  }

  // Flush remaining
  __syncthreads();
  int count = shared_count;
  for (int j = threadIdx.x; j < count; j += BLOCK_THREADS) {
    int out_pos = atomicAdd(&output_counts[row], 1);
    if (out_pos < k) {
      if constexpr (ARG_PARTITION) {
        row_output[out_pos * out_stride_sorted_axis] = shared_idxs[j];
      } else {
        row_output[out_pos * out_stride_sorted_axis] = shared_vals[j];
      }
    }
  }
  __syncthreads();

  // Second pass: fill remaining slots with elements equal to pivot
  if (threadIdx.x == 0) {
    shared_count = 0;
  }
  __syncthreads();

  for (int base = 0; base < size_sorted_axis; base += BLOCK_THREADS) {
    int i = base + threadIdx.x;
    bool valid = i < size_sorted_axis;

    ValT val;
    bool is_equal = false;

    if (valid) {
      val = row_input[i * in_stride_sorted_axis];
      if (!is_nan_value(val)) {
        UnsignedT val_radix = Traits::to_radix(val);
        is_equal = (val_radix == pivot_radix);
      }
    }

    if (is_equal) {
      int out_pos = atomicAdd(&output_counts[row], 1);
      if (out_pos < k) {
        if constexpr (ARG_PARTITION) {
          row_output[out_pos * out_stride_sorted_axis] = i;
        } else {
          row_output[out_pos * out_stride_sorted_axis] = val;
        }
      }
    }
    __syncthreads();

    // Early exit if we have enough
    if (output_counts[row] >= k) {
      break;
    }
  }

  // Fill remaining with elements after k (for partition semantics)
  // The partition operation should have all elements, not just top-k
  __syncthreads();
  int current_count = output_counts[row];

  for (int base = 0; base < size_sorted_axis && current_count < size_sorted_axis;
       base += BLOCK_THREADS) {
    int i = base + threadIdx.x;
    bool valid = i < size_sorted_axis;

    ValT val;
    bool should_add = false;

    if (valid) {
      val = row_input[i * in_stride_sorted_axis];
      if (!is_nan_value(val)) {
        UnsignedT val_radix = Traits::to_radix(val);
        if constexpr (SELECT_LARGEST) {
          should_add = val_radix < pivot_radix;
        } else {
          should_add = val_radix > pivot_radix;
        }
      } else {
        // NaN goes at the end
        should_add = true;
      }
    }

    if (should_add) {
      int out_pos = atomicAdd(&output_counts[row], 1);
      if (out_pos < size_sorted_axis) {
        if constexpr (ARG_PARTITION) {
          row_output[out_pos * out_stride_sorted_axis] = i;
        } else {
          row_output[out_pos * out_stride_sorted_axis] = val;
        }
      }
    }
    __syncthreads();
    current_count = output_counts[row];
  }
}

///////////////////////////////////////////////////////////////////////////////
// Simplified single-pass radix select for small arrays
// Uses block-level sorting when array fits in shared memory
///////////////////////////////////////////////////////////////////////////////

template <
    typename ValT,
    typename OutT,
    bool ARG_PARTITION,
    bool SELECT_LARGEST,
    int BLOCK_THREADS,
    int ITEMS_PER_THREAD>
__global__ void radix_select_small_kernel(
    const ValT* __restrict__ input,
    OutT* __restrict__ output,
    int kth,
    int size_sorted_axis,
    int64_t in_stride_sorted_axis,
    int64_t out_stride_sorted_axis,
    int64_t in_stride_segment_axis,
    int64_t out_stride_segment_axis) {
  using Traits = RadixTraits<ValT>;
  using UnsignedT = typename Traits::UnsignedT;

  constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

  __shared__ UnsignedT shared_keys[TILE_SIZE];
  __shared__ uint32_t shared_idxs[TILE_SIZE];
  __shared__ int shared_hist[RADIX_SIZE];

  int row = blockIdx.y;
  const ValT* row_input = input + row * in_stride_segment_axis;
  OutT* row_output = output + row * out_stride_segment_axis;

  int n = min(size_sorted_axis, TILE_SIZE);

  // Load data into shared memory
  for (int i = threadIdx.x; i < TILE_SIZE; i += BLOCK_THREADS) {
    if (i < n) {
      ValT val = row_input[i * in_stride_sorted_axis];
      UnsignedT key = Traits::to_radix(val);
      if constexpr (SELECT_LARGEST) {
        key = ~key; // Flip for descending order
      }
      // Handle NaN by placing at end
      if (is_nan_value(val)) {
        key = ~UnsignedT(0); // Max value = end
      }
      shared_keys[i] = key;
      shared_idxs[i] = i;
    } else {
      shared_keys[i] = ~UnsignedT(0); // Padding
      shared_idxs[i] = i;
    }
  }
  __syncthreads();

  // Radix select: iterate through digits from MSB to LSB
  int k = kth + 1; // Convert 0-indexed kth to 1-indexed k
  constexpr int NUM_PASSES = (Traits::BITS + RADIX_BITS - 1) / RADIX_BITS;

  UnsignedT prefix_mask = 0;
  int remaining = n;

  for (int pass = NUM_PASSES - 1; pass >= 0 && remaining > 1; pass--) {
    int start_bit = pass * RADIX_BITS;

    // Build histogram
    for (int i = threadIdx.x; i < RADIX_SIZE; i += BLOCK_THREADS) {
      shared_hist[i] = 0;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += BLOCK_THREADS) {
      UnsignedT key = shared_keys[i];
      // Only count if key matches prefix so far
      if ((key & prefix_mask) == (shared_keys[0] & prefix_mask) || prefix_mask == 0) {
        int digit = extract_digit(key, start_bit, RADIX_BITS);
        atomicAdd(&shared_hist[digit], 1);
      }
    }
    __syncthreads();

    // Find target bin via prefix sum
    int target_bin = 0;
    int cumsum = 0;
    for (int bin = 0; bin < RADIX_SIZE; bin++) {
      if (cumsum + shared_hist[bin] >= k) {
        target_bin = bin;
        k = k - cumsum;
        break;
      }
      cumsum += shared_hist[bin];
    }

    // Update prefix mask
    UnsignedT digit_mask = UnsignedT((1 << RADIX_BITS) - 1) << start_bit;
    UnsignedT target_prefix = UnsignedT(target_bin) << start_bit;
    prefix_mask |= digit_mask;

    // Count remaining candidates
    remaining = shared_hist[target_bin];
    __syncthreads();
  }

  // Now we have the pivot - it's the element with the matching prefix
  // Output the partitioned array
  __shared__ int out_count;
  if (threadIdx.x == 0) {
    out_count = 0;
  }
  __syncthreads();

  // Find pivot value
  UnsignedT pivot_key = 0;
  for (int i = 0; i < n; i++) {
    if ((shared_keys[i] & prefix_mask) == (prefix_mask & shared_keys[i])) {
      // This is a candidate for pivot
      // The actual pivot is the k-th one among candidates
      // For simplicity, we'll use the first match as pivot approximation
      pivot_key = shared_keys[i];
      break;
    }
  }
  __syncthreads();

  // Output elements: first those < pivot (or > for largest), then pivot, then rest
  // For partition semantics, we output all elements with proper ordering

  // Phase 1: Elements that should come before pivot
  for (int i = threadIdx.x; i < n; i += BLOCK_THREADS) {
    UnsignedT key = shared_keys[i];
    if (key < pivot_key) {
      int pos = atomicAdd(&out_count, 1);
      if constexpr (ARG_PARTITION) {
        row_output[pos * out_stride_sorted_axis] = shared_idxs[i];
      } else {
        ValT val = row_input[shared_idxs[i] * in_stride_sorted_axis];
        row_output[pos * out_stride_sorted_axis] = val;
      }
    }
  }
  __syncthreads();

  // Phase 2: Elements equal to pivot
  for (int i = threadIdx.x; i < n; i += BLOCK_THREADS) {
    UnsignedT key = shared_keys[i];
    if (key == pivot_key) {
      int pos = atomicAdd(&out_count, 1);
      if constexpr (ARG_PARTITION) {
        row_output[pos * out_stride_sorted_axis] = shared_idxs[i];
      } else {
        ValT val = row_input[shared_idxs[i] * in_stride_sorted_axis];
        row_output[pos * out_stride_sorted_axis] = val;
      }
    }
  }
  __syncthreads();

  // Phase 3: Elements that should come after pivot
  for (int i = threadIdx.x; i < n; i += BLOCK_THREADS) {
    UnsignedT key = shared_keys[i];
    if (key > pivot_key) {
      int pos = atomicAdd(&out_count, 1);
      if constexpr (ARG_PARTITION) {
        row_output[pos * out_stride_sorted_axis] = shared_idxs[i];
      } else {
        ValT val = row_input[shared_idxs[i] * in_stride_sorted_axis];
        row_output[pos * out_stride_sorted_axis] = val;
      }
    }
  }
}

} // namespace cu

namespace {

void single_block_sort(
    const Stream& s,
    const array& in,
    array& out,
    int axis,
    int bn,
    bool argsort) {
  int n_rows = in.size() / in.shape(axis);

  auto in_nc_str = in.strides();
  in_nc_str.erase(in_nc_str.begin() + axis);

  auto out_nc_str = out.strides();
  out_nc_str.erase(out_nc_str.begin() + axis);

  auto nc_shape = in.shape();
  nc_shape.erase(nc_shape.begin() + axis);

  int nc_dim = nc_shape.size();

  int size_sorted_axis = in.shape(axis);
  int64_t in_stride_sorted_axis = in.strides()[axis];
  int64_t out_stride_sorted_axis = out.strides()[axis];

  bool contiguous = in.flags().contiguous;
  auto check_strides = [](const array& x, int64_t sort_stride) {
    int64_t min_stride =
        *std::min_element(x.strides().begin(), x.strides().end());
    int64_t max_stride =
        *std::max_element(x.strides().begin(), x.strides().end());
    return sort_stride == min_stride || sort_stride == max_stride;
  };
  contiguous &= check_strides(in, in_stride_sorted_axis);
  contiguous &= check_strides(out, out_stride_sorted_axis);

  auto& encoder = cu::get_command_encoder(s);
  out.set_data(cu::malloc_async(out.nbytes(), encoder));
  encoder.set_input_array(in);
  encoder.set_output_array(out);

  dispatch_all_types(in.dtype(), [&](auto type_tag) {
    using CTYPE = MLX_GET_TYPE(type_tag);
    if constexpr (!std::is_same_v<CTYPE, complex64_t>) {
      using ValT = cuda_type_t<CTYPE>;
      dispatch_block_dim(bn, [&](auto block_dim) {
        constexpr int BLOCK_THREADS = block_dim();
        if constexpr (BLOCK_THREADS < 1024) {
          dim3 grid(1, n_rows, 1);
          dim3 block(BLOCK_THREADS, 1, 1);

          dispatch_bool(argsort, [&](auto arg_tag) {
            constexpr bool ARG_SORT = decltype(arg_tag)::value;
            using OutT = std::conditional_t<ARG_SORT, uint32_t, ValT>;

            if (contiguous) {
              auto kernel = cu::block_sort_kernel<
                  ValT,
                  OutT,
                  ARG_SORT,
                  BLOCK_THREADS,
                  N_PER_THREAD>;
              int64_t in_stride_segment_axis = INT64_MAX;
              int64_t out_stride_segment_axis = INT64_MAX;
              for (int i = 0; i < nc_shape.size(); i++) {
                if (nc_shape[i] == 1) {
                  continue;
                }
                if (in_nc_str[i] > INT32_MAX || out_nc_str[i] > INT32_MAX) {
                  throw std::runtime_error(
                      "[Sort::eval_gpu] Stride too large.");
                }
                in_stride_segment_axis =
                    std::min(in_stride_segment_axis, in_nc_str[i]);
                out_stride_segment_axis =
                    std::min(out_stride_segment_axis, out_nc_str[i]);
              }
              encoder.add_kernel_node(
                  kernel,
                  grid,
                  block,
                  0,
                  gpu_ptr<ValT>(in),
                  gpu_ptr<OutT>(out),
                  size_sorted_axis,
                  in_stride_sorted_axis,
                  out_stride_sorted_axis,
                  in_stride_segment_axis,
                  out_stride_segment_axis);
            } else {
              auto kernel = cu::block_sort_nc_kernel<
                  ValT,
                  OutT,
                  ARG_SORT,
                  BLOCK_THREADS,
                  N_PER_THREAD>;
              auto nc_shape_param = const_param(nc_shape);
              auto in_nc_strides_param = const_param(in_nc_str);
              auto out_nc_strides_param = const_param(out_nc_str);
              encoder.add_kernel_node(
                  kernel,
                  grid,
                  block,
                  0,
                  gpu_ptr<ValT>(in),
                  gpu_ptr<OutT>(out),
                  size_sorted_axis,
                  in_stride_sorted_axis,
                  out_stride_sorted_axis,
                  nc_shape_param,
                  in_nc_strides_param,
                  out_nc_strides_param,
                  nc_dim);
            }
          });
        }
      });
    } else {
      throw std::runtime_error(
          "CUDA backend does not support sorting complex numbers");
    }
  });
}

void multi_block_sort(
    const Stream& s,
    const array& in,
    array& out,
    int axis,
    int n_blocks,
    bool argsort) {
  int n_rows = in.size() / in.shape(axis);

  auto nc_str = in.strides();
  nc_str.erase(nc_str.begin() + axis);

  auto nc_shape = in.shape();
  nc_shape.erase(nc_shape.begin() + axis);

  int nc_dim = nc_shape.size();

  if (nc_dim == 0) {
    nc_shape = {0};
    nc_str = {1};
  }

  int size_sorted_axis = in.shape(axis);
  int64_t stride_sorted_axis = in.strides()[axis];

  array dev_vals_in({n_rows, size_sorted_axis}, in.dtype(), nullptr, {});
  array dev_vals_out({n_rows, size_sorted_axis}, in.dtype(), nullptr, {});

  array dev_idxs_in({n_rows, size_sorted_axis}, uint32, nullptr, {});
  array dev_idxs_out({n_rows, size_sorted_axis}, uint32, nullptr, {});

  array block_partitions({n_rows, n_blocks + 1}, uint32, nullptr, {});

  auto& encoder = cu::get_command_encoder(s);

  dev_vals_in.set_data(cu::malloc_async(dev_vals_in.nbytes(), encoder));
  dev_vals_out.set_data(cu::malloc_async(dev_vals_out.nbytes(), encoder));
  dev_idxs_in.set_data(cu::malloc_async(dev_idxs_in.nbytes(), encoder));
  dev_idxs_out.set_data(cu::malloc_async(dev_idxs_out.nbytes(), encoder));
  block_partitions.set_data(
      cu::malloc_async(block_partitions.nbytes(), encoder));

  encoder.add_temporary(block_partitions);

  dispatch_all_types(in.dtype(), [&](auto type_tag) {
    using CTYPE = MLX_GET_TYPE(type_tag);
    if constexpr (!std::is_same_v<CTYPE, complex64_t>) {
      using ValT = cuda_type_t<CTYPE>;
      using IdxT = uint32_t;
      constexpr int BLOCK_THREADS = sizeof(ValT) == 8 ? 256 : 512;
      dim3 grid(n_blocks, n_rows, 1);
      dim3 block(BLOCK_THREADS, 1, 1);

      dispatch_bool(argsort, [&](auto arg_tag) {
        constexpr bool ARG_SORT = decltype(arg_tag)::value;
        auto nc_shape_param = const_param(nc_shape);
        auto nc_strides_param = const_param(nc_str);

        auto block_sort_kernel = cu::mb_block_sort_kernel<
            ValT,
            IdxT,
            ARG_SORT,
            BLOCK_THREADS,
            N_PER_THREAD>;
        encoder.set_input_array(in);
        encoder.set_output_array(dev_vals_in);
        encoder.set_output_array(dev_idxs_in);
        encoder.add_kernel_node(
            block_sort_kernel,
            grid,
            block,
            0,
            gpu_ptr<ValT>(in),
            gpu_ptr<ValT>(dev_vals_in),
            gpu_ptr<IdxT>(dev_idxs_in),
            size_sorted_axis,
            stride_sorted_axis,
            nc_shape_param,
            nc_strides_param,
            nc_dim);

        int n_thr_per_group = (n_blocks + 1) < 1024 ? (n_blocks + 1) : 1024;

        for (int merge_tiles = 2; (merge_tiles / 2) < n_blocks;
             merge_tiles *= 2) {
          auto partition_kernel = cu::mb_block_partition_kernel<
              ValT,
              IdxT,
              ARG_SORT,
              BLOCK_THREADS,
              N_PER_THREAD>;

          encoder.set_input_array(dev_vals_in);
          encoder.set_input_array(dev_idxs_in);
          encoder.set_output_array(block_partitions);

          encoder.add_kernel_node(
              partition_kernel,
              dim3(1, n_rows, 1),
              dim3(n_thr_per_group, 1, 1),
              0,
              gpu_ptr<IdxT>(block_partitions),
              gpu_ptr<ValT>(dev_vals_in),
              gpu_ptr<IdxT>(dev_idxs_in),
              size_sorted_axis,
              merge_tiles,
              n_blocks);

          auto merge_kernel = cu::mb_block_merge_kernel<
              ValT,
              IdxT,
              ARG_SORT,
              BLOCK_THREADS,
              N_PER_THREAD>;

          encoder.set_input_array(dev_vals_in);
          encoder.set_input_array(dev_idxs_in);
          encoder.set_input_array(block_partitions);
          encoder.set_output_array(dev_vals_out);
          encoder.set_output_array(dev_idxs_out);

          encoder.add_kernel_node(
              merge_kernel,
              dim3(n_blocks, n_rows, 1),
              dim3(BLOCK_THREADS, 1, 1),
              0,
              gpu_ptr<IdxT>(block_partitions),
              gpu_ptr<ValT>(dev_vals_in),
              gpu_ptr<IdxT>(dev_idxs_in),
              gpu_ptr<ValT>(dev_vals_out),
              gpu_ptr<IdxT>(dev_idxs_out),
              size_sorted_axis,
              merge_tiles,
              n_blocks);
          std::swap(dev_vals_in, dev_vals_out);
          std::swap(dev_idxs_in, dev_idxs_out);
        }
      });
    } else {
      throw std::runtime_error(
          "CUDA backend does not support sorting complex numbers");
    }
  });

  encoder.add_temporary(dev_vals_out);
  encoder.add_temporary(dev_idxs_out);
  encoder.add_temporary(argsort ? dev_vals_in : dev_idxs_in);
  if (axis == in.ndim() - 1) {
    // Copy buffer to out, no need for temporary
    out.copy_shared_buffer(
        argsort ? dev_idxs_in : dev_vals_in,
        out.strides(),
        out.flags(),
        out.size());
  } else {
    encoder.add_temporary(argsort ? dev_idxs_in : dev_vals_in);
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    auto strides = out.strides();
    for (int ax = axis + 1; ax < strides.size(); ax++) {
      strides[ax] *= out.shape(axis);
    }
    strides[axis] = 1;
    copy_gpu_inplace(
        (argsort) ? dev_idxs_in : dev_vals_in,
        out,
        out.shape(),
        strides,
        out.strides(),
        0,
        0,
        CopyType::General,
        s);
  }
}

void gpu_merge_sort(
    const Stream& s,
    const array& in,
    array& out,
    int axis_,
    bool argsort) {
  int axis = axis_ < 0 ? axis_ + in.ndim() : axis_;
  int size_sorted_axis = in.shape(axis);

  constexpr int tn = N_PER_THREAD;
  int potential_bn = (size_sorted_axis + tn - 1) / tn;

  int bn;
  if (potential_bn > 256) {
    bn = 512;
  } else if (potential_bn > 128) {
    bn = 256;
  } else if (potential_bn > 64) {
    bn = 128;
  } else if (potential_bn > 32) {
    bn = 64;
  } else {
    bn = 32;
  }

  if (bn == 512 && size_of(in.dtype()) > 4) {
    bn = 256;
  }

  int n_per_block = bn * tn;
  int n_blocks = (size_sorted_axis + n_per_block - 1) / n_per_block;

  if (n_blocks > 1) {
    return multi_block_sort(s, in, out, axis, n_blocks, argsort);
  }
  return single_block_sort(s, in, out, axis, bn, argsort);
}

void gpu_sort(
    const Stream& s,
    const array& in,
    array& out,
    int axis,
    bool argsort) {
  auto& encoder = cu::get_command_encoder(s);
  gpu_merge_sort(s, in, out, axis, argsort);
}

///////////////////////////////////////////////////////////////////////////////
// Radix Select dispatch for partition operations
///////////////////////////////////////////////////////////////////////////////

void gpu_radix_partition(
    const Stream& s,
    const array& in,
    array& out,
    int axis_,
    int kth,
    bool arg_partition) {
  int axis = axis_ < 0 ? axis_ + in.ndim() : axis_;
  int size_sorted_axis = in.shape(axis);

  // Normalize kth
  if (kth < 0) {
    kth += size_sorted_axis;
  }

  // For very small arrays or when kth is close to n, fall back to full sort
  // as the overhead of radix select setup isn't worth it
  constexpr int RADIX_SELECT_THRESHOLD = 256;
  if (size_sorted_axis <= RADIX_SELECT_THRESHOLD) {
    // Use merge sort for small arrays
    gpu_merge_sort(s, in, out, axis, arg_partition);
    return;
  }

  int n_rows = in.size() / in.shape(axis);

  auto in_nc_str = in.strides();
  in_nc_str.erase(in_nc_str.begin() + axis);

  auto out_nc_str = out.strides();
  out_nc_str.erase(out_nc_str.begin() + axis);

  auto nc_shape = in.shape();
  nc_shape.erase(nc_shape.begin() + axis);

  int nc_dim = nc_shape.size();

  int64_t in_stride_sorted_axis = in.strides()[axis];
  int64_t out_stride_sorted_axis = out.strides()[axis];

  // Check if we can use the contiguous kernel
  bool contiguous = in.flags().contiguous;
  auto check_strides = [](const array& x, int64_t sort_stride) {
    int64_t min_stride =
        *std::min_element(x.strides().begin(), x.strides().end());
    int64_t max_stride =
        *std::max_element(x.strides().begin(), x.strides().end());
    return sort_stride == min_stride || sort_stride == max_stride;
  };
  contiguous &= check_strides(in, in_stride_sorted_axis);
  contiguous &= check_strides(out, out_stride_sorted_axis);

  auto& encoder = cu::get_command_encoder(s);
  out.set_data(cu::malloc_async(out.nbytes(), encoder));
  encoder.set_input_array(in);
  encoder.set_output_array(out);

  // Calculate segment strides for contiguous case
  int64_t in_stride_segment_axis = 0;
  int64_t out_stride_segment_axis = 0;
  if (contiguous) {
    in_stride_segment_axis = INT64_MAX;
    out_stride_segment_axis = INT64_MAX;
    for (size_t i = 0; i < nc_shape.size(); i++) {
      if (nc_shape[i] == 1) {
        continue;
      }
      in_stride_segment_axis = std::min(in_stride_segment_axis, in_nc_str[i]);
      out_stride_segment_axis = std::min(out_stride_segment_axis, out_nc_str[i]);
    }
    if (in_stride_segment_axis == INT64_MAX) {
      in_stride_segment_axis = size_sorted_axis;
    }
    if (out_stride_segment_axis == INT64_MAX) {
      out_stride_segment_axis = size_sorted_axis;
    }
  }

  dispatch_all_types(in.dtype(), [&](auto type_tag) {
    using CTYPE = MLX_GET_TYPE(type_tag);
    if constexpr (!std::is_same_v<CTYPE, complex64_t>) {
      using ValT = cuda_type_t<CTYPE>;

      // Use the small kernel for arrays that fit in shared memory
      constexpr int BLOCK_THREADS = 256;
      constexpr int ITEMS_PER_THREAD = 8;
      constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD; // 2048

      if (size_sorted_axis <= TILE_SIZE) {
        dim3 grid(1, n_rows, 1);
        dim3 block(BLOCK_THREADS, 1, 1);

        dispatch_bool(arg_partition, [&](auto arg_tag) {
          constexpr bool ARG_PARTITION = decltype(arg_tag)::value;
          using OutT = std::conditional_t<ARG_PARTITION, uint32_t, ValT>;

          // SELECT_LARGEST = false for standard partition (ascending order)
          // kth element should be at position kth after partition
          auto kernel = cu::radix_select_small_kernel<
              ValT,
              OutT,
              ARG_PARTITION,
              false, // SELECT_LARGEST = false for ascending
              BLOCK_THREADS,
              ITEMS_PER_THREAD>;

          encoder.add_kernel_node(
              kernel,
              grid,
              block,
              0,
              gpu_ptr<ValT>(in),
              gpu_ptr<OutT>(out),
              kth,
              size_sorted_axis,
              in_stride_sorted_axis,
              out_stride_sorted_axis,
              in_stride_segment_axis,
              out_stride_segment_axis);
        });
      } else {
        // For larger arrays, fall back to merge sort for now
        // TODO: Implement multi-pass radix select for large arrays
        gpu_merge_sort(s, in, out, axis, arg_partition);
      }
    } else {
      throw std::runtime_error(
          "CUDA backend does not support partitioning complex numbers");
    }
  });
}

} // namespace

void ArgSort::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("ArgSort::eval_gpu");
  assert(inputs.size() == 1);
  gpu_sort(stream(), inputs[0], out, axis_, true);
}

void Sort::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Sort::eval_gpu");
  assert(inputs.size() == 1);
  gpu_sort(stream(), inputs[0], out, axis_, false);
}

void ArgPartition::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("ArgPartition::eval_gpu");
  assert(inputs.size() == 1);
  gpu_radix_partition(stream(), inputs[0], out, axis_, kth_, true);
}

void Partition::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Partition::eval_gpu");
  assert(inputs.size() == 1);
  gpu_radix_partition(stream(), inputs[0], out, axis_, kth_, false);
}

} // namespace mlx::core