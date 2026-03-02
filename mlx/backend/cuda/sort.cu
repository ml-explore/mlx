// Copyright © 2025 Apple Inc.

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/fp16_math.cuh"
#include "mlx/backend/cuda/device/radix_select.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype.h"
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

///////////////////////////////////////////////////////////////////////////////
// Radix partition functions
///////////////////////////////////////////////////////////////////////////////

// Upper bound for small-kernel tiling. Keep this aligned with the
// items-per-thread dispatch set and per-block shared-memory budget.
constexpr int MAX_RADIX_ITEMS_PER_THREAD = 64;

int radix_small_block_threads(int size_sorted_axis) {
  int block_threads = 256;
  if (size_sorted_axis <= 128) {
    block_threads = 16;
  } else if (size_sorted_axis <= 256) {
    block_threads = 32;
  } else if (size_sorted_axis <= 512) {
    block_threads = 64;
  } else if (size_sorted_axis <= 1024) {
    block_threads = 128;
  }
  return block_threads;
}

template <typename F>
void dispatch_radix_items_per_thread(
    int size_sorted_axis,
    int block_threads,
    F&& f) {
  int items_per_thread = (size_sorted_axis + block_threads - 1) / block_threads;
  if (items_per_thread <= 1) {
    f(std::integral_constant<int, 1>{});
  } else if (items_per_thread <= 2) {
    f(std::integral_constant<int, 2>{});
  } else if (items_per_thread <= 4) {
    f(std::integral_constant<int, 4>{});
  } else if (items_per_thread <= 8) {
    f(std::integral_constant<int, 8>{});
  } else if (items_per_thread <= 12) {
    f(std::integral_constant<int, 12>{});
  } else if (items_per_thread <= 16) {
    f(std::integral_constant<int, 16>{});
  } else if (items_per_thread <= 24) {
    f(std::integral_constant<int, 24>{});
  } else if (items_per_thread <= 32) {
    f(std::integral_constant<int, 32>{});
  } else if (items_per_thread <= 48) {
    f(std::integral_constant<int, 48>{});
  } else {
    f(std::integral_constant<int, MAX_RADIX_ITEMS_PER_THREAD>{});
  }
}

size_t radix_small_shared_mem_bytes(
    size_t key_size,
    int block_threads,
    int items_per_thread) {
  size_t tile_size = static_cast<size_t>(block_threads) *
      static_cast<size_t>(items_per_thread);
  size_t num_warps = static_cast<size_t>(block_threads / WARP_SIZE);
  return tile_size * key_size + // shared_keys
      tile_size * sizeof(uint32_t) + // shared_idxs
      cu::SMALL_RADIX_SIZE * sizeof(int) + // shared_hist for small kernel
      (2 + 3 * num_warps + 6) * sizeof(int); // shared_count + scatter scratch
}

int radix_max_shared_mem_per_block(const Stream& s) {
  int max_shared_mem_per_block = 0;
  CHECK_CUDA_ERROR(cudaDeviceGetAttribute(
      &max_shared_mem_per_block,
      cudaDevAttrMaxSharedMemoryPerBlock,
      s.device.index));

  int max_shared_mem_per_block_optin = 0;
  cudaError_t optin_err = cudaDeviceGetAttribute(
      &max_shared_mem_per_block_optin,
      cudaDevAttrMaxSharedMemoryPerBlockOptin,
      s.device.index);
  if (optin_err == cudaSuccess) {
    max_shared_mem_per_block =
        std::max(max_shared_mem_per_block, max_shared_mem_per_block_optin);
  } else {
    cudaGetLastError();
  }
  return max_shared_mem_per_block;
}

bool radix_small_fits_shared_memory(
    const Stream& s,
    Dtype dtype,
    int size_sorted_axis) {
  if (size_sorted_axis <= 0) {
    return false;
  }

  int block_threads = radix_small_block_threads(size_sorted_axis);
  int required_items = (size_sorted_axis + block_threads - 1) / block_threads;
  if (required_items > MAX_RADIX_ITEMS_PER_THREAD) {
    return false;
  }

  size_t required_shared_mem = 0;
  dispatch_radix_items_per_thread(
      size_sorted_axis, block_threads, [&](auto items_per_thread_tag) {
        constexpr int ITEMS_PER_THREAD = items_per_thread_tag();
        required_shared_mem = radix_small_shared_mem_bytes(
            size_of(dtype), block_threads, ITEMS_PER_THREAD);
      });

  int max_shared_mem_per_block = radix_max_shared_mem_per_block(s);
  return required_shared_mem <= static_cast<size_t>(max_shared_mem_per_block);
}

void gpu_radix_partition_small(
    const Stream& s,
    const array& in,
    array& out,
    int axis,
    int kth,
    bool arg_partition) {
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

  auto nc_shape_param = const_param(nc_shape);
  auto in_nc_strides_param = const_param(in_nc_str);
  auto out_nc_strides_param = const_param(out_nc_str);

  dispatch_all_types(in.dtype(), [&](auto type_tag) {
    using CTYPE = MLX_GET_TYPE(type_tag);
    if constexpr (!std::is_same_v<CTYPE, complex64_t>) {
      using ValT = cuda_type_t<CTYPE>;

      int block_threads = radix_small_block_threads(size_sorted_axis);

      dispatch_bool(arg_partition, [&](auto arg_tag) {
        constexpr bool ARG_PARTITION = decltype(arg_tag)::value;
        using OutT = std::conditional_t<ARG_PARTITION, uint32_t, ValT>;

        int64_t in_stride_segment_axis = INT64_MAX;
        int64_t out_stride_segment_axis = INT64_MAX;
        if (contiguous) {
          for (size_t i = 0; i < nc_shape.size(); i++) {
            if (nc_shape[i] == 1) {
              continue;
            }
            in_stride_segment_axis =
                std::min(in_stride_segment_axis, in_nc_str[i]);
            out_stride_segment_axis =
                std::min(out_stride_segment_axis, out_nc_str[i]);
          }
        }

        dispatch_block_dim(block_threads, [&](auto block_dim) {
          constexpr int BLOCK_THREADS = block_dim();
          dim3 grid(1, n_rows, 1);
          dim3 block(BLOCK_THREADS, 1, 1);

          dispatch_radix_items_per_thread(
              size_sorted_axis, block_threads, [&](auto items_per_thread_tag) {
                constexpr int ITEMS_PER_THREAD = items_per_thread_tag();

                dispatch_bool(contiguous, [&](auto contiguous_tag) {
                  constexpr bool USE_SIMPLE_STRIDE =
                      decltype(contiguous_tag)::value;

                  auto kernel = cu::radix_select_small_kernel<
                      ValT,
                      OutT,
                      ARG_PARTITION,
                      USE_SIMPLE_STRIDE,
                      BLOCK_THREADS,
                      ITEMS_PER_THREAD>;

                  // Calculate dynamic shared memory size
                  using UnsignedT = typename cu::RadixTraits<ValT>::UnsignedT;
                  constexpr int TILE_SIZE_VAL =
                      BLOCK_THREADS * ITEMS_PER_THREAD;
                  constexpr int NUM_WARPS = BLOCK_THREADS / WARP_SIZE;
                  constexpr size_t shared_mem_bytes =
                      TILE_SIZE_VAL * sizeof(UnsignedT) + // shared_keys
                      TILE_SIZE_VAL * sizeof(uint32_t) + // shared_idxs
                      cu::SMALL_RADIX_SIZE *
                          sizeof(int) + // shared_hist for small kernel
                      (2 + 3 * NUM_WARPS + 6) *
                          sizeof(int); // shared_count + scatter scratch

                  CHECK_CUDA_ERROR(cudaFuncSetAttribute(
                      kernel,
                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                      static_cast<int>(shared_mem_bytes)));

                  encoder.add_kernel_node(
                      kernel,
                      grid,
                      block,
                      shared_mem_bytes,
                      gpu_ptr<ValT>(in),
                      gpu_ptr<OutT>(out),
                      kth,
                      size_sorted_axis,
                      in_stride_sorted_axis,
                      out_stride_sorted_axis,
                      in_stride_segment_axis,
                      out_stride_segment_axis,
                      nc_shape_param,
                      in_nc_strides_param,
                      out_nc_strides_param,
                      nc_dim);
                });
              });
        });
      });
    } else {
      throw std::runtime_error(
          "CUDA backend does not support sorting complex numbers");
    }
  });
}

int64_t segment_stride_for_contiguous(
    const Shape& shape_no_axis,
    const Strides& strides_no_axis) {
  int64_t stride = INT64_MAX;
  for (size_t i = 0; i < shape_no_axis.size(); ++i) {
    if (shape_no_axis[i] == 1) {
      continue;
    }
    stride = std::min(stride, strides_no_axis[i]);
  }
  return (stride == INT64_MAX) ? int64_t(0) : stride;
}

struct RadixLaunchPlan {
  int blocks_per_row{1};

  bool uses_tiled_launch() const {
    return blocks_per_row > 1;
  }
};

RadixLaunchPlan make_radix_tiled_launch_plan(
    const Stream& s,
    int n_rows,
    int size_sorted_axis) {
  if (n_rows <= 0 || size_sorted_axis <= 0 || size_sorted_axis < 8192) {
    return {};
  }

  constexpr int kBlocksPerSmTarget = 4;
  constexpr int kMinElemsPerBlock = 1024;
  constexpr int kMaxBlocksPerRow = 32;

  int sm_count = 0;
  CHECK_CUDA_ERROR(cudaDeviceGetAttribute(
      &sm_count, cudaDevAttrMultiProcessorCount, s.device.index));
  sm_count = std::max(sm_count, 1);

  int target_blocks = std::max(1, sm_count * kBlocksPerSmTarget);
  int needed_blocks_per_row =
      std::max(1, (target_blocks + n_rows - 1) / n_rows);

  int max_blocks_by_work = std::max(
      1, std::min(kMaxBlocksPerRow, size_sorted_axis / kMinElemsPerBlock));
  int blocks_per_row = std::min(needed_blocks_per_row, max_blocks_by_work);

  if (blocks_per_row <= 1) {
    return {};
  }
  return {blocks_per_row};
}

Dtype unsigned_dtype_for_size(int size) {
  switch (size) {
    case 1:
      return uint8;
    case 2:
      return uint16;
    case 4:
      return uint32;
    case 8:
      return uint64;
    default:
      throw std::runtime_error("Unsupported radix key size");
  }
}

void gpu_radix_partition_large_tiled(
    const Stream& s,
    const array& in,
    array& out,
    int kth,
    int n_rows,
    int size_sorted_axis,
    int64_t in_stride_sorted_axis,
    int64_t out_stride_sorted_axis,
    int64_t in_stride_segment_axis,
    int64_t out_stride_segment_axis,
    int blocks_per_row,
    bool arg_partition) {
  auto& encoder = cu::get_command_encoder(s);
  out.set_data(cu::malloc_async(out.nbytes(), encoder));
  encoder.set_input_array(in);
  encoder.set_output_array(out);

  dispatch_all_types(in.dtype(), [&](auto type_tag) {
    using CTYPE = MLX_GET_TYPE(type_tag);
    if constexpr (!std::is_same_v<CTYPE, complex64_t>) {
      using ValT = cuda_type_t<CTYPE>;
      constexpr int BLOCK_THREADS = 256;
      constexpr int NUM_PASSES =
          (cu::RadixTraits<ValT>::BITS + cu::RADIX_BITS - 1) / cu::RADIX_BITS;
      using UnsignedT = typename cu::RadixTraits<ValT>::UnsignedT;

      Dtype unsigned_dtype = unsigned_dtype_for_size(sizeof(UnsignedT));

      array target_prefix_dev({n_rows}, unsigned_dtype, nullptr, {});
      array prefix_mask_dev({n_rows}, unsigned_dtype, nullptr, {});
      array k_values_dev({n_rows}, int32, nullptr, {});
      array row_hist_dev({n_rows, cu::RADIX_SIZE}, int32, nullptr, {});

      int total_blocks = n_rows * blocks_per_row;
      array block_less_dev({total_blocks}, int32, nullptr, {});
      array block_equal_dev({total_blocks}, int32, nullptr, {});

      auto allocate_temporary = [&](array& a) {
        a.set_data(cu::malloc_async(a.nbytes(), encoder));
        encoder.add_temporary(a);
      };
      allocate_temporary(target_prefix_dev);
      allocate_temporary(prefix_mask_dev);
      allocate_temporary(k_values_dev);
      allocate_temporary(row_hist_dev);
      allocate_temporary(block_less_dev);
      allocate_temporary(block_equal_dev);

      dim3 row_grid(1, n_rows, 1);
      dim3 grid(blocks_per_row, n_rows, 1);

      encoder.set_output_array(target_prefix_dev);
      encoder.set_output_array(prefix_mask_dev);
      encoder.set_output_array(k_values_dev);
      encoder.set_output_array(row_hist_dev);
      encoder.add_kernel_node(
          cu::radix_select_tiled_init_state_kernel<UnsignedT>,
          row_grid,
          dim3(32, 1, 1),
          0,
          gpu_ptr<UnsignedT>(target_prefix_dev),
          gpu_ptr<UnsignedT>(prefix_mask_dev),
          gpu_ptr<int>(k_values_dev),
          gpu_ptr<int>(row_hist_dev),
          kth);

      for (int pass = NUM_PASSES - 1; pass >= 0; --pass) {
        int start_bit = pass * cu::RADIX_BITS;

        encoder.set_input_array(in);
        encoder.set_input_array(row_hist_dev);
        encoder.set_input_array(target_prefix_dev);
        encoder.set_input_array(prefix_mask_dev);
        encoder.set_output_array(row_hist_dev);
        encoder.add_kernel_node(
            cu::radix_select_tiled_histogram_kernel<ValT, BLOCK_THREADS>,
            grid,
            dim3(BLOCK_THREADS, 1, 1),
            0,
            gpu_ptr<ValT>(in),
            size_sorted_axis,
            in_stride_sorted_axis,
            in_stride_segment_axis,
            gpu_ptr<UnsignedT>(target_prefix_dev),
            gpu_ptr<UnsignedT>(prefix_mask_dev),
            start_bit,
            blocks_per_row,
            gpu_ptr<int>(row_hist_dev));

        encoder.set_input_array(row_hist_dev);
        encoder.set_input_array(target_prefix_dev);
        encoder.set_input_array(prefix_mask_dev);
        encoder.set_input_array(k_values_dev);
        encoder.set_input_array(row_hist_dev);
        encoder.set_output_array(target_prefix_dev);
        encoder.set_output_array(prefix_mask_dev);
        encoder.set_output_array(k_values_dev);
        encoder.set_output_array(row_hist_dev);
        encoder.add_kernel_node(
            cu::radix_select_tiled_select_bin_kernel<UnsignedT>,
            row_grid,
            dim3(32, 1, 1),
            0,
            gpu_ptr<int>(row_hist_dev),
            gpu_ptr<UnsignedT>(target_prefix_dev),
            gpu_ptr<UnsignedT>(prefix_mask_dev),
            gpu_ptr<int>(k_values_dev),
            pass > 0 ? 1 : 0,
            start_bit);
      }

      encoder.set_input_array(in);
      encoder.set_input_array(target_prefix_dev);
      encoder.set_output_array(block_less_dev);
      encoder.set_output_array(block_equal_dev);
      encoder.add_kernel_node(
          cu::radix_select_tiled_count_kernel<ValT, BLOCK_THREADS>,
          grid,
          dim3(BLOCK_THREADS, 1, 1),
          0,
          gpu_ptr<ValT>(in),
          size_sorted_axis,
          in_stride_sorted_axis,
          in_stride_segment_axis,
          gpu_ptr<UnsignedT>(target_prefix_dev),
          blocks_per_row,
          gpu_ptr<int>(block_less_dev),
          gpu_ptr<int>(block_equal_dev));

      dispatch_bool(arg_partition, [&](auto arg_tag) {
        constexpr bool ARG_PARTITION = decltype(arg_tag)::value;
        using OutT = std::conditional_t<ARG_PARTITION, uint32_t, ValT>;
        encoder.set_input_array(in);
        encoder.set_input_array(target_prefix_dev);
        encoder.set_input_array(block_less_dev);
        encoder.set_input_array(block_equal_dev);
        encoder.set_output_array(out);
        encoder.add_kernel_node(
            cu::radix_select_tiled_scatter_kernel<
                ValT,
                OutT,
                ARG_PARTITION,
                BLOCK_THREADS>,
            grid,
            dim3(BLOCK_THREADS, 1, 1),
            0,
            gpu_ptr<ValT>(in),
            gpu_ptr<OutT>(out),
            size_sorted_axis,
            in_stride_sorted_axis,
            out_stride_sorted_axis,
            in_stride_segment_axis,
            out_stride_segment_axis,
            gpu_ptr<UnsignedT>(target_prefix_dev),
            blocks_per_row,
            gpu_ptr<int>(block_less_dev),
            gpu_ptr<int>(block_equal_dev));
      });
    } else {
      throw std::runtime_error(
          "CUDA backend does not support sorting complex numbers");
    }
  });
}

void gpu_radix_partition_large(
    const Stream& s,
    const array& in,
    array& out,
    int axis,
    int kth,
    bool arg_partition) {
  int n_rows = in.size() / in.shape(axis);

  int size_sorted_axis = in.shape(axis);
  int64_t in_stride_sorted_axis = in.strides()[axis];
  int64_t out_stride_sorted_axis = out.strides()[axis];

  auto in_nc_str = in.strides();
  in_nc_str.erase(in_nc_str.begin() + axis);

  auto out_nc_str = out.strides();
  out_nc_str.erase(out_nc_str.begin() + axis);

  auto nc_shape = in.shape();
  nc_shape.erase(nc_shape.begin() + axis);

  int nc_dim = nc_shape.size();

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

  if (contiguous) {
    const auto plan = make_radix_tiled_launch_plan(s, n_rows, size_sorted_axis);
    if (plan.uses_tiled_launch()) {
      const int64_t in_stride_segment_axis =
          segment_stride_for_contiguous(nc_shape, in_nc_str);
      const int64_t out_stride_segment_axis =
          segment_stride_for_contiguous(nc_shape, out_nc_str);
      return gpu_radix_partition_large_tiled(
          s,
          in,
          out,
          kth,
          n_rows,
          size_sorted_axis,
          in_stride_sorted_axis,
          out_stride_sorted_axis,
          in_stride_segment_axis,
          out_stride_segment_axis,
          plan.blocks_per_row,
          arg_partition);
    }
  }

  auto& encoder = cu::get_command_encoder(s);
  out.set_data(cu::malloc_async(out.nbytes(), encoder));
  encoder.set_input_array(in);
  encoder.set_output_array(out);

  auto nc_shape_param = const_param(nc_shape);
  auto in_nc_strides_param = const_param(in_nc_str);
  auto out_nc_strides_param = const_param(out_nc_str);

  dispatch_all_types(in.dtype(), [&](auto type_tag) {
    using CTYPE = MLX_GET_TYPE(type_tag);
    if constexpr (!std::is_same_v<CTYPE, complex64_t>) {
      using ValT = cuda_type_t<CTYPE>;

      constexpr int BLOCK_THREADS = 256;

      dim3 grid(1, n_rows, 1);
      dim3 block(BLOCK_THREADS, 1, 1);

      dispatch_bool(arg_partition, [&](auto arg_tag) {
        constexpr bool ARG_PARTITION = decltype(arg_tag)::value;
        using OutT = std::conditional_t<ARG_PARTITION, uint32_t, ValT>;

        // Large kernel always uses elem_to_loc addressing
        auto kernel = cu::radix_select_large_streaming_kernel<
            ValT,
            OutT,
            ARG_PARTITION,
            BLOCK_THREADS>;

        encoder.add_kernel_node(
            kernel,
            grid,
            block,
            0,
            gpu_ptr<ValT>(in),
            gpu_ptr<OutT>(out),
            size_sorted_axis,
            kth,
            in_stride_sorted_axis,
            out_stride_sorted_axis,
            nc_shape_param,
            in_nc_strides_param,
            out_nc_strides_param,
            nc_dim);
      });
    } else {
      throw std::runtime_error(
          "CUDA backend does not support sorting complex numbers");
    }
  });
}

void gpu_radix_partition(
    const Stream& s,
    const array& in,
    array& out,
    int axis_,
    int kth_,
    bool arg_partition) {
  int axis = axis_ < 0 ? axis_ + in.ndim() : axis_;
  int size_sorted_axis = in.shape(axis);
  int kth = kth_ < 0 ? kth_ + size_sorted_axis : kth_;
  int nc_dim = static_cast<int>(in.ndim()) - 1;

  // Fixed-size const_param metadata is capped by MAX_NDIM.
  if (nc_dim > MAX_NDIM) {
    return gpu_merge_sort(s, in, out, axis, arg_partition);
  }

  // Dispatch based on whether the small kernel tile fits in shared memory.
  if (radix_small_fits_shared_memory(s, in.dtype(), size_sorted_axis)) {
    return gpu_radix_partition_small(s, in, out, axis, kth, arg_partition);
  } else {
    return gpu_radix_partition_large(s, in, out, axis, kth, arg_partition);
  }
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
  gpu_radix_partition(stream(), inputs[0], out, axis_, kth_, true);
}

void Partition::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Partition::eval_gpu");
  gpu_radix_partition(stream(), inputs[0], out, axis_, kth_, false);
}

} // namespace mlx::core
