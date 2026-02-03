// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>
#include <cassert>
#include <limits>

namespace mlx::core {

constexpr int N_PER_THREAD = 8;

namespace rocm {

template <typename T>
__device__ __forceinline__ T nan_value();

template <>
__device__ __forceinline__ float nan_value<float>() {
  return __builtin_nanf("");
}

template <>
__device__ __forceinline__ double nan_value<double>() {
  return __builtin_nan("");
}

template <>
__device__ __forceinline__ _Float16 nan_value<_Float16>() {
  return static_cast<_Float16>(__builtin_nanf(""));
}

template <>
__device__ __forceinline__ hip_bfloat16 nan_value<hip_bfloat16>() {
  return hip_bfloat16(__builtin_nanf(""));
}

template <typename T, typename = void>
struct InitValue {
  __device__ __forceinline__ static T value() {
    return Limits<T>::max;
  }
};

template <typename T>
struct InitValue<T, std::enable_if_t<std::is_floating_point_v<T>>> {
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
      bool an = isnan(static_cast<float>(a));
      bool bn = isnan(static_cast<float>(b));
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

} // namespace rocm

namespace {

void gpu_sort(const Stream& s, array in, array& out_, int axis, bool argsort) {
  array out = out_;
  auto& encoder = rocm::get_command_encoder(s);
  if (axis < 0) {
    axis += in.ndim();
  }
  
  int size_sorted_axis = in.shape(axis);
  int n_rows = in.size() / size_sorted_axis;
  int last_dim = in.ndim() - 1;

  // If we are not sorting the innermost dimension of a contiguous array,
  // transpose and make a copy.
  bool is_segmented_sort = in.flags().contiguous && in.strides()[axis] == 1;
  if (!is_segmented_sort) {
    array trans = swapaxes_in_eval(in, axis, last_dim);
    in = contiguous_copy_gpu(trans, s);
    encoder.add_temporary(in);
    out = array(allocator::malloc(out.nbytes()), in.shape(), out.dtype());
    encoder.add_temporary(out);
  } else {
    out.set_data(
        allocator::malloc(in.data_size() * out.itemsize()),
        in.data_size(),
        in.strides(),
        in.flags());
  }

  encoder.set_input_array(in);
  encoder.set_output_array(out);
  
  auto& stream = encoder.stream();
  
  // Determine block size
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

  int64_t in_stride_sorted = 1;  // After transpose, always 1
  int64_t out_stride_sorted = 1;
  int64_t in_stride_segment = size_sorted_axis;
  int64_t out_stride_segment = size_sorted_axis;

  dispatch_all_types(in.dtype(), [&](auto type_tag) {
    using CTYPE = MLX_GET_TYPE(type_tag);
    if constexpr (!std::is_same_v<CTYPE, complex64_t>) {
      using ValT = hip_type_t<CTYPE>;
      
      encoder.launch_kernel([&](hipStream_t hip_stream) {
        dim3 grid(1, n_rows, 1);
        
        auto launch_kernel = [&]<bool ARG_SORT, int BLOCK_THREADS>() {
          using OutT = std::conditional_t<ARG_SORT, uint32_t, ValT>;
          constexpr int N_PER_BLOCK = BLOCK_THREADS * tn;
          
          hipLaunchKernelGGL(
              (rocm::block_sort_kernel<ValT, OutT, ARG_SORT, BLOCK_THREADS, tn>),
              grid,
              dim3(BLOCK_THREADS, 1, 1),
              0,
              hip_stream,
              in.data<ValT>(),
              out.data<OutT>(),
              size_sorted_axis,
              in_stride_sorted,
              out_stride_sorted,
              in_stride_segment,
              out_stride_segment);
        };
        
        // Dispatch based on argsort and block size
        if (argsort) {
          switch (bn) {
            case 32: launch_kernel.template operator()<true, 32>(); break;
            case 64: launch_kernel.template operator()<true, 64>(); break;
            case 128: launch_kernel.template operator()<true, 128>(); break;
            case 256: launch_kernel.template operator()<true, 256>(); break;
            case 512: launch_kernel.template operator()<true, 512>(); break;
          }
        } else {
          switch (bn) {
            case 32: launch_kernel.template operator()<false, 32>(); break;
            case 64: launch_kernel.template operator()<false, 64>(); break;
            case 128: launch_kernel.template operator()<false, 128>(); break;
            case 256: launch_kernel.template operator()<false, 256>(); break;
            case 512: launch_kernel.template operator()<false, 512>(); break;
          }
        }
      });
    } else {
      throw std::runtime_error(
          "ROCm backend does not support sorting complex numbers");
    }
  });

  if (!is_segmented_sort) {
    // Swap the sorted axis back.
    copy_gpu(swapaxes_in_eval(out, axis, last_dim), out_, CopyType::General, s);
  }
}

} // namespace

void ArgSort::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  gpu_sort(stream(), inputs[0], out, axis_, true);
}

void Sort::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  gpu_sort(stream(), inputs[0], out, axis_, false);
}

void ArgPartition::eval_gpu(const std::vector<array>& inputs, array& out) {
  gpu_sort(stream(), inputs[0], out, axis_, true);
}

void Partition::eval_gpu(const std::vector<array>& inputs, array& out) {
  gpu_sort(stream(), inputs[0], out, axis_, false);
}

} // namespace mlx::core
