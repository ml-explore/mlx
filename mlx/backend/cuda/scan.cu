// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/binary_ops.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/reduce/reduce_ops.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <nvtx3/nvtx3.hpp>

#include <cassert>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename Op, typename T>
struct ScanResult {
  using type = T;
};

template <>
struct ScanResult<Sum, bool> {
  using type = int32_t;
};

template <typename T>
struct ReduceInit<LogAddExp, T> {
  static constexpr __host__ __device__ T value() {
    return Limits<T>::min();
  }
};

template <bool reverse, typename T, typename U, int N_READS>
inline __device__ void
load_values(int index, const T* in, U (&values)[N_READS], int size, U init) {
  int remaining = size - index * N_READS;
  if constexpr (reverse) {
    in += remaining - N_READS;
    if (remaining < N_READS) {
      for (int i = 0; i < N_READS; ++i) {
        values[N_READS - i - 1] =
            (N_READS - i - 1 < remaining) ? cast_to<U>(in[i]) : init;
      }
    } else {
      for (int i = 0; i < N_READS; ++i) {
        values[N_READS - i - 1] = cast_to<U>(in[i]);
      }
    }
  } else {
    in += index * N_READS;
    if (remaining < N_READS) {
      for (int i = 0; i < N_READS; ++i) {
        values[i] = (i < remaining) ? cast_to<U>(in[i]) : init;
      }
    } else {
      for (int i = 0; i < N_READS; ++i) {
        values[i] = cast_to<U>(in[i]);
      }
    }
  }
}

template <bool reverse, int offset, typename T, int N_READS>
inline __device__ void
store_values(int index, T* out, T (&values)[N_READS], int size) {
  int start = index * N_READS + offset;
  int remaining = size - start;
  if constexpr (reverse) {
    out += remaining - N_READS;
    if (remaining < N_READS) {
      for (int i = 0; i < N_READS; ++i) {
        if (N_READS - i - 1 < remaining) {
          out[i] = values[N_READS - i - 1];
        }
      }
    } else {
      for (int i = 0; i < N_READS; ++i) {
        out[i] = values[N_READS - i - 1];
      }
    }
  } else {
    out += start;
    if (remaining < N_READS) {
      for (int i = 0; i < N_READS; ++i) {
        if (i < remaining) {
          out[i] = values[i];
        }
      }
    } else {
      for (int i = 0; i < N_READS; ++i) {
        out[i] = values[i];
      }
    }
  }
}

template <
    typename T,
    typename U,
    typename Op,
    int N_READS,
    bool inclusive,
    bool reverse>
__global__ void contiguous_scan(const T* in, U* out, int32_t axis_size) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  in += grid.block_rank() * axis_size;
  out += grid.block_rank() * axis_size;

  __shared__ U warp_sums[WARP_SIZE];

  Op op;
  U init = ReduceInit<Op, T>::value();
  U prefix = init;

  // Scan per block.
  for (int r = 0; r < cuda::ceil_div(axis_size, block.size() * N_READS); ++r) {
    int32_t index = r * block.size() + block.thread_rank();
    U values[N_READS];
    load_values<reverse>(index, in, values, axis_size, init);

    // Compute an inclusive scan per thread.
    for (int i = 1; i < N_READS; ++i) {
      values[i] = op(values[i], values[i - 1]);
    }

    // Compute exclusive scan of thread sums.
    U prev_thread_sum = cg::exclusive_scan(warp, values[N_READS - 1], op);
    if (warp.thread_rank() == 0) {
      prev_thread_sum = init;
    }

    // Write wrap's sum to shared memory.
    if (warp.thread_rank() == WARP_SIZE - 1) {
      warp_sums[warp.meta_group_rank()] =
          op(prev_thread_sum, values[N_READS - 1]);
    }
    block.sync();

    // Compute exclusive scan of warp sums.
    if (warp.meta_group_rank() == 0) {
      U prev_warp_sum =
          cg::exclusive_scan(warp, warp_sums[warp.thread_rank()], op);
      if (warp.thread_rank() == 0) {
        prev_warp_sum = init;
      }
      warp_sums[warp.thread_rank()] = prev_warp_sum;
    }
    block.sync();

    // Compute the output.
    for (int i = 0; i < N_READS; ++i) {
      values[i] = op(values[i], prefix);
      values[i] = op(values[i], warp_sums[warp.meta_group_rank()]);
      values[i] = op(values[i], prev_thread_sum);
    }

    // Write the values.
    if (inclusive) {
      store_values<reverse, 0>(index, out, values, axis_size);
    } else {
      store_values<reverse, 1>(index, out, values, axis_size);
      if (reverse) {
        if (block.thread_rank() == 0 && index == 0) {
          out[axis_size - 1] = init;
        }
      } else {
        if (block.thread_rank() == 0 && index == 0) {
          out[0] = init;
        }
      }
    }
    block.sync();

    // Share the prefix.
    if ((warp.meta_group_rank() == warp.meta_group_size() - 1) &&
        (warp.thread_rank() == WARP_SIZE - 1)) {
      warp_sums[0] = values[N_READS - 1];
    }
    block.sync();
    prefix = warp_sums[0];
  }
}

template <
    typename T,
    typename U,
    typename Op,
    int N_READS,
    int BM,
    int BN,
    bool inclusive,
    bool reverse>
__global__ void strided_scan(
    const T* in,
    U* out,
    int32_t axis_size,
    int64_t stride,
    int64_t stride_blocks) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  constexpr int BN_pad = WARP_SIZE + 16 / sizeof(U);
  constexpr int n_warps = BN / N_READS;
  constexpr int n_scans = BN / n_warps;

  __shared__ U read_buffer[BM * BN_pad];

  Op op;
  U init = ReduceInit<Op, T>::value();
  U values[n_scans];
  U prefix[n_scans];
  for (int i = 0; i < n_scans; ++i) {
    prefix[i] = init;
  }

  // Compute offsets.
  int64_t offset = (grid.block_rank() / stride_blocks) * axis_size * stride;
  int64_t global_index_x = (grid.block_rank() % stride_blocks) * BN;
  uint read_offset_y = (block.thread_rank() * N_READS) / BN;
  uint read_offset_x = (block.thread_rank() * N_READS) % BN;
  uint scan_offset_y = warp.thread_rank();
  uint scan_offset_x = warp.meta_group_rank() * n_scans;

  uint stride_limit = stride - global_index_x;
  in += offset + global_index_x + read_offset_x;
  out += offset + global_index_x + read_offset_x;
  U* read_into = read_buffer + read_offset_y * BN_pad + read_offset_x;
  U* read_from = read_buffer + scan_offset_y * BN_pad + scan_offset_x;

  for (uint j = 0; j < axis_size; j += BM) {
    // Calculate the indices for the current thread.
    uint index_y = j + read_offset_y;
    uint check_index_y = index_y;
    if (reverse) {
      index_y = axis_size - 1 - index_y;
    }

    // Read in SM.
    if (check_index_y < axis_size && (read_offset_x + N_READS) < stride_limit) {
      for (int i = 0; i < N_READS; ++i) {
        read_into[i] = in[index_y * stride + i];
      }
    } else {
      for (int i = 0; i < N_READS; ++i) {
        if (check_index_y < axis_size && (read_offset_x + i) < stride_limit) {
          read_into[i] = in[index_y * stride + i];
        } else {
          read_into[i] = init;
        }
      }
    }
    block.sync();

    // Read strided into registers.
    for (int i = 0; i < n_scans; ++i) {
      values[i] = read_from[i];
    }

    // Perform the scan.
    for (int i = 0; i < n_scans; ++i) {
      values[i] = cg::inclusive_scan(warp, values[i], op);
      values[i] = op(values[i], prefix[i]);
      prefix[i] = warp.shfl(values[i], WARP_SIZE - 1);
    }

    // Write to SM.
    for (int i = 0; i < n_scans; ++i) {
      read_from[i] = values[i];
    }
    block.sync();

    // Write to device memory.
    if (!inclusive) {
      if (check_index_y == 0) {
        if ((read_offset_x + N_READS) < stride_limit) {
          for (int i = 0; i < N_READS; ++i) {
            out[index_y * stride + i] = init;
          }
        } else {
          for (int i = 0; i < N_READS; ++i) {
            if ((read_offset_x + i) < stride_limit) {
              out[index_y * stride + i] = init;
            }
          }
        }
      }
      if (reverse) {
        index_y -= 1;
        check_index_y += 1;
      } else {
        index_y += 1;
        check_index_y += 1;
      }
    }
    if (check_index_y < axis_size && (read_offset_x + N_READS) < stride_limit) {
      for (int i = 0; i < N_READS; ++i) {
        out[index_y * stride + i] = read_into[i];
      }
    } else {
      for (int i = 0; i < N_READS; ++i) {
        if (check_index_y < axis_size && (read_offset_x + i) < stride_limit) {
          out[index_y * stride + i] = read_into[i];
        }
      }
    }
  }
}

} // namespace cu

template <typename F>
void dispatch_scan_ops(Scan::ReduceType scan_op, F&& f) {
  if (scan_op == Scan::ReduceType::Max) {
    f(type_identity<cu::Max>{});
  } else if (scan_op == Scan::ReduceType::Min) {
    f(type_identity<cu::Min>{});
  } else if (scan_op == Scan::ReduceType::Sum) {
    f(type_identity<cu::Sum>{});
  } else if (scan_op == Scan::ReduceType::Prod) {
    f(type_identity<cu::Prod>{});
  } else if (scan_op == Scan::ReduceType::LogAddExp) {
    f(type_identity<cu::LogAddExp>{});
  } else {
    throw std::invalid_argument("Unknown reduce type.");
  }
}

template <typename Op>
const char* op_to_string() {
  if (cuda::std::is_same_v<Op, cu::Max>) {
    return "Max";
  } else if (cuda::std::is_same_v<Op, cu::Min>) {
    return "Min";
  } else if (cuda::std::is_same_v<Op, cu::Sum>) {
    return "Sum";
  } else if (cuda::std::is_same_v<Op, cu::Prod>) {
    return "Prod";
  } else if (cuda::std::is_same_v<Op, cu::LogAddExp>) {
    return "LogAddExp";
  } else {
    throw std::invalid_argument("Unknown op.");
  }
}

template <typename Op, typename T>
constexpr bool supports_scan_op() {
  if constexpr (cuda::std::is_same_v<Op, LogAddExp>) {
    return is_inexact_v<T>;
  } else {
    return true;
  }
}

void Scan::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Scan::eval_gpu");
  assert(inputs.size() == 1);
  auto in = inputs[0];
  auto& s = stream();

  if (in.flags().contiguous && in.strides()[axis_] != 0) {
    if (in.is_donatable() && in.itemsize() == out.itemsize()) {
      out.copy_shared_buffer(in);
    } else {
      out.set_data(
          allocator::malloc(in.data_size() * out.itemsize()),
          in.data_size(),
          in.strides(),
          in.flags());
    }
  } else {
    array arr_copy(in.shape(), in.dtype(), nullptr, {});
    copy_gpu(in, arr_copy, CopyType::General, s);
    in = std::move(arr_copy);
    out.copy_shared_buffer(in);
  }

  constexpr int N_READS = 4;
  int32_t axis_size = in.shape(axis_);
  bool contiguous = in.strides()[axis_] == 1;

  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);

  dispatch_all_types(in.dtype(), [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    dispatch_scan_ops(reduce_type_, [&](auto scan_op_tag) {
      using Op = MLX_GET_TYPE(scan_op_tag);
      if constexpr (supports_scan_op<Op, T>) {
        using U = typename cu::ScanResult<Op, T>::type;
        dispatch_bool(inclusive_, [&](auto inclusive) {
          dispatch_bool(reverse_, [&](auto reverse) {
            if (contiguous) {
              auto kernel = cu::contiguous_scan<
                  T,
                  U,
                  Op,
                  N_READS,
                  inclusive.value,
                  reverse.value>;
              int block_dim = cuda::ceil_div(axis_size, N_READS);
              block_dim = cuda::ceil_div(block_dim, WARP_SIZE) * WARP_SIZE;
              block_dim = std::min(block_dim, WARP_SIZE * WARP_SIZE);
              encoder.add_kernel_node(
                  kernel,
                  in.data_size() / axis_size,
                  block_dim,
                  in.data<T>(),
                  out.data<U>(),
                  axis_size);
            } else {
              constexpr int BM = WARP_SIZE;
              constexpr int BN = WARP_SIZE;
              auto kernel = cu::strided_scan<
                  T,
                  U,
                  Op,
                  N_READS,
                  BM,
                  BN,
                  inclusive.value,
                  reverse.value>;
              int64_t stride = in.strides()[axis_];
              int64_t stride_blocks = cuda::ceil_div(stride, BN);
              dim3 num_blocks = get_2d_grid_dims(
                  in.shape(), in.strides(), axis_size * stride);
              if (num_blocks.x * stride_blocks <= UINT32_MAX) {
                num_blocks.x *= stride_blocks;
              } else {
                num_blocks.y *= stride_blocks;
              }
              int block_dim = (BN / N_READS) * WARP_SIZE;
              encoder.add_kernel_node(
                  kernel,
                  num_blocks,
                  block_dim,
                  in.data<T>(),
                  out.data<U>(),
                  axis_size,
                  stride,
                  stride_blocks);
            }
          });
        });
      } else {
        throw std::runtime_error(fmt::format(
            "Can not do scan op {} on inputs of {} with result of {}.",
            op_to_string<Op>(),
            dtype_to_string(in.dtype()),
            dtype_to_string(out.dtype())));
      }
    });
  });
}

} // namespace mlx::core
