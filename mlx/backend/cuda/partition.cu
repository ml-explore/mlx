// Copyright © 2026 Apple Inc.

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/radix_select.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype.h"
#include "mlx/dtype_utils.h"

namespace mlx::core {

void gpu_partition_fallback(
    const Stream& s,
    const array& in,
    array& out,
    int axis,
    bool arg_partition);

namespace {

// Upper bound for small-kernel tiling. Keep this aligned with the
// items-per-thread dispatch set and per-block shared-memory budget.
constexpr int MAX_RADIX_ITEMS_PER_THREAD = 64;
constexpr size_t RADIX_SMALL_SHARED_MEM_BUDGET_BYTES = 48 * 1024;

template <typename F>
void dispatch_radix_small_block_threads(int size_sorted_axis, F&& f) {
  if (size_sorted_axis <= 256) {
    f(std::integral_constant<int, 32>{});
  } else if (size_sorted_axis <= 512) {
    f(std::integral_constant<int, 64>{});
  } else if (size_sorted_axis <= 1024) {
    f(std::integral_constant<int, 128>{});
  } else {
    f(std::integral_constant<int, 256>{});
  }
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
      cu::RADIX_SIZE * sizeof(int) + // shared_hist for small kernel
      (2 + 3 * num_warps + 6) * sizeof(int); // shared_count + scatter scratch
}

bool radix_small_fits_shared_memory(Dtype dtype, int size_sorted_axis) {
  if (size_sorted_axis <= 0) {
    return false;
  }

  size_t required_shared_mem = 0;
  bool fits = false;
  dispatch_radix_small_block_threads(size_sorted_axis, [&](auto block_dim_tag) {
    constexpr int BLOCK_THREADS = block_dim_tag();
    int required_items = (size_sorted_axis + BLOCK_THREADS - 1) / BLOCK_THREADS;
    if (required_items > MAX_RADIX_ITEMS_PER_THREAD) {
      fits = false;
      return;
    }

    dispatch_radix_items_per_thread(
        size_sorted_axis, BLOCK_THREADS, [&](auto items_per_thread_tag) {
          constexpr int ITEMS_PER_THREAD = items_per_thread_tag();
          required_shared_mem = radix_small_shared_mem_bytes(
              size_of(dtype), BLOCK_THREADS, ITEMS_PER_THREAD);
          fits = required_shared_mem <= RADIX_SMALL_SHARED_MEM_BUDGET_BYTES;
        });
  });
  return fits;
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

        dispatch_radix_small_block_threads(
            size_sorted_axis, [&](auto block_dim_tag) {
              constexpr int BLOCK_THREADS = block_dim_tag();
              dim3 grid(1, n_rows, 1);
              dim3 block(BLOCK_THREADS, 1, 1);

              dispatch_radix_items_per_thread(
                  size_sorted_axis,
                  BLOCK_THREADS,
                  [&](auto items_per_thread_tag) {
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
                      using UnsignedT =
                          typename cu::RadixTraits<ValT>::UnsignedT;
                      constexpr int TILE_SIZE_VAL =
                          BLOCK_THREADS * ITEMS_PER_THREAD;
                      constexpr int NUM_WARPS = BLOCK_THREADS / WARP_SIZE;
                      constexpr size_t shared_mem_bytes =
                          TILE_SIZE_VAL * sizeof(UnsignedT) + // shared_keys
                          TILE_SIZE_VAL * sizeof(uint32_t) + // shared_idxs
                          cu::RADIX_SIZE *
                              sizeof(int) + // shared_hist for small kernel
                          (2 + 3 * NUM_WARPS + 6) *
                              sizeof(int); // shared_count + scatter scratch

                      encoder.add_kernel_node_ex(
                          kernel,
                          grid,
                          block,
                          {},
                          static_cast<uint32_t>(shared_mem_bytes),
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

} // namespace

void gpu_partition(
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
    return gpu_partition_fallback(s, in, out, axis, arg_partition);
  }

  // Dispatch based on whether the small kernel tile fits in shared memory.
  if (radix_small_fits_shared_memory(in.dtype(), size_sorted_axis)) {
    return gpu_radix_partition_small(s, in, out, axis, kth, arg_partition);
  } else {
    return gpu_partition_fallback(s, in, out, axis, arg_partition);
  }
}

} // namespace mlx::core
