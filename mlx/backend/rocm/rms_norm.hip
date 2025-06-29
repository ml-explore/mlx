// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/iterators/strided_iterator.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/reduce/reduce.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/fast_primitives.h"

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <rocprim/block/block_load.hpp>
#include <rocprim/block/block_reduce.hpp>

namespace mlx::core {

namespace rocm {

namespace cg = cooperative_groups;

// Similar to rocprim::BlockReduce, but result is broadcasted to every thread.
template <typename T, int BLOCK_DIM>
struct BlockBroadcastReduce {
  static_assert(WARP_SIZE <= BLOCK_DIM && BLOCK_DIM <= WARP_SIZE * WARP_SIZE);
  static_assert(BLOCK_DIM % WARP_SIZE == 0);
  using TempStorage = T[BLOCK_DIM / WARP_SIZE];

  cg::thread_block& block;
  TempStorage& temp;

  template <typename Op>
  __device__ T Reduce(const T& input, const Op& op, const T& init_value) {
    auto warp = cg::tiled_partition<WARP_SIZE>(block);
    T x = cg::reduce(warp, input, op);
    if (warp.thread_rank() == 0) {
      temp[warp.meta_group_rank()] = x;
    }
    block.sync();
    x = warp.thread_rank() < warp.meta_group_size() ? temp[warp.thread_rank()]
                                                    : init_value;
    return cg::reduce(warp, x, op);
  }

  __device__ T Sum(const T& input) {
    return Reduce(input, hip_plus<T>{}, T{});
  }
};

template <typename T, int BLOCK_DIM, int N_READS = 4>
__global__ void rms_norm(
    const T* x,
    const T* w,
    T* out,
    float eps,
    int32_t axis_size,
    int64_t w_stride) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();

  using BlockReduceT = BlockBroadcastReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduceT::TempStorage temp;

  x += grid.block_rank() * axis_size;
  out += grid.block_rank() * axis_size;

  // Sum of squares.
  float sum_sq = 0;
  for (int r = 0; r < hip_ceil_div(axis_size, BLOCK_DIM * N_READS); ++r) {
    auto index = r * BLOCK_DIM + block.thread_rank();
    T xn[N_READS] = {};
    rocprim::block_load_direct_blocked(index, x, xn, axis_size);
    for (int i = 0; i < N_READS; ++i) {
      float val = static_cast<float>(xn[i]);
      sum_sq += val * val;
    }
  }
  sum_sq = BlockReduceT{block, temp}.Sum(sum_sq);

  // RMS normalizer.
  float rms_normalizer = rsqrt(sum_sq / axis_size + eps);

  // Outputs.
  for (int r = 0; r < hip_ceil_div(axis_size, BLOCK_DIM * N_READS); ++r) {
    auto index = r * BLOCK_DIM + block.thread_rank();
    T xn[N_READS];
    T wn[N_READS];
    rocprim::block_load_direct_blocked(index, x, xn, axis_size);
    rocprim::block_load_direct_blocked(index, strided_iterator(w, w_stride), wn, axis_size);
    for (int i = 0; i < N_READS; ++i) {
      float norm = static_cast<float>(xn[i]) * rms_normalizer;
      xn[i] = wn[i] * static_cast<T>(norm);
    }
    rocprim::block_store_direct_blocked(index, out, xn, axis_size);
  }
}

template <typename T, bool HAS_W, int BLOCK_DIM, int N_READS = 4>
__global__ void rms_norm_vjp(
    const T* x,
    const T* w,
    const T* g,
    T* gx,
    T* gw,
    float eps,
    int32_t axis_size,
    int64_t w_stride) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();

  using BlockReduceF = BlockBroadcastReduce<float, BLOCK_DIM>;
  using BlockReduceF2 = BlockBroadcastReduce<float2, BLOCK_DIM>;
  __shared__ union {
    typename BlockReduceF::TempStorage f;
    typename BlockReduceF2::TempStorage f2;
  } temp;

  x += grid.block_rank() * axis_size;
  g += grid.block_rank() * axis_size;
  gx += grid.block_rank() * axis_size;
  gw += grid.block_rank() * axis_size;

  // Sum of squares.
  float sum_sq = 0;
  for (int r = 0; r < hip_ceil_div(axis_size, BLOCK_DIM * N_READS); ++r) {
    auto index = r * BLOCK_DIM + block.thread_rank();
    T xn[N_READS] = {};
    rocprim::block_load_direct_blocked(index, x, xn, axis_size);
    for (int i = 0; i < N_READS; ++i) {
      float val = static_cast<float>(xn[i]);
      sum_sq += val * val;
    }
  }
  sum_sq = BlockReduceF{block, temp.f}.Sum(sum_sq);

  // RMS normalizer.
  float rms_normalizer = rsqrt(sum_sq / axis_size + eps);

  // Compute gradient terms.
  float2 factors = {};
  for (int r = 0; r < hip_ceil_div(axis_size, BLOCK_DIM * N_READS); ++r) {
    T xn[N_READS];
    T wn[N_READS] = {};
    T gn[N_READS] = {};
    auto index = r * BLOCK_DIM + block.thread_rank();
    rocprim::block_load_direct_blocked(index, x, xn, axis_size);
    rocprim::block_load_direct_blocked(index, g, gn, axis_size);
    rocprim::block_load_direct_blocked(index, strided_iterator(w, w_stride), wn, axis_size);
    for (int i = 0; i < N_READS; i++) {
      float xi = static_cast<float>(xn[i]);
      float wi = wn[i];
      float gi = gn[i];
      float wg = wi * gi;
      factors.x += wg;
      factors.y += wg * xi;
    }
  }
  auto plus_f2 = [] __device__ (const float2& a, const float2& b) -> float2 {
    return {a.x + b.x, a.y + b.y};
  };
  factors = BlockReduceF2{block, temp.f2}.Reduce(factors, plus_f2, {});
  float mean_wg = factors.x / axis_size;
  float mean_wgx = factors.y / axis_size;
  float rms3 = rms_normalizer * rms_normalizer * rms_normalizer;

  // Outputs.
  for (int r = 0; r < hip_ceil_div(axis_size, BLOCK_DIM * N_READS); ++r) {
    auto index = r * BLOCK_DIM + block.thread_rank();
    T xn[N_READS];
    T wn[N_READS];
    T gn[N_READS];
    rocprim::block_load_direct_blocked(index, x, xn, axis_size);
    rocprim::block_load_direct_blocked(index, g, gn, axis_size);
    rocprim::block_load_direct_blocked(index, strided_iterator(w, w_stride), wn, axis_size);
    for (int i = 0; i < N_READS; i++) {
      float xi = static_cast<float>(xn[i]);
      float wi = wn[i];
      float gi = gn[i];
      float norm = xi * rms_normalizer;
      xn[i] = rms_normalizer * (wi * gi - mean_wg) - norm * mean_wgx * rms3;
      if constexpr (HAS_W) {
        wn[i] = gi * norm;
      }
    }
    rocprim::block_store_direct_blocked(index, gx, xn, axis_size);
    if constexpr (HAS_W) {
      rocprim::block_store_direct_blocked(index, gw, wn, axis_size);
    }
  }
}

// Utility functions
template <typename T>
struct hip_plus {
  __device__ T operator()(const T& a, const T& b) const {
    return a + b;
  }
};

inline __device__ int hip_ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

template <typename T>
__device__ inline auto strided_iterator(const T* ptr, int64_t stride) {
  return ptr + stride;  // Simplified strided iterator
}

} // namespace rocm

namespace fast {

bool RMSNorm::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void RMSNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& out = outputs[0];

  // Make sure that the last dimension is contiguous.
  auto set_output = [&s, &out](const array& x) {
    bool no_copy = x.flags().contiguous && x.strides()[x.ndim() - 1] == 1;
    if (no_copy && x.ndim() > 1) {
      auto s = x.strides()[x.ndim() - 2];
      no_copy &= (s == 0 || s == x.shape().back());
    }
    if (no_copy) {
      if (x.is_donatable()) {
        out.copy_shared_buffer(x);
      } else {
        out.set_data(
            allocator::malloc(x.data_size() * x.itemsize()),
            x.data_size(),
            x.strides(),
            x.flags());
      }
      return x;
    } else {
      auto x_copy = array(x.shape(), x.dtype(), nullptr, {});
      copy_gpu(x, x_copy, CopyType::General, s);
      out.copy_shared_buffer(x_copy);
      return x_copy;
    }
  };

  const array x = set_output(inputs[0]);
  const array& w = inputs[1];

  int32_t axis_size = x.shape().back();
  int32_t n_rows = x.data_size() / axis_size;
  int64_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;

  auto& encoder = rocm::get_command_encoder(s);
  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_output_array(out);
  encoder.launch_kernel([&](hipStream_t stream) {
    MLX_SWITCH_FLOAT_TYPES_CHECKED(out.dtype(), "rmsnorm", CTYPE, {
      using DataType = hip_type_t<CTYPE>;
      constexpr uint32_t N_READS = 4;
      MLX_SWITCH_BLOCK_DIM(rocm::hip_ceil_div(axis_size, N_READS), BLOCK_DIM, {
        auto kernel = rocm::rms_norm<DataType, BLOCK_DIM, N_READS>;
        hipLaunchKernelGGL(kernel, n_rows, BLOCK_DIM, 0, stream,
            x.data<DataType>(),
            w.data<DataType>(),
            out.data<DataType>(),
            eps_,
            axis_size,
            w_stride);
      });
    });
  });
}

void RMSNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  // Ensure row contiguity. We could relax this step by checking that the array
  // is contiguous (no broadcasts or holes) and that the input strides are the
  // same as the cotangent strides but for now this is simpler.
  auto check_input = [&s](const array& x) -> std::pair<array, bool> {
    if (x.flags().row_contiguous) {
      return {x, false};
    }
    array x_copy(x.shape(), x.dtype(), nullptr, {});
    copy_gpu(x, x_copy, CopyType::General, s);
    return {x_copy, true};
  };
  bool donate_x = inputs[0].is_donatable();
  bool donate_g = inputs[2].is_donatable();
  auto [x, copied] = check_input(inputs[0]);
  donate_x |= copied;
  const array& w = inputs[1];
  auto [g, g_copied] = check_input(inputs[2]);
  donate_g |= g_copied;
  array& gx = outputs[0];
  array& gw = outputs[1];

  // Check whether we had a weight.
  bool has_w = w.ndim() != 0;

  // Allocate space for the outputs.
  bool g_in_gx = false;
  if (donate_x) {
    gx.copy_shared_buffer(x);
  } else if (donate_g) {
    gx.copy_shared_buffer(g);
    g_in_gx = true;
  } else {
    gx.set_data(allocator::malloc(gx.nbytes()));
  }
  if (g_copied && !g_in_gx) {
    encoder.add_temporary(g);
  }

  int32_t axis_size = x.shape().back();
  int32_t n_rows = x.data_size() / axis_size;
  int64_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;

  // Allocate a temporary to store the gradients for w and allocate the output
  // gradient accumulators.
  array gw_temp =
      (has_w) ? array({n_rows, x.shape().back()}, gw.dtype(), nullptr, {}) : w;
  if (has_w) {
    if (!g_in_gx && donate_g) {
      gw_temp.copy_shared_buffer(g);
    } else {
      gw_temp.set_data(allocator::malloc(gw_temp.nbytes()));
      encoder.add_temporary(gw_temp);
    }
  }
  gw.set_data(allocator::malloc(gw.nbytes()));

  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(g);
  encoder.set_output_array(gx);
  encoder.set_output_array(gw_temp);
  encoder.launch_kernel([&, x = x, g = g](hipStream_t stream) {
    MLX_SWITCH_FLOAT_TYPES_CHECKED(gx.dtype(), "rmsnorm_vjp", CTYPE, {
      using DataType = hip_type_t<CTYPE>;
      constexpr int N_READS = 4;
      MLX_SWITCH_BOOL(has_w, HAS_W, {
        MLX_SWITCH_BLOCK_DIM(rocm::hip_ceil_div(axis_size, N_READS), BLOCK_DIM, {
          auto kernel = rocm::rms_norm_vjp<DataType, HAS_W, BLOCK_DIM, N_READS>;
          hipLaunchKernelGGL(kernel, n_rows, BLOCK_DIM, 0, stream,
              x.data<DataType>(),
              w.data<DataType>(),
              g.data<DataType>(),
              gx.data<DataType>(),
              gw_temp.data<DataType>(),
              eps_,
              axis_size,
              w_stride);
        });
      });
    });
  });

  if (has_w) {
    ReductionPlan plan(
        ReductionOpType::ContiguousStridedReduce, {n_rows}, {axis_size});
    col_reduce(encoder, gw_temp, gw, Reduce::ReduceType::Sum, {0}, plan);
  }
}

} // namespace fast

} // namespace mlx::core 