// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/reduce/reduce.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/fast_primitives.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

inline __device__ float2 plus_f2(const float2& a, const float2& b) {
  return {a.x + b.x, a.y + b.y};
}

// Similar to cub::BlockReduce, but result is broadcasted to every thread.
template <typename T, int BLOCK_DIM, int GROUP_DIM = WARP_SIZE>
struct BlockBroadcastReduce {
  using TempStorage = T[std::max(BLOCK_DIM / WARP_SIZE, 1)];

  cg::thread_block& block;
  TempStorage& temp;

  template <typename Op>
  __device__ T Reduce(const T& input, const Op& op, const T& init_value) {
    auto warp = cg::tiled_partition<GROUP_DIM>(block);
    T x = cg::reduce(warp, input, op);
    if constexpr (BLOCK_DIM > GROUP_DIM) {
      if (warp.thread_rank() == 0) {
        temp[warp.meta_group_rank()] = x;
      }
      block.sync();
      x = warp.thread_rank() < warp.meta_group_size() ? temp[warp.thread_rank()]
                                                      : init_value;
      return cg::reduce(warp, x, op);
    } else {
      return x;
    }
  }

  __device__ T Sum(const T& input) {
    return Reduce(input, cg::plus<T>{}, T{});
  }
};

template <typename T, int BLOCK_DIM, int REDUCE_DIM, int N_READS = 4>
__global__ void rms_norm_small(
    const T* x,
    const T* w,
    T* out,
    float eps,
    uint32_t axis_size,
    uint32_t n_rows,
    int64_t w_stride) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();

  using BlockReduceT = BlockBroadcastReduce<float, BLOCK_DIM, REDUCE_DIM>;
  __shared__ typename BlockReduceT::TempStorage temp;

  auto row =
      (grid.block_rank() * block.dim_threads().y) + block.thread_index().y;
  if (row >= n_rows) {
    return;
  }
  x += row * axis_size;
  out += row * axis_size;

  // Normalizer.
  float normalizer = 0;
  auto index = block.thread_index().x;
  auto xn = load_vector<N_READS>(x, index, axis_size, T(0));
#pragma unroll
  for (int i = 0; i < N_READS; ++i) {
    float t = static_cast<float>(xn[i]);
    normalizer += t * t;
  }

  normalizer = BlockReduceT{block, temp}.Sum(normalizer);
  normalizer = rsqrt(normalizer / axis_size + eps);

  // Outputs.
  auto wn = load_vector<N_READS>(w, index, axis_size, w_stride, T(0));
#pragma unroll
  for (int i = 0; i < N_READS; ++i) {
    float y = static_cast<float>(xn[i]) * normalizer;
    xn[i] = wn[i] * static_cast<T>(y);
  }
  store_vector<N_READS>(out, index, xn, axis_size);
}

template <typename T, int BLOCK_DIM, int N_READS = 4>
__global__ void rms_norm(
    const T* x,
    const T* w,
    T* out,
    float eps,
    uint32_t axis_size,
    int64_t w_stride) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();

  using BlockReduceT = BlockBroadcastReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduceT::TempStorage temp;

  x += grid.block_rank() * axis_size;
  out += grid.block_rank() * axis_size;

  // Normalizer.
  float normalizer = 0;
  for (int r = 0; r < cuda::ceil_div(axis_size, BLOCK_DIM * N_READS); ++r) {
    auto index = r * BLOCK_DIM + block.thread_rank();
    auto xn = load_vector<N_READS>(x, index, axis_size, T(0));
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      float t = static_cast<float>(xn[i]);
      normalizer += t * t;
    }
  }
  normalizer = BlockReduceT{block, temp}.Sum(normalizer);
  normalizer = rsqrt(normalizer / axis_size + eps);

  // Outputs.
  for (int r = 0; r < cuda::ceil_div(axis_size, BLOCK_DIM * N_READS); ++r) {
    auto index = r * BLOCK_DIM + block.thread_rank();
    auto xn = load_vector<N_READS>(x, index, axis_size, T(0));
    auto wn = load_vector<N_READS>(w, index, axis_size, w_stride, T(0));
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      float y = static_cast<float>(xn[i]) * normalizer;
      xn[i] = wn[i] * static_cast<T>(y);
    }
    store_vector<N_READS>(out, index, xn, axis_size);
  }
}

template <
    typename T,
    bool HAS_W,
    int BLOCK_DIM,
    int REDUCE_DIM,
    int N_READS = 4>
__global__ void rms_norm_vjp_small(
    const T* x,
    const T* w,
    const T* g,
    T* gx,
    T* gw,
    float eps,
    int32_t axis_size,
    int32_t n_rows,
    int64_t w_stride) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();

  using BlockReduceF2 = BlockBroadcastReduce<float2, BLOCK_DIM, REDUCE_DIM>;
  __shared__ typename BlockReduceF2::TempStorage temp;

  auto row =
      (grid.block_rank() * block.dim_threads().y) + block.thread_index().y;
  if (row >= n_rows) {
    return;
  }

  x += row * axis_size;
  g += row * axis_size;
  gx += row * axis_size;
  gw += row * axis_size;

  // Normalizer.
  float2 factors = {};
  auto index = block.thread_index().x;
  auto xn = load_vector<N_READS>(x, index, axis_size, T(0));
  auto gn = load_vector<N_READS>(g, index, axis_size, T(0));
  auto wn = load_vector<N_READS>(w, index, axis_size, w_stride, T(0));
  for (int i = 0; i < N_READS; i++) {
    float t = static_cast<float>(xn[i]);
    float wi = wn[i];
    float gi = gn[i];
    float wg = wi * gi;
    factors = plus_f2(factors, {wg * t, t * t});
  }

  factors = BlockReduceF2{block, temp}.Reduce(factors, plus_f2, {});
  float meangwx = factors.x / axis_size;
  float normalizer = rsqrt(factors.y / axis_size + eps);
  float normalizer3 = normalizer * normalizer * normalizer;

  // Outputs.
  for (int i = 0; i < N_READS; i++) {
    float xi = xn[i];
    float wi = wn[i];
    float gi = gn[i];
    xn[i] = static_cast<T>(normalizer * wi * gi - xi * meangwx * normalizer3);
    if constexpr (HAS_W) {
      wn[i] = static_cast<T>(gi * xi * normalizer);
    }
  }
  store_vector<N_READS>(gx, index, xn, axis_size);
  if constexpr (HAS_W) {
    store_vector<N_READS>(gw, index, wn, axis_size);
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

  using BlockReduceF2 = BlockBroadcastReduce<float2, BLOCK_DIM>;
  __shared__ typename BlockReduceF2::TempStorage temp;

  x += grid.block_rank() * axis_size;
  g += grid.block_rank() * axis_size;
  gx += grid.block_rank() * axis_size;
  gw += grid.block_rank() * axis_size;

  // Normalizer.
  float2 factors = {};
  for (int r = 0; r < cuda::ceil_div(axis_size, BLOCK_DIM * N_READS); ++r) {
    auto index = r * BLOCK_DIM + block.thread_rank();
    auto xn = load_vector<N_READS>(x, index, axis_size, T(0));
    auto gn = load_vector<N_READS>(g, index, axis_size, T(0));
    auto wn = load_vector<N_READS>(w, index, axis_size, w_stride, T(0));
    for (int i = 0; i < N_READS; i++) {
      float t = static_cast<float>(xn[i]);
      float wi = wn[i];
      float gi = gn[i];
      float wg = wi * gi;
      factors = plus_f2(factors, {wg * t, t * t});
    }
  }
  factors = BlockReduceF2{block, temp}.Reduce(factors, plus_f2, {});
  float meangwx = factors.x / axis_size;
  float normalizer = rsqrt(factors.y / axis_size + eps);
  float normalizer3 = normalizer * normalizer * normalizer;

  // Outputs.
  for (int r = 0; r < cuda::ceil_div(axis_size, BLOCK_DIM * N_READS); ++r) {
    auto index = r * BLOCK_DIM + block.thread_rank();
    auto xn = load_vector<N_READS>(x, index, axis_size, T(0));
    auto gn = load_vector<N_READS>(g, index, axis_size, T(0));
    auto wn = load_vector<N_READS>(w, index, axis_size, w_stride, T(0));
    for (int i = 0; i < N_READS; i++) {
      float xi = xn[i];
      float wi = wn[i];
      float gi = gn[i];
      xn[i] = static_cast<T>(normalizer * wi * gi - xi * meangwx * normalizer3);
      if constexpr (HAS_W) {
        wn[i] = static_cast<T>(gi * xi * normalizer);
      }
    }
    store_vector<N_READS>(gx, index, xn, axis_size);
    if constexpr (HAS_W) {
      store_vector<N_READS>(gw, index, wn, axis_size);
    }
  }
}

} // namespace cu

namespace fast {

bool RMSNorm::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

template <int n_per_thread, typename F>
void dispatch_group_dim(int axis_size, F&& f) {
  if (axis_size <= n_per_thread * 8) {
    f(std::integral_constant<int, 8>{},
      std::integral_constant<int, 1>(),
      std::integral_constant<int, 16>());
  } else if (axis_size <= n_per_thread * 16) {
    f(std::integral_constant<int, 16>{},
      std::integral_constant<int, 1>(),
      std::integral_constant<int, 8>());
  } else if (axis_size <= n_per_thread * 32) {
    f(std::integral_constant<int, 32>{},
      std::integral_constant<int, 1>(),
      std::integral_constant<int, 4>());
  } else if (axis_size <= n_per_thread * 32 * 2) {
    f(std::integral_constant<int, 32>{},
      std::integral_constant<int, 2>(),
      std::integral_constant<int, 2>());
  } else if (axis_size <= n_per_thread * 32 * 4) {
    f(std::integral_constant<int, 32>{},
      std::integral_constant<int, 4>(),
      std::integral_constant<int, 1>());
  } else if (axis_size <= n_per_thread * 32 * 8) {
    f(std::integral_constant<int, 32>{},
      std::integral_constant<int, 8>(),
      std::integral_constant<int, 1>());
  } else if (axis_size <= n_per_thread * 32 * 16) {
    f(std::integral_constant<int, 32>{},
      std::integral_constant<int, 16>(),
      std::integral_constant<int, 1>());
  } else {
    f(std::integral_constant<int, 32>{},
      std::integral_constant<int, 32>(),
      std::integral_constant<int, 1>());
  }
}

// TODO: There are duplicate code with backend/metal/normalization.cpp
void RMSNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("RMSNorm::eval_gpu");
  auto& s = stream();
  auto& out = outputs[0];
  auto& encoder = cu::get_command_encoder(s);

  // Make sure that the last dimension is contiguous.
  auto set_output = [&s, &out, &encoder](const array& x) {
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
            cu::malloc_async(x.data_size() * x.itemsize(), encoder),
            x.data_size(),
            x.strides(),
            x.flags());
      }
      return x;
    } else {
      array x_copy = contiguous_copy_gpu(x, s);
      out.copy_shared_buffer(x_copy);
      return x_copy;
    }
  };

  const array x = set_output(inputs[0]);
  const array& w = inputs[1];

  int32_t axis_size = x.shape().back();
  int32_t n_rows = x.data_size() / axis_size;
  int64_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;

  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_output_array(out);
  dispatch_float_types(out.dtype(), "rms_norm", [&](auto type_tag) {
    using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    constexpr int N_READS = 16 / sizeof(DataType);
    if (axis_size <= N_READS * 1024) {
      dispatch_group_dim<N_READS>(
          axis_size, [&](auto group_dim, auto n_groups, auto groups_per_block) {
            constexpr int block_dim = n_groups() * group_dim();
            auto kernel =
                cu::rms_norm_small<DataType, block_dim, group_dim(), N_READS>;
            auto n_blocks =
                (n_rows + groups_per_block() - 1) / groups_per_block();
            encoder.add_kernel_node(
                kernel,
                n_blocks,
                {block_dim, groups_per_block()},
                0,
                gpu_ptr<DataType>(x),
                gpu_ptr<DataType>(w),
                gpu_ptr<DataType>(out),
                eps_,
                axis_size,
                n_rows,
                w_stride);
          });
    } else {
      auto kernel = cu::rms_norm<DataType, 1024, N_READS>;
      encoder.add_kernel_node(
          kernel,
          n_rows,
          1024,
          0,
          gpu_ptr<DataType>(x),
          gpu_ptr<DataType>(w),
          gpu_ptr<DataType>(out),
          eps_,
          axis_size,
          w_stride);
    }
  });
}

void RMSNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("RMSNormVJP::eval_gpu");
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  // Ensure row contiguity. We could relax this step by checking that the array
  // is contiguous (no broadcasts or holes) and that the input strides are the
  // same as the cotangent strides but for now this is simpler.
  auto check_input = [&s](const array& x, bool& copied) {
    if (x.flags().row_contiguous) {
      copied = false;
      return x;
    }
    copied = true;
    return contiguous_copy_gpu(x, s);
  };
  bool donate_x = inputs[0].is_donatable();
  bool donate_g = inputs[2].is_donatable();
  bool copied;
  auto x = check_input(inputs[0], copied);
  donate_x |= copied;
  const array& w = inputs[1];
  bool g_copied;
  auto g = check_input(inputs[2], g_copied);
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
    gx.set_data(cu::malloc_async(gx.nbytes(), encoder));
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
      gw_temp.set_data(cu::malloc_async(gw_temp.nbytes(), encoder));
      encoder.add_temporary(gw_temp);
    }
  }

  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(g);
  encoder.set_output_array(gx);
  encoder.set_output_array(gw_temp);
  dispatch_float_types(gx.dtype(), "rms_norm_vjp", [&](auto type_tag) {
    dispatch_bool(has_w, [&](auto has_w_constant) {
      using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;
      constexpr int N_READS = 16 / sizeof(DataType);
      if (axis_size <= N_READS * 1024) {
        dispatch_group_dim<N_READS>(
            axis_size,
            [&](auto group_dim, auto n_groups, auto groups_per_block) {
              constexpr int block_dim = group_dim() * n_groups();
              auto kernel = cu::rms_norm_vjp_small<
                  DataType,
                  has_w_constant.value,
                  block_dim,
                  group_dim(),
                  N_READS>;
              auto n_blocks =
                  (n_rows + groups_per_block() - 1) / groups_per_block();
              encoder.add_kernel_node(
                  kernel,
                  n_blocks,
                  {block_dim, groups_per_block()},
                  0,
                  gpu_ptr<DataType>(x),
                  gpu_ptr<DataType>(w),
                  gpu_ptr<DataType>(g),
                  gpu_ptr<DataType>(gx),
                  gpu_ptr<DataType>(gw_temp),
                  eps_,
                  axis_size,
                  n_rows,
                  w_stride);
            });
      } else {
        auto kernel =
            cu::rms_norm_vjp<DataType, has_w_constant.value, 1024, N_READS>;
        encoder.add_kernel_node(
            kernel,
            n_rows,
            1024,
            0,
            gpu_ptr<DataType>(x),
            gpu_ptr<DataType>(w),
            gpu_ptr<DataType>(g),
            gpu_ptr<DataType>(gx),
            gpu_ptr<DataType>(gw_temp),
            eps_,
            axis_size,
            w_stride);
      }
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
