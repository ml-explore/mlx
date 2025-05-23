// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/iterators/strided_iterator.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/reduce/reduce.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/fast_primitives.h"

#include <cooperative_groups.h>
#include <nvtx3/nvtx3.hpp>
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

inline __device__ float2 plus(const float2& a, const float2& b) {
  return float2{a.x + b.x, a.y + b.y};
}

template <typename T, int BLOCK_DIM, int N_READS = 4>
__global__ void layer_norm(
    const T* x,
    const T* w,
    const T* b,
    T* out,
    float eps,
    uint32_t axis_size,
    uint32_t w_stride,
    uint32_t b_stride) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();

  x += grid.block_rank() * axis_size;
  out += grid.block_rank() * axis_size;

  float2 sum = {};
  for (int r = 0; r < cuda::ceil_div(axis_size, BLOCK_DIM * N_READS); r++) {
    auto index = r * BLOCK_DIM + block.thread_rank();
    float xn[N_READS] = {};
    cub::LoadDirectBlocked(index, x, xn, axis_size);
    for (int i = 0; i < N_READS; i++) {
      float xi = xn[i];
      sum = plus(sum, float2{xi, xi * xi});
    }
  }

  using BlockReduceT = cub::BlockReduce<float2, BLOCK_DIM>;
  __shared__ typename BlockReduceT::TempStorage temp;
  sum = BlockReduceT(temp).Reduce(sum, plus);

  __shared__ float local_mean;
  __shared__ float local_normalizer;
  if (block.thread_rank() == 0) {
    float mean = sum.x / axis_size;
    float variance = sum.y / axis_size - mean * mean;
    local_mean = mean;
    local_normalizer = rsqrt(variance + eps);
  }
  block.sync();

  float mean = local_mean;
  float normalizer = local_normalizer;

  for (int r = 0; r < cuda::ceil_div(axis_size, BLOCK_DIM * N_READS); r++) {
    auto index = r * BLOCK_DIM + block.thread_rank();
    T xn[N_READS];
    T wn[N_READS];
    T bn[N_READS];
    cub::LoadDirectBlocked(index, x, xn, axis_size);
    cub::LoadDirectBlocked(index, strided_iterator(w, w_stride), wn, axis_size);
    cub::LoadDirectBlocked(index, strided_iterator(b, b_stride), bn, axis_size);
    for (int i = 0; i < N_READS; i++) {
      float norm = (static_cast<float>(xn[i]) - mean) * normalizer;
      xn[i] = wn[i] * static_cast<T>(norm) + bn[i];
    }
    cub::StoreDirectBlocked(index, out, xn, axis_size);
  }
}

struct SumVJP {
  float x;
  float x2;
  float wg;
  float wgx;
};

inline __device__ SumVJP plus_vjp(const SumVJP& a, const SumVJP& b) {
  return SumVJP{a.x + b.x, a.x2 + b.x2, a.wg + b.wg, a.wgx + b.wgx};
}

template <typename T, bool HAS_W, int BLOCK_DIM, int N_READS = 4>
__global__ void layer_norm_vjp(
    const T* x,
    const T* w,
    const T* g,
    T* gx,
    T* gw,
    float eps,
    uint32_t axis_size,
    uint32_t w_stride) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();

  x += grid.block_rank() * axis_size;
  g += grid.block_rank() * axis_size;
  gx += grid.block_rank() * axis_size;
  gw += grid.block_rank() * axis_size;

  SumVJP sum = {};
  for (int r = 0; r < cuda::ceil_div(axis_size, BLOCK_DIM * N_READS); r++) {
    T xn[N_READS] = {};
    T wn[N_READS] = {};
    T gn[N_READS] = {};
    int index = r * BLOCK_DIM + block.thread_rank();
    cub::LoadDirectBlocked(index, x, xn, axis_size);
    cub::LoadDirectBlocked(index, g, gn, axis_size);
    cub::LoadDirectBlocked(index, strided_iterator(w, w_stride), wn, axis_size);
    for (int i = 0; i < N_READS; i++) {
      float xi = xn[i];
      float wi = wn[i];
      float gi = gn[i];
      float wg = wi * gi;
      sum = plus_vjp(sum, SumVJP{xi, xi * xi, wg, wg * xi});
    }
  }

  using BlockReduceT = cub::BlockReduce<SumVJP, BLOCK_DIM>;
  __shared__ typename BlockReduceT::TempStorage temp;
  sum = BlockReduceT(temp).Reduce(sum, plus_vjp);

  __shared__ float local_mean;
  __shared__ float local_normalizer;
  __shared__ float local_meanwg;
  __shared__ float local_meanwgx;
  if (block.thread_rank() == 0) {
    float mean = sum.x / axis_size;
    float variance = sum.x2 / axis_size - mean * mean;
    local_mean = mean;
    local_normalizer = rsqrt(variance + eps);
    local_meanwg = sum.wg / axis_size;
    local_meanwgx = sum.wgx / axis_size;
  }
  block.sync();

  float mean = local_mean;
  float normalizer = local_normalizer;
  float meanwg = local_meanwg;
  float meanwgxc = local_meanwgx - meanwg * mean;
  ;
  float normalizer2 = normalizer * normalizer;

  for (int r = 0; r < cuda::ceil_div(axis_size, BLOCK_DIM * N_READS); r++) {
    auto index = r * BLOCK_DIM + block.thread_rank();
    T xn[N_READS];
    T wn[N_READS];
    T gn[N_READS];
    cub::LoadDirectBlocked(index, x, xn, axis_size);
    cub::LoadDirectBlocked(index, g, gn, axis_size);
    cub::LoadDirectBlocked(index, strided_iterator(w, w_stride), wn, axis_size);
    for (int i = 0; i < N_READS; i++) {
      float xi = (static_cast<float>(xn[i]) - mean) * normalizer;
      float wi = wn[i];
      float gi = gn[i];
      xn[i] = normalizer * (wi * gi - meanwg) - xi * meanwgxc * normalizer2;
      if constexpr (HAS_W) {
        wn[i] = gi * xi;
      }
    }
    cub::StoreDirectBlocked(index, gx, xn, axis_size);
    if constexpr (HAS_W) {
      cub::StoreDirectBlocked(index, gw, wn, axis_size);
    }
  }
}

} // namespace cu

namespace fast {

// TODO: The implementation is similar to backend/metal/normalization.cpp
void LayerNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("LayerNorm::eval_gpu");
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

  array o = set_output(inputs[0]);
  const array& x = o.data_shared_ptr() ? o : out;
  const array& w = inputs[1];
  const array& b = inputs[2];

  uint32_t axis_size = x.shape().back();
  uint32_t n_rows = x.data_size() / axis_size;
  uint32_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;
  uint32_t b_stride = (b.ndim() == 1) ? b.strides()[0] : 0;

  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(b);
  encoder.set_output_array(out);
  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_FLOAT_TYPES_CHECKED(out.dtype(), "layernorm", CTYPE, {
      using DataType = cuda_type_t<CTYPE>;
      constexpr uint32_t N_READS = 4;
      MLX_SWITCH_BLOCK_DIM(cuda::ceil_div(axis_size, N_READS), BLOCK_DIM, {
        auto kernel = cu::layer_norm<DataType, BLOCK_DIM, N_READS>;
        kernel<<<n_rows, BLOCK_DIM, 0, stream>>>(
            x.data<DataType>(),
            w.data<DataType>(),
            b.data<DataType>(),
            out.data<DataType>(),
            eps_,
            axis_size,
            w_stride,
            b_stride);
      });
    });
  });
}

void LayerNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("LayerNormVJP::eval_gpu");
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

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
  bool donate_g = inputs[3].is_donatable();
  auto [x, copied] = check_input(inputs[0]);
  donate_x |= copied;
  const array& w = inputs[1];
  const array& b = inputs[2];
  auto [g, g_copied] = check_input(inputs[3]);
  donate_g |= g_copied;
  array& gx = outputs[0];
  array& gw = outputs[1];
  array& gb = outputs[2];

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

  uint32_t axis_size = x.shape().back();
  int32_t n_rows = x.data_size() / axis_size;
  uint32_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;

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
  gb.set_data(allocator::malloc(gb.nbytes()));

  // Finish with the gradient for b in case we had a b.
  if (gb.ndim() == 1 && gb.size() == axis_size) {
    ReductionPlan plan(
        ReductionOpType::ContiguousStridedReduce, {n_rows}, {axis_size});
    col_reduce(encoder, g, gb, Reduce::ReduceType::Sum, {0}, plan);
  }

  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(g);
  encoder.set_output_array(gx);
  encoder.set_output_array(gw_temp);
  encoder.launch_kernel([&, x = x, g = g](cudaStream_t stream) {
    MLX_SWITCH_FLOAT_TYPES_CHECKED(gx.dtype(), "layernorm_vjp", CTYPE, {
      using DataType = cuda_type_t<CTYPE>;
      constexpr int N_READS = 4;
      MLX_SWITCH_BOOL(has_w, HAS_W, {
        MLX_SWITCH_BLOCK_DIM(cuda::ceil_div(axis_size, N_READS), BLOCK_DIM, {
          auto kernel = cu::layer_norm_vjp<DataType, HAS_W, BLOCK_DIM, N_READS>;
          kernel<<<n_rows, BLOCK_DIM, 0, stream>>>(
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
