// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/cast_op.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/fast_primitives.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <nvtx3/nvtx3.hpp>

#include <cassert>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

// fused together logsumexp + gather
// cast to float32 inside the kernel
// to avoid logits.astype(mx.float32)
// for each row: loss = logsumexp(x) - x_t
// first we accumulate logsumexp, then we do a gather
template <typename T, int BLOCK_DIM, int N_READS = 4>
__global__ void cross_entropy(
    const T* x, // [M, N]
    const int* y, // [M,]
    float* loss, // [M,] <- will be always in fp32 lse - x
    int axis_size // N
) {
  cg::greater<float> max_op;
  cg::plus<float> plus_op;

  float prevmax;
  float curmax = Limits<float>::finite_min();
  float normalizer = 0;

  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  x += grid.block_rank() * axis_size; // offset input
  for (int r = 0; r < cuda::ceil_div(axis_size, BLOCK_DIM * N_READS); r++) {
    auto index = r * BLOCK_DIM + block.thread_rank();
    auto vals = load_vector<N_READS>(x, index, axis_size, Limits<T>::min());
    prevmax = curmax;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      curmax = max_op(curmax, static_cast<float>(vals[i]));
    }
    // scale already accumulated normiliser
    normalizer = normalizer * __expf(prevmax - curmax);
    // add vals scaled by curmax
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      normalizer += __expf(static_cast<float>(vals[i]) - curmax);
    }
  }
  prevmax = curmax;
  curmax = cg::reduce(warp, curmax, max_op);
  normalizer = normalizer * __expf(prevmax - curmax);
  normalizer = cg::reduce(warp, normalizer, plus_op);
  // second reduce in a block
  __shared__ float warp_max[WARP_SIZE];
  __shared__ float warp_normaliser[WARP_SIZE];

  if (warp.thread_rank() == 0) {
    warp_max[warp.meta_group_rank()] = curmax;
    warp_normaliser[warp.meta_group_rank()] = normalizer;
  }
  block.sync();
  bool is_valid = warp.thread_rank() < warp.meta_group_size();
  curmax =
      is_valid ? warp_max[warp.thread_rank()] : Limits<float>::finite_min();
  prevmax = curmax;
  curmax =
      cg::reduce(warp, curmax, max_op); // max within a block (global row max)
  normalizer = is_valid ? warp_normaliser[warp.thread_rank()] : 0.0f;
  normalizer = normalizer * __expf(prevmax - curmax);
  normalizer = cg::reduce(warp, normalizer, plus_op);
  // gather and writing the output:
  auto row = grid.block_rank();
  if (block.thread_rank() == 0) {
    float gap = curmax - static_cast<float>(x[y[row]]);
    loss[row] = isinf(curmax) ? gap : log(normalizer) + gap;
  }
}

// get loss from the forward
template <typename T, int BLOCK_DIM, int N_READS = 4>
__global__ void cross_entropy_vjp(
    const T* x, // [M, N]
    const int* y, // [M,]
    const float* loss, // [M,]
    const float* gy, // cotangent [M,]
    T* grads, // [M, N] lse is accumulated in float, x is casted to float
    int axis_size // N
) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto row = grid.block_rank();

  x += row * axis_size; // offset input
  grads += row * axis_size; // offset output
  auto y_n = y[row]; // target index [0, N)
  auto g = gy[row]; // cotangent
  auto loss_n = loss[row];
  auto x_t = static_cast<float>(x[y_n]);
  block.sync();
  for (int r = 0; r < cuda::ceil_div(axis_size, BLOCK_DIM * N_READS); r++) {
    auto index = r * BLOCK_DIM + block.thread_rank(); // [0, N)
    auto vals = load_vector<N_READS>(x, index, axis_size, T{});
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      int col = index * N_READS + i;
      float val = __expf((static_cast<float>(vals[i]) - x_t) - loss_n);
      vals[i] = static_cast<T>(g * (val - (col == y_n ? 1.0f : 0.0f)));
    }
    store_vector<N_READS>(grads, index, vals, axis_size);
  }
}
} // namespace cu

namespace fast {

bool CrossEntropy::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void CrossEntropy::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("CrossEntropy::eval_gpu");
  assert(inputs.size() == 2); // logits and target
  auto& s = stream();
  auto& out = outputs[0];
  auto& encoder = cu::get_command_encoder(s);
  auto ensure_row_contiguous = [&s, &encoder](const array& x) {
    if (x.flags().row_contiguous) {
      return x;
    } else {
      array x_copy = contiguous_copy_gpu(x, s);
      encoder.add_temporary(x_copy);
      return x_copy;
    }
  };
  auto in = ensure_row_contiguous(inputs[0]); // [n_rows, V]
  auto target = ensure_row_contiguous(inputs[1]); // [n_rows,]
  out.set_data(cu::malloc_async(out.nbytes(), encoder)); // [n_rows] in fp32

  int axis_size = in.shape().back();
  int n_rows = in.data_size() / axis_size;

  encoder.set_input_array(in);
  encoder.set_input_array(target);
  encoder.set_output_array(out);
  dispatch_float_types(in.dtype(), "cross_entropy", [&](auto type_tag) {
    using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    constexpr int N_READS = 16 / sizeof(DataType);
    dispatch_block_dim(cuda::ceil_div(axis_size, N_READS), [&](auto block_dim) {
      auto kernel = cu::cross_entropy<DataType, block_dim(), N_READS>;
      encoder.add_kernel_node(
          kernel,
          n_rows,
          block_dim(),
          gpu_ptr<DataType>(in),
          gpu_ptr<int>(target),
          gpu_ptr<float>(out),
          axis_size);
    });
  });
}

void CrossEntropyVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("CrossEntropyVJP::eval_gpu");
  assert(inputs.size() == 4); // logits, target, loss, cotangent
  auto& s = stream();
  auto& out = outputs[0];
  auto& encoder = cu::get_command_encoder(s);
  auto ensure_row_contiguous = [&s, &encoder](const array& x) {
    if (x.flags().row_contiguous) {
      return x;
    } else {
      array x_copy = contiguous_copy_gpu(x, s);
      encoder.add_temporary(x_copy);
      return x_copy;
    }
  };

  auto check_input = [&s](const array& x, bool& copied) {
    if (x.flags().row_contiguous) {
      copied = false;
      return x;
    }
    copied = true;
    return contiguous_copy_gpu(x, s);
  };
  bool donate_x = inputs[0].is_donatable();
  bool copied;
  auto in = check_input(inputs[0], copied); // [n_rows, V]
  donate_x |= copied;
  auto target = ensure_row_contiguous(inputs[1]); // [n_rows,]
  auto loss = ensure_row_contiguous(inputs[2]); // [n_rows,] fp32
  auto cotan = ensure_row_contiguous(inputs[3]); // [n_rows,] fp32
  if (donate_x) {
    out.copy_shared_buffer(in);
  } else {
    out.set_data(cu::malloc_async(out.nbytes(), encoder)); // [n_rows, V]
  }

  int axis_size = in.shape().back();
  int n_rows = in.data_size() / axis_size;

  encoder.set_input_array(in);
  encoder.set_input_array(target);
  encoder.set_input_array(loss);
  encoder.set_input_array(cotan);
  encoder.set_output_array(out);
  dispatch_float_types(in.dtype(), "cross_entropy_vjp", [&](auto type_tag) {
    using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    constexpr int N_READS = 16 / sizeof(DataType);
    dispatch_block_dim(cuda::ceil_div(axis_size, N_READS), [&](auto block_dim) {
      auto kernel = cu::cross_entropy_vjp<DataType, block_dim(), N_READS>;
      encoder.add_kernel_node(
          kernel,
          n_rows,
          block_dim(),
          gpu_ptr<DataType>(in),
          gpu_ptr<int>(target),
          gpu_ptr<float>(loss),
          gpu_ptr<float>(cotan),
          gpu_ptr<DataType>(out),
          axis_size);
    });
  });
}

} // namespace fast

} // namespace mlx::core
