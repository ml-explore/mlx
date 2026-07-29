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
// for each row: logsumexp(x) - x_t
// first we accumulate logsumexp, then we do a gather
template <typename T, int BLOCK_DIM, int N_READS = 4>
__global__ void cross_entropy(
    const T* x, // [M, N]
    const int* y, // [M,]
    float* loss, // [M,] <- will be always in fp32 lse - x
    float* lse, // logsumexp for backward [M,] in fp32
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
  // here every thread has it's own normiliser : N_READS values with stride 32
  // and max for all this values we need to exchange it with other threads 1) in
  // a warp 2) in a block first reduce in a warp
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
    float lse_val = isinf(curmax) ? curmax : log(normalizer) + curmax;
    lse[row] = lse_val;
    loss[row] = lse_val - static_cast<float>(x[y[row]]);
  }
}

// get lse from the forward, i think we will assume non negative indc
template <typename T, int BLOCK_DIM, int N_READS = 4>
__global__ void cross_entropy_vjp(
    const T* x, // [M, N]
    const int* y, // [M,]
    const float* lse, // [M,]
    const float* gy, // cotangent [M,]
    T* grads, // [M, N] lse is accumulated in float, x is casted to float
    int axis_size // N
) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto row = grid.block_rank();

  x += row * axis_size; // offset input
  grads += row * axis_size; // offset output
  auto lse_n = lse[row]; // logsumexp
  auto y_n = y[row]; // target index [0, N)
  auto g = gy[row]; // cotangent
  for (int r = 0; r < cuda::ceil_div(axis_size, BLOCK_DIM * N_READS); r++) {
    auto index = r * BLOCK_DIM + block.thread_rank(); // [0, N)
    auto vals = load_vector<N_READS>(x, index, axis_size, T{});
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      // lse(x) >= max(x), so the exponent is <= 0 and __expf is safe
      int col = index * N_READS + i;
      float val = __expf(static_cast<float>(vals[i]) - lse_n);
      vals[i] = static_cast<T>(g * (val - (col == y_n ? 1.0f : 0.0f)));
    }
    store_vector<N_READS>(grads, index, vals, axis_size);
  }
}
} // namespace cu

} // namespace mlx::core
