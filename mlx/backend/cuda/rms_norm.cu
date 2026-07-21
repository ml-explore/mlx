// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/reduce/reduce.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/fast_primitives.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_pipeline.h>
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

// xs and ws stay in registers
// Each thread does N_CHUNKS_THREAD vectorised interleaved loads of width
// N_READS x, w, out in majority of cases should be 16 bytes aligned -> using
// unsafe_load_vector if not aligned -> fall back to load_vector we do it like
// this because of the register presure: load_vector allocates registers for
// loop load fall back and occupancy drops to 30-40% -> x2 slow down
template <
    typename T,
    int BLOCK_SIZE,
    int N_CHUNKS_THREAD,
    bool ALIGNED,
    int N_READS = 8>
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

  using BlockReduceT = BlockBroadcastReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduceT::TempStorage temp;

  auto row = grid.block_rank();
  if (row >= n_rows) {
    return;
  }
  x += row * axis_size;
  out += row * axis_size;

  AlignedVector<T, N_READS> xn[N_CHUNKS_THREAD];

  float normalizer = 0;
  auto index = block.thread_index().x;
#pragma unroll
  for (int i = 0; i < N_CHUNKS_THREAD; i++) {
    int offset = BLOCK_SIZE * i + index;
    if constexpr (ALIGNED) {
      xn[i] = unsafe_load_vector<N_READS>(x, offset);
    } else {
      xn[i] = load_vector<N_READS>(x, offset, axis_size, T(0));
    }
#pragma unroll
    for (int j = 0; j < N_READS; ++j) {
      float t = static_cast<float>(xn[i][j]);
      normalizer += t * t;
    }
  }
  normalizer = BlockReduceT{block, temp}.Sum(normalizer);
  normalizer = rsqrt(normalizer / axis_size + eps);

#pragma unroll
  for (int i = 0; i < N_CHUNKS_THREAD; i++) {
    int offset = BLOCK_SIZE * i + index;
    AlignedVector<T, N_READS> wn;
    if constexpr (ALIGNED) {
      wn = unsafe_load_vector<N_READS>(w, offset);
    } else {
      wn = load_vector<N_READS>(w, offset, axis_size, w_stride, T(0));
    }
#pragma unroll
    for (int j = 0; j < N_READS; j++) {
      float y = static_cast<float>(xn[i][j]) * normalizer;
      xn[i][j] = wn[j] * static_cast<T>(y);
    }
    if constexpr (ALIGNED) {
      unsafe_store_vector<N_READS>(out, offset, xn[i]);
    } else {
      store_vector<N_READS>(out, offset, xn[i], axis_size);
    }
  }
}

// TODO: load x to the shared memory and reload from shared memory after the
// reduction
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
__global__ void rms_norm_vjp_small_fallback(
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

template <typename T, bool HAS_W, int BLOCK_SIZE, int N_CHUNKS, int N_READS = 8>
__global__ void rms_norm_vjp_small(
    const T* x,
    const T* w,
    const T* g,
    T* gx,
    float* gw, // accumulate in float always
    float eps,
    int32_t axis_size,
    int32_t n_rows,
    int64_t w_stride) {
  // persistent kernel, numblocks = number of sms * 2;
  // each block is responsible for a row. 128*2 blocks = 256 stride.
  // we pipeline loads and computation in shared memory:
  // loading the row while doing dw, dx computation for another row

  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();

  using BlockReduceF2 = BlockBroadcastReduce<float2, BLOCK_SIZE>;
  __shared__ typename BlockReduceF2::TempStorage temp;

  // double buffering
  constexpr int STAGES = 2;
  const int num_blocks = static_cast<int>(grid.num_blocks());
  const int num_tiles = cuda::ceil_div(n_rows, num_blocks);
  auto tid = block.thread_index().x;
  auto bid = grid.block_rank();
  constexpr int buffer_size = N_READS * BLOCK_SIZE * N_CHUNKS; // axis_size

  // shared memory is dymanic because we need > 48 kb
  extern __shared__ char smem_raw[];
  T* smem_x = reinterpret_cast<T*>(smem_raw);
  T* smem_dy = smem_x + STAGES * buffer_size;

  AlignedVector<T, N_READS> wn[N_CHUNKS];
  AlignedVector<T, N_READS> xn[N_CHUNKS];
  AlignedVector<T, N_READS> gn[N_CHUNKS];
  // gw_block is summed over every row this block visits
  AlignedVector<float, N_READS> gw_block[N_CHUNKS];
  // zero init
  for (int j = 0; j < N_CHUNKS; j++) {
    for (int k = 0; k < N_READS; k++) {
      gw_block[j][k] = 0.f;
    }
  }
  // shift global pointer for each block
  x += axis_size * bid;
  g += axis_size * bid;
  gx += axis_size * bid;
  gw += axis_size * bid;

  // prefetch first row for each block
  // and load weights to registers in the same loop
  for (int j = 0; j < N_CHUNKS; j++) {
    int offset = (j * BLOCK_SIZE + tid) * N_READS;
    if (bid < n_rows) {
      __pipeline_memcpy_async(&smem_x[offset], &x[offset], sizeof(T) * N_READS);
      __pipeline_memcpy_async(
          &smem_dy[offset], &g[offset], sizeof(T) * N_READS);
    }
    wn[j] = load_vector<N_READS>(
        w, j * BLOCK_SIZE + tid, axis_size, w_stride, T(0));
  }
  // commit all N_CHUNKS 128 byte loads for the first row
  __pipeline_commit();
  // pipelineing
  for (int tile = 0; tile < num_tiles; tile++) {
    x += axis_size * num_blocks;
    g += axis_size * num_blocks;

    int next = tile + 1;
    int index = next % STAGES;
    int64_t next_row =
        static_cast<int64_t>(bid) + static_cast<int64_t>(next) * num_blocks;

    if (next < num_tiles && next_row < n_rows) {
      for (int j = 0; j < N_CHUNKS; j++) {
        int offset = (j * BLOCK_SIZE + tid) * N_READS;
        __pipeline_memcpy_async(
            &smem_x[index * buffer_size + offset],
            &x[offset],
            sizeof(T) * N_READS);
        __pipeline_memcpy_async(
            &smem_dy[index * buffer_size + offset],
            &g[offset],
            sizeof(T) * N_READS);
      }
    }
    __pipeline_commit(); // always commit, empty at the tail
    __pipeline_wait_prior(1); // always wait for 1, the tail is empty

    int64_t cur_row =
        static_cast<int64_t>(bid) + static_cast<int64_t>(tile) * num_blocks;
    if (cur_row < n_rows) {
      // load x and g from shared to registers
      // compute the reduction per row
      float2 factors = {};
      for (int j = 0; j < N_CHUNKS; j++) {
        xn[j] = unsafe_load_vector<N_READS>(
            smem_x + (tile % STAGES) * buffer_size, j * BLOCK_SIZE + tid);
        gn[j] = unsafe_load_vector<N_READS>(
            smem_dy + (tile % STAGES) * buffer_size, j * BLOCK_SIZE + tid);
        for (int k = 0; k < N_READS; k++) {
          float t = static_cast<float>(xn[j][k]);
          float wi = wn[j][k];
          float gi = gn[j][k];
          float wg = wi * gi;
          factors = plus_f2(factors, {wg * t, t * t});
        }
      }
      factors = BlockReduceF2{block, temp}.Reduce(factors, plus_f2, {});
      float meangwx = factors.x / axis_size;
      float normalizer = rsqrt(factors.y / axis_size + eps);
      float normalizer3 = normalizer * normalizer * normalizer;

      // we store dx after processing each row, accumulate dw
      T* gx_row = gx + static_cast<int64_t>(tile) * num_blocks * axis_size;
      for (int j = 0; j < N_CHUNKS; j++) {
        int offset = j * BLOCK_SIZE + tid;
        for (int k = 0; k < N_READS; k++) {
          float xi = static_cast<float>(xn[j][k]);
          float wi = wn[j][k];
          float gi = gn[j][k];
          if constexpr (HAS_W) {
            gw_block[j][k] += gi * xi * normalizer;
          }
          xn[j][k] =
              static_cast<T>(normalizer * wi * gi - xi * meangwx * normalizer3);
        }
        store_vector<N_READS>(gx_row, offset, xn[j], axis_size);
      }
    }
  }
  // store this block's fp32 partial
  if constexpr (HAS_W) {
    for (int j = 0; j < N_CHUNKS; j++) {
      int offset = j * BLOCK_SIZE + tid;
      store_vector<N_READS>(gw, offset, gw_block[j], axis_size);
    }
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
      std::integral_constant<int, 1>());
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

template <int BLOCK_SIZE, typename F>
void dispatch_chunks(int n_chunks, F&& f) {
  auto block_size = std::integral_constant<int, BLOCK_SIZE>{};
  if (n_chunks <= 1) {
    f(block_size, std::integral_constant<int, 1>{});
  } else if (n_chunks <= 2) {
    f(block_size, std::integral_constant<int, 2>{});
  } else if (n_chunks <= 3) {
    f(block_size, std::integral_constant<int, 3>{});
  } else if (n_chunks <= 4) {
    f(block_size, std::integral_constant<int, 4>{});
  } else if (n_chunks <= 5) {
    f(block_size, std::integral_constant<int, 5>{});
  } else if (n_chunks <= 6) {
    f(block_size, std::integral_constant<int, 6>{});
  } else if (n_chunks <= 7) {
    f(block_size, std::integral_constant<int, 7>{});
  } else {
    f(block_size, std::integral_constant<int, 8>{});
  }
}

template <int N_READS, typename F>
void dispatch_num_chunks(int axis_size, F&& f) {
  int nvec = (axis_size + N_READS - 1) / N_READS;
  if (axis_size <= N_READS * 64) {
    f(std::integral_constant<int, 64>{}, std::integral_constant<int, 1>{});
  } else if (nvec % 128 == 0 && nvec / 128 <= 8) {
    dispatch_chunks<128>(nvec / 128, f);
  } else if (nvec % 64 == 0 && nvec / 64 <= 8) {
    dispatch_chunks<64>(nvec / 64, f);
  } else {
    dispatch_chunks<128>((nvec + 127) / 128, f);
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
    if (axis_size <= N_READS * 128 * 8) {
      dispatch_num_chunks<
          N_READS>(axis_size, [&](auto block_size, auto n_chunks) {
        constexpr int BLOCK_SIZE = block_size();
        constexpr int N_CHUNKS = n_chunks();
        bool aligned = (axis_size == N_READS * BLOCK_SIZE * N_CHUNKS) &&
            (w_stride == 1) &&
            (reinterpret_cast<uintptr_t>(gpu_ptr<DataType>(x)) % 16 == 0) &&
            (reinterpret_cast<uintptr_t>(gpu_ptr<DataType>(w)) % 16 == 0) &&
            (reinterpret_cast<uintptr_t>(gpu_ptr<DataType>(out)) % 16 == 0);
        // MSVC can't find N_READS two lambda levels down, dispatch_bool would
        // add that second level
        auto kernel = aligned
            ? cu::rms_norm_small<DataType, BLOCK_SIZE, N_CHUNKS, true, N_READS>
            : cu::rms_norm_small<
                  DataType,
                  BLOCK_SIZE,
                  N_CHUNKS,
                  false,
                  N_READS>;
        encoder.add_kernel_node(
            kernel,
            n_rows,
            BLOCK_SIZE,
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
          gpu_ptr<DataType>(x),
          gpu_ptr<DataType>(w),
          gpu_ptr<DataType>(out),
          eps_,
          axis_size,
          w_stride);
    }
  });
}

template <typename T, int BLOCK_SIZE, int N_CHUNKS, int N_READS>
inline bool use_rmsnorm_vjp_fast(
    const array& x,
    const array& w,
    const array& g,
    const array& gx,
    bool has_w,
    int32_t axis_size,
    int64_t w_stride) {
  if (!has_w || w_stride != 1) {
    return false;
  }
  if (axis_size != N_READS * BLOCK_SIZE * N_CHUNKS) {
    return false;
  }
  auto aligned = [](const array& a) {
    return reinterpret_cast<uintptr_t>(gpu_ptr<T>(a)) % 16 == 0;
  };
  return aligned(x) && aligned(w) && aligned(g) && aligned(gx);
}

template <typename Kernel>
inline int rmsnorm_vjp_num_blocks(
    Kernel kernel,
    const Stream& s,
    int block_size,
    size_t smem_bytes,
    int32_t n_rows) {
  int dev = cu::device(s.device).cuda_device();
  int sm_count = 0;
  CHECK_CUDA_ERROR(
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));
  int blocks_per_sm = 1;
  CHECK_CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, kernel, block_size, smem_bytes));
  int64_t want = static_cast<int64_t>(sm_count) * std::max(blocks_per_sm, 1);
  return std::max<int>(1, static_cast<int>(std::min<int64_t>(want, n_rows)));
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

  bool handled = false;
  if (has_w) {
    dispatch_float_types(gx.dtype(), "rms_norm_vjp", [&](auto type_tag) {
      using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;
      constexpr int N_READS = 16 / sizeof(DataType);
      dispatch_num_chunks<N_READS>(
          axis_size, [&](auto block_size, auto n_chunks) {
            constexpr int BLOCK_SIZE = block_size();
            constexpr int N_CHUNKS = n_chunks();
            if (!use_rmsnorm_vjp_fast<DataType, BLOCK_SIZE, N_CHUNKS, N_READS>(
                    x, w, g, gx, has_w, axis_size, w_stride)) {
              return;
            }
            constexpr int STAGES = 2;
            size_t smem_bytes =
                size_t(2) * STAGES * axis_size * sizeof(DataType);
            int dev = cu::device(s.device).cuda_device();
            int smem_max = 0;
            CHECK_CUDA_ERROR(cudaDeviceGetAttribute(
                &smem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
            // we should not be here often, is never true on Ampere and later
            if (smem_bytes > static_cast<size_t>(smem_max)) {
              return;
            }
            auto kernel = cu::rms_norm_vjp_small<
                DataType,
                /*HAS_W=*/true,
                BLOCK_SIZE,
                N_CHUNKS,
                N_READS>;
            if (smem_bytes > 48000) {
              CHECK_CUDA_ERROR(cudaFuncSetAttribute(
                  reinterpret_cast<const void*>(kernel),
                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                  smem_bytes));
            }
            int num_blocks = rmsnorm_vjp_num_blocks(
                kernel, s, BLOCK_SIZE, smem_bytes, n_rows);

            // fp32 partials (num_blocks, axis_size); narrowed to gw dtype
            // below.
            array gw_temp({num_blocks, axis_size}, float32, nullptr, {});
            gw_temp.set_data(cu::malloc_async(gw_temp.nbytes(), encoder));
            encoder.add_temporary(gw_temp);
            encoder.set_input_array(x);
            encoder.set_input_array(w);
            encoder.set_input_array(g);
            encoder.set_output_array(gx);
            encoder.set_output_array(gw_temp);
            encoder.add_kernel_node_ex(
                kernel,
                dim3{static_cast<uint32_t>(num_blocks)},
                dim3{static_cast<uint32_t>(BLOCK_SIZE)},
                {}, // no cluster
                static_cast<uint32_t>(smem_bytes),
                gpu_ptr<DataType>(x),
                gpu_ptr<DataType>(w),
                gpu_ptr<DataType>(g),
                gpu_ptr<DataType>(gx),
                gpu_ptr<float>(gw_temp),
                eps_,
                axis_size,
                n_rows,
                w_stride);

            ReductionPlan plan(
                ReductionOpType::ContiguousStridedReduce,
                {num_blocks},
                {axis_size});
            if (gw.dtype() == float32) {
              col_reduce(
                  encoder, gw_temp, gw, Reduce::ReduceType::Sum, {0}, plan);
            } else {
              array gw_f32({axis_size}, float32, nullptr, {});
              col_reduce(
                  encoder, gw_temp, gw_f32, Reduce::ReduceType::Sum, {0}, plan);
              encoder.add_temporary(gw_f32);
              copy_gpu(gw_f32, gw, CopyType::General, s);
            }
            handled = true;
          });
    });
  }
  if (handled) {
    return;
  }

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
              static_assert(block_dim <= 32 || groups_per_block() == 1);
              auto kernel = cu::rms_norm_vjp_small_fallback<
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
