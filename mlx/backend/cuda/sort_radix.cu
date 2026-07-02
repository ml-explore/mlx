#include <cassert>
#include <cstdint>

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/utils.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/sort_radix.cuh"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <cooperative_groups.h>
#include <nvtx3/nvtx3.hpp>
#include <cub/block/block_scan.cuh>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

// In MoEs we usually have small number of experts (< 256)
// therefore there is no reason to implement classical radix sort
// with 4-8 passes. Instead we will implement only one pass for lowest 8 bits.
// one pass radix sort for MoE consists of 4 kernels:
// 1. histogram: compute per-tile histograms of the low byte of each key
// 2. scan_tile_column: exclusive scan of each column (bucket) across tiles
// 3. scan_bucket_totals: exclusive scan of the column totals to get bucket
// offsets (tiny kernel)
// 4. radix_scatter: scatter keys to their final positions based on the computed
// offsets Global design: to implement stable radix sort for each element x that
// fall into bucket b we need to compute 3 values:
// - how many elements are in buckets < b (bucket_offsets[b])
// - how many elements are in bucket b in tiles < t (tile_offsets[t, b])
// - how many elements are in bucket b in tile t that are < x (local_rank)
// The final position of x is then: bucket_offsets[b] + tile_offsets[t, b] +
// local_rank First 2 are computed using per tile histogram + scan. The last one
// is more complicated. It is computed inside scatter kernel itself. However,
// since we want sorting to be stable for reproducibility, we can't use
// atomicAdd to compute local_rank. Instead we will use warp-level multi-split
// (WLMS) (see coments in scatter kernel)
// TODO: all loads can be loaded as 4 uint8 packed into int32 / int64 depending
// on N
template <typename T, int N>
__global__ void radix_histogram(const T* input, uint32_t* output, size_t size) {
  // input: keys (expert ids for each token) [size] or [num_tiles * tile_size]
  // output: per-tile histograms of the low byte of each key [num_tiles, RADIX]
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();

  __shared__ uint32_t block_hist[RADIX];

  for (uint32_t b = block.thread_rank(); b < RADIX; b += block.size()) {
    block_hist[b] = 0;
  }
  block.sync();

  const size_t tile_size = block.size() * N;
  const size_t tile_start = grid.block_rank() * tile_size;

  const bool is_full_tile =
      (tile_start + tile_size <= size); // last block might be partial

  if (is_full_tile) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
      uint32_t bucket =
          static_cast<uint32_t>(
              input[tile_start + j * block.size() + block.thread_rank()]) &
          (RADIX - 1);
      atomicAdd(&block_hist[bucket], 1u);
    }
  } else {
    // Partial tile
    const size_t tail_len = size - tile_start;
    for (size_t i = block.thread_rank(); i < tail_len; i += block.size()) {
      uint32_t bucket =
          static_cast<uint32_t>(input[tile_start + i]) & (RADIX - 1);
      atomicAdd(&block_hist[bucket], 1u);
    }
  }

  block.sync();

  size_t out_start_ind = grid.block_rank() * RADIX;

  for (uint32_t i = block.thread_rank(); i < RADIX; i += block.size()) {
    output[out_start_ind + i] = block_hist[i];
  }
}

template <int BLOCK_THREADS, int N>
__global__ void scan_tile_column(
    const uint32_t* per_tile_hist,
    uint32_t* tile_offsets,
    uint32_t* column_totals,
    int num_tiles) {
  // input: per_tile_hist [num_tiles, RADIX] -- computed by radix_histogram
  // output: tile_offsets [num_tiles, RADIX],
  // column_totals [RADIX] -- exclusive scan of per_tile_hist and global
  // histogram
  using BlockScan = cub::BlockScan<uint32_t, BLOCK_THREADS>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  const uint32_t b = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  constexpr int CHUNK = BLOCK_THREADS * N;

  uint32_t running_prefix = 0;

  for (int chunk_start = 0; chunk_start < num_tiles; chunk_start += CHUNK) {
    uint32_t vals[N];
#pragma unroll
    for (int i = 0; i < N; ++i) {
      int t = chunk_start + tid * N + i;
      if (t < num_tiles) {
        vals[i] = per_tile_hist[t * RADIX + b];
      } else {
        vals[i] = 0u;
      }
    }

    uint32_t block_aggregate;
    BlockScan(temp_storage).ExclusiveSum(vals, vals, block_aggregate);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      int t = chunk_start + tid * N + i;
      if (t < num_tiles) {
        tile_offsets[t * RADIX + b] = vals[i] + running_prefix;
      }
    }

    running_prefix += block_aggregate;
    __syncthreads();
  }

  if (tid == 0) {
    column_totals[b] = running_prefix;
  }
}

__global__ void scan_bucket_totals(
    const uint32_t* column_totals,
    uint32_t* bucket_offsets) {
  __shared__ uint32_t s_totals[RADIX];
  const uint32_t b = threadIdx.x;
  // input: column_totals [RADIX] -- computed by scan_tile_column
  // output: bucket_offsets [RADIX] -- exclusive scan of column_totals
  s_totals[b] = column_totals[b];
  __syncthreads();

  if (b == 0) {
    uint32_t acc = 0;
    for (int i = 0; i < RADIX; ++i) {
      uint32_t v = s_totals[i];
      s_totals[i] = acc;
      acc += v;
    }
  }
  __syncthreads();
  bucket_offsets[b] = s_totals[b];
}

template <typename T, int BLOCK_THREADS, int N>
__global__ void radix_scatter(
    const T* input,
    uint32_t* out_indices,
    const uint32_t* tile_offsets,
    const uint32_t* bucket_offsets,
    int n) {
  constexpr int NUM_WARPS = BLOCK_THREADS / 32;
  static_assert(
      BLOCK_THREADS % 32 == 0, "BLOCK_THREADS must be a multiple of 32");

  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();

  __shared__ uint32_t s_tile_off[RADIX];
  __shared__ uint32_t s_bucket_off[RADIX];
  __shared__ uint32_t s_warp_hist[NUM_WARPS * RADIX];
  __shared__ uint32_t s_cursor[RADIX];

  const uint32_t tid = block.thread_rank();
  const uint32_t warp_id = tid >> 5;
  const uint32_t lane_id = tid & 31;
  const uint32_t lanes_lt_me = (1u << lane_id) - 1u;

  const size_t tile_start = grid.block_rank() * BLOCK_THREADS * N;
  const size_t tile_full = BLOCK_THREADS * N;
  const size_t tile_end = (tile_start + tile_full <= static_cast<size_t>(n))
      ? tile_start + tile_full
      : static_cast<size_t>(n);
  const size_t tile_len = tile_end - tile_start;
  const bool is_full_tile = (tile_len == tile_full);

  for (uint32_t i = tid; i < RADIX; i += BLOCK_THREADS) {
    s_tile_off[i] = tile_offsets[grid.block_rank() * RADIX + i];
    s_bucket_off[i] = bucket_offsets[i];
    s_cursor[i] = 0;
  }
  for (uint32_t i = tid; i < NUM_WARPS * RADIX; i += BLOCK_THREADS) {
    s_warp_hist[i] = 0;
  }
  block.sync();

  if (!is_full_tile) {
    if (tid == 0) {
      for (size_t i = 0; i < tile_len; ++i) {
        uint32_t bucket =
            static_cast<uint32_t>(input[tile_start + i]) & (RADIX - 1);
        uint32_t local_rank = s_cursor[bucket]++;
        uint32_t dest = s_bucket_off[bucket] + s_tile_off[bucket] + local_rank;
        out_indices[dest] = static_cast<uint32_t>(tile_start + i);
      }
    }
    return;
  }

  T keys[N];
#pragma unroll
  for (int j = 0; j < N; ++j) {
    keys[j] = input[tile_start + j * BLOCK_THREADS + tid];
  }

  uint32_t my_bucket[N];
  uint32_t offsets[N];

#pragma unroll
  for (int j = 0; j < N; ++j) {
    uint32_t bucket = static_cast<uint32_t>(keys[j]) & (RADIX - 1);
    my_bucket[j] = bucket;

    uint32_t peer_mask = 0xFFFFFFFFu;
#pragma unroll
    for (int k = 0; k < RADIX_BITS; ++k) {
      bool my_bit_k = (bucket >> k) & 1u;
      uint32_t mask_k = __ballot_sync(0xFFFFFFFFu, my_bit_k);
      peer_mask &= my_bit_k ? mask_k : ~mask_k;
    }

    uint32_t bits = __popc(peer_mask & lanes_lt_me);

    uint32_t preIncrementVal = 0;
    uint32_t leader_lane = __ffs(peer_mask) - 1u;
    if (lane_id == leader_lane) {
      preIncrementVal =
          atomicAdd(&s_warp_hist[warp_id * RADIX + bucket], __popc(peer_mask));
    }
    preIncrementVal = __shfl_sync(0xFFFFFFFFu, preIncrementVal, leader_lane);

    offsets[j] = preIncrementVal + bits;
  }
  block.sync();

  if (tid < RADIX) {
    uint32_t running = 0;
#pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) {
      uint32_t v = s_warp_hist[w * RADIX + tid];
      s_warp_hist[w * RADIX + tid] = running;
      running += v;
    }
  }
  block.sync();

#pragma unroll
  for (int j = 0; j < N; ++j) {
    uint32_t bucket = my_bucket[j];
    uint32_t tile_local_rank =
        s_warp_hist[warp_id * RADIX + bucket] + offsets[j];
    uint32_t dest = s_bucket_off[bucket] + s_tile_off[bucket] + tile_local_rank;
    uint32_t input_pos =
        static_cast<uint32_t>(tile_start + j * BLOCK_THREADS + tid);
    out_indices[dest] = input_pos;
  }
}

} // namespace cu

void radix_argsort(const Stream& s, const array& in, array& out) {
  auto& encoder = cu::get_command_encoder(s);

  // Allocate output.
  out.set_data(cu::malloc_async(out.nbytes(), encoder));

  int n = static_cast<int>(in.size());
  int num_tiles = (n + cu::TILE_SIZE - 1) / cu::TILE_SIZE;

  array per_tile_hist({num_tiles, cu::RADIX}, uint32, nullptr, {});
  array tile_offsets({num_tiles, cu::RADIX}, uint32, nullptr, {});
  array bucket_offsets({cu::RADIX}, uint32, nullptr, {});

  per_tile_hist.set_data(cu::malloc_async(per_tile_hist.nbytes(), encoder));
  tile_offsets.set_data(cu::malloc_async(tile_offsets.nbytes(), encoder));
  bucket_offsets.set_data(cu::malloc_async(bucket_offsets.nbytes(), encoder));

  encoder.add_temporary(per_tile_hist);
  encoder.add_temporary(tile_offsets);
  encoder.add_temporary(bucket_offsets);

  // TODO: for now works only for uint8, but we can extend for other types
  auto dispatch_by_dtype = [&](auto key_type_tag) {
    using T = decltype(key_type_tag);
    {
      encoder.set_input_array(in);
      encoder.set_output_array(per_tile_hist);
      auto kernel = cu::radix_histogram<T, cu::N_PER_THREAD>;
      dim3 grid(num_tiles, 1, 1);
      dim3 block(cu::BLOCK_THREADS, 1, 1);
      encoder.add_kernel_node(
          kernel,
          grid,
          block,
          gpu_ptr<T>(in),
          gpu_ptr<uint32_t>(per_tile_hist),
          static_cast<size_t>(n));
    }
    {
      constexpr int SCAN_BLOCK_THREADS = 128;
      constexpr int SCAN_ITEMS_PER_THREAD = 32; // 128 * 32 = 4096 per iteration
      encoder.set_input_array(per_tile_hist);
      encoder.set_output_array(tile_offsets);
      encoder.set_output_array(bucket_offsets);
      auto kernel =
          cu::scan_tile_column<SCAN_BLOCK_THREADS, SCAN_ITEMS_PER_THREAD>;
      dim3 grid(cu::RADIX, 1, 1);
      dim3 block(SCAN_BLOCK_THREADS, 1, 1);
      encoder.add_kernel_node(
          kernel,
          grid,
          block,
          gpu_ptr<uint32_t>(per_tile_hist),
          gpu_ptr<uint32_t>(tile_offsets),
          gpu_ptr<uint32_t>(bucket_offsets),
          num_tiles);
    }
    {
      encoder.set_input_array(bucket_offsets);
      encoder.set_output_array(bucket_offsets);
      auto kernel = cu::scan_bucket_totals;
      dim3 grid(1, 1, 1);
      dim3 block(cu::RADIX, 1, 1);
      encoder.add_kernel_node(
          kernel,
          grid,
          block,
          gpu_ptr<uint32_t>(bucket_offsets),
          gpu_ptr<uint32_t>(bucket_offsets));
    }
    {
      encoder.set_input_array(in);
      encoder.set_input_array(tile_offsets);
      encoder.set_input_array(bucket_offsets);
      encoder.set_output_array(out);
      auto kernel = cu::radix_scatter<T, cu::BLOCK_THREADS, cu::N_PER_THREAD>;
      dim3 grid(num_tiles, 1, 1);
      dim3 block(cu::BLOCK_THREADS, 1, 1);
      encoder.add_kernel_node(
          kernel,
          grid,
          block,
          gpu_ptr<T>(in),
          gpu_ptr<uint32_t>(out),
          gpu_ptr<uint32_t>(tile_offsets),
          gpu_ptr<uint32_t>(bucket_offsets),
          n);
    }
  };

  dispatch_by_dtype(uint8_t{});
}

} // namespace mlx::core
