// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/matmul/mma.cuh"
#include "mlx/backend/cuda/matmul/tiles.cuh"
#include "mlx/backend/cuda/quantized/quantized_utils.cuh"
#include "mlx/dtype_utils.h"

namespace mlx::core {

namespace cu {

template <int NUM_WARPS, int group_size, int bits, typename T, typename Tile>
__device__ inline void load_quantized(
    Tile& tile,
    const uint8_t* x,
    const T* scales,
    const T* biases,
    int N) {
  constexpr int NUM_THREADS = NUM_WARPS * 32;
  constexpr int ELEMENTS_PER_LOAD = sizeof(uint32_t) * get_pack_factor<bits>();
  constexpr int NUM_LOADS = Tile::NUMEL / ELEMENTS_PER_LOAD;
  constexpr int NUM_LOADS_PER_THREAD = NUM_LOADS / NUM_THREADS;
  constexpr int NUM_LOADS_PER_ROW = Tile::COLS / ELEMENTS_PER_LOAD;
  constexpr int STEP_ROWS = NUM_THREADS / NUM_LOADS_PER_ROW;
  constexpr int MASK = (1 << bits) - 1;

  const int row = threadIdx.x / NUM_LOADS_PER_ROW;
  const int col = threadIdx.x % NUM_LOADS_PER_ROW;

  const int Nx = N / get_pack_factor<bits>();
  const int Ng = N / group_size;

  x += row * Nx + col * (ELEMENTS_PER_LOAD / get_pack_factor<bits>());
  scales += row * Ng + col * ELEMENTS_PER_LOAD / group_size;
  biases += row * Ng + col * ELEMENTS_PER_LOAD / group_size;

#pragma unroll
  for (int i = 0; i < NUM_LOADS_PER_THREAD; i++) {
    T vs[ELEMENTS_PER_LOAD];
    uint32_t w = *reinterpret_cast<const uint32_t*>(x + i * STEP_ROWS * Nx);
    T s = scales[i * STEP_ROWS * Ng];
    T b = biases[i * STEP_ROWS * Ng];
#pragma unroll
    for (int j = 0; j < ELEMENTS_PER_LOAD; j++) {
      vs[j] = static_cast<T>((w >> (j * bits)) & MASK) * s + b;
    }
    tile.store(vs, row + i * STEP_ROWS, col * ELEMENTS_PER_LOAD);
  }
}

template <typename T, int BM, int BN, int BK, int group_size, int bits>
__global__ void qmm(
    const T* x,
    const uint8_t* w,
    const T* scales,
    const T* biases,
    T* y,
    int M,
    int N,
    int K) {
  constexpr int NUM_WARPS = 4;
  constexpr int WARP_M = (BM / 16) / (NUM_WARPS / 2);
  constexpr int WARP_N = (BN / 16) / (NUM_WARPS / 2);
  constexpr int WARP_K = BK / 16;
  constexpr int WARP_STEP_M = WARP_M * 16;
  constexpr int WARP_STEP_N = WARP_N * 16;

  const int warpid = threadIdx.x / 32;
  const int laneid = threadIdx.x % 32;
  const int offset_m = (warpid / 2) * WARP_STEP_M;
  const int offset_n = (warpid % 2) * WARP_STEP_N;

  __shared__ SharedTile<T, BM, BK> xs;
  __shared__ SharedTile<T, BN, BK> ws;

  Tile16x16<float> C[WARP_M * WARP_N];
  Tile16x16<T> A[WARP_M];
  Tile16x16<T> B[WARP_N];

  x += blockIdx.y * BM * K;
  w += blockIdx.x * BN * K / get_pack_factor<bits>();
  scales += blockIdx.x * BN * K / group_size;
  biases += blockIdx.x * BN * K / group_size;
  y += blockIdx.y * BM * N + blockIdx.x * BN;

#pragma unroll
  for (int i = 0; i < WARP_M * WARP_N; i++) {
    C[i].clear();
  }

  uint32_t base_addr_xs = __cvta_generic_to_shared(&xs.data[0]);
  uint32_t base_addr_ws = __cvta_generic_to_shared(&ws.data[0]);

  for (int k_block = 0; k_block < K; k_block += BK) {
    load<NUM_WARPS>(xs, x + k_block, K);
    load_quantized<NUM_WARPS, group_size, bits>(
        ws,
        w + k_block / get_pack_factor<bits>(),
        scales + k_block / group_size,
        biases + k_block / group_size,
        K);
    __syncthreads();

#pragma unroll
    for (int k = 0; k < WARP_K; k++) {
#pragma unroll
      for (int i = 0; i < WARP_M; i++) {
        A[i].load(xs.loc(
            base_addr_xs,
            offset_m + i * 16 + laneid % 16,
            k * 16 + laneid / 16 * 8));
      }
#pragma unroll
      for (int i = 0; i < WARP_N; i++) {
        B[i].load(ws.loc(
            base_addr_ws,
            offset_n + i * 16 + laneid % 16,
            k * 16 + laneid / 16 * 8));
      }

#pragma unroll
      for (int i = 0; i < WARP_M; i++) {
#pragma unroll
        for (int j = 0; j < WARP_N; j++) {
          mma(C[i * WARP_N + j], A[i], B[j]);
        }
      }
    }
  }

#pragma unroll
  for (int i = 0; i < WARP_M; i++) {
#pragma unroll
    for (int j = 0; j < WARP_N; j++) {
      C[i * WARP_N + j].store_global(
          y + (offset_m + i * 16) * N + offset_n + j * 16, N);
    }
  }
}

} // namespace cu

void qmm(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    bool transpose_,
    int group_size_,
    int bits_,
    int M,
    int N,
    int K,
    cu::CommandEncoder& enc,
    const Stream& s) {
  dispatch_float_types(x.dtype(), "qmm", [&](auto type_tag) {
    dispatch_groups(group_size_, [&](auto group_size) {
      dispatch_bits(bits_, [&](auto bits) {
        using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;
        constexpr int BM = 64;
        constexpr int BN = 64;
        constexpr int BK = 32;
        auto kernel =
            cu::qmm<DataType, BM, BN, BK, group_size.value, bits.value>;

        dim3 grid(N / BN, M / BM);

        enc.add_kernel_node(
            kernel,
            grid,
            128,
            x.data<DataType>(),
            w.data<uint8_t>(),
            scales.data<DataType>(),
            biases.data<DataType>(),
            out.data<DataType>(),
            M,
            N,
            K);
      });
    });
  });
}

} // namespace mlx::core
