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

  MLX_UNROLL
  for (int i = 0; i < NUM_LOADS_PER_THREAD; i++) {
    T vs[ELEMENTS_PER_LOAD];
    uint32_t w = *reinterpret_cast<const uint32_t*>(x + i * STEP_ROWS * Nx);
    T s = scales[i * STEP_ROWS * Ng];
    T b = biases[i * STEP_ROWS * Ng];
    MLX_UNROLL
    for (int j = 0; j < ELEMENTS_PER_LOAD; j++) {
      vs[j] = static_cast<T>((w >> (j * bits)) & MASK) * s + b;
    }
    tile.store(vs, row + i * STEP_ROWS, col * ELEMENTS_PER_LOAD);
  }
}

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int group_size,
    int bits,
    bool aligned_M>
__global__ void qmm_t(
    const T* x,
    const uint8_t* w,
    const T* scales,
    const T* biases,
    T* y,
    int M,
    int N,
    int K) {
  constexpr int WARPS_M = 2;
  constexpr int WARPS_N = 4;
  constexpr int NUM_WARPS = WARPS_M * WARPS_N;
  constexpr int WARP_STEP_M = BM / WARPS_M;
  constexpr int WARP_STEP_N = BN / WARPS_N;

  const int warpid = threadIdx.x / 32;
  const int laneid = threadIdx.x % 32;
  const int wm = warpid / WARPS_N;
  const int wn = warpid % WARPS_N;
  const int offset_m = wm * WARP_STEP_M;
  const int offset_n = wn * WARP_STEP_N;

  extern __shared__ char shmem[];
  SharedTile<T, BM, BK>(&xs)[1] = *(SharedTile<T, BM, BK>(*)[1])(&shmem[0]);
  SharedTile<T, BN, BK>(&ws)[1] =
      *(SharedTile<T, BN, BK>(*)[1])(&shmem[1 * sizeof(T) * BM * BK]);

  RegisterTile<float, BM / WARPS_M, BN / WARPS_N> C;
  RegisterTile<T, BM / WARPS_M, 16> A;
  RegisterTile<T, BN / WARPS_N, 16> B;

  const int max_rows = M - blockIdx.y * BM;

  x += blockIdx.y * BM * K;
  w += blockIdx.x * BN * K / get_pack_factor<bits>();
  scales += blockIdx.x * BN * K / group_size;
  biases += blockIdx.x * BN * K / group_size;
  y += blockIdx.y * BM * N + blockIdx.x * BN;

  C.fill(0);

  int tic = 0;
  uint32_t base_addr_xs[1], base_addr_ws[1];
  base_addr_xs[0] = __cvta_generic_to_shared(&xs[0].data[0]);
  base_addr_ws[0] = __cvta_generic_to_shared(&ws[0].data[0]);

  if (aligned_M || max_rows >= BM) {
    for (int k_block = 0; k_block < K; k_block += BK) {
      load_async<NUM_WARPS>(xs[tic], base_addr_xs[tic], x + k_block, K);
      cp_async_commit();
      load_quantized<NUM_WARPS, group_size, bits>(
          ws[tic],
          w + k_block / get_pack_factor<bits>(),
          scales + k_block / group_size,
          biases + k_block / group_size,
          K);
      cp_async_wait_all();
      __syncthreads();

      MLX_UNROLL
      for (int k = 0; k < BK / 16; k++) {
        A.load(
            xs[tic],
            base_addr_xs[tic],
            offset_m + laneid % 16,
            k * 16 + laneid / 16 * 8);
        B.load(
            ws[tic],
            base_addr_ws[tic],
            offset_n + laneid % 16,
            k * 16 + laneid / 16 * 8);
        mma_t(C, A, B);
      }
    }
    C.store_global(y, N, offset_m, offset_n);
  } else {
    for (int k_block = 0; k_block < K; k_block += BK) {
      load_async_safe<NUM_WARPS>(
          xs[tic], base_addr_xs[tic], x + k_block, K, max_rows);
      cp_async_commit();
      load_quantized<NUM_WARPS, group_size, bits>(
          ws[tic],
          w + k_block / get_pack_factor<bits>(),
          scales + k_block / group_size,
          biases + k_block / group_size,
          K);
      cp_async_wait_all();
      __syncthreads();

      MLX_UNROLL
      for (int k = 0; k < BK / 16; k++) {
        A.load(
            xs[tic],
            base_addr_xs[tic],
            offset_m + laneid % 16,
            k * 16 + laneid / 16 * 8);
        B.load(
            ws[tic],
            base_addr_ws[tic],
            offset_n + laneid % 16,
            k * 16 + laneid / 16 * 8);
        mma_t(C, A, B);
      }
    }
    C.store_global_safe(y, N, offset_m, offset_n, max_rows);
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
  if (x.dtype() != bfloat16) {
    throw std::invalid_argument("[qmm] Only bfloat16 is supported for now");
  }
  if (!transpose_) {
    throw std::invalid_argument(
        "[qmm] Only transposed matmul is supported for now");
  }

  dispatch_float_types(x.dtype(), "qmm", [&](auto type_tag) {
    dispatch_groups(group_size_, [&](auto group_size) {
      dispatch_bits(bits_, [&](auto bits) {
        using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;

        constexpr int BM = 128;
        constexpr int BN = 128;
        constexpr int BK = 32;
        auto kernel =
            cu::qmm_t<DataType, BM, BN, BK, group_size.value, bits.value, true>;
        if (M % BM != 0) {
          kernel = cu::
              qmm_t<DataType, BM, BN, BK, group_size.value, bits.value, false>;
        }

        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

        enc.add_kernel_node(
            kernel,
            grid,
            2 * 4 * 32,
            1 * sizeof(DataType) * (BM * BK + BN * BK),
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
