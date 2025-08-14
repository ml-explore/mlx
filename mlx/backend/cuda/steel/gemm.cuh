
#include "mlx/backend/cuda/steel/mma.cuh"
#include "mlx/backend/cuda/steel/tiles.cuh"

namespace mlx::core::cu {

template <typename T, int BM, int BN, int BK, int WM, int WN>
__device__ inline void gemm_ab_t(
    RegisterTile<float, BM / WM, BN / WN>& C,
    SharedTile<T, BM, BK>& As,
    SharedTile<T, BN, BK>& Bs,
    RegisterTileLoader<SharedTile<T, BM, BK>>& rloader_a,
    RegisterTileLoader<SharedTile<T, BN, BK>>& rloader_b) {
  RegisterTile<T, BM / WM, 16> A[2];
  RegisterTile<T, BN / WN, 16> B[2];

  rloader_a.load(A[0], As.base_addr(), 0);
  rloader_b.load(B[0], Bs.base_addr(), 0);

  MLX_UNROLL
  for (int k = 1; k < BK / 16; k++) {
    rloader_a.load(A[k & 1], As.base_addr(), k);
    rloader_b.load(B[k & 1], Bs.base_addr(), k);

    mma_t(C, A[(k - 1) & 1], B[(k - 1) & 1]);
  }
  mma_t(C, A[(BK / 16 - 1) & 1], B[(BK / 16 - 1) & 1]);
}

/**
 * An example gemm written with the utils.
 *
 * Computes A @ B.T when A and B are all aligned with the block sizes.
 */
// template <typename T, int BM, int BN, int BK, int WM, int WN, int PIPE>
//__global__ __launch_bounds__(WM * WN * WARP_SIZE, 1)
// void ab_t_aligned(const T* a, const T* b, T* y, int N, int K) {
//   constexpr int NUM_WARPS = WM * WN;
//   constexpr int WARP_STEP_M = BM / WM;
//   constexpr int WARP_STEP_N = BN / WN;
//
//   // Precompute some offsets for each thread
//   const int warpid = threadIdx.x / 32;
//   const int laneid = threadIdx.x % 32;
//   const int wm = warpid / WN;
//   const int wn = warpid % WN;
//   const int offset_m = wm * WARP_STEP_M;
//   const int offset_n = wn * WARP_STEP_N;
//
//   // Allocate shared memory
//   extern __shared__ char shmem[];
//   SharedTile<T, BM, BK>(&as)[PIPE] =
//       *(SharedTile<T, BM, BK>(*)[PIPE])(&shmem[0]);
//   SharedTile<T, BN, BK>(&bs)[PIPE] =
//       *(SharedTile<T, BN, BK>(*)[PIPE])(&shmem[sizeof(T) * PIPE * BM * BK]);
//
//   // Move the global pointers to the tile
//   a += blockIdx.y * BM * K;
//   b += blockIdx.x * BN * K;
//   y += blockIdx.y * BM * N + blockIdx.x * BN;
//
//   // Make the loaders to/from SMEM
//   SharedTileLoader<NUM_WARPS, SharedTile<T, BM, BK>> sloader_a(a, K);
//   SharedTileLoader<NUM_WARPS, SharedTile<T, BN, BK>> sloader_b(b, K);
//   RegisterTileLoader<SharedTile<T, BM, BK>> rloader_a(offset_m, laneid);
//   RegisterTileLoader<SharedTile<T, BN, BK>> rloader_b(offset_n, laneid);
//
//   // Start the SM pipeline
//   MLX_UNROLL
//   for (int i = 0; i < PIPE - 1; i++) {
//     sloader_a.load_async(as[i].base_addr());
//     sloader_b.load_async(bs[i].base_addr());
//     cp_async_commit();
//     sloader_a.next();
//     sloader_b.next();
//   }
//
//   // Allocate and zero the MMA accumulator
//   RegisterTile<float, BM / WM, BN / WN> C;
//   C.fill(0);
//
//   // Matmul loop
//   int num_blocks = K / BK;
//   int sread = 0;
//   int swrite = PIPE - 1;
//   for (int i = 0; i < num_blocks; i++) {
//     cp_async_wait<PIPE - 1>();
//
//     gemm_ab_t<T, BM, BN, BK, WM, WN>(
//         C, as[sread], bs[sread], rloader_a, rloader_b);
//
//     sloader_a.load_async(as[swrite].base_addr());
//     sloader_b.load_async(bs[swrite].base_addr());
//     cp_async_commit();
//     sloader_a.next(i + PIPE < num_blocks);
//     sloader_b.next(i + PIPE < num_blocks);
//
//     swrite = sread;
//     sread = (sread + 1) % PIPE;
//   }
//
//   C.store_global(y, N, offset_m, offset_n);
// }

template <typename T, int BM, int BN, int BK, int WM, int WN, int PIPE>
__global__ __launch_bounds__(
    WM* WN* WARP_SIZE,
    1) void ab_t_aligned(const T* a, const T* b, T* y, int N, int K) {
  constexpr int NUM_WARPS = WM * WN;
  constexpr int WARP_STEP_M = BM / WM;
  constexpr int WARP_STEP_N = BN / WN;

  // Precompute some offsets for each thread
  const int warpid = threadIdx.x / 32;
  const int laneid = threadIdx.x % 32;
  const int wm = warpid / WN;
  const int wn = warpid % WN;
  const int offset_m = wm * WARP_STEP_M;
  const int offset_n = wn * WARP_STEP_N;

  // Allocate shared memory
  extern __shared__ char shmem[];
  SharedTile<T, BM, BK>(&as)[PIPE] =
      *(SharedTile<T, BM, BK>(*)[PIPE])(&shmem[0]);
  SharedTile<T, BN, BK>(&bs)[PIPE] =
      *(SharedTile<T, BN, BK>(*)[PIPE])(&shmem[sizeof(T) * PIPE * BM * BK]);

  // Move the global pointers to the tile
  a += blockIdx.y * BM * K;
  b += blockIdx.x * BN * K;
  y += blockIdx.y * BM * N + blockIdx.x * BN;

  // Make the loaders to/from SMEM
  using sloader = SharedTileLoader<NUM_WARPS, SharedTile<T, BM, BK>>;
  constexpr int SSTEP = sloader::STEP_ROWS * sizeof(T) * BK;
  const int srow = threadIdx.x / sloader::NUM_LOADS_PER_ROW;
  const int scol =
      (threadIdx.x % sloader::NUM_LOADS_PER_ROW) * sloader::ELEMENTS_PER_LOAD;
  a += srow * K + scol;
  b += srow * K + scol;
  uint32_t sm_offsets[PIPE][2];
  MLX_UNROLL
  for (int s = 0; s < PIPE; s++) {
    sm_offsets[s][0] = as[s].loc(as[s].base_addr(), srow, scol);
    sm_offsets[s][1] = bs[s].loc(bs[s].base_addr(), srow, scol);
  }
  RegisterTileLoader<SharedTile<T, BM, BK>> rloader_a(offset_m, laneid);
  RegisterTileLoader<SharedTile<T, BN, BK>> rloader_b(offset_n, laneid);

  // Start the SM pipeline
  MLX_UNROLL
  for (int s = 0; s < PIPE - 1; s++) {
    MLX_UNROLL
    for (int l = 0; l < sloader::NUM_LOADS_PER_THREAD; l++) {
      cp_async<16>(sm_offsets[s][0] + l * SSTEP, a);
      cp_async<16>(sm_offsets[s][1] + l * SSTEP, b);
      a += sloader::STEP_ROWS * K;
      b += sloader::STEP_ROWS * K;
    }
    cp_async_commit();
  }

  // Allocate and zero the MMA accumulator
  RegisterTile<float, BM / WM, BN / WN> C;
  C.fill(0);

  // Matmul loop
  int num_blocks = K / BK;
  int sread = 0;
  int swrite = PIPE - 1;
  for (int i = 0; i < num_blocks; i++) {
    cp_async_wait<PIPE - 1>();

    gemm_ab_t<T, BM, BN, BK, WM, WN>(
        C, as[sread], bs[sread], rloader_a, rloader_b);

    if (false) {
      MLX_UNROLL
      for (int l = 0; l < sloader::NUM_LOADS_PER_THREAD; l++) {
        cp_async<16>(sm_offsets[swrite][0] + l * SSTEP, a);
        cp_async<16>(sm_offsets[swrite][1] + l * SSTEP, b);
        a += sloader::STEP_ROWS * K;
        b += sloader::STEP_ROWS * K;
      }
    }
    cp_async_commit();

    swrite = sread;
    sread = (sread + 1) % PIPE;
  }

  C.store_global(y, N, offset_m, offset_n);
}

} // namespace mlx::core::cu
