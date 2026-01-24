
#include "mlx/backend/cuda/steel/mma.cuh"
#include "mlx/backend/cuda/steel/tiles.cuh"

namespace mlx::core::cu {

/**
 * An example gemm written with the utils.
 *
 * Computes A @ B.T when A and B are all aligned with the block sizes.
 */
template <typename T, int BM, int BN, int BK>
__global__ void ab_t_aligned(const T* a, const T* b, T* y, int N, int K) {
  constexpr int WARPS_M = 2;
  constexpr int WARPS_N = 2;
  constexpr int NUM_WARPS = WARPS_M * WARPS_N;
  constexpr int WARP_STEP_M = BM / WARPS_M;
  constexpr int WARP_STEP_N = BN / WARPS_N;

  // Precompute some offsets for each thread
  const int warpid = threadIdx.x / 32;
  const int laneid = threadIdx.x % 32;
  const int wm = warpid / WARPS_N;
  const int wn = warpid % WARPS_N;
  const int offset_m = wm * WARP_STEP_M;
  const int offset_n = wn * WARP_STEP_N;

  // Allocate shared memory
  extern __shared__ char shmem[];
  SharedTile<T, BM, BK>(&as)[2] = *(SharedTile<T, BM, BK>(*)[2])(&shmem[0]);
  SharedTile<T, BN, BK>(&bs)[2] =
      *(SharedTile<T, BN, BK>(*)[2])(&shmem[sizeof(T) * 2 * BM * BK]);

  // Allocate registers for the MMA
  RegisterTile<float, BM / WARPS_M, BN / WARPS_N> C;
  RegisterTile<T, BM / WARPS_M, 16> A;
  RegisterTile<T, BN / WARPS_N, 16> B;

  // Move the global pointers to the tile
  a += blockIdx.y * BM * K;
  b += blockIdx.x * BN * K;
  y += blockIdx.y * BM * N + blockIdx.x * BN;

  // Zero the accumulators
  C.fill(0);

  // Start the SM pipeline
  load_async<NUM_WARPS>(as[0], as[0].base_addr(), a, K);
  load_async<NUM_WARPS>(bs[0], bs[0].base_addr(), b, K);
  cp_async_commit();

  int tic = 0;
  for (int k_block = BK; k_block < K; k_block += BK) {
    load_async<NUM_WARPS>(as[tic ^ 1], as[tic ^ 1].base_addr(), a + k_block, K);
    load_async<NUM_WARPS>(bs[tic ^ 1], bs[tic ^ 1].base_addr(), b + k_block, K);
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();

    MLX_UNROLL
    for (int k = 0; k < BK / 16; k++) {
      A.load(
          as[tic],
          as[tic].base_addr(),
          offset_m + laneid % 16,
          k * 16 + laneid / 16 * 8);
      B.load(
          bs[tic],
          bs[tic].base_addr(),
          offset_n + laneid % 16,
          k * 16 + laneid / 16 * 8);

      mma_t(C, A, B);
    }

    tic ^= 1;
  }

  // Empty the pipeline
  cp_async_wait_all();
  __syncthreads();
  MLX_UNROLL
  for (int k = 0; k < BK / 16; k++) {
    A.load(
        as[tic],
        as[tic].base_addr(),
        offset_m + laneid % 16,
        k * 16 + laneid / 16 * 8);
    B.load(
        bs[tic],
        bs[tic].base_addr(),
        offset_n + laneid % 16,
        k * 16 + laneid / 16 * 8);

    mma_t(C, A, B);
  }

  C.store_global(y, N, offset_m, offset_n);
}

} // namespace mlx::core::cu
