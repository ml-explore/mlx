
#include "mlx/backend/cuda/steel/mma.cuh"
#include "mlx/backend/cuda/steel/tiles.cuh"

namespace mlx::core::cu {

template <typename T, int BM, int BN, int BK, int WM, int WN>
__device__ inline void gemm_ab_t(
    RegisterTile<float, BM / WM, BN / WN>& C,
    SharedTile<T, BM, BK>& As,
    SharedTile<T, BM, BK>& Bs,
    int lane_row_a,
    int lane_row_b,
    int lane_col) {
  RegisterTile<T, BM / WM, 16> A[2];
  RegisterTile<T, BN / WN, 16> B[2];

  A[0].load(As, As.base_addr(), lane_row_a, lane_col);
  B[0].load(Bs, Bs.base_addr(), lane_row_b, lane_col);

  MLX_UNROLL
  for (int k = 1; k < BK / 16; k++) {
    A[k & 1].load(As, As.base_addr(), lane_row_a, lane_col + k * 16);
    B[k & 1].load(Bs, Bs.base_addr(), lane_row_b, lane_col + k * 16);

    mma_t(C, A[(k - 1) & 1], B[(k - 1) & 1]);
  }
  mma_t(C, A[(BK / 16 - 1) & 1], B[(BK / 16 - 1) & 1]);
}

/**
 * An example gemm written with the utils.
 *
 * Computes A @ B.T when A and B are all aligned with the block sizes.
 */
template <typename T, int BM, int BN, int BK>
__global__ void ab_t_aligned(const T* a, const T* b, T* y, int N, int K) {
  constexpr int WARPS_M = 4;
  constexpr int WARPS_N = 2;
  constexpr int NUM_WARPS = WARPS_M * WARPS_N;
  constexpr int WARP_STEP_M = BM / WARPS_M;
  constexpr int WARP_STEP_N = BN / WARPS_N;
  constexpr int PIPE = 4;

  // Precompute some offsets for each thread
  const int warpid = threadIdx.x / 32;
  const int laneid = threadIdx.x % 32;
  const int wm = warpid / WARPS_N;
  const int wn = warpid % WARPS_N;
  const int offset_m = wm * WARP_STEP_M;
  const int offset_n = wn * WARP_STEP_N;
  const int lane_row_a = offset_m + (laneid & 15);
  const int lane_row_b = offset_n + (laneid & 15);
  const int lane_col = (laneid >> 4) << 3;

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

  // Start the SM pipeline
  MLX_UNROLL
  for (int i = 0; i < PIPE - 1; i++) {
    load_async<NUM_WARPS>(as[i], as[i].base_addr(), a + i * BK, K);
    load_async<NUM_WARPS>(bs[i], bs[i].base_addr(), b + i * BK, K);
    cp_async_commit();
  }

  // Allocate and zero the MMA accumulator
  RegisterTile<float, BM / WARPS_M, BN / WARPS_N> C;
  C.fill(0);

  // Matmul loop
  int num_blocks = K / BK;
  int k_block = (PIPE - 1) * BK;
  int sread = 0;
  int swrite = PIPE - 1;
  for (int i = 0; i < num_blocks; i++) {
    cp_async_wait<PIPE - 2>();

    if (k_block < K) {
      load_async<NUM_WARPS>(as[swrite], as[swrite].base_addr(), a + k_block, K);
      load_async<NUM_WARPS>(bs[swrite], bs[swrite].base_addr(), b + k_block, K);
    }

    gemm_ab_t<T, BM, BN, BK, WARPS_M, WARPS_N>(
        C, as[sread], bs[sread], lane_row_a, lane_row_b, lane_col);

    cp_async_commit();

    swrite = sread;
    sread = (sread + 1) % PIPE;
  }

  C.store_global(y, N, offset_m, offset_n);
}

} // namespace mlx::core::cu
