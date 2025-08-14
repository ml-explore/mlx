#include "mlx/backend/common/matmul.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/utils.cuh"
#include "mlx/backend/cuda/gemms/steel_gemm.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>
#include <numeric>

#include <cooperative_groups.h>

#include "mlx/backend/cuda/steel/gemm.cuh"
#include "mlx/backend/cuda/steel/mma.cuh"
#include "mlx/backend/cuda/steel/tiles.cuh"

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

struct GemmParams {
  int M;
  int N;
  int K;
  int lda;
  int ldb;
  int ldd;

  int NblockM;
  int NblockN;
  int NblockK;
};

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    int SL,
    int Nstages>
__global__ void kernel_steel_gemm(
    const T* a,
    const T* b,
    T* d,
    __grid_constant__ const GemmParams params) {
  const int bM_idx = (blockIdx.y << SL) + (blockIdx.x & ((1 << SL) - 1));
  const int bN_idx = blockIdx.x >> SL;

  if (params.NblockN <= bN_idx || params.NblockM <= bM_idx) {
    return;
  }

  const int d_row = bM_idx * BM;
  const int d_col = bN_idx * BN;
  const size_t d_row_long = size_t(d_row);
  const size_t d_col_long = size_t(d_col);

  a += transpose_a ? d_row_long : d_row_long * params.K;
  b += transpose_b ? d_col_long * params.K : d_col_long;
  d += d_row_long * params.ldd + d_col_long;

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);

  const int lane_idx = warp.thread_rank();
  const int warp_idx = warp.meta_group_rank();

  const int wm = warp_idx / WN;
  const int wn = warp_idx % WN;

  constexpr int SM = BM / WM;
  constexpr int SN = BN / WN;
  constexpr int SK = BK;
  constexpr int TK = SK / 16;

  constexpr int NUM_WARPS = WM * WN;

  // Allocate shared memory
  extern __shared__ char shmem[];
  SharedTile<T, BM, BK>(&as)[Nstages] =
      *(SharedTile<T, BM, BK>(*)[Nstages])(&shmem[0]);
  SharedTile<T, BN, BK>(&bs)[Nstages] = *(SharedTile<T, BN, BK>(*)[Nstages])(
      &shmem[sizeof(T) * Nstages * BM * BK]);

  // Allocate registers for the MMA
  RegisterTile<float, SM, SN> C;
  RegisterTile<T, SM, 16> A[TK];
  RegisterTile<T, SN, 16> B[TK];

  // Zero the accumulators
  C.fill(0);

  // Start gmem -> smem copies
  int k_block_read = 0;

  MLX_UNROLL
  for (int bk = 0; bk < (Nstages - 1); bk++) {
    load_async<NUM_WARPS>(
        as[bk], as[bk].base_addr(), a + k_block_read, params.K);
    load_async<NUM_WARPS>(
        bs[bk], bs[bk].base_addr(), b + k_block_read, params.K);
    k_block_read += BK;
    cp_async_commit();
  }

  int smem_pipe_read = 0;
  int smem_pipe_write = Nstages - 1;

  // Wait till only 1 remains laoding
  cp_async_wait<1>();
  block.sync();

  const int offset_m = wm * SM;
  const int offset_n = wn * SN;

  // Start smem -> register copy
  A[0].load(
      as[smem_pipe_read],
      as[smem_pipe_read].base_addr(),
      offset_m + lane_idx % 16,
      lane_idx / 16 * 8);
  B[0].load(
      bs[smem_pipe_read],
      bs[smem_pipe_read].base_addr(),
      offset_n + lane_idx % 16,
      lane_idx / 16 * 8);

  // Main loop
  for (int kb = 0; kb < params.NblockK; kb++) {
    // Prepare next registers
    {
      A[1].load(
          as[smem_pipe_read],
          as[smem_pipe_read].base_addr(),
          offset_m + lane_idx % 16,
          16 + lane_idx / 16 * 8);
      B[1].load(
          bs[smem_pipe_read],
          bs[smem_pipe_read].base_addr(),
          offset_n + lane_idx % 16,
          16 + lane_idx / 16 * 8);
    }

    // Prepare next smem
    if ((kb + Nstages - 1) < params.NblockK) {
      load_async<NUM_WARPS>(
          as[smem_pipe_write],
          as[smem_pipe_write].base_addr(),
          a + k_block_read,
          params.K);
      load_async<NUM_WARPS>(
          bs[smem_pipe_write],
          bs[smem_pipe_write].base_addr(),
          b + k_block_read,
          params.K);
    }
    k_block_read += BK;

    cp_async_commit();

    smem_pipe_write = smem_pipe_read;
    smem_pipe_read = smem_pipe_read + 1;
    smem_pipe_read = (smem_pipe_read == Nstages) ? 0 : smem_pipe_read;

    // Do current gemm
    mma_t(C, A[0], B[0]);

    // Do wait for next register
    cp_async_wait<1>();
    block.sync();

    // Prepare next register (smem_pipe_read has moved to the next)
    {
      A[0].load(
          as[smem_pipe_read],
          as[smem_pipe_read].base_addr(),
          offset_m + lane_idx % 16,
          lane_idx / 16 * 8);
      B[0].load(
          bs[smem_pipe_read],
          bs[smem_pipe_read].base_addr(),
          offset_n + lane_idx % 16,
          lane_idx / 16 * 8);
    }

    // Do current gemm
    mma_t(C, A[1], B[1]);
  }

  // Wait and clear
  cp_async_wait_all();
  block.sync();

  C.store_global(d, params.ldd, offset_m, offset_n);
}

} // namespace cu

void dispatch_steel_gemm(
    const Stream& s,
    cu::CommandEncoder& encoder,
    const array& a,
    const array& b,
    array& d,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldd,
    bool a_transposed,
    bool b_transposed) {
  using DataType = cuda_type_t<float16_t>;

  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(d);

  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 32;

  constexpr int WM = 2;
  constexpr int WN = 2;

  constexpr int SL = 0;
  constexpr int Nstages = 3;

  constexpr uint32_t smem_bytes = BK * (BM + BN) * Nstages * sizeof(DataType);

  const int NblockM = (M + BM - 1) / BM;
  const int NblockN = (N + BN - 1) / BN;
  const int NblockK = (K + BK - 1) / BK;

  cu::GemmParams params{
      /* int M = */ M,
      /* int N = */ N,
      /* int K = */ K,
      /* int lda = */ lda,
      /* int ldb = */ ldb,
      /* int ldd = */ ldd,

      /* int NblockM = */ NblockM,
      /* int NblockN = */ NblockN,
      /* int NblockK = */ NblockK,
  };

  // Prepare launch grid params
  int tile = 1 << SL;
  int tm = (NblockM + tile - 1) / tile;
  int tn = NblockN * tile;

  dim3 grid_dim(tn, tm, 1);
  dim3 block_dim(32 * WM * WN, 1, 1);

  dispatch_bool(a_transposed, [&](auto ta_) {
    dispatch_bool(b_transposed, [&](auto tb_) {
      constexpr bool ta = ta_.value;
      constexpr bool tb = tb_.value;

      auto kernel = cu::ab_t_aligned<DataType, BM, BN, BK>;
      cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

      encoder.add_kernel_node(
          kernel,
          grid_dim,
          block_dim,
          smem_bytes,
          a.data<DataType>(),
          b.data<DataType>(),
          d.data<DataType>(),
          N,
          K);

      //   auto kernel = cu::kernel_steel_gemm<DataType, BM, BN, BK, WM, WN, ta,
      //   tb, SL, Nstages>;

      //   cudaFuncSetAttribute(kernel,
      //   cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

      //   encoder.add_kernel_node(
      //       kernel,
      //       grid_dim,
      //       block_dim,
      //       smem_bytes,
      //       a.data<DataType>(),
      //       b.data<DataType>(),
      //       d.data<DataType>(),
      //       params);
    });
  });
}

} // namespace mlx::core