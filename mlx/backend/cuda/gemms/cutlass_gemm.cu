// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"

#include <cute/tensor.hpp>

namespace mlx::core::cu {

namespace {

template <typename Kernel>
void configure_matmul(Kernel kernel, int smem_size) {
  static bool initialized = false;
  if (!initialized) {
    initialized = true;
    cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute(
        kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  }
}

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
  cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
};

template <
    class ProblemShape,
    class CtaTiler,
    class TA,
    class AStride,
    class ASmemLayout,
    class TiledCopyA,
    class S2RAtomA,
    class TB,
    class BStride,
    class BSmemLayout,
    class TiledCopyB,
    class S2RAtomB,
    class TC,
    class CStride,
    class CSmemLayout,
    class TiledMma>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value) void gemm_device(
    ProblemShape shape_MNK,
    CtaTiler cta_tiler,
    TA const* A,
    AStride dA,
    ASmemLayout sA_layout,
    TiledCopyA copy_a,
    S2RAtomA s2r_atom_a,
    TB const* B,
    BStride dB,
    BSmemLayout sB_layout,
    TiledCopyB copy_b,
    S2RAtomB s2r_atom_b,
    TC* C,
    CStride dC,
    CSmemLayout,
    TiledMma mma) {
  using namespace cute;

  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{}); // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{}); // (BLK_M, BLK_N, BLK_K)

  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma)); // NumThreads
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma)); // NumThreads

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<CSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler)); // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler)); // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler)); // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler)); // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler)); // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler)); // BLK_K

  CUTE_STATIC_ASSERT_V(
      congruent(select<0, 2>(shape_MNK), dA)); // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(
      congruent(select<1, 2>(shape_MNK), dB)); // dB strides for shape NK
  CUTE_STATIC_ASSERT_V(
      congruent(select<0, 1>(shape_MNK), dC)); // dC strides for shape MN

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  Tensor mA =
      make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // (M,K)
  Tensor mB =
      make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // (N,K)
  Tensor mC =
      make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _); // (m,n,k)
  Tensor gA = local_tile(
      mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(
      mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (BLK_N,BLK_K,k)
  Tensor gC =
      local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_M,BLK_N)

  // Shared memory buffers
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(
      make_smem_ptr(smem.A.begin()), sA_layout); // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(
      make_smem_ptr(smem.B.begin()), sB_layout); // (BLK_N,BLK_K,PIPE)

  //
  // Partition the copying of A and B tiles across the threads
  //

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY,CPY_M,CPY_K,PIPE)

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY,CPY_N,CPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY,CPY_N,CPY_K,PIPE)

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA)); // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA)); // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB)); // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB)); // CPY_K

  //
  // PREFETCH
  //

  auto K_PIPE_MAX = size<3>(tAsA);

  // Total count of tiles
  int k_tile_count = size<3>(tAgA);
  // Current tile index in gmem to read from
  int k_tile_next = 0;

  // Start async loads for all pipes but the last
  CUTE_UNROLL
  for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
    copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe));
    copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe));
    cp_async_fence();
    --k_tile_count;
    if (k_tile_count > 0) {
      ++k_tile_next;
    }
  }

  //
  // Define A/B partitioning and C accumulators
  //

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC); // (MMA,MMA_M,MMA_N)

  // Allocate registers for pipelining
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0)); // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0)); // (MMA,MMA_N,MMA_K)
  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC); // (MMA,MMA_M,MMA_N)

  CUTE_STATIC_ASSERT_V(
      (shape(tCrC) == take<0, 3>(shape(tCgC)))); // (MMA,MMA_M,MMA_N)
  CUTE_STATIC_ASSERT_V((size<1>(tCgC) == size<1>(tCrA))); // MMA_M
  CUTE_STATIC_ASSERT_V((size<2>(tCgC) == size<1>(tCrB))); // MMA_N

  // Clear the accumulators
  clear(tCrC);

  //
  // Copy Atom retiling
  //

  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
  ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA); // (CPY,MMA_M,MMA_K,PIPE)
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA); // (CPY,MMA_M,MMA_K)

  TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
  ThrCopy s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
  Tensor tXsB = s2r_thr_copy_b.partition_S(sB); // (CPY,MMA_N,MMA_K,PIPE)
  Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB); // (CPY,MMA_N,MMA_K)

#if 0
  if(thread0()) {
    print("  mA : "); print(  mA); print("\n");
    print("  gA : "); print(  gA); print("\n");
    print("  sA : "); print(  sA); print("\n");
    print("tAgA : "); print(tAgA); print("\n");
    print("tAsA : "); print(tAsA); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mB : "); print(  mB); print("\n");
    print("  gB : "); print(  gB); print("\n");
    print("  sB : "); print(  sB); print("\n");
    print("tBgB : "); print(tBgB); print("\n");
    print("tBsB : "); print(tBsB); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mC : "); print(  mC); print("\n");
    print("  gC : "); print(  gC); print("\n");
    print("tCgC : "); print(tCgC); print("\n");
    print("tCrA : "); print(tCrA); print("\n");
    print("tCrB : "); print(tCrB); print("\n");
    print("tCrC : "); print(tCrC); print("\n");

    print("tXsA : "); print(tXsA); print("\n");
    print("tXrA : "); print(tXrA); print("\n");
    print("tXsB : "); print(tXsB); print("\n");
    print("tXrB : "); print(tXrB); print("\n");
  }
#endif

#if 1

  // Current pipe index in smem to read from
  int smem_pipe_read = 0;
  // Current pipe index in smem to write to
  int smem_pipe_write = K_PIPE_MAX - 1;

  // Pipe slice
  Tensor tXsA_p = tXsA(_, _, _, smem_pipe_read);
  Tensor tXsB_p = tXsB(_, _, _, smem_pipe_read);

  // Size of the register pipeline
  auto K_BLOCK_MAX = size<2>(tCrA);
  CUTE_STATIC_ASSERT_V(K_BLOCK_MAX == size<2>(tXrA));

  // PREFETCH register pipeline
  if (K_BLOCK_MAX > 1) {
    // Wait until our first prefetched tile is loaded in
    cp_async_wait<K_PIPE_MAX - 2>();
    __syncthreads();

    // Prefetch the first rmem from the first k-tile
    copy(s2r_atom_a, tXsA_p(_, _, Int<0>{}), tXrA(_, _, Int<0>{}));
    copy(s2r_atom_b, tXsB_p(_, _, Int<0>{}), tXrB(_, _, Int<0>{}));
  }

  //
  // PIPELINED MAIN LOOP
  // TUTORIAL: Example of a gemm loop that pipelines shared memory using SM80's
  // cp.async instructions
  //           and explicit pipelines in shared memory.
  //   Data is read from global(k_tile_next) to shared(smem_pipe_write).
  //   Data is read from shared(smem_pipe_read) to registers(k_block_next).
  //   Data is computed on registers(b_block).
  //
  //   This allows all copies and compute to overlap:
  //     Copy from gmem->smem can overlap with copies from smem->rmem and
  //     compute on rmem. Copy from smem->rmem can overlap with compute on rmem.
  //

  CUTE_NO_UNROLL
  while (k_tile_count > -(K_PIPE_MAX - 1)) {
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
      if (k_block == K_BLOCK_MAX - 1) {
        // Slice the smem_pipe_read smem
        tXsA_p = tXsA(_, _, _, smem_pipe_read);
        tXsB_p = tXsB(_, _, _, smem_pipe_read);

        // Commit the smem for smem_pipe_read
        cp_async_wait<K_PIPE_MAX - 2>();
        __syncthreads();
      }

      // Load A, B shmem->regs for k_block+1
      auto k_block_next = (k_block + 1) % K_BLOCK_MAX; // static
      copy(s2r_atom_a, tXsA_p(_, _, k_block_next), tXrA(_, _, k_block_next));
      copy(s2r_atom_b, tXsB_p(_, _, k_block_next), tXrB(_, _, k_block_next));
      // Copy gmem to smem before computing gemm on each k-pipe
      if (k_block == 0) {
        copy(
            copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, smem_pipe_write));
        copy(
            copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, smem_pipe_write));
        cp_async_fence();

        // Advance the gmem tile
        --k_tile_count;
        if (k_tile_count > 0) {
          ++k_tile_next;
        }

        // Advance the smem pipe
        smem_pipe_write = smem_pipe_read;
        smem_pipe_read =
            (smem_pipe_read == K_PIPE_MAX - 1) ? 0 : smem_pipe_read + 1;
      }
      // Thread-level register gemm for k_block
      gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
    }
  }

#endif

  //
  // Epilogue
  //

  copy(tCrC, tCgC);
}

} // namespace

void cutlass_gemm(
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    cu::CommandEncoder& enc) {
  enc.set_input_array(a);
  enc.set_input_array(b);
  enc.set_output_array(out);
  dispatch_float_types(a.dtype(), "simple_gemm", [&](auto type_tag) {
    using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    if constexpr (std::is_same_v<DataType, __nv_bfloat16>) {
      using namespace cute;

      // Define shapes (dynamic)
      auto prob_shape = make_shape(M, N, K);

      // Define TN strides (mixed)
      auto dA = make_stride(K, Int<1>{});
      auto dB = make_stride(K, Int<1>{});
      auto dC = make_stride(N, Int<1>{});

      // Define CTA tile sizes (static)
      auto bM = Int<128>{};
      auto bN = Int<128>{};
      auto bK = Int<64>{};
      auto cta_tiler = make_shape(bM, bN, bK);
      auto bP = Int<3>{};

      // Define the smem layouts (static)
      // Swizzles for LDSM and 128b k-major loads
      auto swizzle_atom = composition(
          Swizzle<3, 3, 3>{},
          Layout<
              cute::Shape<_8, cute::Shape<_8, _8>>,
              cute::Stride<_8, cute::Stride<_1, _64>>>{});

      auto sA = tile_to_shape(swizzle_atom, make_shape(bM, bK, bP));
      auto sB = tile_to_shape(swizzle_atom, make_shape(bN, bK, bP));
      auto sC = make_layout(make_shape(bM, bN));

      // Define the thread layouts (static)

      TiledCopy copyA = make_tiled_copy(
          Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::bfloat16_t>{},
          Layout<cute::Shape<_16, _8>, cute::Stride<_8, _1>>{}, // Thr layout
                                                                // 16x8 k-major
          Layout<cute::Shape<_1, _8>>{}); // Val layout  1x8 k-major
      TiledCopy copyB = make_tiled_copy(
          Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::bfloat16_t>{},
          Layout<cute::Shape<_16, _8>, cute::Stride<_8, _1>>{}, // Thr layout
                                                                // 16x8 k-major
          Layout<cute::Shape<_1, _8>>{}); // Val layout  1x8 n-major

      TiledMMA mmaC = make_tiled_mma(
          SM80_16x8x16_F32BF16BF16F32_TN{},
          Layout<cute::Shape<_2, _2>>{}, // 2x2x1 MMA Atoms
          Tile<_32, _32, _16>{}); // 32x32x16 Tiled MMA for LDSM

      Copy_Atom<SM75_U32x4_LDSM_N, cute::bfloat16_t> s2r_atom_A;
      Copy_Atom<SM75_U32x4_LDSM_N, cute::bfloat16_t> s2r_atom_B;

      int smem_size = int(sizeof(SharedStorage<
                                 cute::bfloat16_t,
                                 cute::bfloat16_t,
                                 decltype(sA),
                                 decltype(sB)>));
      dim3 dimBlock(size(mmaC));
      dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));

      auto kernel = gemm_device<
          decltype(prob_shape),
          decltype(cta_tiler),
          cute::bfloat16_t,
          decltype(dA),
          decltype(sA),
          decltype(copyA),
          decltype(s2r_atom_A),
          cute::bfloat16_t,
          decltype(dB),
          decltype(sB),
          decltype(copyB),
          decltype(s2r_atom_B),
          cute::bfloat16_t,
          decltype(dC),
          decltype(sC),
          decltype(mmaC)>;

      configure_matmul(kernel, smem_size);

      enc.add_kernel_node(
          kernel,
          dimGrid,
          dimBlock,
          smem_size,
          prob_shape,
          cta_tiler,
          a.data<cute::bfloat16_t>(),
          dA,
          sA,
          copyA,
          s2r_atom_A,
          b.data<cute::bfloat16_t>(),
          dB,
          sB,
          copyB,
          s2r_atom_B,
          out.data<cute::bfloat16_t>(),
          dC,
          sC,
          mmaC);
    } else {
      throw std::runtime_error("Only bfloat16 supported");
    }
  });
}

} // namespace mlx::core::cu
