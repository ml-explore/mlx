// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"

#include <cute/tensor.hpp>

namespace mlx::core::cu {

namespace {

template <
    class T,
    class CtaTiler,
    class SmemLayoutA,
    class SmemLayoutB,
    class TiledMMA,
    class G2STiledCopyA,
    class G2STiledCopyB,
    class S2RCopyAtomA,
    class S2RCopyAtomB>
__global__ void cute_gemm_v02(
    unsigned int M,
    unsigned int N,
    unsigned int K,
    const T* A,
    size_t lda,
    const T* B,
    size_t ldb,
    T* C,
    size_t ldc) {
  using namespace cute;

  // global full tensor
  // shape
  auto shape_MNK = make_shape(M, N, K);

  // stride
  // cublas covenience for TN gemm
  // all matrices are in column major
  // A (m,k) --> transpose --> A(k, m) --> cute layout: A (m, k) : (k, 1) -->
  // lda = k B (k,n) --> cute layout: B (n, k) : (k, 1) --> ldb = k C (m,n) -->
  // cute layout: C (m, n) : (1, m) --> ldc = m
  auto dA = make_stride(lda, _1{});
  auto dB = make_stride(ldb, _1{});
  auto dC = make_stride(_1{}, ldc);
  Tensor mA =
      make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // M x K
  Tensor mB =
      make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // N x K
  Tensor mC =
      make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // M x N

  // global tile tensor
  auto cta_tiler = CtaTiler{};
  auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);
  Tensor gA = local_tile(
      mA,
      cta_tiler,
      cta_coord,
      Step<_1, X, _1>{}); // BLOCK_SIZE_M x BLOCK_SIZE_K x NUM_TILES_K
  Tensor gB = local_tile(
      mB,
      cta_tiler,
      cta_coord,
      Step<X, _1, _1>{}); // BLOCK_SIZE_N x BLOCK_SIZE_K x NUM_TILES_K
  Tensor gC = local_tile(
      mC,
      cta_tiler,
      cta_coord,
      Step<_1, _1, X>{}); // BLOCK_SIZE_M x BLOCK_SIZE_N

  // shared memory
  // __shared__ T Asmem[cosize_v<SmemLayoutA>];
  // __shared__ T Bsmem[cosize_v<SmemLayoutB>];

  extern __shared__ T smem[];
  T* Asmem = smem;
  T* Bsmem = smem + cosize_v<SmemLayoutA>;

  Tensor sA = make_tensor(
      make_smem_ptr(Asmem),
      SmemLayoutA{}); // BLOCK_SIZE_M x BLOCK_SIZE_K x NUM_STAGES
  Tensor sB = make_tensor(
      make_smem_ptr(Bsmem),
      SmemLayoutB{}); // BLOCK_SIZE_N x BLOCK_SIZE_K x NUM_STAGES

  // MMA
  // use TiledMMA --> get one thread work
  auto tiled_mma = TiledMMA{};
  ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
  auto tCgC = thr_mma.partition_C(gC); // MMA x MMA_M x MMA_N

  // thread private memory for MMA
  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // MMA x MMA_M x MMA_K
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // MMA x MMA_N x MMA_K

  // thread private memory for accumulator for MMA
  Tensor tCrC = thr_mma.partition_fragment_C(gC); // MMA x MMA_M x MMA_N

  clear(tCrC);

  // initiate copy from global memory to shared memory
  // use G2S TiledCopy --> get one thread copy work
  auto g2s_tiled_copy_a = G2STiledCopyA{};
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(threadIdx.x);
  const auto tAgA =
      g2s_thr_copy_a.partition_S(gA); // CPY x CPY_M x CPY_K x NUM_TILES_K
  auto tAsA =
      g2s_thr_copy_a.partition_D(sA); // CPY x CPY_M x CPY_K x NUM_STAGES

  auto g2s_tiled_copy_b = G2STiledCopyB{};
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(threadIdx.x);
  const auto tBgB =
      g2s_thr_copy_b.partition_S(gB); // CPY x CPY_N x CPY_K x NUM_TILES_K
  auto tBsB =
      g2s_thr_copy_b.partition_D(sB); // CPY x CPY_N x CPY_K x NUM_STAGES

  // initiate copy from shared memory to thread private memory
  // use S2R TiledCopy --> get one thread copy work
  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(threadIdx.x);
  const auto tCsA =
      s2r_thr_copy_a.partition_S(sA); // CPY x CPY_M x CPY_K x NUM_STAGES
  auto tCrA_copy_view = s2r_thr_copy_a.retile_D(tCrA); // CPY x CPY_M x CPY_K

  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(threadIdx.x);
  const auto tCsB =
      s2r_thr_copy_b.partition_S(sB); // CPY x CPY_N x CPY_K x NUM_STAGES
  auto tCrB_copy_view = s2r_thr_copy_b.retile_D(tCrB); // CPY x CPY_N x CPY_K

  // pipeline
  // counter
  int itile_to_read = 0; // read index of the next tile
  // 2 pointers of the buffer
  int ismem_write = 0;
  int ismem_read = 0;

  // NUM_STAGES = 5 --> Prefetech NUM_STAGES-1 = 4 tiles first
  auto NUM_STAGES = size<3>(tAsA);

  CUTE_UNROLL
  for (int stage = 0; stage < NUM_STAGES - 1; ++stage) {
    // prefetch
    // issue copy
    copy(g2s_tiled_copy_a, tAgA(_, _, _, itile_to_read), tAsA(_, _, _, stage));
    copy(g2s_tiled_copy_b, tBgB(_, _, _, itile_to_read), tBsB(_, _, _, stage));

    // commit
    cp_async_fence();

    ismem_write++;
    itile_to_read++;
  }

  // wait for first tile to be prefetched: G^0 -> S^0
  cp_async_wait<NUM_STAGES - 2>();
  __syncthreads();

  // Having S^0, copy from S^0,0 to R^0
  int k = 0;
  copy(s2r_tiled_copy_a, tCsA(_, _, k, ismem_read), tCrA_copy_view(_, _, k));
  copy(s2r_tiled_copy_b, tCsB(_, _, k, ismem_read), tCrB_copy_view(_, _, k));

  // loop over tiles
  auto NUM_TILES_K = size<3>(tAgA);

  CUTE_NO_UNROLL
  for (int tile = 0; tile < NUM_TILES_K; ++tile) {
    auto MMA_K = size<2>(tCrA);
    // loop over MMAs in direction of K

    CUTE_UNROLL
    for (int k = 0; k < MMA_K; ++k) {
      int k_next = (k + 1) % MMA_K;

      // if this is the second last MMA, wait the next tile to be fetched
      if (k == MMA_K - 1) {
        cp_async_wait<NUM_STAGES - 2>();
        __syncthreads();

        ismem_read = (ismem_read + 1) % NUM_STAGES;
      }

      // load data for the next MMA, from S^tile to registers
      copy(
          s2r_tiled_copy_a,
          tCsA(_, _, k_next, ismem_read),
          tCrA_copy_view(_, _, k_next));
      copy(
          s2r_tiled_copy_b,
          tCsB(_, _, k_next, ismem_read),
          tCrB_copy_view(_, _, k_next));

      if (k == 0) {
        // prefetch the next tile
        // issue copy
        if (itile_to_read < NUM_TILES_K) {
          copy(
              g2s_tiled_copy_a,
              tAgA(_, _, _, itile_to_read),
              tAsA(_, _, _, ismem_write));
          copy(
              g2s_tiled_copy_b,
              tBgB(_, _, _, itile_to_read),
              tBsB(_, _, _, ismem_write));

          itile_to_read++;
          ismem_write = (ismem_write + 1) % NUM_STAGES;
        }
        // commit
        cp_async_fence();
      }

      // mma
      gemm(tiled_mma, tCrC, tCrA(_, _, k), tCrB(_, _, k), tCrC);
    }
  }

  copy(tCrC, tCgC);
  // constexpr T alpha{1.0f};
  // constexpr T beta{0.0f};
  // axpby(alpha, tCrC, beta, tCgC);
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

      // block shape and cta tiler
      // additional dim: NUM_STAGES --> This is for later pipelining the k-slice
      // GEMM
      auto BLOCK_SIZE_M = _128{};
      auto BLOCK_SIZE_N = _128{};
      auto BLOCK_SIZE_K = _32{};
      auto NUM_STAGES = _5{};
      using CtaTiler =
          decltype(make_shape(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K));

      // smem layout
      // Swizzle parameters need to be chosen right
      static constexpr int kShmLoadSwizzleM = 3;
      static constexpr int kShmLoadSwizzleS = 3;
      static constexpr int kShmLoadSwizzleB = 3;

      using SmemLayoutAtom = decltype(composition(
          Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
          make_layout(make_shape(_8{}, BLOCK_SIZE_K), LayoutRight{})));

      // what does this do?
      // with BLOCK_SIZE_K = 32, shape: (8, 32)
      // 2^M = 8, 1 new unit = 8 units --> 1 row contains 32/8 = 4 new units
      // 2^S = 8, it will treat 1 row = 8 new units --> do 8-unit swizzle
      // 2^B = 8, it will reset the swizzle pattern after 8 rows
      // print_layout(SmemLayoutAtom{});

      // tile_to_shape extends the layout in LayoutLeft order
      using SmemLayoutA = decltype(tile_to_shape(
          SmemLayoutAtom{},
          make_shape(BLOCK_SIZE_M, BLOCK_SIZE_K, NUM_STAGES)));

      using SmemLayoutB = decltype(tile_to_shape(
          SmemLayoutAtom{},
          make_shape(BLOCK_SIZE_N, BLOCK_SIZE_K, NUM_STAGES)));

      // TiledMMA
      using mma_op = SM80_16x8x16_F32BF16BF16F32_TN;
      using mma_traits = MMA_Traits<mma_op>;
      using mma_atom = MMA_Atom<mma_traits>;

      static constexpr int kMmaEURepeatM = 2;
      static constexpr int kMmaEURepeatN = 2;
      static constexpr int kMmaEURepeatK = 1;
      // 32 x 2 x 2 = 128 threads

      using mma_atom_shape = mma_traits::Shape_MNK;
      static constexpr int MmaVM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
      static constexpr int MmaVN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
      static constexpr int MmaVK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

      // this is for problem shape (16x2) x (8x2x2) x (16x1) = 32x32x16
      using MMA_EU_RepeatT = decltype(make_layout(make_shape(
          Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
      using MMA_V_T = Tile<Int<MmaVM>, Int<MmaVN>, Int<MmaVK>>;

      using TiledMMA =
          decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_V_T{}));

      // TiledCopy from global memory to shared memory
      // uint128_t is 16 bytes = 4 floats = 8 halfs
      static constexpr int NUM_VECTOR_UNITS =
          sizeof(cute::uint128_t) / sizeof(DataType);

      using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
      using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
      using g2s_copy_atom = Copy_Atom<g2s_copy_traits, DataType>;

      // one block contains 128 threads
      // --> find the compatible thread layout
      using G2S_Copy_Thread_Layout = decltype(make_layout(
          make_shape(_32{}, _4{}), // 32x4 = 128 threads
          LayoutRight{} // A is in row-major
          ));

      using G2S_Copy_Value_Layout =
          decltype(make_layout(make_shape(_1{}, Int<NUM_VECTOR_UNITS>{})));

      // This is for copy shape 32x4 of uint128_t
      using G2STiledCopyA = decltype(make_tiled_copy(
          g2s_copy_atom{}, G2S_Copy_Thread_Layout{}, G2S_Copy_Value_Layout{}));

      // Both A and B are in row-major so use the same TiledCopy for B
      using G2STiledCopyB = G2STiledCopyA;

      // CopyAtom from shared memory to registers
      // Why no need to do tiling atom here? Because we will do it later with
      // the information from TiledMMA
      using s2r_copy_op = SM75_U32x4_LDSM_N;
      using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
      using s2r_copy_atom = Copy_Atom<s2r_copy_traits, DataType>;

      using S2RCopyAtomA = s2r_copy_atom;
      using S2RCopyAtomB = s2r_copy_atom;

      // print_latex(
      //     make_tiled_copy(S2RCopyAtomA{}, make_layout(Shape<_32>{}),
      //     make_layout(Shape<_1, _8>{}))
      // );

      // grid, block
      dim3 block{size(TiledMMA{}), 1U, 1U};
      dim3 grid{
          size(ceil_div(static_cast<unsigned int>(N), BLOCK_SIZE_N)),
          size(ceil_div(static_cast<unsigned int>(M), BLOCK_SIZE_M)),
          1U};

      static constexpr int smem_size =
          (cosize_v<SmemLayoutA> + cosize_v<SmemLayoutB>)*sizeof(DataType);

      auto kernel = cute_gemm_v02<
          DataType,
          CtaTiler,
          SmemLayoutA,
          SmemLayoutB,
          TiledMMA,
          G2STiledCopyA,
          G2STiledCopyB,
          S2RCopyAtomA,
          S2RCopyAtomB>;

      cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

      enc.add_kernel_node(
          kernel,
          grid,
          block,
          smem_size,
          M,
          N,
          K,
          a.data<DataType>(),
          K,
          b.data<DataType>(),
          K,
          out.data<DataType>(),
          N);
    } else {
      throw std::runtime_error("Only bfloat16 supported");
    }
  });
}

} // namespace mlx::core::cu
