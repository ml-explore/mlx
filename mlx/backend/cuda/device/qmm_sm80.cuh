// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/device/cute_dequant.cuh"

#include <cuda/cmath>

// clang-format off

namespace mlx::core::cu {

using namespace cute;

template <typename Element,
          typename Quant,
          typename SmemLayoutA,
          typename SmemLayoutB,
          typename SmemLayoutC>
union SharedStorage {
  struct {
    ArrayEngine<Element, cosize_v<SmemLayoutA>> A;
    ArrayEngine<Quant,   cosize_v<SmemLayoutB>> B;
  } mainloop;
  struct {
    ArrayEngine<Element, cosize_v<SmemLayoutC>> C;
  } epilogue;
};

template <typename CtaTiler>
inline constexpr auto make_smem_layouts(CtaTiler cta_tiler) {
  // Note: Kernel launcher assumes cosize being same for all KMajor.
  auto [bM, bN, bK] = cta_tiler;

  // Define the A/B smem layouts (static).
  auto swizzle_ab = composition(Swizzle<3,3,3>{},
                                Layout<Shape <_8,Shape <_8, _8>>,
                                       Stride<_8,Stride<_1,_64>>>{});
  auto bP = Int<3>{}; // pipeline
  auto sA_layout = tile_to_shape(swizzle_ab, make_shape(bM, bK, bP));
  auto sB_layout = tile_to_shape(swizzle_ab, make_shape(bN, bK, bP));

  // Define the C smem layouts (static).
  // TODO: Find a better swizzle.
  auto sC_layout = tile_to_shape(swizzle_ab, make_shape(bM, bN));

  return cute::make_tuple(sA_layout, sB_layout, sC_layout);
}

template <typename Element = half_t, int TileM = 32>
inline constexpr auto make_tiled_mma() {
  // Note: Kernel launcher assumes num_threads being same for all parameters.
  using Atom = cuda::std::conditional_t<
      cuda::std::is_same_v<Element, half_t>,
      SM80_16x8x16_F32F16F16F32_TN,
      SM80_16x8x16_F32BF16BF16F32_TN>;
  if constexpr (TileM >= 32) {
    return make_tiled_mma(Atom{}, Layout<Shape<_2,_2,_1>>{}, Tile<_32,_32,_16>{});
  } else {
    return make_tiled_mma(Atom{}, Layout<Shape<_1,_4,_1>>{}, Tile<_16,_32,_16>{});
  }
}

template <typename T, int bits, template <typename U> typename Atom, typename NumThreads>
inline constexpr auto make_tiled_copy(NumThreads num_threads) {
  return make_tiled_copy(
      Copy_Atom<Atom<uint_bit_t<bits>>, T>{},
      make_layout(make_shape(Int<num_threads / 8>{}, Int<8>{}), LayoutRight{}),
      make_layout(make_shape(Int<1>{}, Int<bits / sizeof_bits_v<T>>{})));
}

template <typename CtaTiler,
          typename TensorA,
          typename TensorB,
          typename TensorS,
          typename TensorZ,
          typename TensorC>
CUTE_DEVICE void qmm_sm80_mainloop(
    CtaTiler cta_tiler,
    TensorA gA,
    TensorB gB,
    TensorS gS,
    TensorZ gZ,
    TensorC gC,
    int m_max_coord,
    int thread_idx) {
  // Get the types of operands.
  using Element = typename decltype(gA)::value_type;
  using Quant = typename decltype(gB)::value_type;
  using Scale = typename decltype(gS)::value_type;

  // Define smem layouts.
  auto [sA_layout, sB_layout, sC_layout] = make_smem_layouts(cta_tiler);

  // Shared memory buffer.
  extern __shared__ char smem_buf[];
  using SharedStorage = SharedStorage<Element, Quant,
                                      decltype(sA_layout),
                                      decltype(sB_layout),
                                      decltype(sC_layout)>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_buf);
  Tensor sA = make_tensor(make_smem_ptr(smem.mainloop.A.begin()), sA_layout); // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smem.mainloop.B.begin()), sB_layout); // (BLK_N,BLK_K)
  Tensor sC = make_tensor(make_smem_ptr(smem.epilogue.C.begin()), sC_layout); // (BLK_M,BLK_N)

  // Define MMA.
  auto mma = make_tiled_mma<Element, size<0>(cta_tiler)>();
  auto num_threads = size(mma);

  // Define copy atoms.
  constexpr int element_bits = sizeof_bits_v<Element>;
  constexpr int quant_bits = sizeof_bits_v<Quant>;
  constexpr int qload = 128 / (element_bits / quant_bits);
  TiledCopy g2s_copy_a = make_tiled_copy<Element, 128, SM80_CP_ASYNC_CACHEALWAYS>(num_threads);
  TiledCopy g2s_copy_b = make_tiled_copy<Quant, qload, SM80_CP_ASYNC_CACHEALWAYS>(num_threads);
  TiledCopy s2g_copy_c = make_tiled_copy<Element, 128, UniversalCopy>(num_threads);

  Copy_Atom<SM75_U32x4_LDSM_N, Element> s2r_atom_a;
  Copy_Atom<UniversalCopy<uint_bit_t<2 * quant_bits>>, Quant> s2r_atom_b;
  Copy_Atom<UniversalCopy<uint_bit_t<2 * element_bits>>, Element> r2s_atom_c;
  Copy_Atom<UniversalCopy<Scale>, Scale> g2r_atom_s;

  // Partition the copying of A/B/C tiles across the threads.
  ThrCopy g2s_thr_copy_a = g2s_copy_a.get_slice(thread_idx);
  Tensor tAgA = g2s_thr_copy_a.partition_S(gA); // (ACPY,ACPY_M,ACPY_K,k)
  Tensor tAsA = g2s_thr_copy_a.partition_D(sA); // (ACPY,ACPY_M,ACPY_K,PIPE)

  ThrCopy g2s_thr_copy_b = g2s_copy_b.get_slice(thread_idx);
  Tensor tBgB = g2s_thr_copy_b.partition_S(gB);  // (BCPY,BCPY_N,BCPY_K,k)
  Tensor tBsB = g2s_thr_copy_b.partition_D(sB);  // (BCPY,BCPY_N,BCPY_K,PIPE)

  ThrCopy s2g_thr_copy_c = s2g_copy_c.get_slice(thread_idx);
  Tensor s2g_tCsC = s2g_thr_copy_c.partition_S(sC); // (CCPY,CCPY_M,CCPY_N)
  Tensor s2g_tCgC = s2g_thr_copy_c.partition_D(gC); // (CCPY,CCPY_M,CCPY_N)

  // MMA.
  ThrMMA thr_mma = mma.get_slice(thread_idx);
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0)); // (MMA,MMA_M,MMA_K)
  Tensor tCsB = thr_mma.partition_B(sB(_,_,0));          // (MMA,MMA_N,MMA_K)
  Tensor tCrB = make_fragment_like<Quant>(tCsB);         // (MMA,MMA_N,MMA_K)
  Tensor tCrB_dq = make_fragment_like<Element>(tCsB);    // (MMA,MMA_N,MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC);                 // (MMA,MMA_M,MMA_N)
  Tensor tCrC_accu = make_fragment_like<float>(tCgC);    // (MMA,MMA_M,MMA_N)
  Tensor tCrC = make_fragment_like<Element>(tCgC);       // (MMA,MMA_M,MMA_N)

  Tensor tCgS = thr_mma.partition_B(gS);         // (MMA,MMA_N,MMA_K,k)
  Tensor tCrS = make_tensor_like(tCgS(_,_,_,0)); // (MMA,MMA_N,MMA_K)
  Tensor tCgZ = thr_mma.partition_B(gZ);         // (MMA,MMA_N,MMA_K,k)
  Tensor tCrZ = make_tensor_like(tCgZ(_,_,_,0)); // (MMA,MMA_N,MMA_K)

  // Copy Atom retiling.
  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
  ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(thread_idx);
  Tensor s2r_tCsA = s2r_thr_copy_a.partition_S(sA); // (ACPY,MMA_M,MMA_K,PIPE)
  Tensor s2r_tCrA = s2r_thr_copy_a.retile_D(tCrA);  // (ACPY,MMA_M,MMA_K)

  TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
  ThrCopy s2r_thr_copy_b = s2r_copy_b.get_slice(thread_idx);
  Tensor s2r_tCsB = s2r_thr_copy_b.partition_S(sB); // (BCPY,MMA_N,MMA_K,PIPE)
  Tensor s2r_tCrB = s2r_thr_copy_b.retile_D(tCrB);  // (BCPY,MMA_N,MMA_K)

  TiledCopy r2s_copy_c = make_tiled_copy_C(r2s_atom_c, mma);
  ThrCopy r2s_thr_copy_c = r2s_copy_c.get_slice(thread_idx);
  Tensor r2s_tCrC = r2s_thr_copy_c.retile_S(tCrC);  // (CCPY,MMA_M,MMA_N)
  Tensor r2s_tCsC = r2s_thr_copy_c.partition_D(sC); // (CCPY,MMA_M,MMA_N)

  TiledCopy g2r_copy_s = make_tiled_copy_B(g2r_atom_s, mma);
  ThrCopy g2r_thr_copy_s = g2r_copy_s.get_slice(thread_idx);
  Tensor g2r_tCgS = g2r_thr_copy_s.partition_S(gS); // (BCPY,MMA_N,MMA_K,k)
  Tensor g2r_tCrS = g2r_thr_copy_s.retile_D(tCrS);  // (BCPY,MMA_N,MMA_K)
  Tensor g2r_tCgZ = g2r_thr_copy_s.partition_S(gZ); // (BCPY,MMA_N,MMA_K,k)
  Tensor g2r_tCrZ = g2r_thr_copy_s.retile_D(tCrZ);  // (BCPY,MMA_N,MMA_K)

  // Predicates for m bound.
  Tensor tApA = make_tensor<bool>(make_shape(size<1>(tAsA), size<2>(tAsA)), Stride<_1,_0>{});         // (CPY_M,CPY_K)
  Tensor tCpC = make_tensor<bool>(make_shape(size<1>(s2g_tCsC), size<2>(s2g_tCsC)), Stride<_1,_0>{}); // (CPY_M,CPY_N)
  Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA))); // (BLK_M,BLK_K)
  Tensor cC = make_identity_tensor(make_shape(size<0>(sC), size<1>(sC))); // (BLK_M,BLK_N)
  Tensor tAcA = g2s_thr_copy_a.partition_D(cA); // (CPY,CPY_M,CPY_K)
  Tensor tCcC = s2g_thr_copy_c.partition_D(cC); // (CPY,CPY_M,CPY_N)
  CUTE_UNROLL
  for (int m = 0; m < size<0>(tApA); ++m) {
    tApA(m,0) = get<0>(tAcA(0,m,0)) < m_max_coord;
  }
  CUTE_UNROLL
  for (int m = 0; m < size<0>(tCpC); ++m) {
    tCpC(m,0) = get<0>(tCcC(0,m,0)) < m_max_coord;
  }

  auto K_PIPE_MAX = size<3>(tAsA);
  int smem_pipe_read = 0;
  int smem_pipe_write = 0;

  // Copy A/B: GMEM => SMEM.
  auto fetch_gmem = [&](int tile) {
    copy_if(g2s_copy_a, tApA, tAgA(_,_,_,tile), tAsA(_,_,_,smem_pipe_write));
    copy(g2s_copy_b, tBgB(_,_,_,tile), tBsB(_,_,_,smem_pipe_write));
    cp_async_fence();
    smem_pipe_write = (smem_pipe_write + 1) % K_PIPE_MAX;
  };
  // Copy S/Z: GMEM => RMEM.
  auto fetch_scales = [&](int tile) {
    copy(g2r_copy_s, g2r_tCgS(_,_,_,tile), g2r_tCrS);
    if constexpr (mlx::core::cu::quant_has_bias_v<Quant>) {
      copy(g2r_copy_s, g2r_tCgZ(_,_,_,tile), g2r_tCrZ);
    }
  };
  // Copy A/B: SMEM => RMEM.
  auto fetch_smem = [&](auto block) {
    copy(s2r_atom_a, s2r_tCsA(_,_,block,smem_pipe_read), s2r_tCrA(_,_,block));
    copy(s2r_atom_b, s2r_tCsB(_,_,block,smem_pipe_read), s2r_tCrB(_,_,block));
    CUTE_UNROLL
    for (int n = 0; n < size<1>(tCrB); ++n) {
      mlx::core::cu::cute_vectorized_dequant(
          tCrB(_,n,block),
          tCrS(_,n,block),
          tCrZ(_,n,block),
          tCrB_dq(_,n,block));
    }
  };

  auto K_TILE_MAX = size<3>(tAgA);
  auto K_BLOCK_MAX = size<2>(tCrA);

  // Prefetch beginning tiles.
  int tile_pipe = 0;
  CUTE_UNROLL
  for (; tile_pipe < K_PIPE_MAX - 1; ++tile_pipe) {
    fetch_gmem(tile_pipe);
  }

  // Clear accumulators.
  clear(tCrC_accu);

  // Prefetch first block.
  if constexpr (K_BLOCK_MAX > 1) {
    cp_async_wait<K_PIPE_MAX - 2>();
    __syncthreads();
    fetch_scales(0);
    fetch_smem(Int<0>{});
  }

  // Loop over CTA tiles.
  for (int tile = 0; tile < K_TILE_MAX; ++tile) {
    // Unroll MMA blocks.
    CUTE_UNROLL
    for (int block = 0; block < K_BLOCK_MAX; ++block) {
      // Wait for last tile.
      if (block == K_BLOCK_MAX - 1) {
        smem_pipe_read = (smem_pipe_read + 1) % K_PIPE_MAX;
        cp_async_wait<K_PIPE_MAX - 2>();
        __syncthreads();
        fetch_scales((tile + 1 < K_TILE_MAX) ? tile + 1 : tile);
      }
      // Prefetch next block.
      fetch_smem((block + 1) % K_BLOCK_MAX);
      // Prefetch next tile.
      if (block == 0) {
        fetch_gmem(tile_pipe);
        tile_pipe = (tile_pipe + 1 < K_TILE_MAX) ? tile_pipe + 1 : tile_pipe;
      }
      // MMA.
      gemm(mma, tCrA(_,_,block), tCrB_dq(_,_,block), tCrC_accu);
    }
  }

  // Epilogue.
  CUTE_UNROLL
  for (int i = 0; i < size(tCrC_accu); i++) {
    tCrC(i) = Element(tCrC_accu(i));
  }
  copy(r2s_copy_c, r2s_tCrC, r2s_tCsC);
  __syncthreads();
  copy_if(s2g_copy_c, tCpC, s2g_tCsC, s2g_tCgC);
}

template <int GroupSize>
inline constexpr auto make_scales_layout(int n, int k, int l) {
  auto group_size = Int<GroupSize>{};
  return make_layout(
      make_shape(n, make_shape(group_size, k / group_size), l),
      make_stride(k / group_size, Stride<_0,_1>{}, n * k / group_size));
}

template <int GroupSize,
          typename Element, typename Quant, typename Scale, typename CtaTiler>
__global__
__launch_bounds__(decltype(size(make_tiled_mma()))::value)
void qmm_sm80_kernel(
    const Element* A,
    const Quant* B,
    const Scale* S,
    const Element* Z,
    const uint32_t* lhs_indices,
    const uint32_t* rhs_indices,
    Element* C,
    int m, int n, int k, int l,
    bool broadcast_b) {
  int thread_idx = int(threadIdx.x);
  int m_coord = int(blockIdx.x);
  int n_coord = int(blockIdx.y);
  int l_coord = int(blockIdx.z);

  // Define layouts (mixed).
  auto dA = make_stride(k, Int<1>{}, m * k); // (dM,dK,dL)
  auto dB = make_stride(k, Int<1>{}, n * k); // (dN,dK,dL)
  auto dC = make_stride(n, Int<1>{}, m * n); // (dM,dN,dL)
  auto S_layout = make_scales_layout<GroupSize>(n, k, l);

  // Handle broadcasting.
  if (broadcast_b) {
    get<2>(dB) = 0;
    get<2>(stride(S_layout)) = 0;
  }

  // Represent the full tensors.
  Tensor mA_mkl = make_tensor(make_gmem_ptr(A),        make_shape(m, k, l), dA); // (M,K,L)
  Tensor mB_nkl = make_tensor(make_gmem_ptr<Quant>(B), make_shape(n, k, l), dB); // (N,K,L)
  Tensor mC_mnl = make_tensor(make_gmem_ptr(C),        make_shape(m, n, l), dC); // (M,N,L)

  Tensor mS_nkl = make_tensor(make_gmem_ptr(S), S_layout); // (N,(group_size,K/group_size),L)
  Tensor mZ_nkl = make_tensor(make_gmem_ptr(Z), S_layout); // (N,(group_size,K/group_size),L)

  // For gather, use index lookup for input batch slicing.
  uint32_t a_batch = lhs_indices ? lhs_indices[l_coord] : l_coord;
  uint32_t b_batch = rhs_indices ? rhs_indices[l_coord] : l_coord;

  // Get batch slice.
  Tensor mA = mA_mkl(_,_,a_batch); // (M,K)
  Tensor mB = mB_nkl(_,_,b_batch); // (N,K)
  Tensor mC = mC_mnl(_,_,l_coord); // (M,N)

  Tensor mS = mS_nkl(_,_,b_batch); // (N,(group_size,K/group_size))
  Tensor mZ = mZ_nkl(_,_,b_batch); // (N,(group_size,K/group_size))

  // Get the appropriate blocks for this thread block.
  auto cta_tiler = CtaTiler{};
  auto cta_coord = make_coord(m_coord, n_coord, _); // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{}); // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{}); // (BLK_M,BLK_N)

  Tensor gS = local_tile(mS, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
  Tensor gZ = local_tile(mZ, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)

  // Compute tile residues for predication.
  auto m_max_coord = m - size<0>(gA) * m_coord; // M - BLK_M * m_coord

  qmm_sm80_mainloop(
      cta_tiler,
      gA,
      gB,
      gS,
      gZ,
      gC,
      m_max_coord,
      thread_idx);
}

} // namespace mlx::core::cu
