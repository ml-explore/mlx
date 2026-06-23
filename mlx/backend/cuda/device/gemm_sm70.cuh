// Copyright © 2026 Apple Inc.

#include <cute/tensor.hpp>

// clang-format off

namespace mlx::core::cu {

using namespace cute;

template <typename Element, typename SmemLayoutA, typename SmemLayoutB>
struct SharedStorage {
  ArrayEngine<Element, cosize_v<SmemLayoutA>> A;
  ArrayEngine<Element, cosize_v<SmemLayoutB>> B;
};

template <bool KMajor, typename TileM, typename TileK>
inline constexpr auto make_smem_layout(TileM bM, TileK bK) {
  // TODO: Calculate swizzle based on tile shape.
  if constexpr (KMajor) {
    auto swizzle = composition(Swizzle<3,3,3>{},
                               Layout<Shape <_8,Shape <_8, _8>>,
                                      Stride<_8,Stride<_1,_64>>>{});
    return tile_to_shape(swizzle, make_shape(bM, bK));
  } else {
    auto swizzle = composition(Swizzle<3,3,3>{},
                               Layout<Shape<_16,_8>, Stride<_8,_1>>{});
    return tile_to_shape(swizzle, make_shape(bM, bK));
  }
}

template <bool KMajorA = true, bool KMajorB = true, typename CtaTiler>
inline constexpr auto make_smem_layouts(CtaTiler cta_tiler) {
  // Note: Kernel launcher assumes num_threads being same for all parameters.
  auto [bM, bN, bK] = cta_tiler;
  auto sA_layout = make_smem_layout<KMajorA>(bM, bK);
  auto sB_layout = make_smem_layout<KMajorB>(bN, bK);
  return cute::make_tuple(sA_layout, sB_layout);
}

template <bool SM80 = false, typename Element = float, typename CtaTiler>
inline constexpr auto make_tiled_mma(CtaTiler cta_tiler) {
  // Note: Kernel launcher assumes num_threads being same for all parameters.
  using Atom = cuda::std::conditional_t<
      SM80,
      cuda::std::conditional_t<
          cuda::std::is_same_v<Element, half_t>,
          SM80_16x8x16_F32F16F16F32_TN,
          cuda::std::conditional_t<
              cuda::std::is_same_v<Element, bfloat16_t>,
              SM80_16x8x16_F32BF16BF16F32_TN,
              UniversalFMA<float>
          >
      >,
      UniversalFMA<float, Element, Element>>;
  if constexpr (!SM80 || cuda::std::is_same_v<Element, float>) {
    return make_tiled_mma(Atom{}, Layout<Shape<_16,_8,_1>>{});
  } else {
    if constexpr (size<0>(cta_tiler) >= 32) {
      return make_tiled_mma(Atom{}, Layout<Shape<_2,_2,_1>>{}, Tile<_32,_32,_16>{});
    } else {
      return make_tiled_mma(Atom{}, Layout<Shape<_1,_4,_1>>{}, Tile<_16,_32,_16>{});
    }
  }
}

template <typename T, bool KMajor, bool Aligned,
          typename NumThreads, typename TileM, typename TileK>
inline constexpr auto make_tiled_copy(NumThreads num_threads, TileM bM, TileK bK) {
  // TODO: Only do 1-element read for the tile of residue.
  auto n_read = Int<Aligned ? 8 : 1>{};
  auto atom = Copy_Atom<UniversalCopy<uint_bit_t<n_read * sizeof_bits_v<T>>>, T>{};
  if constexpr (KMajor) {
    auto k_threads = bK / n_read;
    return make_tiled_copy(
        atom,
        make_layout(make_shape(Int<num_threads / k_threads>{}, k_threads), LayoutRight{}),
        make_layout(make_shape(Int<1>{}, n_read)));
  } else {
    auto m_threads = bM / n_read;
    return make_tiled_copy(
        atom,
        make_layout(make_shape(m_threads, Int<num_threads / m_threads>{}), LayoutLeft{}),
        make_layout(make_shape(n_read, Int<1>{})));
  }
}

template <bool KMajorA, bool KMajorB, bool Aligned, bool SM80,
          typename CtaTiler,
          typename TensorA,
          typename TensorB,
          typename TensorC>
CUTE_DEVICE void gemm_sm70_mainloop(
    CtaTiler cta_tiler,
    TensorA gA,
    TensorB gB,
    TensorC gC,
    int m_max_coord,
    int n_max_coord,
    int k_residue,
    int thread_idx) {
  // Get the types of operands.
  using Element = typename decltype(gA)::value_type;

  // Shift tensor so we handle residue of K in the 0th tile.
  gA = domain_offset(make_coord(0, k_residue, 0), gA);
  gB = domain_offset(make_coord(0, k_residue, 0), gB);

  // Define smem layouts.
  auto [sA_layout, sB_layout] = make_smem_layouts<KMajorA, KMajorB>(cta_tiler);

  // Shared memory buffer.
  extern __shared__ char smem_buf[];
  using SharedStorage = SharedStorage<Element, decltype(sA_layout), decltype(sB_layout)>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_buf);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout); // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout); // (BLK_N,BLK_K)

  // Define MMA.
  auto mma = make_tiled_mma<SM80, Element>(cta_tiler);
  auto num_threads = size(mma);

  // Define copy atoms.
  auto [bM, bN, bK] = cta_tiler;
  TiledCopy copy_a = make_tiled_copy<Element, KMajorA, Aligned>(num_threads, bM, bK);
  TiledCopy copy_b = make_tiled_copy<Element, KMajorB, Aligned>(num_threads, bN, bK);

  // Partition the copying of A/B/C tiles across the threads.
  ThrCopy thr_copy_a = copy_a.get_slice(thread_idx);
  Tensor tAgA = thr_copy_a.partition_S(gA); // (ACPY,ACPY_M,ACPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA); // (ACPY,ACPY_M,ACPY_K)
  Tensor tArA = make_fragment_like(tAsA);   // (ACPY,ACPY_M,ACPY_K)

  ThrCopy thr_copy_b = copy_b.get_slice(thread_idx);
  Tensor tBgB = thr_copy_b.partition_S(gB); // (BCPY,BCPY_N,BCPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB); // (BCPY,BCPY_N,BCPY_K)
  Tensor tBrB = make_fragment_like(tBsB);   // (BCPY,BCPY_M,BCPY_K)

  // MMA.
  ThrMMA thr_mma = mma.get_slice(thread_idx);
  Tensor tCsA = thr_mma.partition_A(sA);       // (MMA,MMA_M,MMA_K)
  Tensor tCsB = thr_mma.partition_B(sB);       // (MMA,MMA_N,MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC);       // (MMA,MMA_M,MMA_N)
  Tensor tCrC = thr_mma.make_fragment_C(tCgC); // (MMA,MMA_M,MMA_N)

  // Predicates for m/n bounds.
  Tensor tApA = make_tensor<bool>(make_shape(size<1>(tAsA), size<2>(tAsA)), Stride<_1,_0>{}); // (CPY_M,CPY_K)
  Tensor tBpB = make_tensor<bool>(make_shape(size<1>(tBsB), size<2>(tBsB)), Stride<_1,_0>{}); // (CPY_N,CPY_K)
  Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA))); // (BLK_M,BLK_K)
  Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB))); // (BLK_N,BLK_K)
  Tensor cC = make_identity_tensor(make_shape(size<0>(gC), size<1>(gC))); // (M,N)
  Tensor tAcA = thr_copy_a.partition_S(cA); // (CPY,CPY_M,CPY_K)
  Tensor tBcB = thr_copy_b.partition_S(cB); // (CPY,CPY_N,CPY_K)
  Tensor tCcC = thr_mma.partition_C(cC);    // (MMA,MMA_M,MMA_N)
  CUTE_UNROLL
  for (int m = 0; m < size<0>(tApA); ++m) {
    tApA(m,0) = get<0>(tAcA(0,m,0)) < m_max_coord;
  }
  CUTE_UNROLL
  for (int n = 0; n < size<0>(tBpB); ++n) {
    tBpB(n,0) = get<0>(tBcB(0,n,0)) < n_max_coord;
  }

  // GMEM => RMEM.
  auto fetch_gmem = [&](int tile) {
    copy_if(copy_a, tApA, tAgA(_,_,_,tile), tArA);
    copy_if(copy_b, tBpB, tBgB(_,_,_,tile), tBrB);
  };
  // RMEM => SMEM.
  auto store_smem = [&]() {
    __syncthreads();
    copy(tArA, tAsA);
    copy(tBrB, tBsB);
    __syncthreads();
  };

  // Clear the rmem tiles to account for predicated off loads.
  clear(tArA);
  clear(tBrB);

  // Prefetch first tile.
  Tensor tAgA_k = tAgA(_,_,_,0);
  Tensor tBgB_k = tBgB(_,_,_,0);
  CUTE_UNROLL
  for (int k = 0; k < size<2>(tArA); ++k) {
    if (get<1>(tAcA(0,0,k)) >= -k_residue) {
      copy_if(copy_a, tApA(_,k), tAgA_k(_,_,k), tArA(_,_,k));
    }
  }
  CUTE_UNROLL
  for (int k = 0; k < size<2>(tBrB); ++k) {
    if (get<1>(tBcB(0,0,k)) >= -k_residue) {
      copy_if(copy_b, tBpB(_,k), tBgB_k(_,_,k), tBrB(_,_,k));
    }
  }

  // Clear accumulators.
  clear(tCrC);

  // Loop over CTA tiles.
  auto K_TILE_MAX = size<3>(tAgA);
  for (int tile = 0; tile < K_TILE_MAX; ++tile) {
    store_smem();
    // Avoid fetching full 0th-tile when there is residue.
    if (K_TILE_MAX > 1) {
      fetch_gmem((tile + 1 < K_TILE_MAX) ? tile + 1 : tile);
    }
    gemm(mma, tCsA, tCsB, tCrC);
  }

  // Epilogue.
  CUTE_UNROLL
  for (int i = 0; i < size(tCrC); ++i) {
    if ((get<0>(tCcC(i)) < m_max_coord) && (get<1>(tCcC(i)) < n_max_coord)) {
      tCgC(i) = Element(tCrC(i));
    }
  }
}

template <bool KMajor>
inline constexpr auto make_matrix_stride(int m, int k) {
  if constexpr (KMajor) {
    return cute::make_stride(k, cute::Int<1>{}, m * k);
  } else {
    return cute::make_stride(cute::Int<1>{}, m, m * k);
  }
}

template <bool KMajorA, bool KMajorB, bool Aligned, bool SM80,
          typename Element, typename CtaTiler>
__global__
__launch_bounds__(decltype(size(make_tiled_mma<SM80, Element>(CtaTiler{})))::value)
void gemm_sm70_kernel(
    const Element* A,
    const Element* B,
    const uint32_t* lhs_indices,
    const uint32_t* rhs_indices,
    Element* C,
    int m, int n, int k, int l) {
  int thread_idx = int(threadIdx.x);
  int m_coord = int(blockIdx.x);
  int n_coord = int(blockIdx.y);
  int l_coord = int(blockIdx.z);

  // Define layouts (mixed).
  auto dA = make_matrix_stride<KMajorA>(m, k); // (dM,dK,dL)
  auto dB = make_matrix_stride<KMajorB>(n, k); // (dN,dK,dL)
  auto dC = make_stride(n, Int<1>{}, m * n);   // (dM,dN,dL)

  // Represent the full tensors.
  Tensor mA_mkl = make_tensor(make_gmem_ptr(A), make_shape(m, k, l), dA); // (M,K,L)
  Tensor mB_nkl = make_tensor(make_gmem_ptr(B), make_shape(n, k, l), dB); // (N,K,L)
  Tensor mC_mnl = make_tensor(make_gmem_ptr(C), make_shape(m, n, l), dC); // (M,N,L)

  // For gather, use index lookup for input batch slicing.
  uint32_t a_batch = lhs_indices ? lhs_indices[l_coord] : l_coord;
  uint32_t b_batch = rhs_indices ? rhs_indices[l_coord] : l_coord;

  // Get batch slice.
  Tensor mA = mA_mkl(_,_,a_batch); // (M,K)
  Tensor mB = mB_nkl(_,_,b_batch); // (N,K)
  Tensor mC = mC_mnl(_,_,l_coord); // (M,N)

  // Get the appropriate blocks for this thread block.
  auto cta_tiler = CtaTiler{};
  auto cta_coord = make_coord(m_coord, n_coord, _); // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{}); // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{}); // (BLK_M,BLK_N)

  // Compute tile residues for predication.
  int m_max_coord = m - size<0>(cta_tiler) * m_coord; // M - BLK_M * m_coord
  int n_max_coord = n - size<1>(cta_tiler) * n_coord; // N - BLK_N * n_coord
  int k_residue = k - size<1>(gA) * size<2>(gA);

  gemm_sm70_mainloop<KMajorA, KMajorB, Aligned, SM80>(
      cta_tiler,
      gA,
      gB,
      gC,
      m_max_coord, n_max_coord, k_residue,
      thread_idx);
}

} // namespace mlx::core::cu
