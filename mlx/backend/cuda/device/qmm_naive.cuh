// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/device/cute_dequant.cuh"

#include <cuda/cmath>

// clang-format off

namespace mlx::core::cu {

using namespace cute;

template <typename Element, typename SmemLayoutA, typename SmemLayoutB>
struct SharedStorage {
  ArrayEngine<Element, cosize_v<SmemLayoutA>> A;
  ArrayEngine<Element, cosize_v<SmemLayoutB>> B;
};

template <bool KMajor = true, typename TileM, typename TileN>
inline constexpr auto make_smem_layout(TileM bM, TileN bK) {
  // TODO: Calculate swizzle based on tile shape.
  if constexpr (KMajor) {
    auto swizzle = composition(Swizzle<3,3,3>{},
                               Layout<Shape <_8,Shape <_8, _8>>,
                                      Stride<_8,Stride<_1,_64>>>{});
    return tile_to_shape(swizzle, make_shape(bM, bK));
  } else {
    auto swizzle = composition(Swizzle<3,3,3>{},
                               Layout<Shape<_64,_1>, Stride<_1,_64>>{});
    return tile_to_shape(swizzle, make_shape(bM, bK));
  }
}

template <bool KMajor = true, typename CtaTiler>
inline constexpr auto make_smem_layouts(CtaTiler cta_tiler) {
  // Note: Kernel launcher assumes cosize being same for all KMajor.
  auto [bM, bN, bK] = cta_tiler;
  auto sA_layout = make_smem_layout(bM, bK);
  auto sB_layout = make_smem_layout<KMajor>(bN, bK);
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

template <typename T, bool KMajor = true, bool HasKResidue = false,
          typename NumThreads, typename TileM, typename TileN>
inline constexpr auto make_tiled_copy(NumThreads num_threads, TileM bM, TileN bK) {
  // TODO: Only do 1-element read for the tile of residue.
  auto n_read = Int<HasKResidue ? 1 : 8>{};
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

template <bool KMajor, bool HasKResidue, bool SM80,
          typename CtaTiler,
          typename TensorA,
          typename TensorB,
          typename TensorS,
          typename TensorZ,
          typename TensorC>
CUTE_DEVICE void qmm_naive_mainloop(
    CtaTiler cta_tiler,
    TensorA gA,
    TensorB gB,
    TensorS gS,
    TensorZ gZ,
    TensorC gC,
    int m_max_coord,
    int n_max_coord,
    int k_residue,
    int thread_idx) {
  // Get the types of operands.
  using Element = typename decltype(gA)::value_type;
  using Quant = typename decltype(gB)::value_type;

  // Shift tensor so we handle residue of K in the 0th tile.
  gA = domain_offset(make_coord(0, k_residue, 0), gA);
  if constexpr (sizeof_bits_v<Quant> % 8 == 0) {
    gB = domain_offset(make_coord(0, k_residue, 0), gB);
  } else {
    // TODO: Figure out why domain_offset is not returning wrong offset.
    gB.data() = recast_ptr<Quant>(raw_pointer_cast(gB.data()) + gB.layout()(0, k_residue, 0) * cuda::std::min(8, sizeof_bits_v<Quant>) / 8);
  }
  gS = domain_offset(make_coord(0, k_residue, 0), gS);
  if constexpr (quant_has_bias_v<Quant>) {
    gZ = domain_offset(make_coord(0, k_residue, 0), gZ);
  }

  // Define smem layouts.
  auto [sA_layout, sB_layout] = make_smem_layouts<KMajor>(cta_tiler);

  // Shared memory buffer.
  extern __shared__ char smem_buf[];
  using SharedStorage = SharedStorage<Element, decltype(sA_layout), decltype(sB_layout)>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_buf);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout); // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout); // (BLK_N,BLK_K)

  // Define MMA.
  auto mma = make_tiled_mma<SM80, Element>(CtaTiler{});
  auto num_threads = size(mma);

  // Define copy atoms.
  auto [bM, bN, bK] = cta_tiler;
  TiledCopy copy_a = make_tiled_copy<Element, true, HasKResidue>(num_threads, bM, bK);
  TiledCopy copy_b = make_tiled_copy<Quant, KMajor>(num_threads, bN, bK);

  // Partition the copying of A/B/C tiles across the threads.
  ThrCopy thr_copy_a = copy_a.get_slice(thread_idx);
  Tensor tAgA = thr_copy_a.partition_S(gA); // (ACPY,ACPY_M,ACPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA); // (ACPY,ACPY_M,ACPY_K)
  Tensor tArA = make_fragment_like(tAsA);   // (ACPY,ACPY_M,ACPY_K)

  ThrCopy thr_copy_b = copy_b.get_slice(thread_idx);
  Tensor tBgB = thr_copy_b.partition_S(gB);        // (BCPY,BCPY_N,BCPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB);        // (BCPY,BCPY_N,BCPY_K)
  Tensor tBrB = make_fragment_like<Quant>(tBsB);   // (BCPY,BCPY_M,BCPY_K)
  Tensor tBrB_dq = make_fragment_like(tBsB);       // (BCPY,BCPY_M,BCPY_K)
  Tensor tBgS = thr_copy_b.partition_S(gS);        // (BCPY,BCPY_N,BCPY_K,k)
  Tensor tBrS = make_fragment_like(tBgS(_,_,_,0)); // (BCPY,BCPY_N,BCPY_K)
  Tensor tBgZ = thr_copy_b.partition_S(gZ);        // (BCPY,BCPY_N,BCPY_K,k)
  Tensor tBrZ = make_fragment_like(tBgZ(_,_,_,0)); // (BCPY,BCPY_N,BCPY_K)

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
    CUTE_UNROLL
    for (int n = 0; n < size<1>(tBrS); ++n) {
      if (tBpB(n,0)) {
        copy(tBgS(_,n,_,tile), tBrS(_,n,_));
        if constexpr (quant_has_bias_v<Quant>) {
          copy(tBgZ(_,n,_,tile), tBrZ(_,n,_));
        }
      }
    }
  };
  // RMEM => SMEM.
  auto store_smem = [&]() {
    __syncthreads();
    copy(tArA, tAsA);
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tBrB); ++k) {
      CUTE_UNROLL
      for (int n = 0; n < size<1>(tBrB); ++n) {
        cute_dequant(tBrB(_,n,k), tBrS(_,n,k), tBrZ(_,n,k), tBrB_dq(_,n,k));
      }
    }
    copy(tBrB_dq, tBsB);
    __syncthreads();
  };

  // Clear the rmem tiles to account for predicated off loads.
  if constexpr (HasKResidue) {
    clear(tArA);
    clear(tBrB);
    clear(tBrS);
    clear(tBrZ);
  }

  // Prefetch first tile.
  if constexpr (HasKResidue) {
    Tensor tAgA_k = tAgA(_,_,_,0);
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tArA); ++k) {
      if (get<1>(tAcA(0,0,k)) >= -k_residue) {
        copy_if(copy_a, tApA(_,k), tAgA_k(_,_,k), tArA(_,_,k));
      }
    }
    Tensor tBgB_k = tBgB(_,_,_,0);
    Tensor tBgS_k = tBgS(_,_,_,0);
    Tensor tBgZ_k = tBgZ(_,_,_,0);
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tBrB); ++k) {
      if (get<1>(tBcB(0,0,k)) >= -k_residue) {
        copy_if(copy_b, tBpB(_,k), tBgB_k(_,_,k), tBrB(_,_,k));
        CUTE_UNROLL
        for (int n = 0; n < size<1>(tBrS); ++n) {
          if (tBpB(n,k)) {
            copy(tBgS_k(_,n,k), tBrS(_,n,k));
            if constexpr (quant_has_bias_v<Quant>) {
              copy(tBgZ_k(_,n,k), tBrZ(_,n,k));
            }
          }
        }
      }
    }
  } else {
    fetch_gmem(0);
  }

  // Clear accumulators.
  clear(tCrC);

  // Loop over CTA tiles.
  auto K_TILE_MAX = size<3>(tAgA);
  for (int tile = 0; tile < K_TILE_MAX; ++tile) {
    store_smem();
    if constexpr (HasKResidue) {
      // Avoid fetching full 0th-tile when there is residue.
      if (K_TILE_MAX > 1) {
        fetch_gmem((tile + 1 < K_TILE_MAX) ? tile + 1 : tile);
      }
    } else {
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
    return make_stride(k, Int<1>{}, m * k);
  } else {
    return make_stride(Int<1>{}, m, m * k);
  }
}

template <int GroupSize, bool KMajor>
inline constexpr auto make_scales_layout(int n, int k, int l) {
  auto group_size = Int<GroupSize>{};
  if constexpr (KMajor) {
    return make_layout(
        make_shape(n, make_shape(group_size, k / group_size), l),
        make_stride(k / group_size, Stride<_0,_1>{}, n * k / group_size));
  } else {
    return make_layout(
        make_shape(make_shape(group_size, n / group_size), k, l),
        make_stride(Stride<_0,_1>{}, n / group_size, n * k / group_size));
  }
}

template <int GroupSize, bool KMajor, bool HasKResidue, bool SM80,
          typename Element, typename Quant, typename Scale, typename CtaTiler>
__global__
__launch_bounds__(decltype(size(make_tiled_mma<SM80, Element>(CtaTiler{})))::value)
void qmm_naive_kernel(
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
  auto dA = make_stride(k, Int<1>{}, m * k);  // (dM,dK,dL)
  auto dB = make_matrix_stride<KMajor>(n, k); // (dN,dK,dL)
  auto dC = make_stride(n, Int<1>{}, m * n);  // (dM,dN,dL)
  auto S_layout = make_scales_layout<GroupSize, KMajor>(n, k, l);

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

  // For gather, use index lookup for input batch slicing.
  uint32_t a_batch = lhs_indices ? lhs_indices[l_coord] : l_coord;
  uint32_t b_batch = rhs_indices ? rhs_indices[l_coord] : l_coord;

  // Get batch slice.
  Tensor mA = mA_mkl(_,_,a_batch); // (M,K)
  Tensor mB = mB_nkl(_,_,b_batch); // (N,K)
  Tensor mC = mC_mnl(_,_,l_coord); // (M,N)

  Tensor mS = mS_nkl(_,_,b_batch); // (N,(group_size,K/group_size))

  // Get the appropriate blocks for this thread block.
  auto cta_tiler = CtaTiler{};
  auto cta_coord = make_coord(m_coord, n_coord, _); // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{}); // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{}); // (BLK_M,BLK_N)

  Tensor gS = local_tile(mS, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
  auto gZ = [&]() {
    if constexpr (quant_has_bias_v<Quant>) {
      Tensor mZ_nkl = make_tensor(make_gmem_ptr(Z), S_layout); // (N,(group_size,K/group_size),L)
      Tensor mZ = mZ_nkl(_,_,b_batch); // (N,(group_size,K/group_size))
      return local_tile(mZ, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
    } else {
      // Dummy tensor; no-bias paths never offset or load gZ.
      return gS;
    }
  }();

  // Compute tile residues for predication.
  int m_max_coord = m - size<0>(cta_tiler) * m_coord; // M - BLK_M * m_coord
  int n_max_coord = n - size<1>(cta_tiler) * n_coord; // N - BLK_N * n_coord
  int k_residue = k - size<1>(gA) * size<2>(gA);

  qmm_naive_mainloop<KMajor, HasKResidue, SM80>(
      cta_tiler,
      gA,
      gB,
      gS,
      gZ,
      gC,
      m_max_coord, n_max_coord, k_residue,
      thread_idx);
}

} // namespace mlx::core::cu
