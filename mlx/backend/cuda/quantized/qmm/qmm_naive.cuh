// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/quantized/qmm/cute_dequant.cuh"
#include "mlx/dtype_utils.h"

// clang-format off

// We can't put kernel code in mlx::core due to name conflicts of "Shape".
namespace cutlass_gemm {

using namespace cute;

template <typename Element, typename SmemLayoutA, typename SmemLayoutB>
struct SharedStorage {
  ArrayEngine<Element, cosize_v<SmemLayoutA>> A;
  ArrayEngine<Element, cosize_v<SmemLayoutB>> B;
};

template <bool KMajor = true>
inline constexpr auto make_smem_layout(auto bM, auto bK) {
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

template <bool KMajor = true>
inline constexpr auto make_smem_layouts(auto cta_tiler) {
  auto [bM, bN, bK] = cta_tiler;
  auto sA_layout = make_smem_layout(bM, bK);
  auto sB_layout = make_smem_layout<KMajor>(bN, bK);
  return std::make_tuple(sA_layout, sB_layout);
}

template <typename T, bool KMajor = true, bool HasKResidue = false>
inline constexpr auto make_tiled_copy(auto num_threads, auto bM, auto bK) {
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


__device__ __forceinline__ void
cute_naive_dequant(auto w, auto s, auto z, auto out) {
  using Element = typename decltype(out)::value_type;
  using Quant = typename decltype(w)::value_type;
  using Scale = typename decltype(s)::value_type;
  transform(w, out, [](Quant q) { return Element(q); } );
  transform(out, s, out, [](Element e, Scale s) { return e * Element(s); });
  if constexpr (quant_has_bias_v<Quant>) {
    transform(out, z, out, plus{});
  }
}

__device__ __forceinline__ void
cute_dequant(auto w, auto s, auto z, auto out) {
  if constexpr (stride(coalesce(w.layout())) == Int<1>{} &&
                is_static_v<decltype(s.layout())>) {
    cute_vectorized_dequant(w, s, z, out);
  } else {
    cute_naive_dequant(w, s, z, out);
  }
}

template <bool KMajor, bool HasKResidue, bool SM80,
          typename CtaTiler,
          typename TensorA,
          typename TensorB,
          typename TensorS,
          typename TensorZ,
          typename TensorC,
          typename TiledMma>
CUTE_DEVICE void qmm_naive_mainloop(
    CtaTiler cta_tiler,
    TensorA gA,
    TensorB gB,
    TensorS gS,
    TensorZ gZ,
    TensorC gC,
    TiledMma mma,
    int m_max_coord,
    int n_max_coord,
    int k_residue,
    int thread_idx) {
  // Get the types of operands.
  using Element = decltype(gA)::value_type;
  using Quant = decltype(gB)::value_type;

  // Shift tensor so we handle residue of K in the 0th tile.
  gA = domain_offset(make_coord(0, k_residue, 0), gA);
  if constexpr (sizeof_bits_v<Quant> % 8 == 0) {
    gB = domain_offset(make_coord(0, k_residue, 0), gB);
  } else {
    gB.data() = recast_ptr<Quant>(raw_pointer_cast(gB.data()) + gB.layout()(0, k_residue, 0) * cuda::std::min(8, sizeof_bits_v<Quant>) / 8);
  }
  gS = domain_offset(make_coord(0, k_residue, 0), gS);
  gZ = domain_offset(make_coord(0, k_residue, 0), gZ);

  // Define smem layouts.
  auto [sA_layout, sB_layout] = make_smem_layouts(cta_tiler);

  // Shared memory buffer.
  extern __shared__ char smem_buf[];
  using SharedStorage = SharedStorage<Element, decltype(sA_layout), decltype(sB_layout)>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_buf);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout); // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout); // (BLK_N,BLK_K)

  // Define copy atoms.
  auto num_threads = size(mma);
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
    copy(tBgS(_,_,_,tile), tBrS);
    copy(tBgZ(_,_,_,tile), tBrZ);
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
        copy(tBgS_k(_,_,k), tBrS(_,_,k));
        copy(tBgZ_k(_,_,k), tBrZ(_,_,k));
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
inline constexpr auto make_matrix_stride(auto m, auto k) {
  if constexpr (KMajor) {
    return cute::make_stride(k, cute::Int<1>{}, m * k);
  } else {
    return cute::make_stride(cute::Int<1>{}, m, m * k);
  }
}

template <bool KMajor>
inline constexpr auto make_scales_layout(auto n, auto k, auto l, auto group_size) {
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

template <int TileM, bool SM80>
inline constexpr auto make_cta_tiler(auto group_size) {
  auto bM = Int<TileM>{};
  auto bN = Int<(!SM80 && group_size > 64) ? 64 : 128>{};
  auto bK = Int<max(64, group_size)>{};
  return make_shape(bM, bN, bK);
}

template <bool SM80, typename Element>
inline constexpr auto make_tiled_mma(auto cta_tiler) {
  using Atom = std::conditional_t<
      SM80,
      std::conditional_t<
          std::is_same_v<Element, half_t>,
          SM80_16x8x16_F32F16F16F32_TN,
          std::conditional_t<
              std::is_same_v<Element, bfloat16_t>,
              SM80_16x8x16_F32BF16BF16F32_TN,
              UniversalFMA<float>
          >
      >,
      UniversalFMA<float, Element, Element>>;
  if constexpr (!SM80 || std::is_same_v<Element, float>) {
    return make_tiled_mma(Atom{}, Layout<Shape<_16,_8,_1>>{});
  } else {
    if constexpr (size<0>(cta_tiler) >= 32) {
      return make_tiled_mma(Atom{}, Layout<Shape<_2,_2,_1>>{}, Tile<_32,_32,_16>{});
    } else {
      return make_tiled_mma(Atom{}, Layout<Shape<_1,_4,_1>>{}, Tile<_16,_32,_16>{});
    }
  }
}

} // namespace cutlass_gemm

// clang-format on

namespace mlx::core {

template <typename F>
inline void dispatch_element_types(Dtype dtype, const char* tag, F&& f) {
  if (dtype == float32) {
    f.template operator()<float>();
  } else if (dtype == float16) {
    f.template operator()<cutlass::half_t>();
  } else if (dtype == bfloat16) {
    f.template operator()<cutlass::bfloat16_t>();
  } else {
    throw std::invalid_argument(
        fmt::format("{} Unsupported dtype: {}.", tag, dtype_to_string(dtype)));
  }
}

template <typename F>
inline void dispatch_groups(int group_size, const char* tag, F&& f) {
  if (group_size == 32) {
    f.template operator()<32>();
  } else if (group_size == 64) {
    f.template operator()<64>();
  } else if (group_size == 128) {
    f.template operator()<128>();
  } else {
    throw std::invalid_argument(
        fmt::format("{} Group size {} is not supported.", tag, group_size));
  }
}

template <typename T, typename F>
inline void dispatch_quant_types(
    int bits,
    int group_size,
    QuantizationMode mode,
    const char* tag,
    F&& f) {
  if (mode == QuantizationMode::Mxfp4) {
    f.template operator()<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32>();
  } else if (mode == QuantizationMode::Mxfp8) {
    f.template operator()<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32>();
  } else if (mode == QuantizationMode::Nvfp4) {
    f.template operator()<cutlass::float_e2m1_t, cutlass::float_e4m3_t, 16>();
  } else {
    dispatch_groups(group_size, tag, [&]<int group_size>() {
      if (bits == 2) {
        f.template operator()<cutlass::uint2b_t, T, group_size>();
      } else if (bits == 3) {
        f.template operator()<cutlass::uint3b_t, T, group_size>();
      } else if (bits == 4) {
        f.template operator()<cutlass::uint4b_t, T, group_size>();
      } else if (bits == 5) {
        f.template operator()<cutlass::uint5b_t, T, group_size>();
      } else if (bits == 6) {
        f.template operator()<cutlass::uint6b_t, T, group_size>();
      } else if (bits == 8) {
        f.template operator()<uint8_t, T, group_size>();
      } else {
        throw std::invalid_argument(
            fmt::format("{} {}-bit quantization is not supported.", tag, bits));
      }
    });
  }
}

} // namespace mlx::core
