// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/quantized/qmm/cute_dequant.cuh"
#include "mlx/dtype_utils.h"

// clang-format off

// We can't put kernel code in mlx::core due to name conflicts of "Shape".
namespace cutlass_gemm {

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

inline constexpr auto make_smem_layouts(auto cta_tiler) {
  // Define the A/B smem layouts (static).
  auto swizzle_ab = composition(Swizzle<3,3,3>{},
                                Layout<Shape <_8,Shape <_8, _8>>,
                                       Stride<_8,Stride<_1,_64>>>{});
  auto [bM, bN, bK] = cta_tiler;
  auto bP = Int<3>{}; // pipeline
  auto sA_layout = tile_to_shape(swizzle_ab, make_shape(bM, bK, bP));
  auto sB_layout = tile_to_shape(swizzle_ab, make_shape(bN, bK, bP));

  // Define the C smem layouts (static).
  // TODO: Find a better swizzle.
  auto sC_layout = tile_to_shape(swizzle_ab, make_shape(bM, bN));

  return std::make_tuple(sA_layout, sB_layout, sC_layout);
}

template <typename T, int bits, template <typename U> typename Atom>
inline constexpr auto make_tiled_copy(auto num_threads) {
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
          typename TensorC,
          typename TiledMma>
CUTE_DEVICE void qmm_sm80_mainloop(
    CtaTiler cta_tiler,
    TensorA gA,
    TensorB gB,
    TensorS gS,
    TensorZ gZ,
    TensorC gC,
    TiledMma mma,
    int m_max_coord,
    int thread_idx) {
  // Get the types of operands.
  using Element = decltype(gA)::value_type;
  using Quant = decltype(gB)::value_type;
  using Scale = decltype(gS)::value_type;

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

  // Define copy atoms.
  constexpr int element_bits = sizeof_bits_v<Element>;
  constexpr int quant_bits = sizeof_bits_v<Quant>;
  constexpr int qload = 128 / (element_bits / quant_bits);
  auto num_threads = size(mma);
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
    if constexpr (quant_has_bias_v<Quant>) {
      copy(g2r_copy_s, g2r_tCgZ(_,_,_,tile), g2r_tCrZ);
    }
  };
  // Copy A/B: SMEM => RMEM.
  auto fetch_smem = [&](auto block) {
    copy(s2r_atom_a, s2r_tCsA(_,_,block,smem_pipe_read), s2r_tCrA(_,_,block));
    copy(s2r_atom_b, s2r_tCsB(_,_,block,smem_pipe_read), s2r_tCrB(_,_,block));
    CUTE_UNROLL
    for (int n = 0; n < size<1>(tCrB); ++n) {
      cute_vectorized_dequant(
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

inline constexpr auto make_scales_layout(auto n, auto k, auto l, auto group_size) {
  return make_layout(
      make_shape(n, make_shape(group_size, k / group_size), l),
      make_stride(k / group_size, Stride<_0,_1>{}, n * k / group_size));
}

template <int TileM>
inline constexpr auto make_cta_tiler(auto group_size) {
  auto bM = Int<TileM>{};
  auto bN = Int<128>{};
  auto bK = Int<max(64, group_size)>{};
  return make_shape(bM, bN, bK);
}

template <int TileM, typename Element>
inline constexpr auto make_tiled_mma() {
  using Atom = std::conditional_t<
      std::is_same_v<Element, half_t>,
      SM80_16x8x16_F32F16F16F32_TN,
      std::conditional_t<
          std::is_same_v<Element, bfloat16_t>,
          SM80_16x8x16_F32BF16BF16F32_TN,
          UniversalFMA<float>>>;
  if constexpr (TileM >= 32) {
    return make_tiled_mma(Atom{}, Layout<Shape<_2,_2,_1>>{}, Tile<_32,_32,_16>{});
  } else {
    return make_tiled_mma(Atom{}, Layout<Shape<_1,_4,_1>>{}, Tile<_16,_32,_16>{});
  }
}

} // namespace cutlass_gemm

// clang-format on

namespace mlx::core {

template <typename F>
inline void dispatch_element_types(Dtype dtype, const char* tag, F&& f) {
  if (dtype == float16) {
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
      if (bits == 4) {
        f.template operator()<cutlass::uint4b_t, T, group_size>();
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
