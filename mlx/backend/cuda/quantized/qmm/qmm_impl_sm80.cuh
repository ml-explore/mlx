// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/quantized/qmm/cute_dequant.cuh"
#include "mlx/backend/cuda/quantized/qmm/qmm.h"
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

template <typename ProblemShape, typename CtaTiler,
          typename Element, typename Quant, typename Scale,
          typename StrideA, typename SmemLayoutA, typename TiledCopyA, typename S2RAtomA,
          typename StrideB, typename SmemLayoutB, typename TiledCopyB, typename S2RAtomB,
          typename StrideC, typename SmemLayoutC, typename TiledCopyC, typename R2SAtomC,
          typename LayoutS, typename G2RAtomS, typename TiledMma>
__global__ void qmm_sm80_kernel(
    ProblemShape shape_MNKL, CtaTiler cta_tiler,
    const Element* A, StrideA dA, SmemLayoutA sA_layout, TiledCopyA g2s_copy_a, S2RAtomA s2r_atom_a,
    const Quant*   B, StrideB dB, SmemLayoutB sB_layout, TiledCopyB g2s_copy_b, S2RAtomB s2r_atom_b,
          Element* C, StrideC dC, SmemLayoutC sC_layout, TiledCopyC s2g_copy_c, R2SAtomC r2s_atom_c,
    const Scale* S, const Element* Z, LayoutS S_layout, G2RAtomS g2r_atom_s, TiledMma mma) {
  CUTE_STATIC_ASSERT_V(size(g2s_copy_a) == size(mma));
  CUTE_STATIC_ASSERT_V(size(g2s_copy_b) == size(mma));
  CUTE_STATIC_ASSERT_V(size(s2g_copy_c) == size(mma));
  CUTE_STATIC_ASSERT_V(congruent(select<0,2,3>(shape_MNKL), dA));
  CUTE_STATIC_ASSERT_V(congruent(select<1,2,3>(shape_MNKL), dB));
  CUTE_STATIC_ASSERT_V(congruent(select<0,1,3>(shape_MNKL), dC));

  int thread_idx = int(threadIdx.x);
  auto [m_coord, n_coord, l_coord] = static_cast<uint3>(blockIdx);

  // Represent the full tensors.
  Tensor mA_mkl = make_tensor(make_gmem_ptr(A),        select<0,2,3>(shape_MNKL), dA); // (M,K,L)
  Tensor mB_nkl = make_tensor(make_gmem_ptr<Quant>(B), select<1,2,3>(shape_MNKL), dB); // (N,K,L)
  Tensor mC_mnl = make_tensor(make_gmem_ptr(C),        select<0,1,3>(shape_MNKL), dC); // (M,N,L)

  Tensor mS_nkl = make_tensor(make_gmem_ptr(S), S_layout); // (N,(group_size,K/group_size),L)
  Tensor mZ_nkl = make_tensor(make_gmem_ptr(Z), S_layout); // (N,(group_size,K/group_size),L)

  // Get batch slice.
  Tensor mA = mA_mkl(_,_,l_coord); // (M,K)
  Tensor mB = mB_nkl(_,_,l_coord); // (N,K)
  Tensor mC = mC_mnl(_,_,l_coord); // (M,N)

  Tensor mS = mS_nkl(_,_,l_coord); // (N,(group_size,K/group_size))
  Tensor mZ = mZ_nkl(_,_,l_coord); // (N,(group_size,K/group_size))

  // Get the appropriate blocks for this thread block.
  auto cta_coord = make_coord(m_coord, n_coord, _); // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{}); // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{}); // (BLK_M,BLK_N)

  Tensor gS = local_tile(mS, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
  Tensor gZ = local_tile(mZ, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)

  // Shared memory buffers.
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<Element, Quant,
                                      SmemLayoutA,
                                      SmemLayoutB,
                                      SmemLayoutC>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.mainloop.A.begin()), sA_layout); // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smem.mainloop.B.begin()), sB_layout); // (BLK_N,BLK_K)
  Tensor sC = make_tensor(make_smem_ptr(smem.epilogue.C.begin()), sC_layout); // (BLK_M,BLK_N)

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
  auto m_max_coord = size<0>(shape_MNKL) - size<0>(gA) * m_coord; // M - BLK_M * m_coord
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

template <typename Element>
inline constexpr auto make_mma_atom() {
  if constexpr (std::is_same_v<Element, half_t>) {
    return SM80_16x8x16_F32F16F16F32_TN{};
  }
  if constexpr (std::is_same_v<Element, bfloat16_t>) {
    return SM80_16x8x16_F32BF16BF16F32_TN{};
  }
}

template <int TileM, typename Element>
inline constexpr auto make_tiled_mma() {
  constexpr auto atom = make_mma_atom<Element>();
  if constexpr (TileM >= 32) {
    return make_tiled_mma(atom, Layout<Shape<_2,_2,_1>>{}, Tile<_32,_32,_16>{});
  } else {
    return make_tiled_mma(atom, Layout<Shape<_1,_4,_1>>{}, Tile<_16,_32,_16>{});
  }
}

template <typename T, int bits, template <typename U> typename Atom, typename NumThreads>
inline auto make_tiled_copy(NumThreads num_threads) {
  return make_tiled_copy(
      Copy_Atom<Atom<uint_bit_t<bits>>, T>{},
      make_layout(make_shape(Int<num_threads / 8>{}, Int<8>{}), LayoutRight{}),
      make_layout(make_shape(Int<1>{}, Int<bits / sizeof_bits_v<T>>{})));
}

template <int TileM = 16, typename Element, typename Quant, typename Scale, typename GroupSize, typename F>
void qmm_sm80(
    const Element* A,
    const Quant*   B,
    const Scale* S,
    const Element* Z,
    Element* C,
    int m, int n, int k, int l,
    bool broadcast_b,
    GroupSize group_size,
    F&& launch_kernel) {
  // Define shapes (dynamic).
  auto prob_shape = make_shape(m, n, k, l); // (M,N,K,L)

  // Define TN strides (mixed).
  auto dA = make_stride(k, Int<1>{}, m * k); // (dM,dK,dL)
  auto dB = make_stride(k, Int<1>{}, n * k); // (dN,dK,dL)
  auto dC = make_stride(n, Int<1>{}, m * n); // (dM,dN,dL)

  // Define layout of scales/biases (mixed).
  auto S_layout = make_layout(
      make_shape(n, make_shape(group_size, k / group_size), l),
      make_stride(k / group_size, Stride<_0, _1>{}, n * k / group_size));

  // Handle broadcasting.
  if (broadcast_b) {
    get<2>(dB) = 0;
    get<2>(stride(S_layout)) = 0;
  }

  // Define CTA tile sizes (static).
  auto bM = Int<TileM>{};
  auto bN = Int<128>{};
  auto bK = Int<max(64, group_size)>{};
  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M,BLK_N,BLK_K)

  // Define MMA.
  TiledMMA mma = make_tiled_mma<TileM, Element>();
  auto num_threads = size(mma);

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

  // Define the scales/biases smem layouts (static).
  auto bS = ceil_div(bK, group_size);
  auto sS_layout = make_layout(make_shape(bN, make_shape(group_size, bS)),
                               make_stride(bS, Stride<_0, _1>{}));

  // Atoms.
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

  auto* kernel = &qmm_sm80_kernel<
      decltype(prob_shape), decltype(cta_tiler),
      Element, Quant, Scale,
      decltype(dA), decltype(sA_layout), decltype(g2s_copy_a), decltype(s2r_atom_a),
      decltype(dB), decltype(sB_layout), decltype(g2s_copy_b), decltype(s2r_atom_b),
      decltype(dC), decltype(sC_layout), decltype(s2g_copy_c), decltype(r2s_atom_c),
      decltype(S_layout), decltype(g2r_atom_s), decltype(mma)>;

  // Set L1 to be SMEM only.
  size_t smem_bytes = sizeof(SharedStorage<Element, Quant,
                                           decltype(sA_layout),
                                           decltype(sB_layout),
                                           decltype(sC_layout)>);
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  dim3 num_blocks(size(ceil_div(m, bM)), size(ceil_div(n, bN)), l);
  dim3 block_dims(num_threads);
  void* args[] = {
      &prob_shape, &cta_tiler,
      &A, &dA, &sA_layout, &g2s_copy_a, &s2r_atom_a,
      &B, &dB, &sB_layout, &g2s_copy_b, &s2r_atom_b,
      &C, &dC, &sC_layout, &s2g_copy_c, &r2s_atom_c,
      &S, &Z, &S_layout, &g2r_atom_s, &mma};
  launch_kernel(reinterpret_cast<void*>(kernel), num_blocks, block_dims, smem_bytes, args);
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

template <int TileM>
void qmm_impl_sm80(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder) {
  const char* tag = "[quantized_matmul]";
  int m = out.ndim() > 1 ? out.shape(-2) : 1;
  int n = out.shape(-1);
  int k = x.shape(-1);
  int l = out.size() / (m * n);
  bool broadcast_b = w.ndim() == 2;

  dispatch_element_types(out.dtype(), tag, [&]<typename Element>() {
    dispatch_quant_types<Element>(
        bits,
        group_size,
        mode,
        tag,
        [&]<typename Quant, typename Scale, int group_size>() {
          encoder.set_input_array(x);
          encoder.set_input_array(w);
          encoder.set_input_array(scales);
          if (biases) {
            encoder.set_input_array(*biases);
          }
          encoder.set_output_array(out);
          cutlass_gemm::qmm_sm80<TileM>(
              gpu_ptr<Element>(x),
              gpu_ptr<Quant>(w),
              gpu_ptr<Scale>(scales),
              biases ? gpu_ptr<Element>(*biases) : nullptr,
              gpu_ptr<Element>(out),
              m,
              n,
              k,
              l,
              broadcast_b,
              cute::Int<group_size>{},
              [&](auto* kernel,
                  dim3 num_blocks,
                  dim3 block_dims,
                  uint32_t smem_bytes,
                  void** args) {
                encoder.add_kernel_node_raw(
                    kernel, num_blocks, block_dims, {}, smem_bytes, args);
              });
        });
  });
}

} // namespace mlx::core

#define QMM_SM80_GPU(TileM)               \
  namespace mlx::core {                   \
  template void qmm_impl_sm80<TileM>(     \
      const array& x,                     \
      const array& w,                     \
      const array& scales,                \
      const std::optional<array>& biases, \
      array& out,                         \
      int bits,                           \
      int group_size,                     \
      QuantizationMode mode,              \
      cu::CommandEncoder& encoder);       \
  }
