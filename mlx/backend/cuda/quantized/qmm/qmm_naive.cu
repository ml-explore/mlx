// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/qmm/qmm.h"
#include "mlx/backend/cuda/quantized/qmm/qmm_naive.cuh"

// clang-format off

// We can't put kernel code in mlx::core due to name conflicts of "Shape".
namespace cutlass_gemm {

using namespace cute;

template <bool KMajor, bool HasKResidue, bool SM80,
          typename Element, typename Quant, typename Scale,
          typename ProblemShape,
          typename CtaTiler,
          typename StrideA,
          typename StrideB,
          typename LayoutS,
          typename StrideC,
          typename TiledMma>
__global__
__launch_bounds__(decltype(size(TiledMma{}))::value)
void qmm_naive_kernel(
    ProblemShape shape_MNKL,
    CtaTiler cta_tiler,
    const Element* A, StrideA dA,
    const Quant* B, StrideB dB,
    const Scale* S, const Element* Z, LayoutS S_layout,
    const uint32_t* lhs_indices, const uint32_t* rhs_indices,
    Element* C, StrideC dC,
    TiledMma mma) {
  CUTE_STATIC_ASSERT_V(congruent(select<0,2,3>(shape_MNKL), dA));
  CUTE_STATIC_ASSERT_V(congruent(select<1,2,3>(shape_MNKL), dB));
  CUTE_STATIC_ASSERT_V(congruent(select<0,1,3>(shape_MNKL), dC));

  int thread_idx = int(threadIdx.x);
  int m_coord = int(blockIdx.x);
  int n_coord = int(blockIdx.y);
  int l_coord = int(blockIdx.z);

  // Represent the full tensors.
  Tensor mA_mkl = make_tensor(make_gmem_ptr(A),        select<0,2,3>(shape_MNKL), dA); // (M,K,L)
  Tensor mB_nkl = make_tensor(make_gmem_ptr<Quant>(B), select<1,2,3>(shape_MNKL), dB); // (N,K,L)
  Tensor mC_mnl = make_tensor(make_gmem_ptr(C),        select<0,1,3>(shape_MNKL), dC); // (M,N,L)

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
  int m_max_coord = size<0>(shape_MNKL) - size<0>(cta_tiler) * m_coord; // M - BLK_M * m_coord
  int n_max_coord = size<1>(shape_MNKL) - size<1>(cta_tiler) * n_coord; // N - BLK_N * n_coord
  int k_residue = size<2>(shape_MNKL) - size<1>(gA) * size<2>(gA);

  qmm_naive_mainloop<KMajor, HasKResidue, SM80>(
      cta_tiler,
      gA,
      gB,
      gS,
      gZ,
      gC,
      mma,
      m_max_coord, n_max_coord, k_residue,
      thread_idx);
}

template <int TileM, bool KMajor, bool HasKResidue, bool SM80,
          typename Element, typename Quant, typename Scale>
void qmm_naive(
    const Element* A,
    const Quant*   B,
    const Scale*   S,
    const Element* Z,
    const uint32_t* lhs_indices,
    const uint32_t* rhs_indices,
    Element* C,
    int m, int n, int k, int l,
    bool broadcast_b,
    auto group_size,
    auto&& launch_kernel) {
  // Define shapes (dynamic).
  auto shape_MNKL = make_shape(m, n, k, l); // (M,N,K,L)

  // Define layouts (mixed).
  auto dA = make_stride(k, Int<1>{}, m * k);  // (dM,dK,dL)
  auto dB = make_matrix_stride<KMajor>(n, k); // (dN,dK,dL)
  auto dC = make_stride(n, Int<1>{}, m * n);  // (dM,dN,dL)
  auto S_layout = make_scales_layout<KMajor>(n, k, l, group_size);

  // Handle broadcasting.
  if (broadcast_b) {
    get<2>(dB) = 0;
    get<2>(stride(S_layout)) = 0;
  }

  // Define CTA tile size (static).
  auto cta_tiler = make_cta_tiler<TileM, SM80>(group_size);

  // Define MMA.
  auto mma = make_tiled_mma<SM80, Element>(cta_tiler);
  auto num_threads = size(mma);

  // Shared memory size.
  auto [sA_layout, sB_layout] = make_smem_layouts<KMajor>(cta_tiler);
  size_t smem_bytes = sizeof(SharedStorage<Element, decltype(sA_layout), decltype(sB_layout)>);

  auto* kernel = &qmm_naive_kernel<
      KMajor, HasKResidue, SM80,
      Element, Quant, Scale,
      decltype(shape_MNKL),
      decltype(cta_tiler),
      decltype(dA),
      decltype(dB),
      decltype(S_layout),
      decltype(dC),
      decltype(mma)>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

  dim3 num_blocks{uint32_t(ceil_div(m, size<0>(cta_tiler))),
                  uint32_t(ceil_div(n, size<1>(cta_tiler))),
                  uint32_t(l)};
  dim3 block_dims{uint32_t(num_threads)};
  void* args[] = {
      &shape_MNKL,
      &cta_tiler,
      &A, &dA,
      &B, &dB,
      &S, &Z, &S_layout,
      &lhs_indices, &rhs_indices,
      &C, &dC,
      &mma};
  launch_kernel(reinterpret_cast<void*>(kernel), num_blocks, block_dims, smem_bytes, args);
}

} // namespace cutlass_gemm

// clang-format on

namespace mlx::core {

template <int TileM, bool KMajor, bool HasKResidue, bool SM80>
void qmm_naive_impl(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& lhs_indices,
    const std::optional<array>& rhs_indices,
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
  bool broadcast_b = (w.ndim() <= 2) || (w.size() != w.data_size());

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
          if (lhs_indices) {
            encoder.set_input_array(*lhs_indices);
          }
          if (rhs_indices) {
            encoder.set_input_array(*rhs_indices);
          }
          encoder.set_output_array(out);
          cutlass_gemm::qmm_naive<TileM, KMajor, HasKResidue, SM80>(
              gpu_ptr<Element>(x),
              gpu_ptr<Quant>(w),
              gpu_ptr<Scale>(scales),
              biases ? gpu_ptr<Element>(*biases) : nullptr,
              lhs_indices ? gpu_ptr<uint32_t>(*lhs_indices) : nullptr,
              rhs_indices ? gpu_ptr<uint32_t>(*rhs_indices) : nullptr,
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
                  size_t smem_bytes,
                  void** args) {
                encoder.add_kernel_node_raw(
                    kernel, num_blocks, block_dims, {}, smem_bytes, args);
              });
        });
  });
}

// clang-format off
template void qmm_naive_impl<@TileM@, @KMajor@, @HasKResidue@, @SM80@>(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& lhs_indices,
    const std::optional<array>& rhs_indices,
    array& out,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder);
// clang-format on

} // namespace mlx::core
