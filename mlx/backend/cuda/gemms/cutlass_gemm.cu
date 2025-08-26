// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"

#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#include <iostream>

namespace mlx::core::cu {

namespace {

using namespace cute;
using bf16 = cute::bfloat16_t;

template <typename Kernel>
void configure_matmul(Kernel kernel, int smem_size) {
  static bool initialized = false;
  if (!initialized) {
    initialized = true;
    cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  }
}

template <bool transpose, typename Tiler>
constexpr int get_feature_size(Tiler smem) {
  int feature_size = (transpose) ? size<0>(smem) : size<1>(smem);
  return (feature_size >= 64) ? 64 : feature_size;
}

constexpr int constexpr_log2(int x) {
  return (x > 0) ? 1 + constexpr_log2(x >> 1) : -1;
}

template <int feature_size, int itemsize, int copy_bits>
constexpr int get_swizzle_bits() {
  constexpr int swizzle_bits =
      constexpr_log2(feature_size * itemsize / copy_bits);
  return (swizzle_bits > 3) ? 3 : swizzle_bits;
}

template <int itemsize, bool transpose, int copy_bits, typename Tiler>
constexpr auto make_smem_layout(Tiler smem) {
  constexpr int feature_size = get_feature_size<transpose>(smem);
  constexpr int swizzle_bits =
      get_swizzle_bits<feature_size, itemsize, copy_bits>();

  using F = Int<feature_size>;
  using BaseLayout = std::conditional_t<
      transpose,
      Layout<cute::Shape<F, _8>, cute::Stride<_1, F>>,
      Layout<cute::Shape<_8, F>, cute::Stride<F, _1>>>;

  auto swizzled =
      make_composed_layout(Swizzle<swizzle_bits, 3, 3>{}, 0, BaseLayout{});

  return tile_to_shape(swizzled, smem);
}

template <int itemsize, bool transpose, int copy_bits, typename Tiler>
constexpr auto make_result_smem_layout(Tiler smem) {
  constexpr int feature_size = get_feature_size<transpose>(smem);
  constexpr int swizzle_bits =
      get_swizzle_bits<feature_size, itemsize, copy_bits>();

  using F = Int<feature_size>;
  using BaseLayout = std::conditional_t<
      transpose,
      Layout<cute::Shape<F, _8>, cute::Stride<_1, F>>,
      Layout<cute::Shape<_8, F>, cute::Stride<F, _1>>>;

  auto swizzled = make_composed_layout(
      Swizzle<transpose ? 0 : swizzle_bits, 3, 4>{}, 0, BaseLayout{});

  return tile_to_shape(swizzled, smem);
}

template <
    int num_threads,
    int itemsize,
    bool transpose,
    int copy_bits,
    typename Copier,
    typename Tiler>
constexpr auto make_tiled_copy(Copier copy_op, Tiler smem) {
  constexpr int num_elements = copy_bits / itemsize;
  constexpr int feature_size = transpose ? size<0>(smem) : size<1>(smem);
  constexpr int copies_per_feature = feature_size / num_elements;

  using E = Int<num_elements>;
  using C = Int<copies_per_feature>;
  using R = Int<num_threads / copies_per_feature>;

  using ThreadLayout = std::conditional_t<
      transpose,
      Layout<cute::Shape<C, R>, cute::Stride<_1, C>>,
      Layout<cute::Shape<R, C>, cute::Stride<C, _1>>>;
  using ValueLayout = std::conditional_t<
      transpose,
      Layout<cute::Shape<E, _1>>,
      Layout<cute::Shape<_1, E>>>;

  return make_tiled_copy(copy_op, ThreadLayout{}, ValueLayout{});
}

template <int rasterization_factor>
__device__ inline int2 raster_tile(int x, int y) {
  return {
      x / rasterization_factor,
      (x % rasterization_factor) + y * rasterization_factor};
}

template <
    typename T,
    typename SLayoutA,
    typename SLayoutB,
    typename SLayoutC,
    typename CopyA,
    typename CopyB,
    typename CopyC,
    typename MMA,
    int rasterization_factor>
__global__ static __launch_bounds__(decltype(size(MMA{}))::value) void matmul_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    SLayoutA SA,
    SLayoutB SB,
    SLayoutC SC,
    CopyA copy_a,
    CopyB copy_b,
    CopyC copy_c,
    MMA mma,
    int M,
    int N,
    int K) {
  constexpr auto BM = size<0>(SA);
  constexpr auto BN = size<0>(SB);
  constexpr auto BK = size<1>(SA);
  constexpr auto PIPE = size<2>(SA);

  const int2 tile = raster_tile<rasterization_factor>(blockIdx.x, blockIdx.y);
  const int blocks_m = ceil_div(M, BM);
  const int blocks_n = ceil_div(N, BN);

  // Exit early if the tile is OOB
  if (tile.x >= blocks_m || tile.y >= blocks_n) {
    return;
  }

  // Make the full tensors
  Tensor full_A =
      make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, _1{}));
  Tensor full_B =
      make_tensor(make_gmem_ptr(B), make_shape(N, K), make_stride(K, _1{}));
  Tensor full_C =
      make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, _1{}));

  // Partition the tensors into tiles and select the ones for this threadblock
  Tensor local_A =
      local_tile(full_A, make_shape(BM, BK), make_coord(tile.x, _));
  Tensor local_B =
      local_tile(full_B, make_shape(BN, BK), make_coord(tile.y, _));
  Tensor local_C =
      local_tile(full_C, make_shape(BM, BN), make_coord(tile.x, tile.y));

  // Make shared memory tensors
  extern __shared__ char shared_memory[];
  T* shared_A_ptr = reinterpret_cast<T*>(shared_memory);
  T* shared_B_ptr =
      reinterpret_cast<T*>(shared_memory + cosize(SA) * sizeof(T));
  T* shared_C_ptr = reinterpret_cast<T*>(shared_memory);
  Tensor shared_A = make_tensor(make_smem_ptr(shared_A_ptr), SA);
  Tensor shared_B = make_tensor(make_smem_ptr(shared_B_ptr), SB);
  Tensor shared_C = make_tensor(make_smem_ptr(shared_C_ptr), SC);

  // Get the copies that correspond to this thread
  auto thread_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor local_A_src = thread_copy_a.partition_S(local_A);
  Tensor local_A_dst = thread_copy_a.partition_D(shared_A);
  auto thread_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor local_B_src = thread_copy_a.partition_S(local_B);
  Tensor local_B_dst = thread_copy_a.partition_D(shared_B);
  auto thread_copy_c = copy_c.get_slice(threadIdx.x);
  Tensor local_C_src = thread_copy_c.partition_S(shared_C);
  Tensor local_C_dst = thread_copy_c.partition_D(local_C);

  // Start fetches
  int k_tile_count = size<2>(local_A);
  int k_tile_next = 0;
  CUTE_UNROLL
  for (int k = 0; k < PIPE - 1; k++) {
    copy(copy_a, local_A_src(_, _, _, k_tile_next), local_A_dst(_, _, _, k));
    copy(copy_b, local_B_src(_, _, _, k_tile_next), local_B_dst(_, _, _, k));
    cp_async_fence();
    k_tile_count--;
    k_tile_next += (k_tile_count > 0);
  }

  // Get the MMA that corresponds to this thread and allocate registers
  auto thread_mma = mma.get_slice(threadIdx.x);
  Tensor mma_shared_A = thread_mma.partition_A(shared_A);
  Tensor mma_shared_B = thread_mma.partition_B(shared_B);
  Tensor mma_shared_C = thread_mma.partition_C(shared_C);
  Tensor mma_global_C = thread_mma.partition_C(local_C);
  Tensor mma_frag_A = mma.make_fragment_A(mma_shared_A(_, _, _, 0));
  Tensor mma_frag_B = mma.make_fragment_B(mma_shared_B(_, _, _, 0));
  Tensor mma_frag_C = mma.make_fragment_C(mma_global_C);
  clear(mma_frag_C);

  // Make shared to register copies
  Copy_Atom<SM75_U32x4_LDSM_N, bf16> s2r_atom_a;
  Copy_Atom<SM75_U32x4_LDSM_N, bf16> s2r_atom_b;
  auto s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
  auto s2r_thread_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  auto s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
  auto s2r_thread_copy_b = s2r_copy_b.get_slice(threadIdx.x);
  Tensor mma_A_src = s2r_thread_copy_a.partition_S(shared_A);
  Tensor mma_A_dst = s2r_thread_copy_a.retile_D(mma_frag_A);
  Tensor mma_B_src = s2r_thread_copy_b.partition_S(shared_B);
  Tensor mma_B_dst = s2r_thread_copy_b.retile_D(mma_frag_B);

  constexpr auto RPIPE = size<2>(mma_shared_A);
  int smem_read = 0;
  int smem_write = PIPE - 1;
  Tensor mma_A_src_p = mma_A_src(_, _, _, smem_read);
  Tensor mma_B_src_p = mma_B_src(_, _, _, smem_read);

  // Start the register pipeline
  if constexpr (RPIPE > 1) {
    cp_async_wait<PIPE - 2>();
    __syncthreads();
    copy(s2r_copy_a, mma_A_src_p(_, _, Int<0>{}), mma_A_dst(_, _, Int<0>{}));
    copy(s2r_copy_b, mma_B_src_p(_, _, Int<0>{}), mma_B_dst(_, _, Int<0>{}));
  }

  CUTE_NO_UNROLL
  while (k_tile_count > -(PIPE - 1)) {
    CUTE_UNROLL
    for (int k_block = 0; k_block < RPIPE; k_block++) {
      if (k_block == RPIPE - 1) {
        mma_A_src_p = mma_A_src(_, _, _, smem_read);
        mma_B_src_p = mma_B_src(_, _, _, smem_read);
        cp_async_wait<PIPE - 2>();
        __syncthreads();
      }

      // Load the next register tile
      auto k_block_next = (k_block + 1) % RPIPE;
      copy(
          s2r_copy_a,
          mma_A_src_p(_, _, k_block_next),
          mma_A_dst(_, _, k_block_next));
      copy(
          s2r_copy_b,
          mma_B_src_p(_, _, k_block_next),
          mma_B_dst(_, _, k_block_next));

      if (k_block == 0) {
        copy(
            copy_a,
            local_A_src(_, _, _, k_tile_next),
            local_A_dst(_, _, _, smem_write));
        copy(
            copy_b,
            local_B_src(_, _, _, k_tile_next),
            local_B_dst(_, _, _, smem_write));
        cp_async_fence();
        k_tile_count--;
        k_tile_next += (k_tile_count > 0);
        smem_write = smem_read;
        smem_read = (smem_read == PIPE - 1) ? 0 : (smem_read + 1);
      }

      gemm(
          mma,
          mma_frag_A(_, _, k_block),
          mma_frag_B(_, _, k_block),
          mma_frag_C);
    }
  }

  copy(mma_frag_C, mma_shared_C);
  __syncthreads();
  copy(copy_c, local_C_src, local_C_dst);

  // if (threadIdx.x == 0) {
  //   print("fC: "); print(mma_frag_C); print("\n");
  //   print("sC: "); print(mma_shared_C); print("\n");
  //   print("dC: "); print(local_C_dst); print("\n");
  //
  //   print(s2r_atom_a); print("\n");
  // }
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

      // Tile definitions
      auto BM = Int<128>{};
      auto BN = Int<128>{};
      auto BK = Int<64>{};
      auto BP = Int<3>{};
      auto GM = Int<8>{};

      // Thread definitions
      using TM = Int<2>;
      using TN = Int<2>;
      using TK = Int<1>;
      constexpr int num_threads = TM::value * TN::value * 32;

      auto SA = make_smem_layout<16, false, 128>(make_shape(BM, BK, BP));
      auto SB = make_smem_layout<16, false, 128>(make_shape(BN, BK, BP));
      auto SC = make_result_smem_layout<16, false, 128>(make_shape(BM, BN));

      constexpr auto smem_size = (cosize(SA) + cosize(SB)) * sizeof(bf16);

      auto async_copy_op =
          Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, bf16>{};
      auto tiled_copy_a = make_tiled_copy<num_threads, 16, false, 128>(
          async_copy_op, make_shape(BM, BK));
      auto tiled_copy_b = make_tiled_copy<num_threads, 16, false, 128>(
          async_copy_op, make_shape(BN, BK));

      auto sync_copy_op = Copy_Atom<UniversalCopy<uint128_t>, bf16>{};
      auto tiled_copy_c = make_tiled_copy<num_threads, 16, false, 128>(
          sync_copy_op, make_shape(BM, BN));

      auto mma_op = SM80_16x8x16_F32BF16BF16F32_TN{};
      auto tiled_mma = make_tiled_mma(
          mma_op, Layout<cute::Shape<TM, TN, TK>>{}, Tile<_32, _32, _16>{});

      auto kernel = matmul_kernel<
          bf16,
          decltype(SA),
          decltype(SB),
          decltype(SC),
          decltype(tiled_copy_a),
          decltype(tiled_copy_b),
          decltype(tiled_copy_c),
          decltype(tiled_mma),
          GM.value>;
      configure_matmul(kernel, smem_size);

      dim3 block(size(tiled_mma));
      dim3 grid(
          size(ceil_div(M, BM) * GM), size(ceil_div(ceil_div(N, BN), GM)));

      enc.add_kernel_node(
          kernel,
          grid,
          block,
          smem_size,
          a.data<bf16>(),
          b.data<bf16>(),
          out.data<bf16>(),
          SA,
          SB,
          SC,
          tiled_copy_a,
          tiled_copy_b,
          tiled_copy_c,
          tiled_mma,
          M,
          N,
          K);
    } else {
      throw std::runtime_error("Only bfloat16 supported");
    }
  });
}

} // namespace mlx::core::cu
