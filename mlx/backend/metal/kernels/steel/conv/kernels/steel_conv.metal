// Copyright © 2024 Apple Inc.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/steel/gemm/mma.h"

#include "mlx/backend/metal/kernels/steel/conv/conv.h"
#include "mlx/backend/metal/kernels/steel/conv/params.h"
#include "mlx/backend/metal/kernels/bf16.h"

using namespace metal;

template <typename T,
          int BM,
          int BN,
          int BK,
          int WM,
          int WN,
          int N_CHANNELS = 0,
          bool SMALL_FILTER = false>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void implicit_gemm_conv_2d(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    device T* C [[buffer(2)]],
    const constant MLXConvParams<2>* params [[buffer(3)]],
    const constant ImplicitGemmConv2DParams* gemm_params [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {


  using namespace mlx::steel;

  (void)lid;
  
  constexpr bool transpose_a = false;
  constexpr bool transpose_b = true;
  constexpr short tgp_padding_a = 16 / sizeof(T);
  constexpr short tgp_padding_b = 16 / sizeof(T);

  constexpr short shape_a_cols = (transpose_a ? BM : BK) + tgp_padding_a;
  constexpr short shape_b_cols = (transpose_b ? BK : BN) + tgp_padding_b;
  constexpr short shape_a_rows = (transpose_a ? BK : BM);
  constexpr short shape_b_rows = (transpose_b ? BN : BK);
  constexpr short tgp_mem_size_a = shape_a_cols * shape_a_rows;
  constexpr short tgp_mem_size_b = shape_b_cols * shape_b_rows;

  constexpr short tgp_size = WM * WN * 32;
  
  // Input loader 

  using loader_a_t = typename metal::conditional_t<
      // Check for small channel specialization
      N_CHANNELS != 0 && N_CHANNELS <= 4,

        // Go to small channel specialization
        Conv2DInputBlockLoaderSmallChannels<
            T, BM, BN, BK, tgp_size, N_CHANNELS, tgp_padding_a>,

        // Else go to general loader 
        typename metal::conditional_t< 
            // Check if filter size is small enough
            SMALL_FILTER,

              // Go to small filter specialization
              Conv2DInputBlockLoaderSmallFilter<
                  T, BM, BN, BK, tgp_size, tgp_padding_a>,

              // Else go to large filter generalization 
              Conv2DInputBlockLoaderLargeFilter<
                  T, BM, BN, BK, tgp_size, tgp_padding_a>
        >
  >;


  // Weight loader
  using loader_b_t = typename metal::conditional_t<
      // Check for small channel specialization
      N_CHANNELS != 0 && N_CHANNELS <= 4,

        // Go to small channel specialization
        Conv2DWeightBlockLoaderSmallChannels<
            T, BM, BN, BK, tgp_size, N_CHANNELS, tgp_padding_b>,

        // Else go to general loader 
        Conv2DWeightBlockLoader<T, BM, BN, BK, tgp_size, tgp_padding_b>
  >;
  
  using mma_t = BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      transpose_a,
      transpose_b,
      shape_a_cols,
      shape_b_cols>;
    
  threadgroup T As[tgp_mem_size_a];
  threadgroup T Bs[tgp_mem_size_b];

  const int tid_y = ((tid.y) << gemm_params->swizzle_log) +
    ((tid.x) & ((1 << gemm_params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> gemm_params->swizzle_log;

  if (gemm_params->tiles_n <= tid_x || gemm_params->tiles_m <= tid_y) {
    return;
  }

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const int K = gemm_params->K;
  const int N = gemm_params->N;

  B += c_col * K;
  C += c_row * N + c_col;

  const int2 offsets_a(0, c_row);
  const int2 offsets_b(0, c_col);

  // Prepare threadgroup loading operations
  loader_a_t loader_a(A, As, offsets_a, params, gemm_params, simd_gid, simd_lid);
  loader_b_t loader_b(B, Bs, offsets_b, params, gemm_params, simd_gid, simd_lid);

  // Prepare threadgroup mma operation
  mma_t mma_op(simd_gid, simd_lid);

  int gemm_k_iterations = gemm_params->gemm_k_iterations;
  for (int k = 0; k < gemm_k_iterations; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Load elements into threadgroup
    loader_a.load_unsafe();
    loader_b.load_unsafe();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Multiply and accumulate threadgroup elements
    mma_op.mma(As, Bs);

    // Prepare for next iteration
    loader_a.next();
    loader_b.next();
  }

  threadgroup_barrier(mem_flags::mem_none);

  // Store results to device memory
  short tgp_bm = min(BM, gemm_params->M - c_row);
  short tgp_bn = min(BN, gemm_params->N - c_col);
  mma_op.store_result_safe(C, N, short2(tgp_bn, tgp_bm));

}

#define instantiate_implicit_conv_2d(name, itype, bm, bn, bk, wm, wn, channel_name, n_channels, filter_name, small_filter) \
  template [[host_name("implicit_gemm_conv_2d_" #name "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn "_channel_" #channel_name "_filter_" #filter_name)]] \
  [[kernel]] void implicit_gemm_conv_2d<itype, bm, bn, bk, wm, wn, n_channels, small_filter>( \
      const device itype* A [[buffer(0)]], \
      const device itype* B [[buffer(1)]], \
      device itype* C [[buffer(2)]], \
      const constant MLXConvParams<2>* params [[buffer(3)]], \
      const constant ImplicitGemmConv2DParams* gemm_params [[buffer(4)]], \
      uint3 tid [[threadgroup_position_in_grid]], \
      uint3 lid [[thread_position_in_threadgroup]], \
      uint simd_gid [[simdgroup_index_in_threadgroup]], \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_implicit_2d_filter(name, itype, bm, bn, bk, wm, wn) \
    instantiate_implicit_conv_2d(name, itype, bm, bn, bk, wm, wn, l, 0, s, true) \
    instantiate_implicit_conv_2d(name, itype, bm, bn, bk, wm, wn, l, 0, l, false) \
    instantiate_implicit_conv_2d(name, itype, bm, bn, bk, wm, wn, 1, 1, l, false) \
    instantiate_implicit_conv_2d(name, itype, bm, bn, bk, wm, wn, 2, 2, l, false) \
    instantiate_implicit_conv_2d(name, itype, bm, bn, bk, wm, wn, 3, 3, l, false) \
    instantiate_implicit_conv_2d(name, itype, bm, bn, bk, wm, wn, 4, 4, l, false)

#define instantiate_implicit_2d_blocks(name, itype) \
    instantiate_implicit_2d_filter(name, itype, 32,  8, 16, 4, 1) \
    instantiate_implicit_2d_filter(name, itype, 64,  8, 16, 4, 1) \
    instantiate_implicit_2d_filter(name, itype, 32, 32, 16, 2, 2) \
    instantiate_implicit_2d_filter(name, itype, 32, 64, 16, 2, 2) \
    instantiate_implicit_2d_filter(name, itype, 64, 32, 16, 2, 2) \
    instantiate_implicit_2d_filter(name, itype, 64, 64, 16, 2, 2)

instantiate_implicit_2d_blocks(float32, float);
instantiate_implicit_2d_blocks(float16, half);
instantiate_implicit_2d_blocks(bfloat16, bfloat16_t);