// Copyright © 2024 Apple Inc.

#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/steel/gemm/mma.h"

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/steel/conv/conv.h"
#include "mlx/backend/metal/kernels/steel/conv/params.h"
#include "mlx/backend/metal/kernels/steel/utils.h"
#include "mlx/backend/metal/kernels/steel/conv/kernels/steel_conv_general.h"

using namespace metal;
using namespace mlx::steel;

#define instantiate_implicit_conv_2d(name, itype, bm, bn, bk, wm, wn)         \
  template                                                                    \
      [[host_name("implicit_gemm_conv_2d_general_" #name "_bm" #bm "_bn" #bn  \
                  "_bk" #bk "_wm" #wm "_wn" #wn)]] [[kernel]] void            \
      implicit_gemm_conv_2d_general<itype, bm, bn, bk, wm, wn>(               \
          const device itype* A [[buffer(0)]],                                \
          const device itype* B [[buffer(1)]],                                \
          device itype* C [[buffer(2)]],                                      \
          const constant MLXConvParams<2>* params [[buffer(3)]],              \
          const constant ImplicitGemmConv2DParams* gemm_params [[buffer(4)]], \
          const constant Conv2DGeneralJumpParams* jump_params [[buffer(5)]],  \
          const constant Conv2DGeneralBaseInfo* base_h [[buffer(6)]],         \
          const constant Conv2DGeneralBaseInfo* base_w [[buffer(7)]],         \
          uint3 tid [[threadgroup_position_in_grid]],                         \
          uint3 lid [[thread_position_in_threadgroup]],                       \
          uint simd_gid [[simdgroup_index_in_threadgroup]],                   \
          uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_implicit_2d_filter(name, itype, bm, bn, bk, wm, wn) \
  instantiate_implicit_conv_2d(name, itype, bm, bn, bk, wm, wn)

#define instantiate_implicit_2d_blocks(name, itype)               \
    instantiate_implicit_2d_filter(name, itype, 32,  8, 16, 4, 1) \
    instantiate_implicit_2d_filter(name, itype, 64,  8, 16, 4, 1) \
    instantiate_implicit_2d_filter(name, itype, 32, 32, 16, 2, 2) \
    instantiate_implicit_2d_filter(name, itype, 32, 64, 16, 2, 2) \
    instantiate_implicit_2d_filter(name, itype, 64, 32, 16, 2, 2) \
    instantiate_implicit_2d_filter(name, itype, 64, 64, 16, 2, 2)

instantiate_implicit_2d_blocks(float32, float);
instantiate_implicit_2d_blocks(float16, half);
instantiate_implicit_2d_blocks(bfloat16, bfloat16_t); // clang-format on
