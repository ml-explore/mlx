// Copyright © 2024 Apple Inc.

// clang-format off
// #include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/attn/attn.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"

#define instantiate_attn(tname, dtype, bq, bk, bd, wm, wn) \
  template [[host_name("steel_attention_" #tname "_bq" #bq "_bk" #bk "_bd" #bd "_wm" #wm "_wn" #wn)]] \
  [[kernel]] void attention<dtype, bq, bk, bd, wm, wn, float>( \
      const device dtype* Q [[buffer(0)]], \
      const device dtype* K [[buffer(1)]], \
      const device dtype* V [[buffer(2)]], \
      device dtype* O [[buffer(3)]],\
      const constant AttnParams* params [[buffer(4)]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simd_group_id [[simdgroup_index_in_threadgroup]], \
      uint3 tid [[threadgroup_position_in_grid]], \
      uint3 lid [[thread_position_in_threadgroup]]);

#define instantiate_attn_shapes_helper(iname, itype) \
    instantiate_attn(iname, itype, 32, 16, 128, 4, 1) \
    instantiate_attn(iname, itype, 32, 32,  80, 4, 1) \
    instantiate_attn(iname, itype, 32, 32,  64, 4, 1)

instantiate_attn_shapes_helper(float16, half);
// instantiate_attn_shapes_helper(bfloat16, bfloat16_t);

instantiate_attn_shapes_helper(float32, float);
// clang-format on
