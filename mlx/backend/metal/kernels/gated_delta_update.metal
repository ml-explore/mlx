#include "mlx/backend/metal/kernels/gated_delta_update_impl.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

#define instantiate_gated_delta_update_seq(in_type, dk, dv, hk, hv) \
  instantiate_kernel(                                               \
      "seq_gated_delta_" #in_type "_" #dk "_" #dv "_" #hk "_" #hv,  \
      gated_delta_seq,                                              \
      in_type,                                                      \
      dk,                                                           \
      dv,                                                           \
      hk,                                                           \
      hv)

#define instantiate_gated_delta_update_fused_chunk(in_type, dk, dv, hk, hv, c) \
  instantiate_kernel(                                                          \
      "gated_delta_fused_chunk_" #in_type "_" #dk "_" #dv "_" #hk "_" #hv      \
      "_" #c,                                                                  \
      gated_delta_fused_chunk,                                                 \
      in_type,                                                                 \
      dk,                                                                      \
      dv,                                                                      \
      hk,                                                                      \
      hv,                                                                      \
      c)

#define instantiate_gated_delta_dims(in_type, dk, dv, hk, hv) \
  instantiate_gated_delta_update_seq(in_type, dk, dv, hk, hv) \
      instantiate_gated_delta_update_fused_chunk(in_type, dk, dv, hk, hv, 8)

#define instantiate_gated_delta(in_type)                                  \
  instantiate_gated_delta_dims(in_type, 128, 128, 24, 24)                 \
      instantiate_gated_delta_dims(in_type, 128, 128, 32, 32)             \
          instantiate_gated_delta_dims(in_type, 128, 128, 16, 32)         \
              instantiate_gated_delta_dims(in_type, 128, 128, 16, 48)     \
                  instantiate_gated_delta_dims(in_type, 128, 128, 16, 16) \
                      instantiate_gated_delta_dims(in_type, 128, 128, 16, 64)

instantiate_gated_delta(float);
instantiate_gated_delta(bfloat16_t);
instantiate_gated_delta(float16_t);