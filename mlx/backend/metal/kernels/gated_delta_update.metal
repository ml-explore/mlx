#include "mlx/backend/metal/kernels/gated_delta_update_impl.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

#define instantiate_gated_delta_update_seq(in_type, st_type, dk, dv, hk, hv) \
  instantiate_kernel(                                                        \
      "seq_gated_delta_" #in_type "_" #st_type "_" #dk "_" #dv "_" #hk       \
      "_" #hv,                                                               \
      gated_delta_seq,                                                       \
      in_type,                                                               \
      st_type,                                                               \
      dk,                                                                    \
      dv,                                                                    \
      hk,                                                                    \
      hv)

#define instantiate_gated_delta_update_seq_dims(in_type, st_type)    \
  instantiate_gated_delta_update_seq(in_type, st_type, 64, 64, 4, 4) \
      instantiate_gated_delta_update_seq(in_type, st_type, 64, 64, 8, 8)

#define instantiate_gated_delta_update_chunk(                            \
    in_type, st_type, dk, dv, hk, hv, c)                                 \
  instantiate_kernel(                                                    \
      "chunk_gated_delta_" #in_type "_" #st_type "_" #dk "_" #dv "_" #hk \
      "_" #hv "_" #c,                                                    \
      gated_delta_chunk,                                                 \
      in_type,                                                           \
      st_type,                                                           \
      dk,                                                                \
      dv,                                                                \
      hk,                                                                \
      hv,                                                                \
      c)

#define instantiate_gated_delta_update_chunk_dims(in_type, st_type)            \
  instantiate_gated_delta_update_chunk(in_type, st_type, 64, 64, 4, 4, 32)     \
      instantiate_gated_delta_update_chunk(in_type, st_type, 64, 64, 4, 4, 16) \
          instantiate_gated_delta_update_chunk(                                \
              in_type, st_type, 64, 64, 4, 4, 8)                               \
              instantiate_gated_delta_update_chunk(                            \
                  in_type, st_type, 64, 64, 8, 8, 8)

#define instantiate_make_wy(in_type, dk, dv, hk, hv, c)           \
  instantiate_kernel(                                             \
      "make_wy_" #in_type "_" #dk "_" #dv "_" #hk "_" #hv "_" #c, \
      make_wy,                                                    \
      in_type,                                                    \
      dk,                                                         \
      dv,                                                         \
      hk,                                                         \
      hv,                                                         \
      c)

#define instantiate_make_wy_dims(in_type)               \
  instantiate_make_wy(in_type, 64, 64, 4, 4, 32)        \
      instantiate_make_wy(in_type, 64, 64, 4, 4, 16)    \
          instantiate_make_wy(in_type, 64, 64, 4, 4, 8) \
              instantiate_make_wy(in_type, 64, 64, 8, 8, 8)

#define instantiate_gated_delta_update_fused_chunk(                            \
    in_type, st_type, dk, dv, hk, hv, c)                                       \
  instantiate_kernel(                                                          \
      "gated_delta_fused_chunk_" #in_type "_" #st_type "_" #dk "_" #dv "_" #hk \
      "_" #hv "_" #c,                                                          \
      gated_delta_fused_chunk,                                                 \
      in_type,                                                                 \
      st_type,                                                                 \
      dk,                                                                      \
      dv,                                                                      \
      hk,                                                                      \
      hv,                                                                      \
      c)

#define instantiate_gated_delta_update_fused_chunk_dims(in_type, st_type) \
  instantiate_gated_delta_update_fused_chunk(                             \
      in_type, st_type, 64, 64, 4, 4, 32)                                 \
      instantiate_gated_delta_update_fused_chunk(                         \
          in_type, st_type, 64, 64, 4, 4, 16)                             \
          instantiate_gated_delta_update_fused_chunk(                     \
              in_type, st_type, 64, 64, 4, 4, 8)                          \
              instantiate_gated_delta_update_fused_chunk(                 \
                  in_type, st_type, 64, 64, 8, 8, 8)

instantiate_gated_delta_update_seq_dims(float, float)
    instantiate_gated_delta_update_chunk_dims(float, float)
        instantiate_make_wy_dims(float)
            instantiate_gated_delta_update_fused_chunk_dims(float, float)