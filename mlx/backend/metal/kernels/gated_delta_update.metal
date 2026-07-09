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

#define instantiate_gated_delta_update_seq_dims(in_type, st_type)              \
  instantiate_gated_delta_update_seq(in_type, st_type, 64, 64, 4, 4)           \
      instantiate_gated_delta_update_seq(in_type, st_type, 64, 64, 8, 8)       \
          instantiate_gated_delta_update_seq(in_type, st_type, 64, 64, 16, 16) \
              instantiate_gated_delta_update_seq(                              \
                  in_type, st_type, 64, 64, 24, 24)                            \
                  instantiate_gated_delta_update_seq(                          \
                      in_type, st_type, 64, 64, 32, 32)                        \
                      instantiate_gated_delta_update_seq(                      \
                          in_type, st_type, 128, 128, 16, 16)                  \
                          instantiate_gated_delta_update_seq(                  \
                              in_type, st_type, 128, 128, 24, 24)              \
                              instantiate_gated_delta_update_seq(              \
                                  in_type, st_type, 128, 128, 32, 32)          \
                                  instantiate_gated_delta_update_seq(          \
                                      in_type, st_type, 128, 128, 16, 32)

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

#define instantiate_gated_delta_update_fused_chunk_dims(in_type, st_type)                 \
  instantiate_gated_delta_update_fused_chunk(                                             \
      in_type, st_type, 128, 128, 16, 16, 8)                                              \
      instantiate_gated_delta_update_fused_chunk(                                         \
          in_type, st_type, 128, 128, 24, 24, 8)                                          \
          instantiate_gated_delta_update_fused_chunk(                                     \
              in_type, st_type, 128, 128, 32, 32, 8)                                      \
              instantiate_gated_delta_update_fused_chunk(                                 \
                  in_type, st_type, 64, 64, 4, 4, 8)                                      \
                  instantiate_gated_delta_update_fused_chunk(                             \
                      in_type, st_type, 64, 64, 8, 8, 8)                                  \
                      instantiate_gated_delta_update_fused_chunk(                         \
                          in_type, st_type, 64, 64, 16, 16, 8)                            \
                          instantiate_gated_delta_update_fused_chunk(                     \
                              in_type, st_type, 64, 64, 24, 24, 8)                        \
                              instantiate_gated_delta_update_fused_chunk(                 \
                                  in_type, st_type, 64, 64, 32, 32, 8)                    \
                                  instantiate_gated_delta_update_fused_chunk(             \
                                      in_type, st_type, 128, 128, 16, 32, 8)              \
                                      instantiate_gated_delta_update_fused_chunk(         \
                                          in_type, st_type, 32, 32, 16, 32, 8)            \
                                          instantiate_gated_delta_update_fused_chunk(     \
                                              in_type,                                    \
                                              st_type,                                    \
                                              32,                                         \
                                              32,                                         \
                                              32,                                         \
                                              32,                                         \
                                              8)                                          \
                                              instantiate_gated_delta_update_fused_chunk( \
                                                  in_type,                                \
                                                  st_type,                                \
                                                  16,                                     \
                                                  16,                                     \
                                                  16,                                     \
                                                  32,                                     \
                                                  8)

#define instantiate_gated_delta_update_fused_nax(                            \
    in_type, st_type, dk, dv, hk, hv, c)                                     \
  instantiate_kernel(                                                        \
      "gated_delta_fused_nax_" #in_type "_" #st_type "_" #dk "_" #dv "_" #hk \
      "_" #hv "_" #c,                                                        \
      gated_delta_fused_nax,                                                 \
      in_type,                                                               \
      st_type,                                                               \
      dk,                                                                    \
      dv,                                                                    \
      hk,                                                                    \
      hv,                                                                    \
      c)

#define instantiate_gated_delta_update_fused_nax_dims(in_type, st_type)                 \
  instantiate_gated_delta_update_fused_nax(                                             \
      in_type, st_type, 128, 128, 16, 16, 16)                                           \
      instantiate_gated_delta_update_fused_nax(                                         \
          in_type, st_type, 128, 128, 24, 24, 16)                                       \
          instantiate_gated_delta_update_fused_nax(                                     \
              in_type, st_type, 128, 128, 32, 32, 16)                                   \
              instantiate_gated_delta_update_fused_nax(                                 \
                  in_type, st_type, 64, 64, 4, 4, 16)                                   \
                  instantiate_gated_delta_update_fused_nax(                             \
                      in_type, st_type, 64, 64, 8, 8, 16)                               \
                      instantiate_gated_delta_update_fused_nax(                         \
                          in_type, st_type, 64, 64, 16, 16, 16)                         \
                          instantiate_gated_delta_update_fused_nax(                     \
                              in_type, st_type, 64, 64, 24, 24, 16)                     \
                              instantiate_gated_delta_update_fused_nax(                 \
                                  in_type, st_type, 64, 64, 32, 32, 16)                 \
                                  instantiate_gated_delta_update_fused_nax(             \
                                      in_type, st_type, 128, 128, 16, 32, 16)           \
                                      instantiate_gated_delta_update_fused_nax(         \
                                          in_type,                                      \
                                          st_type,                                      \
                                          32,                                           \
                                          32,                                           \
                                          16,                                           \
                                          32,                                           \
                                          16)                                           \
                                          instantiate_gated_delta_update_fused_nax(     \
                                              in_type,                                  \
                                              st_type,                                  \
                                              32,                                       \
                                              32,                                       \
                                              32,                                       \
                                              32,                                       \
                                              16)                                       \
                                              instantiate_gated_delta_update_fused_nax( \
                                                  in_type,                              \
                                                  st_type,                              \
                                                  16,                                   \
                                                  16,                                   \
                                                  16,                                   \
                                                  32,                                   \
                                                  16)

instantiate_gated_delta_update_seq_dims(float, float)
    instantiate_gated_delta_update_fused_chunk_dims(float, float)
        instantiate_gated_delta_update_fused_nax_dims(float, float)