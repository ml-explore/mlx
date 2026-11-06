#include "mlx/backend/metal/kernels/gated_delta_update_nax_vjp.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

#define instantiate_gdu_vjp_nax(in_type, dk, dv, hk, hv, c, ckpt) \
  instantiate_kernel(                                             \
      "gated_delta_vjp_fused_nax_" #in_type "_" #dk "_" #dv       \
      "_" #hk "_" #hv "_" #c "_" #ckpt,                           \
      gated_delta_vjp_fused_nax,                                  \
      in_type,                                                    \
      dk,                                                         \
      dv,                                                         \
      hk,                                                         \
      hv,                                                         \
      c,                                                          \
      ckpt)

#define instantiate_gdu_vjp_dims(in_type, dk, dv, hk, hv)      \
  instantiate_gdu_vjp_nax(in_type, dk, dv, hk, hv, 16, 1)      \
      instantiate_gdu_vjp_nax(in_type, dk, dv, hk, hv, 16, 4)  \
          instantiate_gdu_vjp_nax(in_type, dk, dv, hk, hv, 16, 8) \
              instantiate_gdu_vjp_nax(in_type, dk, dv, hk, hv, 16, 16)

#define instantiate_gdu_vjp(in_type)                          \
  instantiate_gdu_vjp_dims(in_type, 128, 128, 24, 24)         \
      instantiate_gdu_vjp_dims(in_type, 128, 128, 32, 32)     \
          instantiate_gdu_vjp_dims(in_type, 128, 128, 16, 32) \
              instantiate_gdu_vjp_dims(in_type, 128, 128, 16, 16) \
                  instantiate_gdu_vjp_dims(in_type, 128, 128, 16, 48)

instantiate_gdu_vjp(float);
instantiate_gdu_vjp(bfloat16_t);

// Postprocessing: converts dL/dgamma to dL/dg. Not templated on ckpt.
#define instantiate_gdu_dgamma(in_type, c)         \
  instantiate_kernel(                              \
      "gated_delta_dgamma_to_dg_" #in_type "_" #c, \
      gated_delta_dgamma_to_dg,                    \
      in_type,                                     \
      c)

instantiate_gdu_dgamma(float, 16);
instantiate_gdu_dgamma(bfloat16_t, 16);
