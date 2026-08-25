#include "mlx/backend/metal/kernels/gated_delta_update_nax_vjp.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

#define instantiate_gated_delta_vjp_fused_nax(in_type, dk, dv, hk, hv, c)   \
  instantiate_kernel(                                                       \
      "gated_delta_vjp_fused_nax_" #in_type "_" #dk "_" #dv "_" #hk "_" #hv \
      "_" #c,                                                               \
      gated_delta_vjp_fused_nax,                                            \
      in_type,                                                              \
      dk,                                                                   \
      dv,                                                                   \
      hk,                                                                   \
      hv,                                                                   \
      c)

#define instantiate_gated_delta_dims(in_type, dk, dv, hk, hv) \
  instantiate_gated_delta_vjp_fused_nax(in_type, dk, dv, hk, hv, 16)

#define instantiate_gated_delta(in_type)                          \
  instantiate_gated_delta_dims(in_type, 128, 128, 24, 24)         \
      instantiate_gated_delta_dims(in_type, 128, 128, 32, 32)     \
          instantiate_gated_delta_dims(in_type, 128, 128, 16, 32) \
              instantiate_gated_delta_dims(in_type, 128, 128, 16, 48)

instantiate_gated_delta(float);
instantiate_gated_delta(bfloat16_t);

#define instantiate_gated_delta_dgamma(in_type, c) \
  instantiate_kernel(                              \
      "gated_delta_dgamma_to_dg_" #in_type "_" #c, \
      gated_delta_dgamma_to_dg,                    \
      in_type,                                     \
      c)

instantiate_gated_delta_dgamma(float, 16);
instantiate_gated_delta_dgamma(bfloat16_t, 16);
