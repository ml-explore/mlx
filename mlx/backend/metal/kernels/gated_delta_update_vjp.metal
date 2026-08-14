#include "mlx/backend/metal/kernels/gated_delta_update_vjp_impl.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

#define instantiate_gdn_vjp_seq(in_type, dk, dv, hk, hv)               \
  instantiate_kernel(                                                  \
      "seq_gated_delta_vjp_" #in_type "_" #dk "_" #dv "_" #hk "_" #hv, \
      gated_delta_vjp_seq,                                             \
      in_type,                                                         \
      dk,                                                              \
      dv,                                                              \
      hk,                                                              \
      hv)

#define instantiate_gated_delta_vjp_dims(in_type, dk, dv, hk, hv) \
  instantiate_gdn_vjp_seq(in_type, dk, dv, hk, hv)

#define instantiate_gated_delta_vjp(in_type)                          \
  instantiate_gated_delta_vjp_dims(in_type, 128, 128, 24, 24)         \
      instantiate_gated_delta_vjp_dims(in_type, 128, 128, 32, 32)     \
          instantiate_gated_delta_vjp_dims(in_type, 128, 128, 16, 32) \
              instantiate_gated_delta_vjp_dims(in_type, 128, 128, 16, 48)

instantiate_gated_delta_vjp(float);
// instantiate_gated_delta_vjp(bfloat16_t);