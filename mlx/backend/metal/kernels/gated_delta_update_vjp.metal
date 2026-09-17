#include "mlx/backend/metal/kernels/gated_delta_update_vjp_impl.h"
#include "mlx/backend/metal/kernels/gated_delta_update.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

#define instantiate_gdn_vjp_seq(in_type, dk, dv, hk, hv, ckpt)         \
  instantiate_kernel(                                                  \
      "seq_gated_delta_vjp_" #in_type "_" #dk "_" #dv "_" #hk "_" #hv  \
      "_" #ckpt,                                                       \
      gated_delta_vjp_seq,                                             \
      in_type,                                                         \
      dk,                                                              \
      dv,                                                              \
      hk,                                                              \
      hv,                                                              \
      ckpt)

#define instantiate_gated_delta_vjp_dims(in_type, dk, dv, hk, hv) \
  instantiate_gdn_vjp_seq(in_type, dk, dv, hk, hv, 1) \
  instantiate_gdn_vjp_seq(in_type, dk, dv, hk, hv, 4) \
  instantiate_gdn_vjp_seq(in_type, dk, dv, hk, hv, 8) \
  instantiate_gdn_vjp_seq(in_type, dk, dv, hk, hv, 16)

#define instantiate_gated_delta_vjp(in_type)                          \
  instantiate_gated_delta_vjp_dims(in_type, 128, 128, 24, 24)         \
      instantiate_gated_delta_vjp_dims(in_type, 128, 128, 32, 32)     \
          instantiate_gated_delta_vjp_dims(in_type, 128, 128, 16, 32) \
          instantiate_gated_delta_vjp_dims(in_type, 128, 128, 16, 48) \
          instantiate_gated_delta_vjp_dims(in_type, 128, 128, 16, 16) \
              instantiate_gated_delta_vjp_dims(in_type, 128, 128, 16, 64)

instantiate_gated_delta_vjp(float);
instantiate_gated_delta_vjp(bfloat16_t);
instantiate_gated_delta_vjp(float16_t);