#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/gated_delta_update_impl.h"

using namespace metal;

#define instantiate_gated_delta_update(in_type, st_type, dk, dv, hk, hv)  \
  instantiate_kernel(                                                        \
      "gated_delta_step_" #in_type "_" #st_type "_" #dk "_" #dv "_" #hk "_" #hv, \
      gated_delta_step,                                                      \
      in_type,                                                               \
      st_type,                                                               \
      dk,                                                                    \
      dv,                                                                    \
      hk,                                                                    \
      hv)

#define instantiate_gated_delta_update_dims(in_type, st_type) \
  instantiate_gated_delta_update(in_type, st_type, 64, 64, 4, 4)

instantiate_gated_delta_update_dims(float, float)
