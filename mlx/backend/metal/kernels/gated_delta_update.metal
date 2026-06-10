// mlx/backend/metal/kernels/gated_delta_update.metal

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/gated_delta_update_impl.h"

using namespace metal;

#define instantiate_gated_delta_update(type, dk, dv)          \
  instantiate_kernel(                                          \
      "gated_delta_update_fwd_" #type "_" #dk "_" #dv,        \
      gated_delta_update_fwd,                                  \
      type,                                                    \
      dk,                                                      \
      dv)

#define instantiate_gated_delta_update_dims(type) \
  instantiate_gated_delta_update(type, 64,  64)  

instantiate_gated_delta_update_dims(float)
