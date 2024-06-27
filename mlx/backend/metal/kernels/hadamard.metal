#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/hadamard.h"

// clang-format off
#define instantiate_hadamard(name, type, N) \
  instantiate_kernel("hadamard_" #N "_" #name, \
                     hadamard, type, N)

#define instantiate_hadamard_types(N) \
  instantiate_hadamard(float32, float, N) \
  instantiate_hadamard(float16, half, N) \
  instantiate_hadamard(bfloat16, bfloat16_t, N)

instantiate_hadamard_types(2)
instantiate_hadamard_types(4)
instantiate_hadamard_types(8)
instantiate_hadamard_types(16)
instantiate_hadamard_types(32)
instantiate_hadamard_types(64)
instantiate_hadamard_types(128)
instantiate_hadamard_types(256)
instantiate_hadamard_types(512)
instantiate_hadamard_types(1024)
instantiate_hadamard_types(2048)
instantiate_hadamard_types(4096)
instantiate_hadamard_types(8192)

instantiate_hadamard(float16, half, 16384)
instantiate_hadamard(bfloat16, bfloat16_t, 16384) // clang-format on