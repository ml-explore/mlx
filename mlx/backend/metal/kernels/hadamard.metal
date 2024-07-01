#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/hadamard.h"

// clang-format off
#define instantiate_hadamard(name, type, N, max_radix) \
  instantiate_kernel("hadamard_" #N "_" #name, \
                     hadamard, type, N)

#define instantiate_hadamard_types(N, max_radix) \
  instantiate_hadamard(float32, float, N, max_radix) \
  instantiate_hadamard(float16, half, N, max_radix) \
  instantiate_hadamard(bfloat16, bfloat16_t, N, max_radix)

instantiate_hadamard_types(2, 2)
instantiate_hadamard_types(4, 4)
instantiate_hadamard_types(8, 8)
instantiate_hadamard_types(16, 16)
instantiate_hadamard_types(32, 16)
instantiate_hadamard_types(64, 16)
instantiate_hadamard_types(128, 16)
instantiate_hadamard_types(256, 16)
instantiate_hadamard_types(512, 16)
instantiate_hadamard_types(1024, 16)
instantiate_hadamard_types(2048, 16)
instantiate_hadamard_types(4096, 16)
instantiate_hadamard_types(8192, 16)

instantiate_hadamard(float16, half, 16384, 16)
instantiate_hadamard(bfloat16, bfloat16_t, 16384, 16) // clang-format on