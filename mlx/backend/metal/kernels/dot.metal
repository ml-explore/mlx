#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/dot.h"

#define instantiate_dot_product_kernel(                           \
    name, itype, items_per_thread, tg_size, simd_groups)          \
  instantiate_kernel(                                             \
      "dot_product_" #name "_it" #items_per_thread "_tg" #tg_size \
      "_sg" #simd_groups,                                         \
      dot_product,                                                \
      itype,                                                      \
      items_per_thread,                                           \
      tg_size,                                                    \
      simd_groups)

instantiate_dot_product_kernel(float32, float, 32, 512, 16);
instantiate_dot_product_kernel(float16, half, 32, 512, 16);
instantiate_dot_product_kernel(bfloat16, bfloat16_t, 32, 512, 16);
