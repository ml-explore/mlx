// Copyright © 2024 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/copy.h"

#define instantiate_copy_all(tname, itype, otype)    \
  instantiate_kernel("s_copy" #tname, copy_s, itype, otype) \
  instantiate_kernel("v_copy" #tname, copy_v, itype, otype) \
  instantiate_kernel("s2_copy" #tname, copy_s2, itype, otype) \
  instantiate_kernel("v2_copy" #tname, copy_v2, itype, otype) \
  instantiate_kernel("g1_copy" #tname, copy_g_nd1, itype, otype) \
  instantiate_kernel("g2_copy" #tname, copy_g_nd2, itype, otype) \
  instantiate_kernel("g3_copy" #tname, copy_g_nd3, itype, otype) \
  instantiate_kernel("gg1_copy" #tname, copy_gg_nd1, itype, otype) \
  instantiate_kernel("gg2_copy" #tname, copy_gg_nd2, itype, otype) \
  instantiate_kernel("gg3_copy" #tname, copy_gg_nd3, itype, otype) \
  instantiate_kernel("gn4_copy" #tname, copy_g, itype, otype, 4) \
  instantiate_kernel("ggn4_copy" #tname, copy_gg, itype, otype, 4)

#define instantiate_copy_itype(itname, itype)                \
  instantiate_copy_all(itname ##bool_, itype, bool)          \
  instantiate_copy_all(itname ##uint8, itype, uint8_t)       \
  instantiate_copy_all(itname ##uint16, itype, uint16_t)     \
  instantiate_copy_all(itname ##uint32, itype, uint32_t)     \
  instantiate_copy_all(itname ##uint64, itype, uint64_t)     \
  instantiate_copy_all(itname ##int8, itype, int8_t)         \
  instantiate_copy_all(itname ##int16, itype, int16_t)       \
  instantiate_copy_all(itname ##int32, itype, int32_t)       \
  instantiate_copy_all(itname ##int64, itype, int64_t)       \
  instantiate_copy_all(itname ##float16, itype, half)        \
  instantiate_copy_all(itname ##float32, itype, float)       \
  instantiate_copy_all(itname ##bfloat16, itype, bfloat16_t) \
  instantiate_copy_all(itname ##complex64, itype, complex64_t)

instantiate_copy_itype(bool_, bool)
instantiate_copy_itype(uint8, uint8_t)
instantiate_copy_itype(uint16, uint16_t)
instantiate_copy_itype(uint32, uint32_t)
instantiate_copy_itype(uint64, uint64_t)
instantiate_copy_itype(int8, int8_t)
instantiate_copy_itype(int16, int16_t)
instantiate_copy_itype(int32, int32_t)
instantiate_copy_itype(int64, int64_t)
instantiate_copy_itype(float16, half)
instantiate_copy_itype(float32, float)
instantiate_copy_itype(bfloat16, bfloat16_t)
instantiate_copy_itype(complex64, complex64_t) // clang-format on
