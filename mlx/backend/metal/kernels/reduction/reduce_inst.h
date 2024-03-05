// Copyright © 2024 Apple Inc.

#pragma once

#include <metal_atomic>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/reduction/ops.h"

#define instantiate_reduce_helper_floats(inst_f, name, op)         \
  inst_f(name, float16, half, op) inst_f(name, float32, float, op) \
      inst_f(name, bfloat16, bfloat16_t, op)

#define instantiate_reduce_helper_uints(inst_f, name, op)             \
  inst_f(name, uint8, uint8_t, op) inst_f(name, uint16, uint16_t, op) \
      inst_f(name, uint32, uint32_t, op)

#define instantiate_reduce_helper_ints(inst_f, name, op)          \
  inst_f(name, int8, int8_t, op) inst_f(name, int16, int16_t, op) \
      inst_f(name, int32, int32_t, op)

#define instantiate_reduce_helper_64b(inst_f, name, op) \
  inst_f(name, int64, int64_t, op) inst_f(name, uint64, uint64_t, op)

#define instantiate_reduce_helper_types(inst_f, name, op) \
  instantiate_reduce_helper_floats(inst_f, name, op)      \
      instantiate_reduce_helper_uints(inst_f, name, op)   \
          instantiate_reduce_helper_ints(inst_f, name, op)

#define instantiate_reduce_ops(inst_f, type_f)        \
  type_f(inst_f, sum, Sum) type_f(inst_f, prod, Prod) \
      type_f(inst_f, min_, Min) type_f(inst_f, max_, Max)

// Special case for bool reductions
#define instantiate_reduce_from_types_helper( \
    inst_f, name, tname, itype, otype, op)    \
  inst_f(name##tname, itype, otype, op)

#define instantiate_reduce_from_types(inst_f, name, otype, op)                  \
  instantiate_reduce_from_types_helper(inst_f, name, bool_, bool, otype, op)    \
      instantiate_reduce_from_types_helper(                                     \
          inst_f, name, uint8, uint8_t, otype, op)                              \
          instantiate_reduce_from_types_helper(                                 \
              inst_f, name, uint16, uint16_t, otype, op)                        \
              instantiate_reduce_from_types_helper(                             \
                  inst_f, name, uint32, uint32_t, otype, op)                    \
                  instantiate_reduce_from_types_helper(                         \
                      inst_f, name, int8, int8_t, otype, op)                    \
                      instantiate_reduce_from_types_helper(                     \
                          inst_f, name, int16, int16_t, otype, op)              \
                          instantiate_reduce_from_types_helper(                 \
                              inst_f, name, int32, int32_t, otype, op)          \
                              instantiate_reduce_from_types_helper(             \
                                  inst_f, name, int64, int64_t, otype, op)      \
                                  instantiate_reduce_from_types_helper(         \
                                      inst_f, name, float16, half, otype, op)   \
                                      instantiate_reduce_from_types_helper(     \
                                          inst_f,                               \
                                          name,                                 \
                                          float32,                              \
                                          float,                                \
                                          otype,                                \
                                          op)                                   \
                                          instantiate_reduce_from_types_helper( \
                                              inst_f,                           \
                                              name,                             \
                                              bfloat16,                         \
                                              bfloat16_t,                       \
                                              otype,                            \
                                              op)