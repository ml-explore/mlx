// Copyright © 2024 Apple Inc.

#include <metal_atomic>
#include <metal_simdgroup>

// clang-format off
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/atomic.h"
#include "mlx/backend/metal/kernels/reduction/ops.h"
#include "mlx/backend/metal/kernels/reduce.h"

#define instantiate_init_reduce(name, tname, type, op) \
  instantiate_kernel("init_reduce_" #name #tname, init_reduce, type, op<type>)

instantiate_init_reduce(and, bool_, bool, And)
instantiate_init_reduce(or, bool_, bool, Or)

#define instantiate_init_sum_prod(name, op)                 \
  instantiate_init_reduce(name, int32, int32_t, op)         \
  instantiate_init_reduce(name, int64, int64_t, op)         \
  instantiate_init_reduce(name, float16, float16_t, op)     \
  instantiate_init_reduce(name, bfloat16, bfloat16_t, op)   \
  instantiate_init_reduce(name, float32, float, op)         \
  instantiate_init_reduce(name, complex64, complex64_t, op)

instantiate_init_sum_prod(sum, Sum)
instantiate_init_sum_prod(prod, Prod)

#define instantiate_init_min_max(name, op)                   \
  instantiate_init_reduce(name, bool_, bool, op)             \
  instantiate_init_reduce(name, int8, int8_t, op)            \
  instantiate_init_reduce(name, int16, int16_t, op)          \
  instantiate_init_reduce(name, int32, int32_t, op)          \
  instantiate_init_reduce(name, int64, int64_t, op)          \
  instantiate_init_reduce(name, uint8, uint8_t, op)          \
  instantiate_init_reduce(name, uint16, uint16_t, op)        \
  instantiate_init_reduce(name, uint32, uint32_t, op)        \
  instantiate_init_reduce(name, uint64, uint64_t, op)        \
  instantiate_init_reduce(name, float16, float16_t, op)      \
  instantiate_init_reduce(name, bfloat16, bfloat16_t, op)    \
  instantiate_init_reduce(name, float32, float, op)          \
  instantiate_init_reduce(name, complex64, complex64_t, op)

instantiate_init_min_max(min, Min)
instantiate_init_min_max(max, Max)

#define instantiate_all_reduce(name, itype, otype, op) \
  instantiate_kernel("all_reduce_" #name,              \
                     all_reduce,                       \
                     itype, otype, op)

#define instantiate_col_reduce_small(name, itype, otype, op, dim)          \
  instantiate_kernel("col_reduce_small_" #dim "_reduce_" #name,            \
                     col_reduce_small,                                     \
                     itype, otype, op, int, dim)                           \
  instantiate_kernel("col_reduce_longcolumn_" #dim "_reduce_" #name,       \
                     col_reduce_longcolumn,                                \
                     itype, otype, op, int, dim)                           \
  instantiate_kernel("col_reduce_small_large_" #dim "_reduce_" #name,      \
                     col_reduce_small,                                     \
                     itype, otype, op, int64_t, dim)                       \
  instantiate_kernel("col_reduce_longcolumn_large_" #dim "_reduce_" #name, \
                     col_reduce_longcolumn,                                \
                     itype, otype, op, int64_t, dim)

#define instantiate_col_reduce_looped_tile(name, itype, otype, op, dim, bm, bn)        \
  instantiate_kernel("col_reduce_looped_" #dim "_" #bm "_" #bn "_reduce_" #name,       \
                     col_reduce_looped,                                                \
                     itype, otype, op, int, dim, bm, bn)                               \
  instantiate_kernel("col_reduce_looped_large_" #dim "_" #bm "_" #bn "_reduce_" #name, \
                     col_reduce_looped,                                                \
                     itype, otype, op, int64_t, dim, bm, bn)

#define instantiate_col_reduce_2pass_tile(name, itype, otype, op, dim, bm, bn)        \
  instantiate_kernel("col_reduce_2pass_" #dim "_" #bm "_" #bn "_reduce_" #name,       \
                     col_reduce_2pass,                                                \
                     itype, otype, op, int, dim, bm, bn)                              \
  instantiate_kernel("col_reduce_2pass_large_" #dim "_" #bm "_" #bn "_reduce_" #name, \
                     col_reduce_2pass,                                                \
                     itype, otype, op, int64_t, dim, bm, bn)

#define instantiate_col_reduce_looped(name, itype, otype, op, dim)        \
  instantiate_col_reduce_looped_tile(name, itype, otype, op, dim, 32, 32) \
  instantiate_col_reduce_2pass_tile(name, itype, otype, op, dim, 32, 32)

#define instantiate_col_reduce_general(name, itype, otype, op) \
  instantiate_col_reduce_small(name, itype, otype, op, 1)      \
  instantiate_col_reduce_small(name, itype, otype, op, 2)      \
  instantiate_col_reduce_small(name, itype, otype, op, 5)      \
  instantiate_col_reduce_looped(name, itype, otype, op, 1)     \
  instantiate_col_reduce_looped(name, itype, otype, op, 2)     \
  instantiate_col_reduce_looped(name, itype, otype, op, 5)

#define instantiate_row_reduce_small(name, itype, otype, op, dim)     \
  instantiate_kernel("row_reduce_small_" #dim "_reduce_" #name,       \
                     row_reduce_small,                                \
                     itype, otype, op, int, dim)                      \
  instantiate_kernel("row_reduce_small_large_" #dim "_reduce_" #name, \
                     row_reduce_small,                                \
                     itype, otype, op, int64_t, dim)

#define instantiate_row_reduce_looped(name, itype, otype, op, dim)       \
  instantiate_kernel("row_reduce_looped_" #dim "_reduce_" #name,         \
                     row_reduce_looped,                                  \
                     itype, otype, op, int, dim)                         \
  instantiate_kernel("row_reduce_looped_large_" #dim "_reduce_" #name,   \
                     row_reduce_looped,                                  \
                     itype, otype, op, int64_t, dim)

#define instantiate_row_reduce_general(name, itype, otype, op) \
  instantiate_row_reduce_small(name, itype, otype, op, 1)      \
  instantiate_row_reduce_small(name, itype, otype, op, 2)      \
  instantiate_row_reduce_small(name, itype, otype, op, 5)      \
  instantiate_row_reduce_looped(name, itype, otype, op, 1)     \
  instantiate_row_reduce_looped(name, itype, otype, op, 2)     \
  instantiate_row_reduce_looped(name, itype, otype, op, 5)     \
  instantiate_kernel("row_reduce_simple_" #name,               \
                     row_reduce_simple,                        \
                     itype, otype, op)

#define instantiate_reduce_functions(name, tname, itype, otype, op)    \
  instantiate_all_reduce(name##tname, itype, otype, op<otype>)         \
  instantiate_row_reduce_general(name##tname, itype, otype, op<otype>) \
  instantiate_col_reduce_general(name##tname, itype, otype, op<otype>)

#define instantiate_and_or(name, op)                           \
  instantiate_reduce_functions(name, bool_, bool, bool, op)    \
  instantiate_reduce_functions(name, int16, int16_t, bool, op) \
  instantiate_reduce_functions(name, int32, int32_t, bool, op) \
  instantiate_reduce_functions(name, int64, int64_t, bool, op)

instantiate_and_or(and, And)
instantiate_and_or(or, Or)

#define instantiate_sum_prod(name, op)                                       \
  instantiate_reduce_functions(name, int8, int8_t, int32_t, op)              \
  instantiate_reduce_functions(name, int16, int16_t, int32_t, op)            \
  instantiate_reduce_functions(name, int32, int32_t, int32_t, op)            \
  instantiate_reduce_functions(name, int64, int64_t, int64_t, op)            \
  instantiate_reduce_functions(name, float16, float16_t, float16_t, op)      \
  instantiate_reduce_functions(name, bfloat16, bfloat16_t, bfloat16_t, op)   \
  instantiate_reduce_functions(name, float32, float, float, op)              \
  instantiate_reduce_functions(name, complex64, complex64_t, complex64_t, op)

instantiate_sum_prod(sum, Sum)
instantiate_sum_prod(prod, Prod)

#define instantiate_min_max(name, op)                                        \
  instantiate_reduce_functions(name, int8, int8_t, int8_t, op)               \
  instantiate_reduce_functions(name, int16, int16_t, int16_t, op)            \
  instantiate_reduce_functions(name, int32, int32_t, int32_t, op)            \
  instantiate_reduce_functions(name, int64, int64_t, int64_t, op)            \
  instantiate_reduce_functions(name, uint8, uint8_t, uint8_t, op)            \
  instantiate_reduce_functions(name, uint16, uint16_t, uint16_t, op)         \
  instantiate_reduce_functions(name, uint32, uint32_t, uint32_t, op)         \
  instantiate_reduce_functions(name, uint64, uint64_t, uint64_t, op)         \
  instantiate_reduce_functions(name, float16, float16_t, float16_t, op)      \
  instantiate_reduce_functions(name, bfloat16, bfloat16_t, bfloat16_t, op)   \
  instantiate_reduce_functions(name, float32, float, float, op)              \
  instantiate_reduce_functions(name, complex64, complex64_t, complex64_t, op)

instantiate_min_max(min, Min)
instantiate_min_max(max, Max)
    // clang-format on
