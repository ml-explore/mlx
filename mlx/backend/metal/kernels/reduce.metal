// Copyright © 2024 Apple Inc.

#include <metal_atomic>
#include <metal_simdgroup>

// clang-format off
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/atomic.h"
#include "mlx/backend/metal/kernels/reduction/ops.h"
#include "mlx/backend/metal/kernels/reduce.h"

// sum / prod get bool, signed integers, floats
// sum output type is special case for bool and int

// prod gets bool, signed integers, floats
// and / or get bool, signed integers
// min / max get everything

#define instantiate_init_reduce(name, tname, type, op) \
  instantiate_kernel("init_reduce_" #name #tname, init_reduce, type, op<type>)

instantiate_init_reduce(and, bool_, bool, And)
instantiate_init_reduce(or, bool_, bool, Or)
instantiate_init_reduce(sum, int8, int8_t, Sum)
instantiate_init_reduce(sum, int16, int16_t, Sum)
instantiate_init_reduce(sum, int32, int32_t, Sum)
instantiate_init_reduce(sum, int64, int64_t, Sum)
instantiate_init_reduce(sum, float16, float16_t, Sum)
instantiate_init_reduce(sum, bfloat16, bfloat16_t, Sum)
instantiate_init_reduce(sum, float32, float, Sum)
instantiate_init_reduce(sum, complex64, complex64_t, Sum)
instantiate_init_reduce(prod, int8, int8_t, Prod)
instantiate_init_reduce(prod, int16, int16_t, Prod)
instantiate_init_reduce(prod, int32, int32_t, Prod)
instantiate_init_reduce(prod, int64, int64_t, Prod)
instantiate_init_reduce(prod, float16, float16_t, Prod)
instantiate_init_reduce(prod, bfloat16, bfloat16_t, Prod)
instantiate_init_reduce(prod, float32, float, Prod)
instantiate_init_reduce(prod, complex64, complex64_t, Prod)
instantiate_init_reduce(min, bool_, bool, Min)
instantiate_init_reduce(min, int8, int8_t, Min)
instantiate_init_reduce(min, int16, int16_t, Min)
instantiate_init_reduce(min, int32, int32_t, Min)
instantiate_init_reduce(min, int64, int64_t, Min)
instantiate_init_reduce(min, uint8, uint8_t, Min)
instantiate_init_reduce(min, uint16, uint16_t, Min)
instantiate_init_reduce(min, uint32, uint32_t, Min)
instantiate_init_reduce(min, uint64, uint64_t, Min)
instantiate_init_reduce(min, float16, float16_t, Min)
instantiate_init_reduce(min, bfloat16, bfloat16_t, Min)
instantiate_init_reduce(min, float32, float, Min)
instantiate_init_reduce(min, complex64, complex64_t, Min)
instantiate_init_reduce(max, bool_, bool, Max)
instantiate_init_reduce(max, int8, int8_t, Max)
instantiate_init_reduce(max, int16, int16_t, Max)
instantiate_init_reduce(max, int32, int32_t, Max)
instantiate_init_reduce(max, int64, int64_t, Max)
instantiate_init_reduce(max, uint8, uint8_t, Max)
instantiate_init_reduce(max, uint16, uint16_t, Max)
instantiate_init_reduce(max, uint32, uint32_t, Max)
instantiate_init_reduce(max, uint64, uint64_t, Max)
instantiate_init_reduce(max, float16, float16_t, Max)
instantiate_init_reduce(max, bfloat16, bfloat16_t, Max)
instantiate_init_reduce(max, float32, float, Max)
instantiate_init_reduce(max, complex64, complex64_t, Max)

#define instantiate_all_reduce(name, itype, otype, op) \
  instantiate_kernel("all_reduce_" #name,              \
                     all_reduce,                       \
                     itype, otype, op)

#define instantiate_col_reduce_small(name, itype, otype, op, dim) \
  instantiate_kernel("col_reduce_small_" #dim "_reduce_" #name,        \
                     col_reduce_small,                                 \
                     itype, otype, op, uint, dim)                      \
  instantiate_kernel("col_reduce_longcolumn_" #dim "_reduce_" #name,   \
                     col_reduce_longcolumn,                            \
                     itype, otype, op, uint, dim)     \
  instantiate_kernel("col_reduce_small_large_" #dim "_reduce_" #name,        \
                     col_reduce_small,                                 \
                     itype, otype, op, size_t, dim)                            \
  instantiate_kernel("col_reduce_longcolumn_large_" #dim "_reduce_" #name,   \
                     col_reduce_longcolumn,                            \
                     itype, otype, op, size_t, dim)

#define instantiate_col_reduce_looped_tile(name, itype, otype, op, dim, bm, bn)  \
  instantiate_kernel("col_reduce_looped_" #dim "_" #bm "_" #bn "_reduce_" #name, \
                     col_reduce_looped,                                          \
                     itype, otype, op, uint, dim, bm, bn) \
  instantiate_kernel("col_reduce_looped_large_" #dim "_" #bm "_" #bn "_reduce_" #name, \
                     col_reduce_looped,                                          \
                     itype, otype, op, size_t, dim, bm, bn)

#define instantiate_col_reduce_2pass_tile(name, itype, otype, op, dim, bm, bn)  \
  instantiate_kernel("col_reduce_2pass_" #dim "_" #bm "_" #bn "_reduce_" #name, \
                     col_reduce_2pass,                                          \
                     itype, otype, op, uint, dim, bm, bn) \
  instantiate_kernel("col_reduce_2pass_large_" #dim "_" #bm "_" #bn "_reduce_" #name, \
                     col_reduce_2pass,                                          \
                     itype, otype, op, size_t, dim, bm, bn)

#define instantiate_col_reduce_looped(name, itype, otype, op, dim)        \
  instantiate_col_reduce_looped_tile(name, itype, otype, op, dim, 32, 32) \
  instantiate_col_reduce_2pass_tile(name, itype, otype, op, dim, 32, 32)

#define instantiate_col_reduce_general(name, itype, otype, op) \
  instantiate_col_reduce_small(name, itype, otype, op, 1)      \
  instantiate_col_reduce_small(name, itype, otype, op, 2)      \
  instantiate_col_reduce_small(name, itype, otype, op, 3)      \
  instantiate_col_reduce_looped(name, itype, otype, op, 1)     \
  instantiate_col_reduce_looped(name, itype, otype, op, 2)     \
  instantiate_col_reduce_looped(name, itype, otype, op, 3)

#define instantiate_row_reduce_small(name, itype, otype, op, dim) \
  instantiate_kernel("row_reduce_small_" #dim "_reduce_" #name,   \
                     row_reduce_small,                            \
                     itype, otype, op, uint, dim)         \
  instantiate_kernel("row_reduce_small_large_" #dim "_reduce_" #name,   \
                     row_reduce_small,                            \
                     itype, otype, op, size_t, dim)

#define instantiate_row_reduce_looped(name, itype, otype, op, dim) \
  instantiate_kernel("row_reduce_looped_" #dim "_reduce_" #name,   \
                     row_reduce_looped,                            \
                     itype, otype, op, uint, dim)                  \
  instantiate_kernel("row_reduce_looped_large_" #dim "_reduce_" #name,   \
                     row_reduce_looped,                            \
                     itype, otype, op, size_t, dim)

#define instantiate_row_reduce_general(name, itype, otype, op) \
  instantiate_row_reduce_small(name, itype, otype, op, 1)      \
  instantiate_row_reduce_small(name, itype, otype, op, 2)      \
  instantiate_row_reduce_small(name, itype, otype, op, 3)      \
  instantiate_row_reduce_looped(name, itype, otype, op, 1)     \
  instantiate_row_reduce_looped(name, itype, otype, op, 2)     \
  instantiate_row_reduce_looped(name, itype, otype, op, 3)     \
  instantiate_kernel("row_reduce_simple_" #name,               \
                     row_reduce_simple,                        \
                     itype, otype, op)

#define instantiate_reduce_functions(name, tname, itype, otype, op)    \
  instantiate_all_reduce(name##tname, itype, otype, op<otype>)         \
  instantiate_row_reduce_general(name##tname, itype, otype, op<otype>) \
  instantiate_col_reduce_general(name##tname, itype, otype, op<otype>)

instantiate_reduce_functions(and, bool_, bool, bool, And)
instantiate_reduce_functions(and, int8, int8_t, bool, And)
instantiate_reduce_functions(and, int16, int16_t, bool, And)
instantiate_reduce_functions(and, int32, int32_t, bool, And)
instantiate_reduce_functions(and, int64, int64_t, bool, And)
instantiate_reduce_functions(or, bool_, bool, bool, Or)
instantiate_reduce_functions(or, int8, int8_t, bool, Or)
instantiate_reduce_functions(or, int16, int16_t, bool, Or)
instantiate_reduce_functions(or, int32, int32_t, bool, Or)
instantiate_reduce_functions(or, int64, int64_t, bool, Or)

instantiate_reduce_functions(sum, bool_, bool, int32_t, Sum)
instantiate_reduce_functions(sum, int8, int8_t, int8_t, Sum)
instantiate_reduce_functions(sum, int16, int16_t, int16_t, Sum)
instantiate_reduce_functions(sum, int32, int32_t, int32_t, Sum)
instantiate_reduce_functions(sum, int64, int64_t, int64_t, Sum)
instantiate_reduce_functions(sum, float16, float16_t, float16_t, Sum)
instantiate_reduce_functions(sum, bfloat16, bfloat16_t, bfloat16_t, Sum)
instantiate_reduce_functions(sum, float32, float, float, Sum)
instantiate_reduce_functions(sum, complex64, complex64_t, complex64_t, Sum)

instantiate_reduce_functions(prod, int8, int8_t, int8_t, Prod)
instantiate_reduce_functions(prod, int16, int16_t, int16_t, Prod)
instantiate_reduce_functions(prod, int32, int32_t, int32_t, Prod)
instantiate_reduce_functions(prod, int64, int64_t, int64_t, Prod)
instantiate_reduce_functions(prod, float16, float16_t, float16_t, Prod)
instantiate_reduce_functions(prod, bfloat16, bfloat16_t, bfloat16_t, Prod)
instantiate_reduce_functions(prod, float32, float, float, Prod)
instantiate_reduce_functions(prod, complex64, complex64_t, complex64_t, Prod)

instantiate_reduce_functions(min, int8, int8_t, int8_t, Min)
instantiate_reduce_functions(min, int16, int16_t, int16_t, Min)
instantiate_reduce_functions(min, int32, int32_t, int32_t, Min)
instantiate_reduce_functions(min, int64, int64_t, int64_t, Min)
instantiate_reduce_functions(min, uint8, uint8_t, uint8_t, Min)
instantiate_reduce_functions(min, uint16, uint16_t, uint16_t, Min)
instantiate_reduce_functions(min, uint32, uint32_t, uint32_t, Min)
instantiate_reduce_functions(min, uint64, uint64_t, uint64_t, Min)
instantiate_reduce_functions(min, float16, float16_t, float16_t, Min)
instantiate_reduce_functions(min, bfloat16, bfloat16_t, bfloat16_t, Min)
instantiate_reduce_functions(min, float32, float, float, Min)
instantiate_reduce_functions(min, complex64, complex64_t, complex64_t, Min)

instantiate_reduce_functions(max, int8, int8_t, int8_t, Max)
instantiate_reduce_functions(max, int16, int16_t, int16_t, Max)
instantiate_reduce_functions(max, int32, int32_t, int32_t, Max)
instantiate_reduce_functions(max, int64, int64_t, int64_t, Max)
instantiate_reduce_functions(max, uint8, uint8_t, uint8_t, Max)
instantiate_reduce_functions(max, uint16, uint16_t, uint16_t, Max)
instantiate_reduce_functions(max, uint32, uint32_t, uint32_t, Max)
instantiate_reduce_functions(max, uint64, uint64_t, uint64_t, Max)
instantiate_reduce_functions(max, float16, float16_t, float16_t, Max)
instantiate_reduce_functions(max, bfloat16, bfloat16_t, bfloat16_t, Max)
instantiate_reduce_functions(max, float32, float, float, Max)
instantiate_reduce_functions(max, complex64, complex64_t, complex64_t, Max)

    // clang-format on
