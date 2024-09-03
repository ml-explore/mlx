// Copyright © 2024 Apple Inc.

#include <metal_atomic>
#include <metal_simdgroup>

// clang-format off
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/atomic.h"
#include "mlx/backend/metal/kernels/reduction/ops.h"
#include "mlx/backend/metal/kernels/reduce.h"

#define instantiate_reduce_helper_floats(inst_f, name, op) \
  inst_f(name, float16, half, op)                          \
  inst_f(name, float32, float, op)                         \
  inst_f(name, bfloat16, bfloat16_t, op)

#define instantiate_reduce_helper_uints(inst_f, name, op)  \
  inst_f(name, uint8, uint8_t, op)                         \
  inst_f(name, uint16, uint16_t, op)                       \
  inst_f(name, uint32, uint32_t, op)

#define instantiate_reduce_helper_ints(inst_f, name, op) \
  inst_f(name, int8, int8_t, op)                         \
  inst_f(name, int16, int16_t, op)                       \
  inst_f(name, int32, int32_t, op)

#define instantiate_reduce_helper_64b(inst_f, name, op) \
  inst_f(name, int64, int64_t, op)                      \
  inst_f(name, uint64, uint64_t, op)                    \
  inst_f(name, complex64, complex64_t, op)

#define instantiate_reduce_helper_types(inst_f, name, op) \
  instantiate_reduce_helper_floats(inst_f, name, op)      \
  instantiate_reduce_helper_uints(inst_f, name, op)       \
  instantiate_reduce_helper_ints(inst_f, name, op)

#define instantiate_reduce_ops(inst_f, type_f) \
  type_f(inst_f, sum, Sum)                     \
  type_f(inst_f, prod, Prod)                   \
  type_f(inst_f, min, Min)                     \
  type_f(inst_f, max, Max)

// Special case for bool reductions
#define instantiate_reduce_from_types_helper( \
    inst_f, name, tname, itype, otype, op)    \
    inst_f(name##tname, itype, otype, op)

#define instantiate_reduce_from_types(inst_f, name, otype, op)  \
  instantiate_reduce_from_types_helper(                         \
    inst_f, name, bool_, bool, otype, op)                       \
  instantiate_reduce_from_types_helper(                         \
    inst_f, name, uint8, uint8_t, otype, op)                    \
  instantiate_reduce_from_types_helper(                         \
    inst_f, name, uint16, uint16_t, otype, op)                  \
  instantiate_reduce_from_types_helper(                         \
    inst_f, name, uint32, uint32_t, otype, op)                  \
  instantiate_reduce_from_types_helper(                         \
    inst_f, name, uint64, uint64_t, otype, op)                  \
  instantiate_reduce_from_types_helper(                         \
    inst_f, name, int8, int8_t, otype, op)                      \
  instantiate_reduce_from_types_helper(                         \
    inst_f, name, int16, int16_t, otype, op)                    \
  instantiate_reduce_from_types_helper(                         \
    inst_f, name, int32, int32_t, otype, op)                    \
  instantiate_reduce_from_types_helper(                         \
    inst_f, name, int64, int64_t, otype, op)                    \
  instantiate_reduce_from_types_helper(                         \
    inst_f, name, float16, half, otype, op)                     \
  instantiate_reduce_from_types_helper(                         \
    inst_f,                                                     \
    name,                                                       \
    float32,                                                    \
    float,                                                      \
    otype,                                                      \
    op)                                                         \
  instantiate_reduce_from_types_helper(                         \
    inst_f,                                                     \
    name,                                                       \
    bfloat16,                                                   \
    bfloat16_t,                                                 \
    otype,                                                      \
    op)

#define instantiate_init_reduce(name, otype, op)             \
  instantiate_kernel("init_reduce_" #name, \
                     init_reduce, \
                     otype, op)

#define instantiate_init_reduce_helper(name, tname, type, op) \
  instantiate_init_reduce(name##tname, type, op<type>)

instantiate_reduce_ops(instantiate_init_reduce_helper, instantiate_reduce_helper_types)
instantiate_reduce_ops(instantiate_init_reduce_helper, instantiate_reduce_helper_64b)

instantiate_init_reduce(andbool_, bool, And<bool>)
instantiate_init_reduce(orbool_, bool, Or<bool>)

#define instantiate_all_reduce(name, itype, otype, op)        \
  instantiate_kernel("allReduce_" #name, \
                     all_reduce, \
                     itype, otype, op)

#define instantiate_same_all_reduce_helper(name, tname, type, op) \
  instantiate_all_reduce(name##tname, type, type, op<type>)

instantiate_reduce_ops(instantiate_same_all_reduce_helper, instantiate_reduce_helper_types)
instantiate_reduce_ops(instantiate_same_all_reduce_helper, instantiate_reduce_helper_64b)

instantiate_reduce_from_types(instantiate_all_reduce, and, bool, And<bool>)
instantiate_reduce_from_types(instantiate_all_reduce, or, bool, Or<bool>)

// special case bool with larger output type
instantiate_all_reduce(sumbool_, bool, uint32_t, Sum<uint32_t>)

#define instantiate_col_reduce_small(name, itype, otype, op, dim) \
  instantiate_kernel("colReduceSmall_" #dim "_reduce_" #name, \
                     col_reduce_small, \
                     itype, otype, op, dim)

#define instantiate_col_reduce_looped_tile(name, itype, otype, op, dim, bm, bn) \
  instantiate_kernel("colReduceLooped_" #dim "_" #bm "_" #bn "_reduce_" #name, \
                     col_reduce_looped, \
                     itype, otype, op, dim, bm, bn)

#define instantiate_col_reduce_atomic_tile(name, itype, otype, op, dim, bm, bn) \
  template  [[host_name("colAtomic" #dim "_" #bm "_" #bn "_reduce_" #name)]]    \
  [[kernel]] void col_reduce_atomics<itype, otype, op, dim, bm, bn>(            \
      const device itype* in [[buffer(0)]],                                     \
      device mlx_atomic<otype>* out [[buffer(1)]],                              \
      const constant size_t& reduction_size [[buffer(2)]],                      \
      const constant size_t& reduction_stride [[buffer(3)]],                    \
      const constant int* shape [[buffer(4)]],                                  \
      const constant size_t* strides [[buffer(5)]],                             \
      const constant int& ndim [[buffer(6)]],                                   \
      const constant int* reduce_shape [[buffer(7)]],                           \
      const constant size_t* reduce_strides [[buffer(8)]],                      \
      const constant int& reduce_ndim [[buffer(9)]],                            \
      const constant size_t& non_col_reductions [[buffer(10)]],                 \
      uint3 gid [[threadgroup_position_in_grid]],                               \
      uint3 gsize [[threadgroups_per_grid]],                                    \
      uint simd_lane_id [[thread_index_in_simdgroup]],                          \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_col_reduce_looped(name, itype, otype, op, dim) \
  instantiate_col_reduce_looped_tile(name, itype, otype, op, dim, 8, 128) \
  instantiate_col_reduce_looped_tile(name, itype, otype, op, dim, 32, 32)

#define instantiate_col_reduce_general(name, itype, otype, op) \
  instantiate_col_reduce_small(name, itype, otype, op, 0)      \
  instantiate_col_reduce_small(name, itype, otype, op, 1)      \
  instantiate_col_reduce_small(name, itype, otype, op, 2)      \
  instantiate_col_reduce_small(name, itype, otype, op, 3)      \
  instantiate_col_reduce_small(name, itype, otype, op, 4)      \
  instantiate_col_reduce_looped(name, itype, otype, op, 0)     \
  instantiate_col_reduce_looped(name, itype, otype, op, 1)     \
  instantiate_col_reduce_looped(name, itype, otype, op, 2)     \
  instantiate_col_reduce_looped(name, itype, otype, op, 3)     \
  instantiate_col_reduce_looped(name, itype, otype, op, 4)

#define instantiate_same_col_reduce_helper(name, tname, type, op) \
  instantiate_col_reduce_general(name##tname, type, type, op<type>)

#define instantiate_same_col_reduce_atomics_helper(name, tname, type, op) \
  instantiate_col_reduce_atomic_tile(name##tname, type, type, op<type>, 1, 8, 128)

instantiate_reduce_ops(instantiate_same_col_reduce_helper, instantiate_reduce_helper_types)
instantiate_reduce_ops(instantiate_same_col_reduce_atomics_helper, instantiate_reduce_helper_types)
instantiate_reduce_ops(instantiate_same_col_reduce_helper, instantiate_reduce_helper_64b)

instantiate_col_reduce_general(sumbool_, bool, uint32_t, Sum<uint32_t>)
instantiate_reduce_from_types(instantiate_col_reduce_general, and, bool, And<bool>)
instantiate_reduce_from_types(instantiate_col_reduce_general, or, bool, Or<bool>)

#define instantiate_row_reduce_small(name, itype, otype, op, dim)          \
  instantiate_kernel("rowReduceSmall_" #dim "_reduce_" #name, \
                     row_reduce_small, \
                     itype, otype, op, dim)

#define instantiate_row_reduce_looped(name, itype, otype, op, dim)      \
  instantiate_kernel("rowReduceLooped_" #dim "_reduce_" #name, \
                     row_reduce_looped, \
                     itype, otype, op, dim)

#define instantiate_row_reduce_general(name, itype, otype, op)     \
  instantiate_row_reduce_small(name, itype, otype, op, 0)          \
  instantiate_row_reduce_small(name, itype, otype, op, 1)          \
  instantiate_row_reduce_small(name, itype, otype, op, 2)          \
  instantiate_row_reduce_small(name, itype, otype, op, 3)          \
  instantiate_row_reduce_small(name, itype, otype, op, 4)          \
  instantiate_row_reduce_looped(name, itype, otype, op, 0)         \
  instantiate_row_reduce_looped(name, itype, otype, op, 1)         \
  instantiate_row_reduce_looped(name, itype, otype, op, 2)         \
  instantiate_row_reduce_looped(name, itype, otype, op, 3)         \
  instantiate_row_reduce_looped(name, itype, otype, op, 4)         \
  instantiate_kernel("rowReduceSimple_" #name, \
                     row_reduce_simple, \
                     itype, otype, op)

#define instantiate_same_row_reduce_helper(name, tname, type, op) \
  instantiate_row_reduce_general(name##tname, type, type, op<type>)

instantiate_reduce_ops(instantiate_same_row_reduce_helper, instantiate_reduce_helper_types)
instantiate_reduce_ops(instantiate_same_row_reduce_helper, instantiate_reduce_helper_64b)

instantiate_reduce_from_types(instantiate_row_reduce_general, and, bool, And<bool>)
instantiate_reduce_from_types(instantiate_row_reduce_general, or, bool, Or<bool>)

instantiate_row_reduce_general(sumbool_, bool, uint32_t, Sum<uint32_t>)
    // clang-format on
