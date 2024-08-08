// Copyright © 2024 Apple Inc.

#include <metal_atomic>
#include <metal_simdgroup>

// clang-format off
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/atomic.h"
#include "mlx/backend/metal/kernels/reduction/ops.h"
#include "mlx/backend/metal/kernels/reduction/reduce_init.h"
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
  inst_f(name, uint64, uint64_t, op)

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
  template [[host_name("i_reduce_" #name)]] [[kernel]] void  \
  init_reduce<otype, op>(                                    \
      device otype * out [[buffer(1)]], uint tid [[thread_position_in_grid]]);

#define instantiate_init_reduce_helper(name, tname, type, op) \
  instantiate_init_reduce(name##tname, type, op<type>)

instantiate_reduce_ops(instantiate_init_reduce_helper, instantiate_reduce_helper_types)
instantiate_reduce_ops(instantiate_init_reduce_helper, instantiate_reduce_helper_64b)

instantiate_init_reduce(andbool_, bool, And<bool>)
instantiate_init_reduce(orbool_, bool, Or<bool>)

#define instantiate_all_reduce(name, itype, otype, op)        \
  template [[host_name("all_reduce_" #name)]] [[kernel]] void \
  all_reduce<itype, otype, op>(                               \
      const device itype* in [[buffer(0)]],                   \
      device mlx_atomic<otype>* out [[buffer(1)]],            \
      const device size_t& in_size [[buffer(2)]],             \
      uint gid [[thread_position_in_grid]],                   \
      uint lid [[thread_position_in_threadgroup]],            \
      uint grid_size [[threads_per_grid]],                    \
      uint simd_per_group [[simdgroups_per_threadgroup]],     \
      uint simd_lane_id [[thread_index_in_simdgroup]],        \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_all_reduce_no_atomics(name, itype, otype, op)        \
  template [[host_name("allNoAtomics_reduce_" #name)]] [[kernel]] void   \
  all_reduce_no_atomics<itype, otype, op>(                               \
      const device itype* in [[buffer(0)]],                              \
      device otype* out [[buffer(1)]],                                   \
      const device size_t& in_size [[buffer(2)]],                        \
      uint gid [[thread_position_in_grid]],                              \
      uint lid [[thread_position_in_threadgroup]],                       \
      uint grid_size [[threads_per_grid]],                               \
      uint simd_per_group [[simdgroups_per_threadgroup]],                \
      uint simd_lane_id [[thread_index_in_simdgroup]],                   \
      uint simd_group_id [[simdgroup_index_in_threadgroup]],             \
      uint thread_group_id [[threadgroup_position_in_grid]]);

#define instantiate_same_all_reduce_helper(name, tname, type, op) \
  instantiate_all_reduce(name##tname, type, type, op<type>)

#define instantiate_same_all_reduce_na_helper(name, tname, type, op) \
  instantiate_all_reduce_no_atomics(name##tname, type, type, op<type>)

instantiate_reduce_ops(instantiate_same_all_reduce_helper, instantiate_reduce_helper_types)
instantiate_reduce_ops(instantiate_same_all_reduce_na_helper, instantiate_reduce_helper_64b)

instantiate_reduce_from_types(instantiate_all_reduce, and, bool, And<bool>)
instantiate_reduce_from_types(instantiate_all_reduce, or, bool, Or<bool>)

// special case bool with larger output type
instantiate_all_reduce(sumbool_, bool, uint32_t, Sum<uint32_t>)

#define instantiate_col_reduce_general(name, itype, otype, op)        \
  template [[host_name("colGeneral_reduce_" #name)]] [[kernel]] void  \
  col_reduce_general<itype, otype, op>(                               \
      const device itype* in [[buffer(0)]],                           \
      device mlx_atomic<otype>* out [[buffer(1)]],                    \
      const constant size_t& reduction_size [[buffer(2)]],            \
      const constant size_t& reduction_stride [[buffer(3)]],          \
      const constant size_t& out_size [[buffer(4)]],                  \
      const constant int* shape [[buffer(5)]],                        \
      const constant size_t* strides [[buffer(6)]],                   \
      const constant int& ndim [[buffer(7)]],                         \
      threadgroup otype* local_data [[threadgroup(0)]],               \
      uint3 tid [[threadgroup_position_in_grid]],                     \
      uint3 lid [[thread_position_in_threadgroup]],                   \
      uint3 lsize [[threads_per_threadgroup]]);

#define instantiate_col_reduce_general_no_atomics(name, itype, otype, op)   \
  template                                                                  \
      [[host_name("colGeneralNoAtomics_reduce_" #name)]] [[kernel]] void    \
      col_reduce_general_no_atomics<itype, otype, op>(                      \
          const device itype* in [[buffer(0)]],                             \
          device otype* out [[buffer(1)]],                                  \
          const constant size_t& reduction_size [[buffer(2)]],              \
          const constant size_t& reduction_stride [[buffer(3)]],            \
          const constant size_t& out_size [[buffer(4)]],                    \
          const constant int* shape [[buffer(5)]],                          \
          const constant size_t* strides [[buffer(6)]],                     \
          const constant int& ndim [[buffer(7)]],                           \
          threadgroup otype* local_data [[threadgroup(0)]],                 \
          uint3 tid [[threadgroup_position_in_grid]],                       \
          uint3 lid [[thread_position_in_threadgroup]],                     \
          uint3 gid [[thread_position_in_grid]],                            \
          uint3 lsize [[threads_per_threadgroup]],                          \
          uint3 gsize [[threads_per_grid]]);

#define instantiate_col_reduce_small(name, itype, otype, op)        \
  template [[host_name("colSmall_reduce_" #name)]] [[kernel]] void \
  col_reduce_small<itype, otype, op>(                               \
      const device itype* in [[buffer(0)]],                         \
      device otype* out [[buffer(1)]],                              \
      const constant size_t& reduction_size [[buffer(2)]],          \
      const constant size_t& reduction_stride [[buffer(3)]],        \
      const constant size_t& out_size [[buffer(4)]],                \
      const constant int* shape [[buffer(5)]],                      \
      const constant size_t* strides [[buffer(6)]],                 \
      const constant int& ndim [[buffer(7)]],                       \
      const constant size_t& non_col_reductions [[buffer(8)]],      \
      const constant int* non_col_shapes [[buffer(9)]],             \
      const constant size_t* non_col_strides [[buffer(10)]],        \
      const constant int& non_col_ndim [[buffer(11)]],              \
      uint tid [[thread_position_in_grid]]);

#define instantiate_same_col_reduce_helper(name, tname, type, op)  \
  instantiate_col_reduce_small(name ##tname, type, type, op<type>) \
  instantiate_col_reduce_general(name ##tname, type, type, op<type>)

#define instantiate_same_col_reduce_na_helper(name, tname, type, op) \
  instantiate_col_reduce_small(name ##tname, type, type, op<type>)   \
  instantiate_col_reduce_general_no_atomics(name ##tname, type, type, op<type>)

instantiate_reduce_ops(instantiate_same_col_reduce_helper, instantiate_reduce_helper_types)
instantiate_reduce_ops(instantiate_same_col_reduce_na_helper, instantiate_reduce_helper_64b)

instantiate_col_reduce_general(sumbool_, bool, uint32_t, Sum<uint32_t>)
instantiate_reduce_from_types(instantiate_col_reduce_general, and, bool, And<bool>)
instantiate_reduce_from_types(instantiate_col_reduce_general, or, bool, Or<bool>)

instantiate_col_reduce_small(sumbool_, bool, uint32_t, Sum<uint32_t>)
instantiate_reduce_from_types(instantiate_col_reduce_small, and, bool, And<bool>)
instantiate_reduce_from_types(instantiate_col_reduce_small, or, bool, Or<bool>)

#define instantiate_row_reduce_small(name, itype, otype, op)                \
  template [[host_name("rowGeneralSmall_reduce_" #name)]] [[kernel]] void   \
  row_reduce_general_small<itype, otype, op>(                               \
      const device itype* in [[buffer(0)]],                                 \
      device otype* out [[buffer(1)]],                                      \
      const constant size_t& reduction_size [[buffer(2)]],                  \
      const constant size_t& out_size [[buffer(3)]],                        \
      const constant size_t& non_row_reductions [[buffer(4)]],              \
      const constant int* shape [[buffer(5)]],                              \
      const constant size_t* strides [[buffer(6)]],                         \
      const constant int& ndim [[buffer(7)]],                               \
      uint lid [[thread_position_in_grid]]);                                \
  template [[host_name("rowGeneralMed_reduce_" #name)]] [[kernel]] void     \
  row_reduce_general_med<itype, otype, op>(                                 \
      const device itype* in [[buffer(0)]],                                 \
      device otype* out [[buffer(1)]],                                      \
      const constant size_t& reduction_size [[buffer(2)]],                  \
      const constant size_t& out_size [[buffer(3)]],                        \
      const constant size_t& non_row_reductions [[buffer(4)]],              \
      const constant int* shape [[buffer(5)]],                              \
      const constant size_t* strides [[buffer(6)]],                         \
      const constant int& ndim [[buffer(7)]],                               \
      uint tid [[threadgroup_position_in_grid]],                            \
      uint simd_lane_id [[thread_index_in_simdgroup]],                      \
      uint simd_per_group [[dispatch_simdgroups_per_threadgroup]],          \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_row_reduce_general(name, itype, otype, op)     \
  instantiate_row_reduce_small(name, itype, otype, op)             \
  template                                                         \
      [[host_name("rowSimple_reduce_" #name)]] [[kernel]] void     \
      row_reduce_simple<itype, otype, op>(                         \
          const device itype* in [[buffer(0)]],                    \
          device otype* out [[buffer(1)]],                         \
          const constant int& reduction_size [[buffer(2)]],        \
          const constant size_t& out_size [[buffer(3)]],           \
          uint3 gid [[threadgroup_position_in_grid]],              \
          uint3 gsize [[threadgroups_per_grid]],                   \
          uint3 lid [[thread_position_in_threadgroup]],            \
          uint3 lsize [[threads_per_threadgroup]],                 \
          uint simd_lane_id [[thread_index_in_simdgroup]],         \
          uint simd_per_group [[simdgroups_per_threadgroup]],      \
          uint simd_group_id [[simdgroup_index_in_threadgroup]]);  \
  template                                                         \
      [[host_name("rowLooped_reduce_" #name)]] [[kernel]] void     \
      row_reduce_looped<itype, otype, op>(                         \
          const device itype* in [[buffer(0)]],                    \
          device otype* out [[buffer(1)]],                         \
          const constant int& row_size [[buffer(2)]],              \
          const constant size_t& non_row_reductions [[buffer(3)]], \
          const constant int* shape [[buffer(4)]],                 \
          const constant size_t* strides [[buffer(5)]],            \
          const constant int& ndim [[buffer(6)]],                  \
          const constant int* reduce_shape [[buffer(7)]],          \
          const constant size_t* reduce_strides [[buffer(8)]],     \
          const constant int& reduce_ndim [[buffer(9)]],           \
          uint3 gid [[threadgroup_position_in_grid]],              \
          uint3 gsize [[threadgroups_per_grid]],                   \
          uint3 lid [[thread_position_in_threadgroup]],            \
          uint3 lsize [[threads_per_threadgroup]],                 \
          uint simd_lane_id [[thread_index_in_simdgroup]],         \
          uint simd_per_group [[simdgroups_per_threadgroup]],      \
          uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_row_reduce_general_no_atomics(name, itype, otype, op)   \
  instantiate_row_reduce_small(name, itype, otype, op)                      \
  template                                                                  \
      [[host_name("rowGeneralNoAtomics_reduce_" #name)]] [[kernel]] void    \
      row_reduce_general_no_atomics<itype, otype, op>(                      \
          const device itype* in [[buffer(0)]],                             \
          device otype* out [[buffer(1)]],                                  \
          const constant int& reduction_size [[buffer(2)]],              \
          const constant size_t& out_size [[buffer(3)]],                    \
          const constant size_t& non_row_reductions [[buffer(4)]],          \
          const constant int* shape [[buffer(5)]],                          \
          const constant size_t* strides [[buffer(6)]],                     \
          const constant int& ndim [[buffer(7)]],                           \
          uint3 lid [[thread_position_in_threadgroup]],                     \
          uint3 lsize [[threads_per_threadgroup]],                          \
          uint3 gsize [[threads_per_grid]],                                 \
          uint3 tid [[threadgroup_position_in_grid]],                       \
          uint simd_lane_id [[thread_index_in_simdgroup]],                  \
          uint simd_per_group [[simdgroups_per_threadgroup]],               \
          uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_same_row_reduce_helper(name, tname, type, op) \
  instantiate_row_reduce_general(name##tname, type, type, op<type>)

#define instantiate_same_row_reduce_na_helper(name, tname, type, op) \
  instantiate_row_reduce_general_no_atomics(name##tname, type, type, op<type>)

instantiate_reduce_ops(instantiate_same_row_reduce_helper, instantiate_reduce_helper_types)
instantiate_reduce_ops(instantiate_same_row_reduce_helper, instantiate_reduce_helper_64b)

instantiate_reduce_from_types(instantiate_row_reduce_general, and, bool, And<bool>)
instantiate_reduce_from_types(instantiate_row_reduce_general, or, bool, Or<bool>)

instantiate_row_reduce_general(sumbool_, bool, uint32_t, Sum<uint32_t>)
    // clang-format on
