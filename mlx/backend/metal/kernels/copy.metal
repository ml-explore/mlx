// Copyright © 2024 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/copy.h"

#define instantiate_copy(name, itype, otype, ctype)                        \
  template [[host_name(name)]] [[kernel]] void copy_##ctype<itype, otype>( \
      device const itype* src [[buffer(0)]],                               \
      device otype* dst [[buffer(1)]],                                     \
      uint index [[thread_position_in_grid]]);

#define instantiate_copy_g_dim(name, itype, otype, dims)      \
  template [[host_name("g" #dims "_" name)]] [[kernel]] void  \
  copy_g_nd<itype, otype, dims>(                              \
      device const itype* src [[buffer(0)]],                  \
      device otype* dst [[buffer(1)]],                        \
      constant const int* src_shape [[buffer(2)]],            \
      constant const int64_t* src_strides [[buffer(3)]],      \
      uint3 index [[thread_position_in_grid]],                \
      uint3 grid_dim [[threads_per_grid]]);                   \
  template [[host_name("gg" #dims "_" name)]] [[kernel]] void \
  copy_gg_nd<itype, otype, dims>(                             \
      device const itype* src [[buffer(0)]],                  \
      device otype* dst [[buffer(1)]],                        \
      constant const int* src_shape [[buffer(2)]],            \
      constant const int64_t* src_strides [[buffer(3)]],      \
      constant const int64_t* dst_strides [[buffer(4)]],      \
      uint3 index [[thread_position_in_grid]]);

#define instantiate_copy_g_nd(name, itype, otype)                              \
  template [[host_name("g1_" name)]] [[kernel]] void copy_g_nd1<itype, otype>( \
      device const itype* src [[buffer(0)]],                                   \
      device otype* dst [[buffer(1)]],                                         \
      constant const int64_t& src_stride [[buffer(3)]],                        \
      uint index [[thread_position_in_grid]]);                                 \
  template [[host_name("g2_" name)]] [[kernel]] void copy_g_nd2<itype, otype>( \
      device const itype* src [[buffer(0)]],                                   \
      device otype* dst [[buffer(1)]],                                         \
      constant const int64_t* src_strides [[buffer(3)]],                       \
      uint2 index [[thread_position_in_grid]],                                 \
      uint2 grid_dim [[threads_per_grid]]);                                    \
  template [[host_name("g3_" name)]] [[kernel]] void copy_g_nd3<itype, otype>( \
      device const itype* src [[buffer(0)]],                                   \
      device otype* dst [[buffer(1)]],                                         \
      constant const int64_t* src_strides [[buffer(3)]],                       \
      uint3 index [[thread_position_in_grid]],                                 \
      uint3 grid_dim [[threads_per_grid]]);                                    \
  template [[host_name("gg1_" name )]] [[kernel]] void                         \
  copy_gg_nd1<itype, otype>(                                                   \
      device const itype* src [[buffer(0)]],                                   \
      device otype* dst [[buffer(1)]],                                         \
      constant const int64_t& src_stride [[buffer(3)]],                        \
      constant const int64_t& dst_stride [[buffer(4)]],                        \
      uint index [[thread_position_in_grid]]);                                 \
  template [[host_name("gg2_" name)]] [[kernel]] void                          \
  copy_gg_nd2<itype, otype>(                                                   \
      device const itype* src [[buffer(0)]],                                   \
      device otype* dst [[buffer(1)]],                                         \
      constant const int64_t* src_strides [[buffer(3)]],                       \
      constant const int64_t* dst_strides [[buffer(4)]],                       \
      uint2 index [[thread_position_in_grid]]);                                \
  template [[host_name("gg3_" name)]] [[kernel]] void                          \
  copy_gg_nd3<itype, otype>(                                                   \
      device const itype* src [[buffer(0)]],                                   \
      device otype* dst [[buffer(1)]],                                         \
      constant const int64_t* src_strides [[buffer(3)]],                       \
      constant const int64_t* dst_strides [[buffer(4)]],                       \
      uint3 index [[thread_position_in_grid]]);                                \
  instantiate_copy_g_dim(name, itype, otype, 4)                                \
  instantiate_copy_g_dim(name, itype, otype, 5)

#define instantiate_copy_g(name, itype, otype)                              \
  template [[host_name("g_" name)]] [[kernel]] void copy_g<itype, otype>(   \
      device const itype* src [[buffer(0)]],                                \
      device otype* dst [[buffer(1)]],                                      \
      constant const int* src_shape [[buffer(2)]],                          \
      constant const int64_t* src_strides [[buffer(3)]],                    \
      constant const int& ndim [[buffer(5)]],                               \
      uint3 index [[thread_position_in_grid]],                              \
      uint3 grid_dim [[threads_per_grid]]);                                 \
  template [[host_name("gg_" name)]] [[kernel]] void copy_gg<itype, otype>( \
      device const itype* src [[buffer(0)]],                                \
      device otype* dst [[buffer(1)]],                                      \
      constant const int* src_shape [[buffer(2)]],                          \
      constant const int64_t* src_strides [[buffer(3)]],                    \
      constant const int64_t* dst_strides [[buffer(4)]],                    \
      constant const int& ndim [[buffer(5)]],                               \
      uint3 index [[thread_position_in_grid]]);

#define instantiate_copy_all(tname, itype, otype)    \
  instantiate_copy("s_copy" #tname, itype, otype, s) \
  instantiate_copy("v_copy" #tname, itype, otype, v) \
  instantiate_copy_g("copy" #tname, itype, otype)    \
  instantiate_copy_g_nd("copy" #tname, itype, otype)

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
