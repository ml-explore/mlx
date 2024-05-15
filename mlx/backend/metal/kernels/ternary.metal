// Copyright © 2024 Apple Inc.

#include <metal_integer>
#include <metal_math>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/ternary.h"
#include "mlx/backend/metal/kernels/ternarys.h"
#include "mlx/backend/metal/kernels/utils.h"

// clang-format off
#define instantiate_ternary_v(name, type, op)                          \
  template [[host_name("v_" name)]] [[kernel]] void ternary_v<type, op>( \
      device const bool* a,                                            \
      device const type* b,                                            \
      device const type* c,                                            \
      device type* d,                                                  \
      uint index [[thread_position_in_grid]]);

#define instantiate_ternary_g(name, type, op)                          \
  template [[host_name("g_" name)]] [[kernel]] void ternary_g<type, op>( \
      device const bool* a,                                            \
      device const type* b,                                            \
      device const type* c,                                            \
      device type* d,                                                  \
      constant const int* shape,                                       \
      constant const size_t* a_strides,                                \
      constant const size_t* b_strides,                                \
      constant const size_t* c_strides,                                \
      constant const int& ndim,                                        \
      uint3 index [[thread_position_in_grid]],                         \
      uint3 grid_dim [[threads_per_grid]]);

#define instantiate_ternary_g_dim(name, type, op, dims)  \
  template [[host_name("g" #dims "_" name )]] [[kernel]] void \
  ternary_g_nd<type, op, dims>(                       \
      device const bool* a,                              \
      device const type* b,                              \
      device const type* c,                              \
      device type* d,                                    \
      constant const int shape[dims],                    \
      constant const size_t a_strides[dims],             \
      constant const size_t b_strides[dims],             \
      constant const size_t c_strides[dims],             \
      uint3 index [[thread_position_in_grid]],           \
      uint3 grid_dim [[threads_per_grid]]);

#define instantiate_ternary_g_nd(name, type, op)    \
  template [[host_name("g1_" name)]] [[kernel]] void \
  ternary_g_nd1<type, op>(                       \
      device const bool* a,                         \
      device const type* b,                         \
      device const type* c,                         \
      device type* d,                               \
      constant const size_t& a_strides,             \
      constant const size_t& b_strides,             \
      constant const size_t& c_strides,             \
      uint index [[thread_position_in_grid]]);      \
  template [[host_name("g2_" name)]] [[kernel]] void \
  ternary_g_nd2<type, op>(                       \
      device const bool* a,                         \
      device const type* b,                         \
      device const type* c,                         \
      device type* d,                               \
      constant const size_t a_strides[2],           \
      constant const size_t b_strides[2],           \
      constant const size_t c_strides[2],           \
      uint2 index [[thread_position_in_grid]],      \
      uint2 grid_dim [[threads_per_grid]]);         \
  template [[host_name("g3_" name)]] [[kernel]] void \
  ternary_g_nd3<type, op>(                       \
      device const bool* a,                         \
      device const type* b,                         \
      device const type* c,                         \
      device type* d,                               \
      constant const size_t a_strides[3],           \
      constant const size_t b_strides[3],           \
      constant const size_t c_strides[3],           \
      uint3 index [[thread_position_in_grid]],      \
      uint3 grid_dim [[threads_per_grid]]);         \
  instantiate_ternary_g_dim(name, type, op, 4)      \
  instantiate_ternary_g_dim(name, type, op, 5)

#define instantiate_ternary_all(name, tname, type, op) \
  instantiate_ternary_v(#name #tname, type, op)    \
  instantiate_ternary_g(#name #tname, type, op)    \
  instantiate_ternary_g_nd(#name #tname, type, op)

#define instantiate_ternary_types(name, op)               \
  instantiate_ternary_all(name, bool_, bool, op)          \
  instantiate_ternary_all(name, uint8, uint8_t, op)       \
  instantiate_ternary_all(name, uint16, uint16_t, op)     \
  instantiate_ternary_all(name, uint32, uint32_t, op)     \
  instantiate_ternary_all(name, uint64, uint64_t, op)     \
  instantiate_ternary_all(name, int8, int8_t, op)         \
  instantiate_ternary_all(name, int16, int16_t, op)       \
  instantiate_ternary_all(name, int32, int32_t, op)       \
  instantiate_ternary_all(name, int64, int64_t, op)       \
  instantiate_ternary_all(name, float16, half, op)        \
  instantiate_ternary_all(name, float32, float, op)       \
  instantiate_ternary_all(name, bfloat16, bfloat16_t, op) \
  instantiate_ternary_all(name, complex64, complex64_t, op) // clang-format on

instantiate_ternary_types(select, Select)
