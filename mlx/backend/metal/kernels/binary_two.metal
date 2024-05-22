// Copyright © 2024 Apple Inc.
#include <metal_integer>
#include <metal_math>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/binary_ops.h"
#include "mlx/backend/metal/kernels/binary_two.h"

#define instantiate_binary(name, itype, otype, op, bopt)       \
  template [[host_name(name)]] [[kernel]] void                 \
      binary_##bopt<itype, otype, op>(                         \
          device const itype* a,                               \
          device const itype* b,                               \
          device otype* c,                                     \
          device otype* d,                                     \
          uint index [[thread_position_in_grid]]);

#define instantiate_binary_g_dim(name, itype, otype, op, dims)       \
  template [[host_name("g" #dims name)]] [[kernel]] void             \
  binary_g_nd<itype, otype, op, dims>(                               \
      device const itype* a,                                         \
      device const itype* b,                                         \
      device otype* c,                                               \
      device otype* d,                                               \
      constant const int shape[dims],                                \
      constant const size_t a_strides[dims],                         \
      constant const size_t b_strides[dims],                         \
      uint3 index [[thread_position_in_grid]],                       \
      uint3 grid_dim [[threads_per_grid]]);

#define instantiate_binary_g_nd(name, itype, otype, op)       \
  template [[host_name("g1" name)]] [[kernel]] void           \
  binary_g_nd1<itype, otype, op>(                             \
      device const itype* a,                                  \
      device const itype* b,                                  \
      device otype* c,                                        \
      device otype* d,                                        \
      constant const size_t& a_stride,                        \
      constant const size_t& b_stride,                        \
      uint index [[thread_position_in_grid]]);                \
  template [[host_name("g2" name)]] [[kernel]] void           \
  binary_g_nd2<itype, otype, op>(                             \
      device const itype* a,                                  \
      device const itype* b,                                  \
      device otype* c,                                        \
      device otype* d,                                        \
      constant const size_t a_strides[2],                     \
      constant const size_t b_strides[2],                     \
      uint2 index [[thread_position_in_grid]],                \
      uint2 grid_dim [[threads_per_grid]]);                   \
  template [[host_name("g3" name)]] [[kernel]] void           \
  binary_g_nd3<itype, otype, op>(                             \
      device const itype* a,                                  \
      device const itype* b,                                  \
      device otype* c,                                        \
      device otype* d,                                        \
      constant const size_t a_strides[3],                     \
      constant const size_t b_strides[3],                     \
      uint3 index [[thread_position_in_grid]],                \
      uint3 grid_dim [[threads_per_grid]]);                   \
  instantiate_binary_g_dim(name, itype, otype, op, 4)         \
  instantiate_binary_g_dim(name, itype, otype, op, 5)

#define instantiate_binary_g(name, itype, otype, op)       \
  template [[host_name("gn" name)]] [[kernel]] void        \
  binary_g<itype, otype, op>(                              \
      device const itype* a,                               \
      device const itype* b,                               \
      device otype* c,                                     \
      device otype* d,                                     \
      constant const int* shape,                           \
      constant const size_t* a_strides,                    \
      constant const size_t* b_strides,                    \
      constant const int& ndim,                            \
      uint3 index [[thread_position_in_grid]],             \
      uint3 grid_dim [[threads_per_grid]]);

#define instantiate_binary_all(name, tname, itype, otype, op) \
  instantiate_binary("ss" #name #tname, itype, otype, op, ss) \
  instantiate_binary("sv" #name #tname, itype, otype, op, sv) \
  instantiate_binary("vs" #name #tname, itype, otype, op, vs) \
  instantiate_binary("vv" #name #tname, itype, otype, op, vv) \
  instantiate_binary_g(#name #tname, itype, otype, op)    \
  instantiate_binary_g_nd(#name #tname, itype, otype, op)

#define instantiate_binary_float(name, op)                \
  instantiate_binary_all(name, float16, half, half, op)   \
  instantiate_binary_all(name, float32, float, float, op) \
  instantiate_binary_all(name, bfloat16, bfloat16_t, bfloat16_t, op)

#define instantiate_binary_types(name, op)                              \
  instantiate_binary_all(name, bool_, bool, bool, op)                   \
  instantiate_binary_all(name, uint8, uint8_t, uint8_t, op)             \
  instantiate_binary_all(name, uint16, uint16_t, uint16_t, op)          \
  instantiate_binary_all(name, uint32, uint32_t, uint32_t, op)          \
  instantiate_binary_all(name, uint64, uint64_t, uint64_t, op)          \
  instantiate_binary_all(name, int8, int8_t, int8_t, op)                \
  instantiate_binary_all(name, int16, int16_t, int16_t, op)             \
  instantiate_binary_all(name, int32, int32_t, int32_t, op)             \
  instantiate_binary_all(name, int64, int64_t, int64_t, op)             \
  instantiate_binary_all(name, complex64, complex64_t, complex64_t, op) \
  instantiate_binary_float(name, op)

instantiate_binary_types(divmod, DivMod) // clang-format on
