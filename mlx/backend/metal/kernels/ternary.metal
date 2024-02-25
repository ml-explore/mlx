// Copyright © 2023 Apple Inc.

#include <metal_integer>
#include <metal_math>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/ternary.h"

template <typename T, typename Op>
[[kernel]] void ternary_op_v(
    device const bool* a,
    device const T* b,
    device const T* c,
    device T* d,
    uint index [[thread_position_in_grid]]) {
  d[index] = Op()(a[index], b[index], c[index]);
}

template <typename T, typename Op>
[[kernel]] void ternary_op_g_nd1(
    device const bool* a,
    device const T* b,
    device const T* c,
    device T* d,
    constant const size_t& a_strides,
    constant const size_t& b_strides,
    constant const size_t& c_strides,
    uint index [[thread_position_in_grid]]) {
  auto a_idx = elem_to_loc_1(index, a_strides);
  auto b_idx = elem_to_loc_1(index, b_strides);
  auto c_idx = elem_to_loc_1(index, c_strides);
  d[index] = Op()(a[a_idx], b[b_idx], c[c_idx]);
}

template <typename T, typename Op>
[[kernel]] void ternary_op_g_nd2(
    device const bool* a,
    device const T* b,
    device const T* c,
    device T* d,
    constant const size_t a_strides[2],
    constant const size_t b_strides[2],
    constant const size_t c_strides[2],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  auto a_idx = elem_to_loc_2(index, a_strides);
  auto b_idx = elem_to_loc_2(index, b_strides);
  auto c_idx = elem_to_loc_2(index, c_strides);
  size_t out_idx = index.x + (size_t)grid_dim.x * index.y;
  d[out_idx] = Op()(a[a_idx], b[b_idx], c[c_idx]);
}

template <typename T, typename Op>
[[kernel]] void ternary_op_g_nd3(
    device const bool* a,
    device const T* b,
    device const T* c,
    device T* d,
    constant const size_t a_strides[3],
    constant const size_t b_strides[3],
    constant const size_t c_strides[3],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto a_idx = elem_to_loc_3(index, a_strides);
  auto b_idx = elem_to_loc_3(index, b_strides);
  auto c_idx = elem_to_loc_3(index, c_strides);
  size_t out_idx = index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
  d[out_idx] = Op()(a[a_idx], b[b_idx], c[c_idx]);
}

template <typename T, typename Op, int DIM>
[[kernel]] void ternary_op_g_nd(
    device const bool* a,
    device const T* b,
    device const T* c,
    device T* d,
    constant const int shape[DIM],
    constant const size_t a_strides[DIM],
    constant const size_t b_strides[DIM],
    constant const size_t c_strides[DIM],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto idx = elem_to_loc_3_nd<DIM>(index, shape, a_strides, b_strides, c_strides);
  size_t out_idx = index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
  d[out_idx] = Op()(a[idx.x], b[idx.y], c[idx.z]);
}

template <typename T, typename Op>
[[kernel]] void ternary_op_g(
    device const bool* a,
    device const T* b,
    device const T* c,
    device T* d,
    constant const int* shape,
    constant const size_t* a_strides,
    constant const size_t* b_strides,
    constant const size_t* c_strides,
    constant const int& ndim,
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto idx = elem_to_loc_3_nd(index, shape, a_strides, b_strides, c_strides, ndim);
  size_t out_idx = index.x + grid_dim.x * (index.y + grid_dim.y * index.z);
  d[out_idx] = Op()(a[idx.x], b[idx.y], c[idx.z]);
}

#define instantiate_ternary_v(name, type, op) \
  template [[host_name(name)]] \
  [[kernel]] void ternary_op_v<type, op>( \
      device const bool* a, \
      device const type* b, \
      device const type* c, \
      device type* d, \
      uint index [[thread_position_in_grid]]); \

#define instantiate_ternary_g(name, type, op) \
  template [[host_name(name)]] \
  [[kernel]] void ternary_op_g<type, op>( \
      device const bool* a, \
      device const type* b, \
      device const type* c, \
      device type* d, \
      constant const int* shape, \
      constant const size_t* a_strides, \
      constant const size_t* b_strides, \
      constant const size_t* c_strides, \
      constant const int& ndim, \
      uint3 index [[thread_position_in_grid]], \
      uint3 grid_dim [[threads_per_grid]]); \

#define instantiate_ternary_g_dim(name, type, op, dims) \
  template [[host_name(name "_" #dims)]] \
  [[kernel]] void ternary_op_g_nd<type, op, dims>( \
      device const bool* a, \
      device const type* b, \
      device const type* c, \
      device type* d, \
      constant const int shape[dims], \
      constant const size_t a_strides[dims], \
      constant const size_t b_strides[dims], \
      constant const size_t c_strides[dims], \
      uint3 index [[thread_position_in_grid]], \
      uint3 grid_dim [[threads_per_grid]]); \

#define instantiate_ternary_g_nd(name, type, op) \
  template [[host_name(name "_1")]] \
  [[kernel]] void ternary_op_g_nd1<type, op>( \
      device const bool* a, \
      device const type* b, \
      device const type* c, \
      device type* d, \
      constant const size_t& a_strides, \
      constant const size_t& b_strides, \
      constant const size_t& c_strides, \
      uint index [[thread_position_in_grid]]); \
  template [[host_name(name "_2")]] \
  [[kernel]] void ternary_op_g_nd2<type, op>( \
      device const bool* a, \
      device const type* b, \
      device const type* c, \
      device type* d, \
      constant const size_t a_strides[2], \
      constant const size_t b_strides[2], \
      constant const size_t c_strides[2], \
      uint2 index [[thread_position_in_grid]], \
      uint2 grid_dim [[threads_per_grid]]); \
  template [[host_name(name "_3")]] \
  [[kernel]] void ternary_op_g_nd3<type, op>( \
      device const bool* a, \
      device const type* b, \
      device const type* c, \
      device type* d, \
      constant const size_t a_strides[3], \
      constant const size_t b_strides[3], \
      constant const size_t c_strides[3], \
      uint3 index [[thread_position_in_grid]], \
      uint3 grid_dim [[threads_per_grid]]); \
  instantiate_ternary_g_dim(name, type, op, 4) \
  instantiate_ternary_g_dim(name, type, op, 5) \

#define instantiate_ternary_all(name, tname, type, op) \
  instantiate_ternary_v("v" #name #tname, type, op) \
  instantiate_ternary_g("g" #name #tname, type, op) \
  instantiate_ternary_g_nd("g" #name #tname, type, op) \

#define instantiate_ternary_types(name, op) \
  instantiate_ternary_all(name, bool_, bool, op) \
  instantiate_ternary_all(name, uint8, uint8_t, op) \
  instantiate_ternary_all(name, uint16, uint16_t, op) \
  instantiate_ternary_all(name, uint32, uint32_t, op) \
  instantiate_ternary_all(name, uint64, uint64_t, op) \
  instantiate_ternary_all(name, int8, int8_t, op) \
  instantiate_ternary_all(name, int16, int16_t, op) \
  instantiate_ternary_all(name, int32, int32_t, op) \
  instantiate_ternary_all(name, int64, int64_t, op) \
  instantiate_ternary_all(name, float16, half, op) \
  instantiate_ternary_all(name, float32, float, op) \
  instantiate_ternary_all(name, bfloat16, bfloat16_t, op) \
  instantiate_ternary_all(name, complex64, complex64_t, op) \

instantiate_ternary_types(select, Select)
