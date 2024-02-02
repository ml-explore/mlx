// Copyright © 2023 Apple Inc.

#include <metal_integer>
#include <metal_math>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/bf16.h"

template <typename T>
[[kernel]] void select_predicate_op_ss(
    device const bool* a,
    device const T* b,
    device T* c,
    uint index [[thread_position_in_grid]]) {
  if (a[0]) c[index] = b[0];
}

template <typename T>
[[kernel]] void select_inverted_predicate_op_ss(
    device const bool* a,
    device const T* b,
    device T* c,
    uint index [[thread_position_in_grid]]) {
  if (!a[0]) c[index] = b[0];
}

template <typename T>
[[kernel]] void select_predicate_op_sv(
    device const bool* a,
    device const T* b,
    device T* c,
    uint index [[thread_position_in_grid]]) {
  if (a[0]) c[index] = b[index];
}

template <typename T>
[[kernel]] void select_inverted_predicate_op_sv(
    device const bool* a,
    device const T* b,
    device T* c,
    uint index [[thread_position_in_grid]]) {
  if (!a[0]) c[index] = b[index];
}

template <typename T>
[[kernel]] void select_predicate_op_vs(
    device const bool* a,
    device const T* b,
    device T* c,
    uint index [[thread_position_in_grid]]) {
  if (a[index]) c[index] = b[0];
}

template <typename T>
[[kernel]] void select_inverted_predicate_op_vs(
    device const bool* a,
    device const T* b,
    device T* c,
    uint index [[thread_position_in_grid]]) {
  if (!a[index]) c[index] = b[0];
}

template <typename T>
[[kernel]] void select_predicate_op_vv(
    device const bool* a,
    device const T* b,
    device T* c,
    uint index [[thread_position_in_grid]]) {
  if (a[index]) c[index] = b[index];
}

template <typename T>
[[kernel]] void select_inverted_predicate_op_vv(
    device const bool* a,
    device const T* b,
    device T* c,
    uint index [[thread_position_in_grid]]) {
  if (!a[index]) c[index] = b[index];
}

template <typename T>
[[kernel]] void select_predicate_op_g_nd1(
    device const bool* a,
    device const T* b,
    device T* c,
    constant const size_t& a_stride,
    constant const size_t& b_stride,
    uint index [[thread_position_in_grid]]) {
  auto a_idx = elem_to_loc_1(index, a_stride);
  auto b_idx = elem_to_loc_1(index, b_stride);
  if (a[a_idx]) c[index] = b[b_idx];
}

template <typename T>
[[kernel]] void select_inverted_predicate_op_g_nd1(
    device const bool* a,
    device const T* b,
    device T* c,
    constant const size_t& a_stride,
    constant const size_t& b_stride,
    uint index [[thread_position_in_grid]]) {
  auto a_idx = elem_to_loc_1(index, a_stride);
  auto b_idx = elem_to_loc_1(index, b_stride);
  if (!a[a_idx]) c[index] = b[b_idx];
}

template <typename T>
[[kernel]] void select_predicate_op_g_nd2(
    device const bool* a,
    device const T* b,
    device T* c,
    constant const size_t a_strides[2],
    constant const size_t b_strides[2],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  auto a_idx = elem_to_loc_2(index, a_strides);
  auto b_idx = elem_to_loc_2(index, b_strides);
  size_t out_idx = index.x + (size_t)grid_dim.x * index.y;
  if (a[a_idx]) c[out_idx] = b[b_idx];
}

template <typename T>
[[kernel]] void select_inverted_predicate_op_g_nd2(
    device const bool* a,
    device const T* b,
    device T* c,
    constant const size_t a_strides[2],
    constant const size_t b_strides[2],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  auto a_idx = elem_to_loc_2(index, a_strides);
  auto b_idx = elem_to_loc_2(index, b_strides);
  size_t out_idx = index.x + (size_t)grid_dim.x * index.y;
  if (!a[a_idx]) c[out_idx] = b[b_idx];
}

template <typename T>
[[kernel]] void select_predicate_op_g_nd3(
    device const bool* a,
    device const T* b,
    device T* c,
    constant const size_t a_strides[3],
    constant const size_t b_strides[3],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto a_idx = elem_to_loc_3(index, a_strides);
  auto b_idx = elem_to_loc_3(index, b_strides);
  size_t out_idx = index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
    if (a[a_idx]) c[out_idx] = b[b_idx];
}

template <typename T>
[[kernel]] void select_inverted_predicate_op_g_nd3(
    device const bool* a,
    device const T* b,
    device T* c,
    constant const size_t a_strides[3],
    constant const size_t b_strides[3],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto a_idx = elem_to_loc_3(index, a_strides);
  auto b_idx = elem_to_loc_3(index, b_strides);
  size_t out_idx = index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
    if (!a[a_idx]) c[out_idx] = b[b_idx];
}

template <typename T, int DIM>
[[kernel]] void select_predicate_op_g_nd(
    device const bool* a,
    device const T* b,
    device T* c,
    constant const int shape[DIM],
    constant const size_t a_strides[DIM],
    constant const size_t b_strides[DIM],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto idx = elem_to_loc_2_nd<DIM>(index, shape, a_strides, b_strides);
  size_t out_idx = index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
  if (a[idx.x]) c[out_idx] = b[idx.y];
}

template <typename T, int DIM>
[[kernel]] void select_inverted_predicate_op_g_nd(
    device const bool* a,
    device const T* b,
    device T* c,
    constant const int shape[DIM],
    constant const size_t a_strides[DIM],
    constant const size_t b_strides[DIM],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto idx = elem_to_loc_2_nd<DIM>(index, shape, a_strides, b_strides);
  size_t out_idx = index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
  if (!a[idx.x]) c[out_idx] = b[idx.y];
}

template <typename T>
[[kernel]] void select_predicate_op_g(
    device const bool* a,
    device const T* b,
    device T* c,
    constant const int* shape,
    constant const size_t* a_strides,
    constant const size_t* b_strides,
    constant const int& ndim,
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto idx = elem_to_loc_2_nd(index, shape, a_strides, b_strides, ndim);
  size_t out_idx = index.x + grid_dim.x * (index.y + grid_dim.y * index.z);
  if (a[idx.x]) c[out_idx] = b[idx.y];
}

template <typename T>
[[kernel]] void select_inverted_predicate_op_g(
    device const bool* a,
    device const T* b,
    device T* c,
    constant const int* shape,
    constant const size_t* a_strides,
    constant const size_t* b_strides,
    constant const int& ndim,
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto idx = elem_to_loc_2_nd(index, shape, a_strides, b_strides, ndim);
  size_t out_idx = index.x + grid_dim.x * (index.y + grid_dim.y * index.z);
  if (!a[idx.x]) c[out_idx] = b[idx.y];
}

#define instantiate_select(name, type, bopt) \
  template [[host_name(name)]] \
  [[kernel]] void select_predicate_op_##bopt<type>( \
      device const bool* a, \
      device const type* b, \
      device type* c, \
      uint index [[thread_position_in_grid]]); \
  template [[host_name(name "_invert_predicate")]] \
  [[kernel]] void select_inverted_predicate_op_##bopt<type>( \
      device const bool* a, \
      device const type* b, \
      device type* c, \
      uint index [[thread_position_in_grid]]); \

#define instantiate_select_g(name, type) \
  template [[host_name(name)]] \
  [[kernel]] void select_predicate_op_g<type>( \
      device const bool* a, \
      device const type* b, \
      device type* c, \
      constant const int* shape, \
      constant const size_t* a_strides, \
      constant const size_t* b_strides, \
      constant const int& ndim, \
      uint3 index [[thread_position_in_grid]], \
      uint3 grid_dim [[threads_per_grid]]); \
  template [[host_name(name "_invert_predicate")]] \
  [[kernel]] void select_inverted_predicate_op_g<type>( \
      device const bool* a, \
      device const type* b, \
      device type* c, \
      constant const int* shape, \
      constant const size_t* a_strides, \
      constant const size_t* b_strides, \
      constant const int& ndim, \
      uint3 index [[thread_position_in_grid]], \
      uint3 grid_dim [[threads_per_grid]]); \

#define instantiate_select_g_dim(name, type, dims) \
  template [[host_name(name "_" #dims)]] \
  [[kernel]] void select_predicate_op_g_nd<type, dims>( \
      device const bool* a, \
      device const type* b, \
      device type* c, \
      constant const int shape[dims], \
      constant const size_t a_strides[dims], \
      constant const size_t b_strides[dims], \
      uint3 index [[thread_position_in_grid]], \
      uint3 grid_dim [[threads_per_grid]]); \
        template [[host_name(name "_" #dims "_invert_predicate")]] \
  [[kernel]] void select_inverted_predicate_op_g_nd<type, dims>( \
      device const bool* a, \
      device const type* b, \
      device type* c, \
      constant const int shape[dims], \
      constant const size_t a_strides[dims], \
      constant const size_t b_strides[dims], \
      uint3 index [[thread_position_in_grid]], \
      uint3 grid_dim [[threads_per_grid]]); \

#define instantiate_select_g_nd(name, type) \
  template [[host_name(name "_1")]] \
  [[kernel]] void select_predicate_op_g_nd1<type>( \
      device const bool* a, \
      device const type* b, \
      device type* c, \
      constant const size_t& a_strides, \
      constant const size_t& b_strides, \
      uint index [[thread_position_in_grid]]); \
  template [[host_name(name "_1" "_invert_predicate")]] \
  [[kernel]] void select_inverted_predicate_op_g_nd1<type>( \
      device const bool* a, \
      device const type* b, \
      device type* c, \
      constant const size_t& a_strides, \
      constant const size_t& b_strides, \
      uint index [[thread_position_in_grid]]); \
  template [[host_name(name "_2")]] \
  [[kernel]] void select_predicate_op_g_nd2<type>( \
      device const bool* a, \
      device const type* b, \
      device type* c, \
      constant const size_t a_strides[2], \
      constant const size_t b_strides[2], \
      uint2 index [[thread_position_in_grid]], \
      uint2 grid_dim [[threads_per_grid]]); \
  template [[host_name(name "_2" "_invert_predicate")]] \
  [[kernel]] void select_inverted_predicate_op_g_nd2<type>( \
      device const bool* a, \
      device const type* b, \
      device type* c, \
      constant const size_t a_strides[2], \
      constant const size_t b_strides[2], \
      uint2 index [[thread_position_in_grid]], \
      uint2 grid_dim [[threads_per_grid]]); \
  template [[host_name(name "_3")]] \
  [[kernel]] void select_predicate_op_g_nd3<type>( \
      device const bool* a, \
      device const type* b, \
      device type* c, \
      constant const size_t a_strides[3], \
      constant const size_t b_strides[3], \
      uint3 index [[thread_position_in_grid]], \
      uint3 grid_dim [[threads_per_grid]]); \
  template [[host_name(name "_3" "_invert_predicate")]] \
  [[kernel]] void select_inverted_predicate_op_g_nd3<type>( \
      device const bool* a, \
      device const type* b, \
      device type* c, \
      constant const size_t a_strides[3], \
      constant const size_t b_strides[3], \
      uint3 index [[thread_position_in_grid]], \
      uint3 grid_dim [[threads_per_grid]]); \
  instantiate_select_g_dim(name, type, 4) \
  instantiate_select_g_dim(name, type, 5) \

#define instantiate_select_all(name, tname, type) \
  instantiate_select("select_ss" #name #tname, type, ss) \
  instantiate_select("select_sv" #name #tname, type, sv) \
  instantiate_select("select_vs" #name #tname, type, vs) \
  instantiate_select("select_vv" #name #tname, type, vv) \
  instantiate_select_g("select_g" #name #tname, type) \
  instantiate_select_g_nd("select_g" #name #tname, type) \

#define instantiate_select_float(name) \
  instantiate_select_all(name, float16, half) \
  instantiate_select_all(name, float32, float) \
  instantiate_select_all(name, bfloat16, bfloat16_t)

#define instantiate_select_types(name) \
  instantiate_select_all(name, bool_, bool) \
  instantiate_select_all(name, uint8, uint8_t) \
  instantiate_select_all(name, uint16, uint16_t) \
  instantiate_select_all(name, uint32, uint32_t) \
  instantiate_select_all(name, uint64, uint64_t) \
  instantiate_select_all(name, int8, int8_t) \
  instantiate_select_all(name, int16, int16_t) \
  instantiate_select_all(name, int32, int32_t) \
  instantiate_select_all(name, int64, int64_t) \
  instantiate_select_all(name, complex64, complex64_t) \
  instantiate_select_float(name)

instantiate_select_types(select)