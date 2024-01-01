// Copyright © 2023 Apple Inc.

#include <metal_integer>
#include <metal_math>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/bf16.h"

struct FloorDivide {
  template <typename T> T operator()(T x, T y) { return x / y; }
  template <> float operator()(float x, float y) { return trunc(x / y); }
  template <> half operator()(half x, half y) { return trunc(x / y); }
  template <> bfloat16_t operator()(bfloat16_t x, bfloat16_t y) { return trunc(x / y); }
};

struct Remainder {
  template <typename T> T operator()(T x, T y) { return x % y; }
  template <> float operator()(float x, float y) { return fmod(x, y); }
  template <> half operator()(half x, half y) { return fmod(x, y); }
  template <> bfloat16_t operator()(bfloat16_t x, bfloat16_t y) { return fmod(x, y); }
};

template <typename T, typename U, typename Op1, typename Op2>
[[kernel]] void binary_op_s2s(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    uint index [[thread_position_in_grid]]) {
  c[index] = Op1()(a[0], b[0]);
  d[index] = Op2()(a[0], b[0]);
}


template <typename T, typename U, typename Op1, typename Op2>
[[kernel]] void binary_op_ss(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    uint index [[thread_position_in_grid]]) {
  c[index] = Op1()(a[0], b[0]);
  d[index] = Op2()(a[0], b[0]);
}

template <typename T, typename U, typename Op1, typename Op2>
[[kernel]] void binary_op_sv(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    uint index [[thread_position_in_grid]]) {
  c[index] = Op1()(a[0], b[index]);
  d[index] = Op2()(a[0], b[index]);
}

template <typename T, typename U, typename Op1, typename Op2>
[[kernel]] void binary_op_vs(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    uint index [[thread_position_in_grid]]) {
  c[index] = Op1()(a[index], b[0]);
  d[index] = Op2()(a[index], b[0]);
}

template <typename T, typename U, typename Op1, typename Op2>
[[kernel]] void binary_op_vv(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    uint index [[thread_position_in_grid]]) {
  c[index] = Op1()(a[index], b[index]);
  d[index] = Op2()(a[index], b[index]);
}

template <typename T, typename U, typename Op1, typename Op2>
[[kernel]] void binary_op_g_nd1(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    constant const size_t& a_stride,
    constant const size_t& b_stride,
    uint index [[thread_position_in_grid]]) {
  auto a_idx = elem_to_loc_1(index, a_stride);
  auto b_idx = elem_to_loc_1(index, b_stride);
  c[index] = Op1()(a[a_idx], b[b_idx]);
  d[index] = Op2()(a[a_idx], b[b_idx]);
}

template <typename T, typename U, typename Op1, typename Op2>
[[kernel]] void binary_op_g_nd2(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    constant const size_t a_strides[2],
    constant const size_t b_strides[2],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  auto a_idx = elem_to_loc_2(index, a_strides);
  auto b_idx = elem_to_loc_2(index, b_strides);
  size_t out_idx = index.x + (size_t)grid_dim.x * index.y;
  c[out_idx] = Op1()(a[a_idx], b[b_idx]);
  d[out_idx] = Op2()(a[a_idx], b[b_idx]);
}

template <typename T, typename U, typename Op1, typename Op2>
[[kernel]] void binary_op_g_nd3(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    constant const size_t a_strides[3],
    constant const size_t b_strides[3],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto a_idx = elem_to_loc_3(index, a_strides);
  auto b_idx = elem_to_loc_3(index, b_strides);
  size_t out_idx = index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
  c[out_idx] = Op1()(a[a_idx], b[b_idx]);
  d[out_idx] = Op2()(a[a_idx], b[b_idx]);
}

template <typename T, typename U, typename Op1, typename Op2, int DIM>
[[kernel]] void binary_op_g_nd(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    constant const int shape[DIM],
    constant const size_t a_strides[DIM],
    constant const size_t b_strides[DIM],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto idx = elem_to_loc_2_nd<DIM>(index, shape, a_strides, b_strides);
  size_t out_idx = index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
  c[out_idx] = Op1()(a[idx.x], b[idx.y]);
  d[out_idx] = Op2()(a[idx.x], b[idx.y]);
}

template <typename T, typename U, typename Op1, typename Op2>
[[kernel]] void binary_op_g(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    constant const int* shape,
    constant const size_t* a_strides,
    constant const size_t* b_strides,
    constant const int& ndim,
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto idx = elem_to_loc_2_nd(index, shape, a_strides, b_strides, ndim);
  size_t out_idx = index.x + grid_dim.x * (index.y + grid_dim.y * index.z);
  c[out_idx] = Op1()(a[idx.x], b[idx.y]);
  d[out_idx] = Op2()(a[idx.x], b[idx.y]);
}

#define instantiate_binary(name, itype, otype, op1, op2, bopt) \
  template [[host_name(name)]] \
  [[kernel]] void binary_op_##bopt<itype, otype, op1, op2>( \
      device const itype* a, \
      device const itype* b, \
      device otype* c, \
      device otype* d, \
      uint index [[thread_position_in_grid]]);

#define instantiate_binary_g_dim(name, itype, otype, op1, op2, dims) \
  template [[host_name(name "_" #dims)]] \
  [[kernel]] void binary_op_g_nd<itype, otype, op1, op2, dims>( \
      device const itype* a, \
      device const itype* b, \
      device otype* c, \
      device otype* d, \
      constant const int shape[dims], \
      constant const size_t a_strides[dims], \
      constant const size_t b_strides[dims], \
      uint3 index [[thread_position_in_grid]], \
      uint3 grid_dim [[threads_per_grid]]);

#define instantiate_binary_g_nd(name, itype, otype, op1, op2) \
  template [[host_name(name "_1")]] \
  [[kernel]] void binary_op_g_nd1<itype, otype, op1, op2>( \
      device const itype* a, \
      device const itype* b, \
      device otype* c, \
      device otype* d, \
      constant const size_t& a_stride, \
      constant const size_t& b_stride, \
      uint index [[thread_position_in_grid]]); \
  template [[host_name(name "_2")]] \
  [[kernel]] void binary_op_g_nd2<itype, otype, op1, op2>( \
      device const itype* a, \
      device const itype* b, \
      device otype* c, \
      device otype* d, \
      constant const size_t a_strides[2], \
      constant const size_t b_strides[2], \
      uint2 index [[thread_position_in_grid]], \
      uint2 grid_dim [[threads_per_grid]]); \
  template [[host_name(name "_3")]] \
  [[kernel]] void binary_op_g_nd3<itype, otype, op1, op2>( \
      device const itype* a, \
      device const itype* b, \
      device otype* c, \
      device otype* d, \
      constant const size_t a_strides[3], \
      constant const size_t b_strides[3], \
      uint3 index [[thread_position_in_grid]], \
      uint3 grid_dim [[threads_per_grid]]); \
  instantiate_binary_g_dim(name, itype, otype, op1, op2, 4) \
  instantiate_binary_g_dim(name, itype, otype, op1, op2, 5)


#define instantiate_binary_g(name, itype, otype, op1, op2) \
  template [[host_name(name)]] \
  [[kernel]] void binary_op_g<itype, otype, op2, op2>( \
      device const itype* a, \
      device const itype* b, \
      device otype* c, \
      device otype* d, \
      constant const int* shape, \
      constant const size_t* a_strides, \
      constant const size_t* b_strides, \
      constant const int& ndim, \
      uint3 index [[thread_position_in_grid]], \
      uint3 grid_dim [[threads_per_grid]]);

#define instantiate_binary_all(name, tname, itype, otype, op1, op2) \
  instantiate_binary("ss" #name #tname, itype, otype, op1, op2, ss) \
  instantiate_binary("sv" #name #tname, itype, otype, op1, op2, sv) \
  instantiate_binary("vs" #name #tname, itype, otype, op1, op2, vs) \
  instantiate_binary("vv" #name #tname, itype, otype, op1, op2, vv) \
  instantiate_binary_g("g" #name #tname, itype, otype, op1, op2) \
  instantiate_binary_g_nd("g" #name #tname, itype, otype, op1, op2)

#define instantiate_binary_float(name, op1, op2) \
  instantiate_binary_all(name, float16, half, half, op1, op2) \
  instantiate_binary_all(name, float32, float, float, op1, op2) \
  instantiate_binary_all(name, bfloat16, bfloat16_t, bfloat16_t, op1, op2)

#define instantiate_binary_types(name, op1, op2) \
  instantiate_binary_all(name, bool_, bool, bool, op1, op2) \
  instantiate_binary_all(name, uint8, uint8_t, uint8_t, op1, op2) \
  instantiate_binary_all(name, uint16, uint16_t, uint16_t, op1, op2) \
  instantiate_binary_all(name, uint32, uint32_t, uint32_t, op1, op2) \
  instantiate_binary_all(name, uint64, uint64_t, uint64_t, op1, op2) \
  instantiate_binary_all(name, int8, int8_t, int8_t, op1, op2) \
  instantiate_binary_all(name, int16, int16_t, int16_t, op1, op2) \
  instantiate_binary_all(name, int32, int32_t, int32_t, op1, op2) \
  instantiate_binary_all(name, int64, int64_t, int64_t, op1, op2) \
  instantiate_binary_all(name, complex64, complex64_t, complex64_t, op1, op2) \
  instantiate_binary_float(name, op1, op2)

instantiate_binary_types(divmod, FloorDivide, Remainder)
