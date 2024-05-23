// Copyright © 2023-2024 Apple Inc.

#include <metal_math>
#include <metal_simdgroup>

// clang-format off

using namespace metal;

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/scan.h"

#define instantiate_contiguous_scan(                                    \
    name, itype, otype, op, inclusive, reverse, nreads)                 \
  template [[host_name("contig_scan_" #name)]] [[kernel]] void      \
  contiguous_scan<itype, otype, op<otype>, nreads, inclusive, reverse>( \
      const device itype* in [[buffer(0)]],                             \
      device otype* out [[buffer(1)]],                                  \
      const constant size_t& axis_size [[buffer(2)]],                   \
      uint gid [[thread_position_in_grid]],                             \
      uint lid [[thread_position_in_threadgroup]],                      \
      uint lsize [[threads_per_threadgroup]],                           \
      uint simd_size [[threads_per_simdgroup]],                         \
      uint simd_lane_id [[thread_index_in_simdgroup]],                  \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_strided_scan(                                    \
    name, itype, otype, op, inclusive, reverse, nreads)              \
  template [[host_name("strided_scan_" #name)]] [[kernel]] void      \
  strided_scan<itype, otype, op<otype>, nreads, inclusive, reverse>( \
      const device itype* in [[buffer(0)]],                          \
      device otype* out [[buffer(1)]],                               \
      const constant size_t& axis_size [[buffer(2)]],                \
      const constant size_t& stride [[buffer(3)]],                   \
      uint2 gid [[thread_position_in_grid]],                         \
      uint2 lid [[thread_position_in_threadgroup]],                  \
      uint2 lsize [[threads_per_threadgroup]],                       \
      uint simd_size [[threads_per_simdgroup]]);

#define instantiate_scan_helper(name, itype, otype, op, nreads)                                \
  instantiate_contiguous_scan(inclusive_##name, itype, otype, op, true, false, nreads)         \
  instantiate_contiguous_scan(exclusive_##name, itype, otype, op, false, false, nreads)        \
  instantiate_contiguous_scan(reverse_inclusive_##name, itype, otype, op, true, true, nreads)  \
  instantiate_contiguous_scan(reverse_exclusive_##name, itype, otype, op, false, true, nreads) \
  instantiate_strided_scan(inclusive_##name, itype, otype, op, true, false, nreads)            \
  instantiate_strided_scan(exclusive_##name, itype, otype, op, false, false, nreads)           \
  instantiate_strided_scan(reverse_inclusive_##name, itype, otype, op, true, true, nreads)     \
  instantiate_strided_scan(reverse_exclusive_##name, itype, otype, op, false, true, nreads)

instantiate_scan_helper(sum_bool__int32,         bool,        int32_t,     CumSum, 4)
instantiate_scan_helper(sum_uint8_uint8,         uint8_t,     uint8_t,     CumSum, 4)
instantiate_scan_helper(sum_uint16_uint16,       uint16_t,    uint16_t,    CumSum, 4)
instantiate_scan_helper(sum_uint32_uint32,       uint32_t,    uint32_t,    CumSum, 4)
//instantiate_scan_helper(sum_uint64_uint64,       uint64_t,    uint64_t,    CumSum, 2)
instantiate_scan_helper(sum_int8_int8,           int8_t,      int8_t,      CumSum, 4)
instantiate_scan_helper(sum_int16_int16,         int16_t,     int16_t,     CumSum, 4)
instantiate_scan_helper(sum_int32_int32,         int32_t,     int32_t,     CumSum, 4)
//instantiate_scan_helper(sum_int64_int64,         int64_t,     int64_t,     CumSum, 2)
instantiate_scan_helper(sum_float16_float16,     half,        half,        CumSum, 4)
instantiate_scan_helper(sum_float32_float32,     float,       float,       CumSum, 4)
instantiate_scan_helper(sum_bfloat16_bfloat16,   bfloat16_t,  bfloat16_t,  CumSum, 4)
//instantiate_scan_helper(sum_complex64_complex64, complex64_t, complex64_t, CumSum)
//instantiate_scan_helper(prod_bool__bool_,         bool,        bool,        CumProd, 4)
instantiate_scan_helper(prod_uint8_uint8,         uint8_t,     uint8_t,     CumProd, 4)
instantiate_scan_helper(prod_uint16_uint16,       uint16_t,    uint16_t,    CumProd, 4)
instantiate_scan_helper(prod_uint32_uint32,       uint32_t,    uint32_t,    CumProd, 4)
//instantiate_scan_helper(prod_uint64_uint64,       uint64_t,    uint64_t,    CumProd, 2)
instantiate_scan_helper(prod_int8_int8,           int8_t,      int8_t,      CumProd, 4)
instantiate_scan_helper(prod_int16_int16,         int16_t,     int16_t,     CumProd, 4)
instantiate_scan_helper(prod_int32_int32,         int32_t,     int32_t,     CumProd, 4)
//instantiate_scan_helper(prod_int64_int64,         int64_t,     int64_t,     CumProd, 2)
instantiate_scan_helper(prod_float16_float16,     half,        half,        CumProd, 4)
instantiate_scan_helper(prod_float32_float32,     float,       float,       CumProd, 4)
instantiate_scan_helper(prod_bfloat16_bfloat16,   bfloat16_t,  bfloat16_t,  CumProd, 4)
//instantiate_scan_helper(prod_complex64_complex64, complex64_t, complex64_t, CumProd)
//instantiate_scan_helper(max_bool__bool_,         bool,        bool,        CumMax, 4)
instantiate_scan_helper(max_uint8_uint8,         uint8_t,     uint8_t,     CumMax, 4)
instantiate_scan_helper(max_uint16_uint16,       uint16_t,    uint16_t,    CumMax, 4)
instantiate_scan_helper(max_uint32_uint32,       uint32_t,    uint32_t,    CumMax, 4)
//instantiate_scan_helper(max_uint64_uint64,       uint64_t,    uint64_t,    CumMax, 2)
instantiate_scan_helper(max_int8_int8,           int8_t,      int8_t,      CumMax, 4)
instantiate_scan_helper(max_int16_int16,         int16_t,     int16_t,     CumMax, 4)
instantiate_scan_helper(max_int32_int32,         int32_t,     int32_t,     CumMax, 4)
//instantiate_scan_helper(max_int64_int64,         int64_t,     int64_t,     CumMax, 2)
instantiate_scan_helper(max_float16_float16,     half,        half,        CumMax, 4)
instantiate_scan_helper(max_float32_float32,     float,       float,       CumMax, 4)
instantiate_scan_helper(max_bfloat16_bfloat16,   bfloat16_t,  bfloat16_t,  CumMax, 4)
//instantiate_scan_helper(max_complex64_complex64, complex64_t, complex64_t, CumMax)
//instantiate_scan_helper(min_bool__bool_,         bool,        bool,        CumMin, 4)
instantiate_scan_helper(min_uint8_uint8,         uint8_t,     uint8_t,     CumMin, 4)
instantiate_scan_helper(min_uint16_uint16,       uint16_t,    uint16_t,    CumMin, 4)
instantiate_scan_helper(min_uint32_uint32,       uint32_t,    uint32_t,    CumMin, 4)
//instantiate_scan_helper(min_uint64_uint64,       uint64_t,    uint64_t,    CumMin, 2)
instantiate_scan_helper(min_int8_int8,           int8_t,      int8_t,      CumMin, 4)
instantiate_scan_helper(min_int16_int16,         int16_t,     int16_t,     CumMin, 4)
instantiate_scan_helper(min_int32_int32,         int32_t,     int32_t,     CumMin, 4)
//instantiate_scan_helper(min_int64_int64,         int64_t,     int64_t,     CumMin, 2)
instantiate_scan_helper(min_float16_float16,     half,        half,        CumMin, 4)
instantiate_scan_helper(min_float32_float32,     float,       float,       CumMin, 4)
instantiate_scan_helper(min_bfloat16_bfloat16,   bfloat16_t,  bfloat16_t,  CumMin, 4)
//instantiate_scan_helper(min_complex64_complex64, complex64_t, complex64_t, CumMin) // clang-format on
