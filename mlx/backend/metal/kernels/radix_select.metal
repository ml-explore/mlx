// Copyright © 2025 Apple Inc.

#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/radix_select.h"

///////////////////////////////////////////////////////////////////////////////
// Single-pass Radix Select Kernel Instantiations
///////////////////////////////////////////////////////////////////////////////

#define instantiate_radix_select(name, itname, itype, otname, otype, arg_part, bn, tn) \
  instantiate_kernel(                                                                   \
      "c" #name "_" #itname "_" #otname "_bn" #bn "_tn" #tn,                            \
      radix_select_partition,                                                           \
      itype, otype, arg_part, bn, tn)                                                   \
  instantiate_kernel(                                                                   \
      "nc" #name "_" #itname "_" #otname "_bn" #bn "_tn" #tn,                           \
      radix_select_partition_nc,                                                        \
      itype, otype, arg_part, bn, tn)

#define instantiate_radix_select_arg(itname, itype, bn, tn) \
  instantiate_radix_select(arg_radix_select, itname, itype, uint32, uint32_t, true, bn, tn)

#define instantiate_radix_select_val(itname, itype, bn, tn) \
  instantiate_radix_select(_radix_select, itname, itype, itname, itype, false, bn, tn)

#define instantiate_radix_select_tn(itname, itype, bn) \
  instantiate_radix_select_arg(itname, itype, bn, 8)   \
  instantiate_radix_select_val(itname, itype, bn, 8)

#define instantiate_radix_select_bn(itname, itype) \
  instantiate_radix_select_tn(itname, itype, 256)

instantiate_radix_select_bn(uint8, uint8_t)
instantiate_radix_select_bn(uint16, uint16_t)
instantiate_radix_select_bn(uint32, uint32_t)
instantiate_radix_select_bn(int8, int8_t)
instantiate_radix_select_bn(int16, int16_t)
instantiate_radix_select_bn(int32, int32_t)
instantiate_radix_select_bn(float16, half)
instantiate_radix_select_bn(float32, float)
instantiate_radix_select_bn(bfloat16, bfloat16_t)

#define instantiate_radix_select_long(itname, itype) \
  instantiate_radix_select_tn(itname, itype, 128)

instantiate_radix_select_long(uint64, uint64_t)
instantiate_radix_select_long(int64, int64_t)

///////////////////////////////////////////////////////////////////////////////
// Large Array Streaming Radix Select Kernel Instantiations
///////////////////////////////////////////////////////////////////////////////

#define instantiate_radix_large_streaming(itname, itype, otname, otype, arg_part, bn) \
  instantiate_kernel(                                                                  \
      "radix_select_large_" #itname "_" #otname "_" #arg_part "_bn" #bn,               \
      radix_select_large_streaming,                                                    \
      itype, otype, arg_part, bn)

#define instantiate_radix_large_streaming_all(itname, itype, bn) \
  instantiate_radix_large_streaming(itname, itype, uint32, uint32_t, true, bn) \
  instantiate_radix_large_streaming(itname, itype, itname, itype, false, bn)

instantiate_radix_large_streaming_all(uint8, uint8_t, 256)
instantiate_radix_large_streaming_all(uint16, uint16_t, 256)
instantiate_radix_large_streaming_all(uint32, uint32_t, 256)
instantiate_radix_large_streaming_all(int8, int8_t, 256)
instantiate_radix_large_streaming_all(int16, int16_t, 256)
instantiate_radix_large_streaming_all(int32, int32_t, 256)
instantiate_radix_large_streaming_all(float16, half, 256)
instantiate_radix_large_streaming_all(float32, float, 256)
instantiate_radix_large_streaming_all(bfloat16, bfloat16_t, 256)
instantiate_radix_large_streaming_all(uint64, uint64_t, 128)
instantiate_radix_large_streaming_all(int64, int64_t, 128)

///////////////////////////////////////////////////////////////////////////////
// Large Array Non-Contiguous Streaming Radix Select Kernel Instantiations
///////////////////////////////////////////////////////////////////////////////

#define instantiate_radix_large_streaming_nc(itname, itype, otname, otype, arg_part, bn) \
  instantiate_kernel(                                                                     \
      "radix_select_large_nc_" #itname "_" #otname "_" #arg_part "_bn" #bn,               \
      radix_select_large_streaming_nc,                                                    \
      itype, otype, arg_part, bn)

#define instantiate_radix_large_streaming_nc_all(itname, itype, bn) \
  instantiate_radix_large_streaming_nc(itname, itype, uint32, uint32_t, true, bn) \
  instantiate_radix_large_streaming_nc(itname, itype, itname, itype, false, bn)

instantiate_radix_large_streaming_nc_all(uint8, uint8_t, 256)
instantiate_radix_large_streaming_nc_all(uint16, uint16_t, 256)
instantiate_radix_large_streaming_nc_all(uint32, uint32_t, 256)
instantiate_radix_large_streaming_nc_all(int8, int8_t, 256)
instantiate_radix_large_streaming_nc_all(int16, int16_t, 256)
instantiate_radix_large_streaming_nc_all(int32, int32_t, 256)
instantiate_radix_large_streaming_nc_all(float16, half, 256)
instantiate_radix_large_streaming_nc_all(float32, float, 256)
instantiate_radix_large_streaming_nc_all(bfloat16, bfloat16_t, 256)
instantiate_radix_large_streaming_nc_all(uint64, uint64_t, 128)
instantiate_radix_large_streaming_nc_all(int64, int64_t, 128)

///////////////////////////////////////////////////////////////////////////////
// Multi-pass Radix Select Kernel Instantiations
///////////////////////////////////////////////////////////////////////////////

#define instantiate_radix_histogram(itname, itype, bn) \
  instantiate_kernel("radix_histogram_" #itname "_bn" #bn, radix_histogram_kernel, itype, bn)

#define instantiate_radix_histogram_all(itname, itype) \
  instantiate_radix_histogram(itname, itype, 256)

instantiate_radix_histogram_all(uint8, uint8_t)
instantiate_radix_histogram_all(uint16, uint16_t)
instantiate_radix_histogram_all(uint32, uint32_t)
instantiate_radix_histogram_all(int8, int8_t)
instantiate_radix_histogram_all(int16, int16_t)
instantiate_radix_histogram_all(int32, int32_t)
instantiate_radix_histogram_all(float16, half)
instantiate_radix_histogram_all(float32, float)
instantiate_radix_histogram_all(bfloat16, bfloat16_t)
instantiate_radix_histogram_all(uint64, uint64_t)
instantiate_radix_histogram_all(int64, int64_t)

#define instantiate_radix_find_bin(itname, itype) \
  instantiate_kernel("radix_find_bin_" #itname, radix_find_bin_kernel, itype)

instantiate_radix_find_bin(uint8, uint8_t)
instantiate_radix_find_bin(uint16, uint16_t)
instantiate_radix_find_bin(uint32, uint32_t)
instantiate_radix_find_bin(int8, int8_t)
instantiate_radix_find_bin(int16, int16_t)
instantiate_radix_find_bin(int32, int32_t)
instantiate_radix_find_bin(float16, half)
instantiate_radix_find_bin(float32, float)
instantiate_radix_find_bin(bfloat16, bfloat16_t)
instantiate_radix_find_bin(uint64, uint64_t)
instantiate_radix_find_bin(int64, int64_t)

#define instantiate_partition_output(itname, itype, otname, otype, arg_part, bn) \
  instantiate_kernel(                                                             \
      "radix_partition_output_" #itname "_" #otname "_" #arg_part "_bn" #bn,      \
      radix_partition_output_kernel, itype, otype, arg_part, bn)                  \
  instantiate_kernel(                                                             \
      "radix_partition_equal_" #itname "_" #otname "_" #arg_part "_bn" #bn,       \
      radix_partition_equal_kernel, itype, otype, arg_part, bn)                   \
  instantiate_kernel(                                                             \
      "radix_partition_greater_" #itname "_" #otname "_" #arg_part "_bn" #bn,     \
      radix_partition_greater_kernel, itype, otype, arg_part, bn)

#define instantiate_partition_output_all(itname, itype) \
  instantiate_partition_output(itname, itype, uint32, uint32_t, true, 256) \
  instantiate_partition_output(itname, itype, itname, itype, false, 256)

instantiate_partition_output_all(uint8, uint8_t)
instantiate_partition_output_all(uint16, uint16_t)
instantiate_partition_output_all(uint32, uint32_t)
instantiate_partition_output_all(int8, int8_t)
instantiate_partition_output_all(int16, int16_t)
instantiate_partition_output_all(int32, int32_t)
instantiate_partition_output_all(float16, half)
instantiate_partition_output_all(float32, float)
instantiate_partition_output_all(bfloat16, bfloat16_t)
instantiate_partition_output_all(uint64, uint64_t)
instantiate_partition_output_all(int64, int64_t)
    // clang-format on
