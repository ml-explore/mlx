// Copyright © 2026 Apple Inc.

#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/partition.h"

#define instantiate_radix_partition_bn(tname, type, bn)        \
  instantiate_kernel("radix_partition_" #tname "_bn" #bn,      \
                     radix_partition, type, type, false, bn)   \
  instantiate_kernel("radix_argpartition_" #tname "_bn" #bn,   \
                     radix_partition, type, uint32_t, true, bn)

#define instantiate_radix_partition(tname, type)   \
  instantiate_radix_partition_bn(tname, type, 64)  \
  instantiate_radix_partition_bn(tname, type, 128) \
  instantiate_radix_partition_bn(tname, type, 256)

instantiate_radix_partition(uint8, uint8_t)
instantiate_radix_partition(uint16, uint16_t)
instantiate_radix_partition(uint32, uint32_t)
instantiate_radix_partition(uint64, uint64_t)
instantiate_radix_partition(int8, int8_t)
instantiate_radix_partition(int16, int16_t)
instantiate_radix_partition(int32, int32_t)
instantiate_radix_partition(int64, int64_t)
instantiate_radix_partition(float16, half)
instantiate_radix_partition(float32, float)
instantiate_radix_partition(bfloat16, bfloat16_t)
instantiate_radix_partition(complex64, complex64_t) // clang-format on
