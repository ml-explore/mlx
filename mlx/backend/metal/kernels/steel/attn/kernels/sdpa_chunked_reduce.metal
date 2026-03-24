// Copyright © 2025 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/sdpa_chunked_reduce.h"

#define instantiate_chunked_reduce(tname, type)               \
  template [[host_name("sdpa_chunked_reduce_" #tname)]]       \
  [[kernel]] decltype(sdpa_chunked_reduce<type>)              \
      sdpa_chunked_reduce<type>;

instantiate_chunked_reduce(float32, float)
instantiate_chunked_reduce(float16, half)
instantiate_chunked_reduce(bfloat16, bfloat16_t)
// clang-format on
