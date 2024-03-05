// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/metal/kernels/reduction/utils.h"
#include "mlx/backend/metal/kernels/reduction/ops.h"
#include "mlx/backend/metal/kernels/reduction/reduce_inst.h"

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// Reduce init
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename Op>
[[kernel]] void init_reduce(
    device T *out [[buffer(0)]],
    uint tid [[thread_position_in_grid]]) {
  out[tid] = Op::init;
}

#define instantiate_init_reduce(name, otype, op) \
  template [[host_name("i" #name)]] \
    [[kernel]] void init_reduce<otype, op>( \
      device otype *out [[buffer(1)]], \
      uint tid [[thread_position_in_grid]]);

#define instantiate_init_reduce_helper(name, tname, type, op) \
  instantiate_init_reduce(name ##tname, type, op<type>)

instantiate_reduce_ops(instantiate_init_reduce_helper, instantiate_reduce_helper_types)
instantiate_reduce_ops(instantiate_init_reduce_helper, instantiate_reduce_helper_64b)

instantiate_init_reduce(andbool_, bool, And)
instantiate_init_reduce(orbool_, bool, Or)