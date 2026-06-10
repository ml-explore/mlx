#pragma once

#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

template <typename T, int Dk, int Dv>
[[kernel]] void gated_delta_update_fwd(
    const device T* q        [[buffer(0)]],
    const device T* k        [[buffer(1)]],
    const device T* v        [[buffer(2)]],
    const device T* g        [[buffer(3)]],
    const device T* beta     [[buffer(4)]],
    const device T* h0       [[buffer(5)]],
          device T* out      [[buffer(6)]],
          device T* hf       [[buffer(7)]],
    constant int& B          [[buffer(8)]],
    constant int& H          [[buffer(9)]],
    constant int& T_len      [[buffer(10)]],
    uint3 tid                [[thread_position_in_grid]]
) {
    // kernel implementation
}
