#include <metal_math>
#include <metal_integer>

#include "mlx/backend/metal/kernels/bf16.h"

using namespace metal;

template <typename T>
[[kernel]] void max_pool_1d(
    device const T* in,
    device T* out,
    constant const int& kernel_size,
    constant const int& stride,
    constant const int& padding,
    constant const int& in_height,
    constant const uint& out_height, 
    constant const size_t in_strides[3],
    constant const size_t out_strides[3],
    uint3 pos [[thread_position_in_grid]]
) {
    // Get the max
    if(pos.y >= out_height) return;

    int start = pos.y * stride - padding;
    int end = min(start + kernel_size, in_height);
    start = max(0, start);
    int bx = pos.x * in_strides[0];
    T val = numeric_limits<T>::lowest();
    for (int i = start; i < end; i++) {
        val = max(val, in[bx + i * in_strides[1] + pos.z]);
    }
    out[pos.x * out_strides[0] + pos.y * out_strides[1] + pos.z] = val;
}

#define instantiate_max_pool_1d(name, type) \
    template [[host_name("max_pool_1d_" #name)]] \
    [[kernel]] void max_pool_1d<type>( \
        device const type* in, \
        device type* out, \
        constant const int& kernel_size, \
        constant const int& stride, \
        constant const int& padding, \
        constant const int& in_height, \
        constant const uint& out_height, \
        constant const size_t in_strides[3], \
        constant const size_t out_strides[3], \
        uint3 pos [[thread_position_in_grid]] \
    );

instantiate_max_pool_1d(float16, half);
instantiate_max_pool_1d(bfloat16, bfloat16_t);
instantiate_max_pool_1d(float32, float);