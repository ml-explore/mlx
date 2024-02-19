#include <metal_math>
#include <metal_integer>

using namespace metal;

[[kernel]] void max_pool_1d_float(
    device const float* in,
    device float* out,
    constant const int& kernel_size,
    constant const int& stride,
    constant const int& padding,
    constant const int& in_height,
    constant const int& out_height, 
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
    float val = -INFINITY;
    for (int i = start; i < end; i++) {
        val = max(val, in[bx + i * in_strides[1] + pos.z]);
    }
    out[pos.x * out_strides[0] + pos.y * out_strides[1] + pos.z] = val;
}