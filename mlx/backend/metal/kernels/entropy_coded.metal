// Copyright © 2024 Apple Inc.
// Entropy-Coded Quantization Metal Kernels

#include "entropy_coded.h"
#include "entropy_decode_async.h"

// Instantiate for float and bfloat16
template [[host_name("entropy_coded_qmv_float")]] [[kernel]] void
entropy_coded_qmv<float>(
    device const uint8_t* compressed [[buffer(0)]],
    device const uint* row_offsets [[buffer(1)]],
    device const uint* row_stream_lens [[buffer(2)]],
    device const uint16_t* freq [[buffer(3)]],
    device const uint16_t* cumfreq [[buffer(4)]],
    device const uint8_t* sym_table [[buffer(5)]],
    device const float* input [[buffer(6)]],
    device float* output [[buffer(7)]],
    device const float* scales [[buffer(8)]],
    device const float* biases [[buffer(9)]],
    constant uint& n_streams [[buffer(10)]],
    constant uint& in_vec_size [[buffer(11)]],
    constant uint& out_vec_size [[buffer(12)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint thread_idx [[thread_index_in_threadgroup]]);

template [[host_name("entropy_coded_qmv_bfloat16")]] [[kernel]] void
entropy_coded_qmv<bfloat>(
    device const uint8_t* compressed [[buffer(0)]],
    device const uint* row_offsets [[buffer(1)]],
    device const uint* row_stream_lens [[buffer(2)]],
    device const uint16_t* freq [[buffer(3)]],
    device const uint16_t* cumfreq [[buffer(4)]],
    device const uint8_t* sym_table [[buffer(5)]],
    device const bfloat* input [[buffer(6)]],
    device bfloat* output [[buffer(7)]],
    device const bfloat* scales [[buffer(8)]],
    device const bfloat* biases [[buffer(9)]],
    constant uint& n_streams [[buffer(10)]],
    constant uint& in_vec_size [[buffer(11)]],
    constant uint& out_vec_size [[buffer(12)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint thread_idx [[thread_index_in_threadgroup]]);
