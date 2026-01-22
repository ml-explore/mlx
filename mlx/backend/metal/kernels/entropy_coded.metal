// Copyright © 2024 Apple Inc.
// Entropy-Coded Quantization Metal Kernels

#include "entropy_coded.h"
#include "entropy_coded_v2.h"
#include "entropy_decode_async.h"

// Instantiate for float and bfloat16
template [[host_name("entropy_coded_qmv_float")]] [[kernel]] void
entropy_coded_qmv<float>(
    device const uint8_t* compressed [[buffer(0)]],
    device const uint* stream_lengths [[buffer(1)]],
    device const uint16_t* freq [[buffer(2)]],
    device const uint16_t* cumfreq [[buffer(3)]],
    device const uint8_t* sym_table [[buffer(4)]],
    device const float* input [[buffer(5)]],
    device float* output [[buffer(6)]],
    device const float* scales [[buffer(7)]],
    device const float* biases [[buffer(8)]],
    constant uint& n_streams [[buffer(9)]],
    constant uint& n_symbols [[buffer(10)]],
    constant uint& max_stream_len [[buffer(11)]],
    constant uint& out_vec_size [[buffer(12)]],
    constant uint& in_vec_size [[buffer(13)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint thread_idx [[thread_index_in_threadgroup]]);

template [[host_name("entropy_coded_qmv_bfloat16")]] [[kernel]] void
entropy_coded_qmv<bfloat>(
    device const uint8_t* compressed [[buffer(0)]],
    device const uint* stream_lengths [[buffer(1)]],
    device const uint16_t* freq [[buffer(2)]],
    device const uint16_t* cumfreq [[buffer(3)]],
    device const uint8_t* sym_table [[buffer(4)]],
    device const bfloat* input [[buffer(5)]],
    device bfloat* output [[buffer(6)]],
    device const bfloat* scales [[buffer(7)]],
    device const bfloat* biases [[buffer(8)]],
    constant uint& n_streams [[buffer(9)]],
    constant uint& n_symbols [[buffer(10)]],
    constant uint& max_stream_len [[buffer(11)]],
    constant uint& out_vec_size [[buffer(12)]],
    constant uint& in_vec_size [[buffer(13)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint thread_idx [[thread_index_in_threadgroup]]);

template [[host_name("dequantize_4bit_float")]] [[kernel]] void
dequantize_4bit<float>(
    device const uint8_t* indices [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* biases [[buffer(3)]],
    constant uint& n_elements [[buffer(4)]],
    constant uint& group_size [[buffer(5)]],
    uint tid [[thread_position_in_grid]]);

template [[host_name("dequantize_4bit_bfloat16")]] [[kernel]] void
dequantize_4bit<bfloat>(
    device const uint8_t* indices [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    device const bfloat* scales [[buffer(2)]],
    device const bfloat* biases [[buffer(3)]],
    constant uint& n_elements [[buffer(4)]],
    constant uint& group_size [[buffer(5)]],
    uint tid [[thread_position_in_grid]]);

// Per-row encoded kernel instantiations
template [[host_name("entropy_coded_qmv_per_row_float")]] [[kernel]] void
entropy_coded_qmv_per_row<float>(
    device const uint8_t* compressed [[buffer(0)]],
    device const uint* row_offsets [[buffer(1)]],
    device const uint* stream_lengths [[buffer(2)]],
    device const uint16_t* freq [[buffer(3)]],
    device const uint16_t* cumfreq [[buffer(4)]],
    device const uint8_t* sym_table [[buffer(5)]],
    device const float* input [[buffer(6)]],
    device float* output [[buffer(7)]],
    device const float* scales [[buffer(8)]],
    device const float* biases [[buffer(9)]],
    constant uint& n_streams [[buffer(10)]],
    constant uint& in_vec_size [[buffer(11)]],
    constant uint& max_stream_len [[buffer(12)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint thread_idx [[thread_index_in_threadgroup]]);

template [[host_name("entropy_coded_qmv_per_row_bfloat16")]] [[kernel]] void
entropy_coded_qmv_per_row<bfloat>(
    device const uint8_t* compressed [[buffer(0)]],
    device const uint* row_offsets [[buffer(1)]],
    device const uint* stream_lengths [[buffer(2)]],
    device const uint16_t* freq [[buffer(3)]],
    device const uint16_t* cumfreq [[buffer(4)]],
    device const uint8_t* sym_table [[buffer(5)]],
    device const bfloat* input [[buffer(6)]],
    device bfloat* output [[buffer(7)]],
    device const bfloat* scales [[buffer(8)]],
    device const bfloat* biases [[buffer(9)]],
    constant uint& n_streams [[buffer(10)]],
    constant uint& in_vec_size [[buffer(11)]],
    constant uint& max_stream_len [[buffer(12)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint thread_idx [[thread_index_in_threadgroup]]);

// V2 Per-Row Kernels with proper buffer layout
template [[host_name("entropy_coded_qmv_v2_float")]] [[kernel]] void
entropy_coded_qmv_v2<float>(
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

template [[host_name("entropy_coded_qmv_v2_bfloat16")]] [[kernel]] void
entropy_coded_qmv_v2<bfloat>(
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
