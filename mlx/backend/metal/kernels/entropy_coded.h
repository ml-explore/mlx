// Copyright © 2024 Apple Inc.

#pragma once

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

constant uint PROB_BITS = 14;
constant uint PROB_SCALE = 1 << PROB_BITS;
constant uint RANS_L = 1 << 23;

// Fused rANS decode + dequantize + GEMV kernel.
// Each row is independently encoded, so each threadgroup decodes only in_vec_size symbols.
template <typename T>
[[kernel]] void entropy_coded_qmv(
    device const uint8_t* compressed [[buffer(0)]],
    device const uint* row_offsets [[buffer(1)]],
    device const uint* row_stream_lens [[buffer(2)]],
    device const uint16_t* freq [[buffer(3)]],
    device const uint16_t* cumfreq [[buffer(4)]],
    device const uint8_t* sym_table [[buffer(5)]],
    device const T* input [[buffer(6)]],
    device T* output [[buffer(7)]],
    device const T* scales [[buffer(8)]],
    device const T* biases [[buffer(9)]],
    constant uint& n_streams [[buffer(10)]],
    constant uint& in_vec_size [[buffer(11)]],
    constant uint& out_vec_size [[buffer(12)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint thread_idx [[thread_index_in_threadgroup]]
) {
    uint row = tgid.x;
    if (row >= out_vec_size) return;
    
    uint16_t local_freq[16];
    uint16_t local_cumfreq[16];
    for (int i = 0; i < 16; i++) {
        local_freq[i] = freq[i];
        local_cumfreq[i] = cumfreq[i];
    }
    
    threadgroup uint8_t shared_sym_table[PROB_SCALE];
    uint load_start = thread_idx * 64;
    if (load_start < PROB_SCALE) {
        for (uint i = 0; i < 64 && (load_start + i) < PROB_SCALE; i++) {
            shared_sym_table[load_start + i] = sym_table[load_start + i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    threadgroup float partial_sums[32];
    
    uint stream_idx = simd_lane + simd_group * 32;
    float acc = 0.0f;
    
    T scale = scales[row];
    T bias = biases[row];
    
    device const uint8_t* row_data = compressed + row_offsets[row];
    device const uint* row_lens = row_stream_lens + row * n_streams;
    
    if (stream_idx < n_streams) {
        uint stream_len = row_lens[stream_idx];
        
        if (stream_len >= 4) {
            uint b0 = row_data[stream_idx + 0 * n_streams];
            uint b1 = row_data[stream_idx + 1 * n_streams];
            uint b2 = row_data[stream_idx + 2 * n_streams];
            uint b3 = row_data[stream_idx + 3 * n_streams];
            uint state = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
            uint ptr = 4;
            
            uint symbols_per_stream = (in_vec_size - stream_idx + n_streams - 1) / n_streams;
            
            for (uint i = 0; i < symbols_per_stream; i++) {
                uint col = stream_idx + i * n_streams;
                if (col >= in_vec_size) break;
                
                uint slot = state & (PROB_SCALE - 1);
                uint8_t s = shared_sym_table[slot];
                
                uint freq_s = local_freq[s];
                uint start_s = local_cumfreq[s];
                state = freq_s * (state >> PROB_BITS) + slot - start_s;
                
                while (state < RANS_L && ptr < stream_len) {
                    uint8_t b = row_data[stream_idx + ptr * n_streams];
                    state = (state << 8) | b;
                    ptr++;
                }
                
                T weight = T(s) * scale + bias;
                acc += float(weight * input[col]);
            }
        }
    }
    
    acc = simd_sum(acc);
    if (simd_lane == 0) {
        partial_sums[simd_group] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_group == 0) {
        float final_sum = 0.0f;
        uint n_simd_groups = (min(uint(256), n_streams) + 31) / 32;
        if (simd_lane < n_simd_groups) {
            final_sum = partial_sums[simd_lane];
        }
        final_sum = simd_sum(final_sum);
        if (simd_lane == 0) {
            output[row] = T(final_sum);
        }
    }
}

// Decode-only kernel for async prefetching
template <typename T>
[[kernel]] void entropy_decode(
    device const uint8_t* compressed [[buffer(0)]],
    device const uint* row_offsets [[buffer(1)]],
    device const uint* row_stream_lens [[buffer(2)]],
    device const uint16_t* freq [[buffer(3)]],
    device const uint16_t* cumfreq [[buffer(4)]],
    device const uint8_t* sym_table [[buffer(5)]],
    device uint8_t* output [[buffer(6)]],
    device const T* scales [[buffer(7)]],
    device const T* biases [[buffer(8)]],
    device T* dequantized [[buffer(9)]],
    constant uint& n_streams [[buffer(10)]],
    constant uint& in_vec_size [[buffer(11)]],
    constant uint& out_vec_size [[buffer(12)]],
    constant bool& do_dequantize [[buffer(13)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint thread_idx [[thread_index_in_threadgroup]]
) {
    uint row = tgid.x;
    if (row >= out_vec_size) return;
    
    uint16_t local_freq[16];
    uint16_t local_cumfreq[16];
    for (int i = 0; i < 16; i++) {
        local_freq[i] = freq[i];
        local_cumfreq[i] = cumfreq[i];
    }
    
    threadgroup uint8_t shared_sym_table[PROB_SCALE];
    uint load_start = thread_idx * 64;
    if (load_start < PROB_SCALE) {
        for (uint i = 0; i < 64 && (load_start + i) < PROB_SCALE; i++) {
            shared_sym_table[load_start + i] = sym_table[load_start + i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    uint stream_idx = simd_lane + simd_group * 32;
    
    T scale = scales[row];
    T bias = biases[row];
    
    device const uint8_t* row_data = compressed + row_offsets[row];
    device const uint* row_lens = row_stream_lens + row * n_streams;
    
    if (stream_idx < n_streams) {
        uint stream_len = row_lens[stream_idx];
        
        if (stream_len >= 4) {
            uint b0 = row_data[stream_idx + 0 * n_streams];
            uint b1 = row_data[stream_idx + 1 * n_streams];
            uint b2 = row_data[stream_idx + 2 * n_streams];
            uint b3 = row_data[stream_idx + 3 * n_streams];
            uint state = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
            uint ptr = 4;
            
            uint symbols_per_stream = (in_vec_size - stream_idx + n_streams - 1) / n_streams;
            
            for (uint i = 0; i < symbols_per_stream; i++) {
                uint col = stream_idx + i * n_streams;
                if (col >= in_vec_size) break;
                
                uint slot = state & (PROB_SCALE - 1);
                uint8_t s = shared_sym_table[slot];
                
                uint output_idx = row * in_vec_size + col;
                output[output_idx] = s;
                
                if (do_dequantize && dequantized != nullptr) {
                    T weight = T(s) * scale + bias;
                    dequantized[output_idx] = weight;
                }
                
                uint freq_s = local_freq[s];
                uint start_s = local_cumfreq[s];
                state = freq_s * (state >> PROB_BITS) + slot - start_s;
                
                while (state < RANS_L && ptr < stream_len) {
                    uint8_t b = row_data[stream_idx + ptr * n_streams];
                    state = (state << 8) | b;
                    ptr++;
                }
            }
        }
    }
}

template <typename T>
[[kernel]] void dequantize_indices(
    device const uint8_t* indices [[buffer(0)]],
    device T* output [[buffer(1)]],
    device const T* scales [[buffer(2)]],
    device const T* biases [[buffer(3)]],
    constant uint& n_elements [[buffer(4)]],
    constant uint& in_vec_size [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n_elements) return;
    
    uint row = tid / in_vec_size;
    T scale = scales[row];
    T bias = biases[row];
    
    output[tid] = T(indices[tid]) * scale + bias;
}
