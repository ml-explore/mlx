// Copyright © 2024 Apple Inc.
// Entropy-Coded Quantization V2: Per-Row Encoding for O(n) decode
//
// Key optimization: Each row is encoded independently, so each threadgroup
// only decodes in_vec_size symbols instead of out_vec_size * in_vec_size.
// This reduces decode work from O(rows * total) to O(total).

#pragma once

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// rANS constants
constant uint PROB_BITS_V2 = 14;
constant uint PROB_SCALE_V2 = 1 << PROB_BITS_V2;  // 16384
constant uint RANS_L_V2 = 1 << 23;

// ============================================================================
// Per-Row Fused Decode + Dequantize + GEMV Kernel
//
// Each row is independently encoded, so each threadgroup decodes exactly
// in_vec_size symbols - no wasted work!
//
// Data layout:
//   compressed: [row0_data | row1_data | ... | rowN_data]
//   row_offsets: [offset_to_row0, offset_to_row1, ...]
//   Each row's data is physically interleaved across n_streams
// ============================================================================

template <typename T>
[[kernel]] void entropy_coded_qmv_v2(
    device const uint8_t* compressed [[buffer(0)]],      // All rows concatenated
    device const uint* row_offsets [[buffer(1)]],        // Byte offset to each row
    device const uint* row_stream_lens [[buffer(2)]],    // Per-row stream lengths [n_rows * n_streams]
    device const uint16_t* freq [[buffer(3)]],           // Frequency table (16 entries)
    device const uint16_t* cumfreq [[buffer(4)]],        // Cumulative freq (16 entries)
    device const uint8_t* sym_table [[buffer(5)]],       // Symbol lookup (16384 entries)
    device const T* input [[buffer(6)]],                 // Input vector
    device T* output [[buffer(7)]],                      // Output vector
    device const T* scales [[buffer(8)]],                // Per-row scales
    device const T* biases [[buffer(9)]],                // Per-row biases
    constant uint& n_streams [[buffer(10)]],             // Parallel streams per row
    constant uint& in_vec_size [[buffer(11)]],           // Columns per row
    constant uint& out_vec_size [[buffer(12)]],          // Number of rows
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint thread_idx [[thread_index_in_threadgroup]]
) {
    uint row = tgid.x;
    if (row >= out_vec_size) return;
    
    // ========================================================================
    // OPTIMIZATION 1: Register-cached frequency tables
    // ========================================================================
    uint16_t local_freq[16];
    uint16_t local_cumfreq[16];
    for (int i = 0; i < 16; i++) {
        local_freq[i] = freq[i];
        local_cumfreq[i] = cumfreq[i];
    }
    
    // ========================================================================
    // OPTIMIZATION 2: Threadgroup-cached symbol table
    // ========================================================================
    threadgroup uint8_t shared_sym_table[PROB_SCALE_V2];
    uint load_start = thread_idx * 64;
    if (load_start < PROB_SCALE_V2) {
        for (uint i = 0; i < 64 && (load_start + i) < PROB_SCALE_V2; i++) {
            shared_sym_table[load_start + i] = sym_table[load_start + i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    threadgroup float partial_sums[32];
    
    uint stream_idx = simd_lane + simd_group * 32;
    float acc = 0.0f;
    
    T scale = scales[row];
    T bias = biases[row];
    
    // ========================================================================
    // KEY OPTIMIZATION: Only decode THIS ROW's data
    // ========================================================================
    device const uint8_t* row_data = compressed + row_offsets[row];
    device const uint* row_lens = row_stream_lens + row * n_streams;
    
    if (stream_idx < n_streams) {
        uint stream_len = row_lens[stream_idx];
        
        if (stream_len >= 4) {
            // Coalesced state init from this row's interleaved data
            uint b0 = row_data[stream_idx + 0 * n_streams];
            uint b1 = row_data[stream_idx + 1 * n_streams];
            uint b2 = row_data[stream_idx + 2 * n_streams];
            uint b3 = row_data[stream_idx + 3 * n_streams];
            uint state = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
            uint ptr = 4;
            
            // Only decode symbols for THIS row (in_vec_size total)
            uint symbols_per_stream = (in_vec_size - stream_idx + n_streams - 1) / n_streams;
            
            for (uint i = 0; i < symbols_per_stream; i++) {
                uint col = stream_idx + i * n_streams;
                if (col >= in_vec_size) break;
                
                // Decode from threadgroup cache
                uint slot = state & (PROB_SCALE_V2 - 1);
                uint8_t s = shared_sym_table[slot];
                
                // State update from register cache
                uint freq_s = local_freq[s];
                uint start_s = local_cumfreq[s];
                state = freq_s * (state >> PROB_BITS_V2) + slot - start_s;
                
                // Coalesced renormalization
                while (state < RANS_L_V2 && ptr < stream_len) {
                    uint8_t b = row_data[stream_idx + ptr * n_streams];
                    state = (state << 8) | b;
                    ptr++;
                }
                
                // Fused dequantize + MAC
                T weight = T(s) * scale + bias;
                acc += float(weight * input[col]);
            }
        }
    }
    
    // SIMD reduction
    acc = simd_sum(acc);
    if (simd_lane == 0) {
        partial_sums[simd_group] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Final reduction
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

// ============================================================================
// Shared frequency table variant - when all rows use same distribution
// ============================================================================

template <typename T>
[[kernel]] void entropy_coded_qmv_v2_shared_freq(
    device const uint8_t* compressed [[buffer(0)]],
    device const uint* row_offsets [[buffer(1)]],
    device const uint* max_stream_len_per_row [[buffer(2)]],  // Single value per row
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
    
    // Register-cached tables
    uint16_t local_freq[16];
    uint16_t local_cumfreq[16];
    for (int i = 0; i < 16; i++) {
        local_freq[i] = freq[i];
        local_cumfreq[i] = cumfreq[i];
    }
    
    // Threadgroup-cached symbol table
    threadgroup uint8_t shared_sym_table[PROB_SCALE_V2];
    uint load_start = thread_idx * 64;
    if (load_start < PROB_SCALE_V2) {
        for (uint i = 0; i < 64 && (load_start + i) < PROB_SCALE_V2; i++) {
            shared_sym_table[load_start + i] = sym_table[load_start + i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    threadgroup float partial_sums[32];
    
    uint stream_idx = simd_lane + simd_group * 32;
    float acc = 0.0f;
    
    T scale = scales[row];
    T bias = biases[row];
    
    // This row's compressed data
    device const uint8_t* row_data = compressed + row_offsets[row];
    uint max_len = max_stream_len_per_row[row];
    
    if (stream_idx < n_streams && max_len >= 4) {
        // All streams in this row have same max length (padded)
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
            
            uint slot = state & (PROB_SCALE_V2 - 1);
            uint8_t s = shared_sym_table[slot];
            
            uint freq_s = local_freq[s];
            uint start_s = local_cumfreq[s];
            state = freq_s * (state >> PROB_BITS_V2) + slot - start_s;
            
            while (state < RANS_L_V2 && ptr < max_len) {
                uint8_t b = row_data[stream_idx + ptr * n_streams];
                state = (state << 8) | b;
                ptr++;
            }
            
            T weight = T(s) * scale + bias;
            acc += float(weight * input[col]);
        }
    }
    
    // Reduction
    acc = simd_sum(acc);
    if (simd_lane == 0) partial_sums[simd_group] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_group == 0) {
        float final_sum = 0.0f;
        uint n_simd_groups = (min(uint(256), n_streams) + 31) / 32;
        if (simd_lane < n_simd_groups) final_sum = partial_sums[simd_lane];
        final_sum = simd_sum(final_sum);
        if (simd_lane == 0) output[row] = T(final_sum);
    }
}
