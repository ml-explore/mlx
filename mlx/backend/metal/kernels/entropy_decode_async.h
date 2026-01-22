// Copyright © 2024 Apple Inc.
// Entropy Decode Kernel for GPU_ASYNC Mode
//
// This kernel decodes compressed weights to 4-bit indices without doing the
// matmul. Used for async prefetching: decode layer N+1 while computing layer N.
//
// Timeline:
//   GPU Queue 1: [Compute L0] [Compute L1] [Compute L2] ...
//   GPU Queue 2: [Decode L1]  [Decode L2]  [Decode L3]  ...
//                     ↓           ↓           ↓
//                Ready before GPU needs it!

#pragma once

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// rANS constants
constant uint PROB_BITS_ASYNC = 14;
constant uint PROB_SCALE_ASYNC = 1 << PROB_BITS_ASYNC;  // 16384
constant uint RANS_L_ASYNC = 1 << 23;

// ============================================================================
// Decode-Only Kernel (V1: Flat Encoding)
//
// Decodes interleaved rANS streams to output buffer.
// Each thread handles one stream.
// ============================================================================

template <typename T>
[[kernel]] void entropy_decode_flat(
    device const uint8_t* compressed [[buffer(0)]],
    device const uint* stream_lengths [[buffer(1)]],
    device const uint16_t* freq [[buffer(2)]],
    device const uint16_t* cumfreq [[buffer(3)]],
    device const uint8_t* sym_table [[buffer(4)]],
    device uint8_t* output [[buffer(5)]],           // Decoded 4-bit indices
    device const T* scales [[buffer(6)]],           // For dequantization
    device const T* biases [[buffer(7)]],
    device T* dequantized [[buffer(8)]],            // Dequantized weights (optional)
    constant uint& n_streams [[buffer(9)]],
    constant uint& n_symbols [[buffer(10)]],
    constant uint& max_stream_len [[buffer(11)]],
    constant uint& out_vec_size [[buffer(12)]],
    constant uint& in_vec_size [[buffer(13)]],
    constant uint& group_size [[buffer(14)]],
    constant bool& do_dequantize [[buffer(15)]],    // Whether to also dequantize
    uint tid [[thread_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]]
) {
    uint stream_idx = tid;
    if (stream_idx >= n_streams) return;
    
    // Cache frequency tables in registers
    uint16_t local_freq[16];
    uint16_t local_cumfreq[16];
    for (int i = 0; i < 16; i++) {
        local_freq[i] = freq[i];
        local_cumfreq[i] = cumfreq[i];
    }
    
    // Threadgroup-cached symbol table
    threadgroup uint8_t shared_sym_table[PROB_SCALE_ASYNC];
    uint load_start = thread_idx * 64;
    if (load_start < PROB_SCALE_ASYNC) {
        for (uint i = 0; i < 64 && (load_start + i) < PROB_SCALE_ASYNC; i++) {
            shared_sym_table[load_start + i] = sym_table[load_start + i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    uint stream_len = stream_lengths[stream_idx];
    if (stream_len < 4) return;
    
    // Initialize state from physically interleaved data
    uint b0 = compressed[stream_idx + 0 * n_streams];
    uint b1 = compressed[stream_idx + 1 * n_streams];
    uint b2 = compressed[stream_idx + 2 * n_streams];
    uint b3 = compressed[stream_idx + 3 * n_streams];
    uint state = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
    uint ptr = 4;
    
    // Calculate symbols this stream handles
    uint symbols_per_stream = (n_symbols - stream_idx + n_streams - 1) / n_streams;
    
    for (uint i = 0; i < symbols_per_stream; i++) {
        uint output_idx = stream_idx + i * n_streams;
        if (output_idx >= n_symbols) break;
        
        // Decode symbol
        uint slot = state & (PROB_SCALE_ASYNC - 1);
        uint8_t s = shared_sym_table[slot];
        
        // Write decoded symbol
        output[output_idx] = s;
        
        // Optionally dequantize
        if (do_dequantize && dequantized != nullptr) {
            uint row = output_idx / in_vec_size;
            uint col = output_idx % in_vec_size;
            uint group = col / group_size;
            uint scale_idx = row * ((in_vec_size + group_size - 1) / group_size) + group;
            T weight = T(s) * scales[scale_idx] + biases[scale_idx];
            dequantized[output_idx] = weight;
        }
        
        // Update state
        uint freq_s = local_freq[s];
        uint start_s = local_cumfreq[s];
        state = freq_s * (state >> PROB_BITS_ASYNC) + slot - start_s;
        
        // Renormalize
        while (state < RANS_L_ASYNC && ptr < stream_len) {
            uint8_t b = compressed[stream_idx + ptr * n_streams];
            state = (state << 8) | b;
            ptr++;
        }
    }
}

// ============================================================================
// Decode-Only Kernel (V2: Per-Row Encoding)
//
// Each threadgroup decodes one row independently.
// Much faster for async decode since O(n) work.
// ============================================================================

template <typename T>
[[kernel]] void entropy_decode_per_row(
    device const uint8_t* compressed [[buffer(0)]],
    device const uint* row_offsets [[buffer(1)]],
    device const uint* row_stream_lens [[buffer(2)]],
    device const uint16_t* freq [[buffer(3)]],
    device const uint16_t* cumfreq [[buffer(4)]],
    device const uint8_t* sym_table [[buffer(5)]],
    device uint8_t* output [[buffer(6)]],           // Decoded indices [out_vec_size, in_vec_size]
    device const T* scales [[buffer(7)]],
    device const T* biases [[buffer(8)]],
    device T* dequantized [[buffer(9)]],            // Dequantized weights (optional)
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
    
    // Register-cached frequency tables
    uint16_t local_freq[16];
    uint16_t local_cumfreq[16];
    for (int i = 0; i < 16; i++) {
        local_freq[i] = freq[i];
        local_cumfreq[i] = cumfreq[i];
    }
    
    // Threadgroup-cached symbol table
    threadgroup uint8_t shared_sym_table[PROB_SCALE_ASYNC];
    uint load_start = thread_idx * 64;
    if (load_start < PROB_SCALE_ASYNC) {
        for (uint i = 0; i < 64 && (load_start + i) < PROB_SCALE_ASYNC; i++) {
            shared_sym_table[load_start + i] = sym_table[load_start + i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    uint stream_idx = simd_lane + simd_group * 32;
    
    T scale = scales[row];
    T bias = biases[row];
    
    // This row's compressed data
    device const uint8_t* row_data = compressed + row_offsets[row];
    device const uint* row_lens = row_stream_lens + row * n_streams;
    
    if (stream_idx < n_streams) {
        uint stream_len = row_lens[stream_idx];
        
        if (stream_len >= 4) {
            // Initialize state
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
                
                // Decode symbol
                uint slot = state & (PROB_SCALE_ASYNC - 1);
                uint8_t s = shared_sym_table[slot];
                
                // Write to output
                uint output_idx = row * in_vec_size + col;
                output[output_idx] = s;
                
                // Optionally dequantize
                if (do_dequantize && dequantized != nullptr) {
                    T weight = T(s) * scale + bias;
                    dequantized[output_idx] = weight;
                }
                
                // Update state
                uint freq_s = local_freq[s];
                uint start_s = local_cumfreq[s];
                state = freq_s * (state >> PROB_BITS_ASYNC) + slot - start_s;
                
                // Renormalize
                while (state < RANS_L_ASYNC && ptr < stream_len) {
                    uint8_t b = row_data[stream_idx + ptr * n_streams];
                    state = (state << 8) | b;
                    ptr++;
                }
            }
        }
    }
}

// Kernel instantiations
#define INSTANTIATE_DECODE_KERNELS(T)                              \
    template [[host_name("entropy_decode_flat_" #T)]]              \
    [[kernel]] void entropy_decode_flat<T>(                        \
        device const uint8_t*, device const uint*,                 \
        device const uint16_t*, device const uint16_t*,            \
        device const uint8_t*, device uint8_t*,                    \
        device const T*, device const T*, device T*,               \
        constant uint&, constant uint&, constant uint&,            \
        constant uint&, constant uint&, constant uint&,            \
        constant bool&, uint, uint);                               \
    template [[host_name("entropy_decode_per_row_" #T)]]           \
    [[kernel]] void entropy_decode_per_row<T>(                     \
        device const uint8_t*, device const uint*, device const uint*, \
        device const uint16_t*, device const uint16_t*,            \
        device const uint8_t*, device uint8_t*,                    \
        device const T*, device const T*, device T*,               \
        constant uint&, constant uint&, constant uint&,            \
        constant bool&, uint3, uint, uint, uint);

INSTANTIATE_DECODE_KERNELS(float)
INSTANTIATE_DECODE_KERNELS(half)
