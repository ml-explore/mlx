// Copyright © 2024 Apple Inc.
// Entropy-Coded Quantization: rANS decode + dequantize for 2x compression

#pragma once

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// rANS constants
constant uint PROB_BITS = 14;
constant uint PROB_SCALE = 1 << PROB_BITS;  // 16384
constant uint RANS_L = 1 << 23;

// Tile header for compressed weights
struct EntropyTileHeader {
    uint n_streams;
    uint n_symbols;
    uint max_stream_len;
};

// ============================================================================
// rANS Decode + Dequantize + GEMV Kernel (Optimized)
// 
// Optimizations applied:
// 1. Physical interleaving for coalesced memory access
// 2. Register-cached frequency tables (eliminates ~600 cycles/symbol)
// 3. Threadgroup-cached symbol table (eliminates ~300 cycles/lookup)
// 4. SIMD reduction for dot product
// 5. 256 parallel rANS streams
//
// NOTE: This kernel expects the ENTIRE weight matrix to be encoded as a
// single flat array. Each output row shares the same compressed data but
// uses different scale/bias for that row.
// ============================================================================

template <typename T>
[[kernel]] void entropy_coded_qmv(
    device const uint8_t* compressed [[buffer(0)]],
    device const uint* stream_lengths [[buffer(1)]],
    device const uint16_t* freq [[buffer(2)]],
    device const uint16_t* cumfreq [[buffer(3)]],
    device const uint8_t* sym_table [[buffer(4)]],
    device const T* input [[buffer(5)]],
    device T* output [[buffer(6)]],
    device const T* scales [[buffer(7)]],
    device const T* biases [[buffer(8)]],
    constant uint& n_streams [[buffer(9)]],
    constant uint& n_symbols [[buffer(10)]],
    constant uint& max_stream_len [[buffer(11)]],
    constant uint& out_vec_size [[buffer(12)]],
    constant uint& in_vec_size [[buffer(13)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint thread_idx [[thread_index_in_threadgroup]]
) {
    // Each threadgroup handles one output row
    uint row = tgid.x;
    if (row >= out_vec_size) return;
    
    // ========================================================================
    // OPTIMIZATION 1: Register-cached frequency tables (32 bytes total)
    // Eliminates ~600 cycles/symbol of VRAM latency
    // ========================================================================
    uint16_t local_freq[16];
    uint16_t local_cumfreq[16];
    for (int i = 0; i < 16; i++) {
        local_freq[i] = freq[i];
        local_cumfreq[i] = cumfreq[i];
    }
    
    // ========================================================================
    // OPTIMIZATION 2: Threadgroup-cached symbol lookup table
    // 16KB table cached in fast threadgroup memory (~10 cycles vs ~300 device)
    // Each thread loads 64 entries (256 threads * 64 = 16384 = PROB_SCALE)
    // ========================================================================
    threadgroup uint8_t shared_sym_table[PROB_SCALE];
    
    // Cooperative load: each thread loads 64 consecutive entries
    uint load_start = thread_idx * 64;
    if (load_start < PROB_SCALE) {
        for (uint i = 0; i < 64 && (load_start + i) < PROB_SCALE; i++) {
            shared_sym_table[load_start + i] = sym_table[load_start + i];
        }
    }
    
    // Ensure all threads have loaded before proceeding
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Shared memory for SIMD reduction
    threadgroup float partial_sums[32];
    
    uint stream_idx = simd_lane + simd_group * 32;
    float acc = 0.0f;
    
    // Row-specific scale and bias
    T scale = scales[row];
    T bias = biases[row];
    
    // Calculate offsets for this row's weights in the flat encoding
    // Weights are stored row-major: [row0_col0, row0_col1, ..., row0_colN, row1_col0, ...]
    uint row_start_symbol = row * in_vec_size;
    
    if (stream_idx < n_streams) {
        uint stream_len = stream_lengths[stream_idx];
        
        if (stream_len >= 4) {
            // ================================================================
            // OPTIMIZATION 3: Coalesced state initialization
            // Physical interleaving ensures adjacent threads read adjacent bytes
            // ================================================================
            uint b0 = compressed[stream_idx + 0 * n_streams];
            uint b1 = compressed[stream_idx + 1 * n_streams];
            uint b2 = compressed[stream_idx + 2 * n_streams];
            uint b3 = compressed[stream_idx + 3 * n_streams];
            uint state = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
            uint ptr = 4;
            
            // Calculate how many symbols this stream handles for the entire matrix
            uint total_symbols_in_stream = (n_symbols - stream_idx + n_streams - 1) / n_streams;
            
            // Decode all symbols in this stream, but only accumulate those for our row
            for (uint i = 0; i < total_symbols_in_stream; i++) {
                uint global_sym_idx = stream_idx + i * n_streams;
                if (global_sym_idx >= n_symbols) break;
                
                // Decode symbol using threadgroup-cached table (fast!)
                uint slot = state & (PROB_SCALE - 1);
                uint8_t s = shared_sym_table[slot];
                
                // State update with register-cached tables (no memory access)
                uint freq_s = local_freq[s];
                uint start_s = local_cumfreq[s];
                state = freq_s * (state >> PROB_BITS) + slot - start_s;
                
                // ============================================================
                // OPTIMIZATION 4: Coalesced renormalization reads
                // Adjacent threads read adjacent bytes during renormalization
                // ============================================================
                while (state < RANS_L && ptr < stream_len) {
                    uint8_t b = compressed[stream_idx + ptr * n_streams];
                    state = (state << 8) | b;
                    ptr++;
                }
                
                // Check if this symbol belongs to our row
                if (global_sym_idx >= row_start_symbol && 
                    global_sym_idx < row_start_symbol + in_vec_size) {
                    uint col = global_sym_idx - row_start_symbol;
                    
                    // Dequantize + MAC (fused, no intermediate storage)
                    T weight = T(s) * scale + bias;
                    acc += float(weight * input[col]);
                }
            }
        }
    }
    
    // ========================================================================
    // OPTIMIZATION 5: SIMD reduction (hardware-accelerated)
    // ========================================================================
    acc = simd_sum(acc);
    
    if (simd_lane == 0) {
        partial_sums[simd_group] = acc;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Final reduction across SIMD groups
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
// Per-Row Encoded rANS Decode + Dequantize + GEMV Kernel
// 
// This variant expects each row to be independently encoded, which allows:
// 1. Parallel decode of all rows simultaneously
// 2. Better compression (per-row frequency tables)
// 3. Simpler memory access pattern
// ============================================================================

template <typename T>
[[kernel]] void entropy_coded_qmv_per_row(
    device const uint8_t* compressed [[buffer(0)]],      // All rows concatenated
    device const uint* row_offsets [[buffer(1)]],        // Offset to each row's data
    device const uint* stream_lengths [[buffer(2)]],     // Per-stream lengths (global)
    device const uint16_t* freq [[buffer(3)]],           // Single freq table
    device const uint16_t* cumfreq [[buffer(4)]],
    device const uint8_t* sym_table [[buffer(5)]],
    device const T* input [[buffer(6)]],
    device T* output [[buffer(7)]],
    device const T* scales [[buffer(8)]],
    device const T* biases [[buffer(9)]],
    constant uint& n_streams [[buffer(10)]],
    constant uint& in_vec_size [[buffer(11)]],
    constant uint& max_stream_len [[buffer(12)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint thread_idx [[thread_index_in_threadgroup]]
) {
    uint row = tgid.x;
    
    // Register-cached frequency tables
    uint16_t local_freq[16];
    uint16_t local_cumfreq[16];
    for (int i = 0; i < 16; i++) {
        local_freq[i] = freq[i];
        local_cumfreq[i] = cumfreq[i];
    }
    
    // Threadgroup-cached symbol table
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
    
    // Get this row's compressed data
    device const uint8_t* row_data = compressed + row_offsets[row];
    
    if (stream_idx < n_streams) {
        uint stream_len = stream_lengths[stream_idx];
        
        if (stream_len >= 4) {
            // Coalesced state init from this row's data
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
                
                // Dequantize + MAC
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
// Standalone rANS Decode Kernel (for GPU_ASYNC strategy)
// Decodes to indices without fusing with GEMV
// ============================================================================

[[kernel]] void entropy_decode_only(
    device const uint8_t* compressed [[buffer(0)]],
    device const uint* stream_lengths [[buffer(1)]],
    device const uint16_t* freq [[buffer(2)]],
    device const uint16_t* cumfreq [[buffer(3)]],
    device const uint8_t* sym_table [[buffer(4)]],
    device uint8_t* output [[buffer(5)]],
    constant uint& n_streams [[buffer(6)]],
    constant uint& n_symbols [[buffer(7)]],
    constant uint& max_stream_len [[buffer(8)]],
    uint tid [[thread_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]]
) {
    uint stream_idx = tid;
    if (stream_idx >= n_streams) return;
    
    // Register-cached tables
    uint16_t local_freq[16];
    uint16_t local_cumfreq[16];
    for (int i = 0; i < 16; i++) {
        local_freq[i] = freq[i];
        local_cumfreq[i] = cumfreq[i];
    }
    
    // Note: For standalone decode, we use device memory for sym_table
    // (threadgroup caching less beneficial for 1D dispatch)
    
    uint stream_len = stream_lengths[stream_idx];
    if (stream_len < 4) return;
    
    // Coalesced state init
    uint b0 = compressed[stream_idx + 0 * n_streams];
    uint b1 = compressed[stream_idx + 1 * n_streams];
    uint b2 = compressed[stream_idx + 2 * n_streams];
    uint b3 = compressed[stream_idx + 3 * n_streams];
    uint state = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
    uint ptr = 4;
    
    uint symbols_in_stream = (n_symbols - stream_idx + n_streams - 1) / n_streams;
    
    for (uint i = 0; i < symbols_in_stream; i++) {
        uint output_idx = stream_idx + i * n_streams;
        if (output_idx >= n_symbols) break;
        
        uint slot = state & (PROB_SCALE - 1);
        uint8_t s = sym_table[slot];
        output[output_idx] = s;
        
        uint freq_s = local_freq[s];
        uint start_s = local_cumfreq[s];
        state = freq_s * (state >> PROB_BITS) + slot - start_s;
        
        while (state < RANS_L && ptr < stream_len) {
            uint8_t b = compressed[stream_idx + ptr * n_streams];
            state = (state << 8) | b;
            ptr++;
        }
    }
}

// ============================================================================
// Dequantize kernel (for pre-decoded weights)
// ============================================================================

template <typename T>
[[kernel]] void dequantize_4bit(
    device const uint8_t* indices [[buffer(0)]],
    device T* output [[buffer(1)]],
    device const T* scales [[buffer(2)]],
    device const T* biases [[buffer(3)]],
    constant uint& n_elements [[buffer(4)]],
    constant uint& group_size [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n_elements) return;
    
    uint group_id = tid / group_size;
    T scale = scales[group_id];
    T bias = biases[group_id];
    
    output[tid] = T(indices[tid]) * scale + bias;
}
