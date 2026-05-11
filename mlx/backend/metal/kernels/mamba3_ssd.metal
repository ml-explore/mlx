// V3: Vectorized float4 + SRAM Bank Padding + Loop Unrolling
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_math>
#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

constant int CHUNK_SIZE = 64;
// SRAM Padding: We add 1 to the width to misalign the memory banks and prevent traffic jams
constant int PADDED_CHUNK = CHUNK_SIZE + 1; 

constant int HEADDIM_QK = 128; 
constant int HEADDIM_V = 64;

struct Mamba3Params {
    int seqlen, headdim_qk, headdim_v, chunk_size, num_chunks;
};

template <typename T>
[[kernel, max_total_threads_per_threadgroup(256)]] void mamba3_ssd_fused(
    device const T* q_in [[buffer(0)]],
    device const T* k_in [[buffer(1)]],
    device const T* v_in [[buffer(2)]],
    device const float* dt_in [[buffer(3)]],
    device const float* trap_in [[buffer(4)]],
    device const float* angles_in [[buffer(5)]], 
    device T* out [[buffer(6)]],
    constant Mamba3Params& params [[buffer(7)]],
    uint3 gid [[threadgroup_position_in_grid]], 
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 grid_size [[threadgroups_per_grid]]
) {
    const int head_idx = gid.x;
    const int batch_idx = gid.y;
    const int tid = lid.x; 
    
    const int stride_seq_qk = params.headdim_qk * grid_size.x; 
    const int stride_seq_v = params.headdim_v * grid_size.x;
    const int batch_offset_qk = batch_idx * params.seqlen * stride_seq_qk;
    const int batch_offset_v = batch_idx * params.seqlen * stride_seq_v;
    const int head_offset_qk = head_idx * params.headdim_qk;
    const int head_offset_v = head_idx * params.headdim_v;

    // TARGET 1: SRAM Padding applied to allocation
    threadgroup float shared_s[CHUNK_SIZE * PADDED_CHUNK];
    threadgroup float shared_da_cs[CHUNK_SIZE];  

    for (int chunk = 0; chunk < params.num_chunks; chunk++) {
        int seq_start = chunk * CHUNK_SIZE;
        int seq_offset_qk = seq_start * stride_seq_qk;
        int seq_offset_v = seq_start * stride_seq_v;

        // 1. INLINE SCAN
        if (tid < CHUNK_SIZE) {
            float da = dt_in[batch_idx * params.seqlen * grid_size.x + head_idx * params.seqlen + seq_start + tid] * 1.442695f; 
            shared_da_cs[tid] = da; 
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. VECTORIZED COOPERATIVE MATMUL
        for (int i = tid; i < (CHUNK_SIZE * CHUNK_SIZE); i += 256) {
            int row = i / CHUNK_SIZE;
            int col = i % CHUNK_SIZE;
            
            // TARGET 1: Map the logical 64x64 grid to the physical 64x65 padded SRAM
            int pad_idx = row * PADDED_CHUNK + col;
            
            if (row < col) {
                shared_s[pad_idx] = 0.0f; 
            } else {
                float sum = 0.0f;
                
                // TARGET 2: Force the compiler to copy-paste the instructions
                #pragma unroll
                for (int d = 0; d < HEADDIM_QK / 4; d++) {
                    int q_idx = batch_offset_qk + seq_offset_qk + (row * stride_seq_qk) + head_offset_qk + (d * 4);
                    int k_idx = batch_offset_qk + seq_offset_qk + (col * stride_seq_qk) + head_offset_qk + (d * 4);
                    
                    float4 q_vec = *(device const float4*)(q_in + q_idx);
                    float4 k_vec = *(device const float4*)(k_in + k_idx);
                    
                    sum += dot(q_vec, k_vec);
                }
                float decay = fast::exp2(min(shared_da_cs[row] - shared_da_cs[col], 0.0f));
                shared_s[pad_idx] = sum * decay;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. FAST OUTPUT MATMUL
        for (int i = tid; i < (CHUNK_SIZE * HEADDIM_V); i += 256) {
            int row = i / HEADDIM_V;
            int dim = i % HEADDIM_V;
            
            float sv_sum = 0.0f;
            
            // TARGET 2: Unroll the inner causal sequence loop
            #pragma unroll
            for (int t = 0; t <= row; t++) { 
                // TARGET 1: Read using the PADDED stride
                sv_sum += shared_s[row * PADDED_CHUNK + t] * (float)v_in[batch_offset_v + seq_offset_v + (t * stride_seq_v) + head_offset_v + dim];
            }
            int global_out_idx = batch_offset_v + seq_offset_v + (row * stride_seq_v) + head_offset_v + dim;
            out[global_out_idx] = (T)sv_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template [[host_name("mamba3_ssd_fused_float32")]] [[kernel]] 
void mamba3_ssd_fused<float>(
    device const float* q_in [[buffer(0)]], device const float* k_in [[buffer(1)]],
    device const float* v_in [[buffer(2)]], device const float* dt_in [[buffer(3)]],
    device const float* trap_in [[buffer(4)]], device const float* angles_in [[buffer(5)]],
    device float* out [[buffer(6)]], constant Mamba3Params& params [[buffer(7)]],
    uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
    uint3 grid_size [[threadgroups_per_grid]]);

template [[host_name("mamba3_ssd_fused_float16")]] [[kernel]] 
void mamba3_ssd_fused<half>(
    device const half* q_in [[buffer(0)]], device const half* k_in [[buffer(1)]],
    device const half* v_in [[buffer(2)]], device const float* dt_in [[buffer(3)]],
    device const float* trap_in [[buffer(4)]], device const float* angles_in [[buffer(5)]],
    device half* out [[buffer(6)]], constant Mamba3Params& params [[buffer(7)]],
    uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
    uint3 grid_size [[threadgroups_per_grid]]);