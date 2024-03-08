#include <metal_stdlib>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/scaled_dot_product_attention_params.h"
using namespace metal;

template<typename T, typename T2, typename T4, uint16_t TILE_SIZE_CONST, uint16_t NSIMDGROUPS>
[[kernel]] void fast_inference_sdpa_compute_partials_template(const device T *Q [[buffer(0)]],
                              const device T *K [[buffer(1)]],
                              const device T *V [[buffer(2)]],
                              const device uint64_t& L [[buffer(3)]],
                              const device MLXScaledDotProductAttentionParams& params [[buffer(4)]],
                              device float* O_partials [[buffer(5)]],
                              device float* p_lse [[buffer(6)]],
                              device float* p_maxes [[buffer(7)]],
                              threadgroup T* threadgroup_block [[threadgroup(0)]],
                              uint simd_lane_id [[thread_index_in_simdgroup]],
                              uint simd_group_id [[simdgroup_index_in_threadgroup]],
                              uint3 tid [[threadgroup_position_in_grid]]) {
    constexpr const size_t DK = 128;
    constexpr const ulong SIMDGROUP_MATRIX_LOAD_FACTOR = 8;
    constexpr const size_t THREADS_PER_SIMDGROUP = 32;
    constexpr const uint iter_offset = NSIMDGROUPS * 4;
    const bool is_gqa = params.N_KV_HEADS != params.N_Q_HEADS;
    uint kv_head_offset_factor = tid.x;
    if(is_gqa) {
        int q_kv_head_ratio = params.N_Q_HEADS / params.N_KV_HEADS;
        kv_head_offset_factor = tid.x / q_kv_head_ratio;
    }
    constexpr const uint16_t P_VEC4 = TILE_SIZE_CONST / NSIMDGROUPS / 4;
    constexpr const size_t MATRIX_LOADS_PER_SIMDGROUP = TILE_SIZE_CONST / (SIMDGROUP_MATRIX_LOAD_FACTOR * NSIMDGROUPS);
    constexpr const size_t MATRIX_COLS = DK / SIMDGROUP_MATRIX_LOAD_FACTOR;
    constexpr const uint totalSmemV = SIMDGROUP_MATRIX_LOAD_FACTOR * SIMDGROUP_MATRIX_LOAD_FACTOR * (MATRIX_LOADS_PER_SIMDGROUP + 1) * NSIMDGROUPS;

    threadgroup T4* smemFlush = (threadgroup T4*)threadgroup_block;
    #pragma clang loop unroll(full)
    for(uint i = 0; i < 8; i++) {
        smemFlush[simd_lane_id + simd_group_id * THREADS_PER_SIMDGROUP + i * NSIMDGROUPS * THREADS_PER_SIMDGROUP] = T4(0.f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // TODO: multiple query sequence length for speculative decoding
    const uint tgroup_query_head_offset = tid.x * DK + tid.z * (params.N_Q_HEADS * DK);

    const uint tgroup_k_head_offset = kv_head_offset_factor * DK * L;
    const uint tgroup_k_tile_offset = tid.y * TILE_SIZE_CONST * DK;
    const uint tgroup_k_batch_offset = tid.z * L * params.N_KV_HEADS * DK;

    const device T* baseK = K + tgroup_k_batch_offset + tgroup_k_tile_offset + tgroup_k_head_offset;
    const device T* baseQ = Q + tgroup_query_head_offset;

    device T4* simdgroupQueryData = (device T4*)baseQ;

    constexpr const size_t ACCUM_PER_GROUP = TILE_SIZE_CONST / NSIMDGROUPS;
    float threadAccum[ACCUM_PER_GROUP];

    #pragma clang loop unroll(full)
    for(size_t threadAccumIndex = 0; threadAccumIndex < ACCUM_PER_GROUP; threadAccumIndex++) {
        threadAccum[threadAccumIndex] = -INFINITY;
    }

    uint KROW_ACCUM_INDEX = 0;

    const int32_t SEQUENCE_LENGTH_LESS_TILE_SIZE = L - TILE_SIZE_CONST;
    const bool LAST_TILE = (tid.y + 1) * TILE_SIZE_CONST >= L;
    const bool LAST_TILE_ALIGNED = (SEQUENCE_LENGTH_LESS_TILE_SIZE == int32_t(tid.y * TILE_SIZE_CONST));

    T4 thread_data_x4;
    T4 thread_data_y4;
    if(!LAST_TILE || LAST_TILE_ALIGNED) {
        thread_data_x4 = *(simdgroupQueryData + simd_lane_id);
        #pragma clang loop unroll(full)
        for(size_t KROW = simd_group_id; KROW < TILE_SIZE_CONST; KROW += NSIMDGROUPS) {
            const uint KROW_OFFSET = KROW * DK;
            const device T* baseKRow = baseK + KROW_OFFSET;
            device T4* keysData = (device T4*)baseKRow;
            thread_data_y4 = *(keysData + simd_lane_id);
            T kq_scalar = dot(thread_data_x4, thread_data_y4);
            threadAccum[KROW_ACCUM_INDEX] = float(kq_scalar);
            KROW_ACCUM_INDEX++;
        }
    } else {
        thread_data_x4 = *(simdgroupQueryData + simd_lane_id);
        const uint START_ROW = tid.y * TILE_SIZE_CONST;
        const device T* baseKThisHead = K + tgroup_k_batch_offset + tgroup_k_head_offset;

        for(size_t KROW = START_ROW + simd_group_id; KROW < L; KROW += NSIMDGROUPS) {
            const uint KROW_OFFSET = KROW * DK;
            const device T* baseKRow = baseKThisHead + KROW_OFFSET;
            device T4* keysData = (device T4*)baseKRow;
            thread_data_y4 = *(keysData + simd_lane_id);
            T kq_scalar = dot(thread_data_x4, thread_data_y4);
            threadAccum[KROW_ACCUM_INDEX] = float(kq_scalar);
            KROW_ACCUM_INDEX++;
        }
    }
    threadgroup float* smemP = (threadgroup float*)threadgroup_block;

    #pragma clang loop unroll(full)
    for(size_t i = 0; i < P_VEC4; i++) {
        thread_data_x4 = T4(threadAccum[4 * i], threadAccum[4 * i + 1], threadAccum[4 * i + 2], threadAccum[4 * i + 3]);
        simdgroup_barrier(mem_flags::mem_none);
        thread_data_y4 = simd_sum(thread_data_x4);
        if(simd_lane_id == 0) {
            const uint base_smem_p_offset = i * iter_offset + simd_group_id;
            smemP[base_smem_p_offset + NSIMDGROUPS * 0] = float(thread_data_y4.x);
            smemP[base_smem_p_offset + NSIMDGROUPS * 1] = float(thread_data_y4.y);
            smemP[base_smem_p_offset + NSIMDGROUPS * 2] = float(thread_data_y4.z);
            smemP[base_smem_p_offset + NSIMDGROUPS * 3] = float(thread_data_y4.w);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float groupMax;
    float lse = 0.f;

    constexpr const size_t THREADS_PER_THREADGROUP_TIMES_4 = 4 * 32;
    constexpr const size_t ACCUM_ARRAY_LENGTH = TILE_SIZE_CONST / THREADS_PER_THREADGROUP_TIMES_4 + 1;
    float4 pvals[ACCUM_ARRAY_LENGTH];

    #pragma clang loop unroll(full)
    for(uint accum_array_iter = 0; accum_array_iter < ACCUM_ARRAY_LENGTH; accum_array_iter++) {
        pvals[accum_array_iter] = float4(-INFINITY);
    }

    if (TILE_SIZE_CONST == 64) {
        threadgroup float2* smemPtrFlt2 = (threadgroup float2*)threadgroup_block;
        float2 vals = smemPtrFlt2[simd_lane_id];
        vals *= params.INV_ALPHA;
        float maxval = max(vals.x, vals.y);
        simdgroup_barrier(mem_flags::mem_none);
        groupMax = simd_max(maxval);

        float2 expf_shifted = exp(vals - groupMax);
        float sumExpLocal = expf_shifted.x + expf_shifted.y;
        simdgroup_barrier(mem_flags::mem_none);
        float tgroupExpSum = simd_sum(sumExpLocal);

        lse = log(tgroupExpSum);
        float2 local_p_hat = expf_shifted / tgroupExpSum;
        pvals[0].x = local_p_hat.x;
        pvals[0].y = local_p_hat.y;
        smemPtrFlt2[simd_lane_id] = float2(0.f);
    }
    constexpr const bool TILE_SIZE_LARGER_THAN_64 = TILE_SIZE_CONST > 64;
    constexpr const int TILE_SIZE_ITERS_128 = TILE_SIZE_CONST / 128;

    if (TILE_SIZE_LARGER_THAN_64) {
        float maxval = -INFINITY;
        threadgroup float4* smemPtrFlt4 = (threadgroup float4*)threadgroup_block;
        #pragma clang loop unroll(full)
        for(int i = 0; i < TILE_SIZE_ITERS_128; i++) {
            float4 vals = smemPtrFlt4[simd_lane_id + i * THREADS_PER_SIMDGROUP];
            vals *= params.INV_ALPHA;
            pvals[i] = vals;
            maxval = fmax3(vals.x, vals.y, maxval);
            maxval = fmax3(vals.z, vals.w, maxval);
        }
        simdgroup_barrier(mem_flags::mem_none);
        groupMax = simd_max(maxval);

        float sumExpLocal = 0.f;
        #pragma clang loop unroll(full)
        for(int i = 0; i < TILE_SIZE_ITERS_128; i++) {
            pvals[i] = exp(pvals[i] - groupMax);
            sumExpLocal += pvals[i].x + pvals[i].y + pvals[i].z + pvals[i].w;
        }
        simdgroup_barrier(mem_flags::mem_none);
        float tgroupExpSum = simd_sum(sumExpLocal);
        lse = log(tgroupExpSum);
        #pragma clang loop unroll(full)
        for(int i = 0; i < TILE_SIZE_ITERS_128; i++) {
            pvals[i] = pvals[i] / tgroupExpSum;
            smemPtrFlt4[simd_lane_id + i * THREADS_PER_SIMDGROUP] = float4(0.f);
        }
    }

    threadgroup T* smemV = (threadgroup T*)threadgroup_block;

    const size_t v_batch_offset = tid.z * params.N_KV_HEADS * L * DK;
    const size_t v_head_offset = kv_head_offset_factor * L * DK;

    const size_t v_tile_offset = tid.y * TILE_SIZE_CONST * DK;
    const size_t v_offset = v_batch_offset + v_head_offset + v_tile_offset;
    device T* baseV = (device T*)V + v_offset;

    threadgroup float* smemOpartial = (threadgroup float*)(smemV + totalSmemV);

    if (!LAST_TILE || LAST_TILE_ALIGNED) {
        #pragma clang loop unroll(full)
        for(size_t col = 0; col < MATRIX_COLS; col++) {
            uint matrix_load_loop_iter = 0;
            constexpr const size_t TILE_SIZE_CONST_DIV_8 = TILE_SIZE_CONST / 8;
            
            for(size_t tile_start = simd_group_id; tile_start < TILE_SIZE_CONST_DIV_8; tile_start += NSIMDGROUPS) {
                simdgroup_matrix<T, 8, 8> tmp;
                ulong simdgroup_matrix_offset = matrix_load_loop_iter * NSIMDGROUPS * SIMDGROUP_MATRIX_LOAD_FACTOR + simd_group_id * SIMDGROUP_MATRIX_LOAD_FACTOR;
                ulong2 matrixOrigin = ulong2(col * SIMDGROUP_MATRIX_LOAD_FACTOR, simdgroup_matrix_offset);
                simdgroup_load(tmp, baseV, DK, matrixOrigin, true);
                const ulong2 matrixOriginSmem = ulong2(simdgroup_matrix_offset, 0);
                const ulong elemsPerRowSmem = TILE_SIZE_CONST;
                simdgroup_store(tmp, smemV, elemsPerRowSmem, matrixOriginSmem, false);
                matrix_load_loop_iter++;
            };
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            if (TILE_SIZE_CONST == 64) {
                T2 local_p_hat = T2(pvals[0].x, pvals[0].y);
                uint loop_iter = 0;
                threadgroup float* oPartialSmem = smemOpartial + SIMDGROUP_MATRIX_LOAD_FACTOR * col;
                
                #pragma clang loop unroll(full)
                for(size_t row = simd_group_id; row < SIMDGROUP_MATRIX_LOAD_FACTOR; row += NSIMDGROUPS) {
                    threadgroup T* smemV_row = smemV + (TILE_SIZE_CONST * row);
                    threadgroup T2* smemV2 = (threadgroup T2*)smemV_row;
                    T2 v_local = *(smemV2 + simd_lane_id);
                    
                    T val = dot(local_p_hat, v_local);
                    simdgroup_barrier(mem_flags::mem_none);

                    T row_sum = simd_sum(val);
                    oPartialSmem[simd_group_id + loop_iter * NSIMDGROUPS] = float(row_sum);
                    loop_iter++;
                }
            }
            
            if (TILE_SIZE_CONST > 64) {
                constexpr const size_t TILE_SIZE_CONST_DIV_128 = (TILE_SIZE_CONST + 1) / 128;
                threadgroup float* oPartialSmem = smemOpartial + SIMDGROUP_MATRIX_LOAD_FACTOR * col;
                uint loop_iter = 0;
                for(size_t row = simd_group_id; row < SIMDGROUP_MATRIX_LOAD_FACTOR; row += NSIMDGROUPS) {
                    threadgroup T* smemV_row = smemV + (TILE_SIZE_CONST * row);
                    
                    T row_sum = 0.f;
                    for(size_t i = 0; i < TILE_SIZE_CONST_DIV_128; i++) {
                        threadgroup T4* smemV2 = (threadgroup T4*)smemV_row;
                        T4 v_local = *(smemV2 + simd_lane_id + i * THREADS_PER_SIMDGROUP);
                        T4 p_local = T4(pvals[i]);
                        T val = dot(p_local, v_local);
                        row_sum += val;
                    }
                    simdgroup_barrier(mem_flags::mem_none);
                    row_sum = simd_sum(row_sum);
                    oPartialSmem[simd_group_id + loop_iter * NSIMDGROUPS] = float(row_sum);
                    loop_iter++;
                }
            }
        }
    } else {
        const int32_t START_ROW = tid.y * TILE_SIZE_CONST;
        const int32_t MAX_START_ROW = L - SIMDGROUP_MATRIX_LOAD_FACTOR + 1;
        const device T* baseVThisHead = V + v_batch_offset + v_head_offset;
        constexpr const int ROWS_PER_ITER = 8;
        #pragma clang loop unroll(full)
        for(size_t col = 0; col < MATRIX_COLS; col++) {
            uint smem_col_index = simd_group_id * SIMDGROUP_MATRIX_LOAD_FACTOR;
            int32_t tile_start;
            for(tile_start = START_ROW + simd_group_id * SIMDGROUP_MATRIX_LOAD_FACTOR; tile_start < MAX_START_ROW; tile_start += NSIMDGROUPS * SIMDGROUP_MATRIX_LOAD_FACTOR) {
                simdgroup_matrix<T, 8, 8> tmp;
                ulong2 matrixOrigin = ulong2(col * SIMDGROUP_MATRIX_LOAD_FACTOR, tile_start);
                simdgroup_load(tmp, baseVThisHead, DK, matrixOrigin, /* transpose */ true);
                const ulong2 matrixOriginSmem = ulong2(smem_col_index, 0);
                constexpr const ulong elemsPerRowSmem = TILE_SIZE_CONST;
                simdgroup_store(tmp, smemV, elemsPerRowSmem, matrixOriginSmem, /* transpose */ false);
                smem_col_index += NSIMDGROUPS * SIMDGROUP_MATRIX_LOAD_FACTOR;
            };

            tile_start = ((L / SIMDGROUP_MATRIX_LOAD_FACTOR) * SIMDGROUP_MATRIX_LOAD_FACTOR);

            const int32_t INT_L = int32_t(L);
            for(int row_index  = tile_start + simd_group_id ; row_index < INT_L; row_index += NSIMDGROUPS) {
                if(simd_lane_id < SIMDGROUP_MATRIX_LOAD_FACTOR) {
                    const uint elems_per_row_gmem = DK;
                    const uint col_index_v_gmem = col * SIMDGROUP_MATRIX_LOAD_FACTOR + simd_lane_id;
                    const uint row_index_v_gmem = row_index;

                    const uint elems_per_row_smem = TILE_SIZE_CONST;
                    const uint col_index_v_smem = row_index % TILE_SIZE_CONST;
                    const uint row_index_v_smem = simd_lane_id;

                    const uint scalar_offset_gmem = row_index_v_gmem * elems_per_row_gmem + col_index_v_gmem;
                    const uint scalar_offset_smem = row_index_v_smem * elems_per_row_smem + col_index_v_smem;
                    T vdata = T(*(baseVThisHead + scalar_offset_gmem));
                    smemV[scalar_offset_smem] = vdata;
                    smem_col_index += NSIMDGROUPS;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (TILE_SIZE_CONST == 64) {
                T2 local_p_hat = T2(pvals[0].x, pvals[0].y);
                threadgroup float* oPartialSmem = smemOpartial + SIMDGROUP_MATRIX_LOAD_FACTOR * col;
                for(size_t smem_row_index = simd_group_id;
                    smem_row_index < ROWS_PER_ITER; smem_row_index += NSIMDGROUPS) {
                    threadgroup T* smemV_row = smemV + (TILE_SIZE_CONST * smem_row_index);
                    threadgroup T2* smemV2 = (threadgroup T2*)smemV_row;
                    T2 v_local = *(smemV2 + simd_lane_id);
                    T val = dot(local_p_hat, v_local);
                    simdgroup_barrier(mem_flags::mem_none);
                    T row_sum = simd_sum(val);
                    oPartialSmem[smem_row_index] = float(row_sum);
                }
            }

            if (TILE_SIZE_CONST > 64) {
                threadgroup float* oPartialSmem = smemOpartial + SIMDGROUP_MATRIX_LOAD_FACTOR * col;
                uint loop_count = 0;
                for(size_t row_index = simd_group_id;
                    row_index < ROWS_PER_ITER; row_index += NSIMDGROUPS) {
                    T row_sum = 0.f;
                    for(size_t tile_iters = 0; tile_iters < TILE_SIZE_ITERS_128; tile_iters++) {
                        threadgroup T* smemV_row = smemV + (TILE_SIZE_CONST * row_index);
                        threadgroup T4* smemV2 = (threadgroup T4*)smemV_row;
                        T4 v_local = *(smemV2 + simd_lane_id + tile_iters * THREADS_PER_SIMDGROUP);
                        T4 p_local = T4(pvals[tile_iters]);
                        row_sum += dot(p_local, v_local);
                        
                    }
                    simdgroup_barrier(mem_flags::mem_none);
                    row_sum = simd_sum(row_sum);
                    oPartialSmem[simd_group_id + NSIMDGROUPS * loop_count] = float(row_sum);
                    loop_count++;
                }
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if(simd_group_id == 0) {
        threadgroup float4* oPartialVec4 = (threadgroup float4*)smemOpartial;
        float4 vals = *(oPartialVec4 + simd_lane_id);
        device float* oPartialGmem = O_partials + tid.x * DK * params.KV_TILES + tid.y * DK;
        device float4* oPartialGmemVec4 = (device float4*)oPartialGmem;
        oPartialGmemVec4[simd_lane_id] = vals;
    }

    if(simd_group_id == 0 && simd_lane_id == 0) {
        const uint tileIndex = tid.y;
        const uint gmem_partial_scalar_offset = tid.z * params.N_Q_HEADS * params.KV_TILES + tid.x * params.KV_TILES + tileIndex;
        p_lse[gmem_partial_scalar_offset] = lse;
        p_maxes[gmem_partial_scalar_offset] = groupMax;
    }
}

#define instantiate_fast_inference_sdpa_to_partials_kernel(itype, itype2, itype4, tile_size, nsimdgroups) \
template [[host_name("fast_inference_sdpa_compute_partials_" #itype "_" #tile_size "_" #nsimdgroups )]] \
[[kernel]] void fast_inference_sdpa_compute_partials_template<itype, itype2, itype4, tile_size, nsimdgroups>( \
    const device itype *Q [[buffer(0)]], \
    const device itype *K [[buffer(1)]], \
    const device itype *V [[buffer(2)]], \
    const device uint64_t& L [[buffer(3)]], \
    const device MLXScaledDotProductAttentionParams& params [[buffer(4)]], \
    device float* O_partials [[buffer(5)]], \
    device float* p_lse [[buffer(6)]], \
    device float* p_maxes [[buffer(7)]], \
    threadgroup itype *threadgroup_block [[threadgroup(0)]], \
    uint simd_lane_id [[thread_index_in_simdgroup]], \
    uint simd_group_id [[simdgroup_index_in_threadgroup]], \
    uint3 tid [[threadgroup_position_in_grid]]);


#define instantiate_fast_inference_sdpa_to_partials_shapes_helper(itype, itype2, itype4, tile_size) \
    instantiate_fast_inference_sdpa_to_partials_kernel(itype, itype2, itype4, tile_size, 4) \
    instantiate_fast_inference_sdpa_to_partials_kernel(itype, itype2, itype4, tile_size, 8) \

instantiate_fast_inference_sdpa_to_partials_shapes_helper(float, float2, float4, 64);
instantiate_fast_inference_sdpa_to_partials_shapes_helper(float, float2, float4, 128);
instantiate_fast_inference_sdpa_to_partials_shapes_helper(float, float2, float4, 256);
instantiate_fast_inference_sdpa_to_partials_shapes_helper(float, float2, float4, 512);

instantiate_fast_inference_sdpa_to_partials_shapes_helper(half, half2, half4, 64);
instantiate_fast_inference_sdpa_to_partials_shapes_helper(half, half2, half4, 128);
instantiate_fast_inference_sdpa_to_partials_shapes_helper(half, half2, half4, 256);
instantiate_fast_inference_sdpa_to_partials_shapes_helper(half, half2, half4, 512);


template <typename T>
void fast_inference_sdpa_reduce_tiles_template(
    const device float *O_partials [[buffer(0)]],
    const device float *p_lse[[buffer(1)]],
    const device float *p_maxes [[buffer(2)]],
    const device MLXScaledDotProductAttentionParams& params [[buffer(3)]],
    device T* O [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {

    constexpr const int DK = 128;
    const ulong offset_rows = tid.z * params.KV_TILES * params.N_Q_HEADS + tid.x * params.KV_TILES;
    const device float* p_lse_row = p_lse + offset_rows;
    const device float* p_rowmax_row = p_maxes + offset_rows;
    // reserve some number of registers.  this constitutes an assumption on max value of KV TILES.
    constexpr const uint8_t reserve = 128;
    float p_lse_regs[reserve];
    float p_rowmax_regs[reserve];
    float weights[reserve];

    float true_max = -INFINITY;
    for(size_t i = 0; i < params.KV_TILES; i++) {
        p_lse_regs[i] = float(*(p_lse_row + i));
        p_rowmax_regs[i] = float(*(p_rowmax_row + i));
        true_max = fmax(p_rowmax_regs[i], true_max);
        weights[i] = exp(p_lse_regs[i]);
    }

    float denom = 0.f;
    for(size_t i = 0; i < params.KV_TILES; i++) {
        weights[i] *= exp(p_rowmax_regs[i]-true_max);
        denom += weights[i];
    }

    const device float* O_partials_with_offset = O_partials + tid.z * params.N_Q_HEADS * DK * params.KV_TILES + tid.x * DK * params.KV_TILES;

    float o_value = 0.f;
    for(size_t i = 0; i < params.KV_TILES; i++) {
        float val = *(O_partials_with_offset + i * DK + lid.x);
        o_value += val * weights[i] / denom;
    }
    device T* O_gmem = O + tid.z * params.N_Q_HEADS * DK + tid.x * DK;
    O_gmem[lid.x] = T(o_value);
    return;
}


kernel void fast_inference_sdpa_reduce_tiles_float(
    const device float *O_partials [[buffer(0)]],
    const device float *p_lse[[buffer(1)]],
    const device float *p_maxes [[buffer(2)]],
    const device MLXScaledDotProductAttentionParams& params [[buffer(3)]],
    device float* O [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
    fast_inference_sdpa_reduce_tiles_template<float>(O_partials, p_lse, p_maxes, params,
                                     O, tid, lid);
}

kernel void fast_inference_sdpa_reduce_tiles_half(
    const device float *O_partials [[buffer(0)]],
    const device float *p_lse[[buffer(1)]],
    const device float *p_maxes [[buffer(2)]],
    const device MLXScaledDotProductAttentionParams& params [[buffer(3)]],
    device half* O [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
    fast_inference_sdpa_reduce_tiles_template<half>(O_partials, p_lse, p_maxes, params,
                                     O, tid, lid);
}
