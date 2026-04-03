// Copyright © 2026 Apple Inc. (sdpa_vector.h base) + MLA modifications

// Fused quantized MLA SDPA for decode (L==1).
// Based on sdpa_vector.h with:
//   - INT4 affine dequant replacing fp16 key loads
//   - Split nope/rope scoring
//   - Latent reuse for value accumulation (dequant once, use twice)
//   - Shared latent across heads (H_TILE heads per threadgroup)
//
// Replaces 5+ separate kernel dispatches with one fused kernel.
// All intermediates stay in registers/threadgroup memory.

#include <metal_stdlib>
using namespace metal;

// MLA dimensions (Mistral Small 4)
constant uint MLA_D       = 256;  // kv_lora_rank (nope scoring + value dim)
constant uint MLA_RD      = 64;   // qk_rope_head_dim
constant uint MLA_GS      = 64;   // quantization group size
constant uint MLA_NGROUPS = 4;    // MLA_D / MLA_GS
constant uint MLA_WORDS   = 32;   // MLA_D / 8 (8 values per uint32 at 4-bit)

// Thread organization
// BN = 32 simdgroups process KV positions in parallel (same as sdpa_vector)
// BD = 32 threads per simdgroup
// Each thread handles 8 nope dims (256/32) and 2 rope dims (64/32)
// H_TILE heads share the same dequanted latent via threadgroup memory
constant uint BN = 32;
constant uint BD = 32;

template <typename T>
[[kernel]] void mla_fused_sdpa(
    const device T*        q_nope      [[buffer(0)]],   // [B, H, 256] post-embed_q, NOT pre-scaled
    const device T*        q_pe        [[buffer(1)]],   // [B, H, 64] NOT pre-scaled
    const device uint32_t* lat_packed  [[buffer(2)]],   // [B, S, 32] INT4 packed latent (shared)
    const device T*        lat_scales  [[buffer(3)]],   // [B, S, 4] per-group scales
    const device T*        lat_biases  [[buffer(4)]],   // [B, S, 4] per-group biases
    const device T*        k_pe        [[buffer(5)]],   // [B, S, 64] fp16 RoPE keys (shared)
    device T*              out         [[buffer(6)]],   // [B, H, 256] latent attention output
    const constant uint&   B           [[buffer(7)]],
    const constant uint&   H           [[buffer(8)]],
    const constant uint&   S           [[buffer(9)]],
    const constant float&  scale       [[buffer(10)]],  // attention scale, applied at query load
    uint3  tid       [[threadgroup_position_in_grid]],
    uint   simd_gid  [[simdgroup_index_in_threadgroup]],
    uint   simd_lid  [[thread_index_in_simdgroup]]) {

    // tid.x = head index (one per head, not head-tiled for v1)
    // tid.y = batch index
    const uint head_idx = tid.x;
    const uint batch_idx = tid.y;

    if (head_idx >= H || batch_idx >= B) return;

    typedef float U;

    // --- Load query into registers with scale applied (sdpa_vector.h pattern) ---
    // Scale applied once at load, not per KV position
    // 8 nope dims per thread (256 / 32 = 8)
    thread U q_n[8];
    const device T* q_nope_ptr = q_nope + (batch_idx * H + head_idx) * MLA_D;
    for (uint i = 0; i < 8; i++) {
        q_n[i] = static_cast<U>(scale) * static_cast<U>(q_nope_ptr[simd_lid * 8 + i]);
    }

    // 2 rope dims per thread (64 / 32 = 2)
    thread U q_r[2];
    const device T* q_pe_ptr = q_pe + (batch_idx * H + head_idx) * MLA_RD;
    for (uint i = 0; i < 2; i++) {
        q_r[i] = static_cast<U>(scale) * static_cast<U>(q_pe_ptr[simd_lid * 2 + i]);
    }

    // --- Output accumulator (256 latent dims, 8 per thread) ---
    thread U o[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // --- Online softmax state ---
    U max_score = -1e20f;
    U sum_exp_score = 0;

    // --- Threadgroup memory for cross-simdgroup reduction ---
    threadgroup U tg_max[BN];
    threadgroup U tg_sum[BN];
    threadgroup U tg_out[BN * BD];  // BN simdgroups × 32 threads for output transpose

    // --- Base pointers for this batch (latent is shared across heads) ---
    const uint lat_base = batch_idx * S;
    const uint kpe_base = batch_idx * S * MLA_RD;

    // --- Main loop: process KV positions distributed across simdgroups ---
    // Each simdgroup handles positions: simd_gid, simd_gid + BN, simd_gid + 2*BN, ...
    for (uint s = simd_gid; s < S; s += BN) {

        // 1. DEQUANT latent[s] — one uint32 word per thread = 8 values
        //    Thread simd_lid reads word simd_lid (32 words = 256 values)
        uint word = lat_packed[(lat_base + s) * MLA_WORDS + simd_lid];
        uint group = simd_lid / 8;  // 8 words per group, 4 groups total
        U scale = static_cast<U>(lat_scales[(lat_base + s) * MLA_NGROUPS + group]);
        U bias  = static_cast<U>(lat_biases[(lat_base + s) * MLA_NGROUPS + group]);

        thread U lat[8];
        for (uint i = 0; i < 8; i++) {
            uint raw = (word >> (i * 4)) & 0xFu;
            lat[i] = static_cast<U>(raw) * scale + bias;
        }

        // 2. NOPE SCORE — dot(q_nope, dequanted_latent) via simd_sum
        U nope_partial = 0;
        for (uint i = 0; i < 8; i++) {
            nope_partial += q_n[i] * lat[i];
        }
        U nope_score = simd_sum(nope_partial);

        // 3. ROPE SCORE — dot(q_pe, k_pe[s]) via simd_sum
        U rope_partial = 0;
        for (uint i = 0; i < 2; i++) {
            uint elem = simd_lid * 2 + i;
            rope_partial += q_r[i] * static_cast<U>(k_pe[kpe_base + s * MLA_RD + elem]);
        }
        U rope_score = simd_sum(rope_partial);

        // 4. COMBINED SCORE + ONLINE SOFTMAX
        U score = nope_score + rope_score;
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        // 5. ACCUMULATE VALUE — reuse dequanted latent (no re-read!)
        for (uint i = 0; i < 8; i++) {
            o[i] = o[i] * factor + exp_score * lat[i];
        }
    }

    // --- Cross-simdgroup reduction (same pattern as sdpa_vector.h) ---

    // Store per-simdgroup state
    if (simd_lid == 0) {
        tg_max[simd_gid] = max_score;
        tg_sum[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Find global max and compute correction factors
    max_score = tg_max[simd_lid];
    U new_max = simd_max(max_score);
    U factor = fast::exp(max_score - new_max);
    sum_exp_score = simd_sum(tg_sum[simd_lid] * factor);

    // Aggregate outputs across simdgroups
    // Each thread has 8 output dims. We need to combine across BN simdgroups.
    // Use the sdpa_vector transpose trick: each iteration handles one of the 8 dims.
    for (uint i = 0; i < 8; i++) {
        tg_out[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(tg_out[simd_gid * BD + simd_lid] * factor);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Write output ---
    // Output: [B, H, 256] — each head's latent attention result
    if (simd_lid == 0) {
        device T* out_ptr = out + (batch_idx * H + head_idx) * MLA_D;
        for (uint i = 0; i < 8; i++) {
            out_ptr[simd_gid * 8 + i] = static_cast<T>(o[i]);
        }
    }
}

// =============================================================================
// V2: Fused SDPA + direct cache update (eliminates SliceUpdate copy)
//
// Same SDPA logic as v1, plus:
//   - Quantizes new token in-kernel → threadgroup memory (source of truth)
//   - Writes quantized data to cache at position S (persistence only)
//   - SDPA loop reads new token from threadgroup memory, NOT from cache
//   - Cache buffers are read+write (non-const) for direct append
//
// Contract: decode only, B=1, exact MLA dims, append exactly 1 token.
// =============================================================================

template <typename T>
[[kernel]] void mla_fused_sdpa_v2(
    const device T*        q_nope      [[buffer(0)]],   // [B, H, 256]
    const device T*        q_pe        [[buffer(1)]],   // [B, H, 64]
    device uint32_t*       cache_packed [[buffer(2)]],   // [B, S_alloc, 32] read+write
    device T*              cache_scales [[buffer(3)]],   // [B, S_alloc, 4]  read+write
    device T*              cache_biases [[buffer(4)]],   // [B, S_alloc, 4]  read+write
    device T*              cache_kpe    [[buffer(5)]],   // [B, S_alloc, 64] read+write
    const device T*        new_latent   [[buffer(6)]],   // [B, 1, 256] raw fp16
    const device T*        new_kpe      [[buffer(7)]],   // [B, 1, 64]  fp16
    device T*              out          [[buffer(8)]],   // [B, H, 256]
    const constant uint&   B            [[buffer(9)]],
    const constant uint&   H            [[buffer(10)]],
    const constant uint&   S            [[buffer(11)]],  // current occupancy (0..S-1 valid)
    const constant uint&   S_alloc      [[buffer(12)]],  // allocated cache dimension
    const constant float&  attn_scale   [[buffer(13)]],
    uint3  tid       [[threadgroup_position_in_grid]],
    uint   simd_gid  [[simdgroup_index_in_threadgroup]],
    uint   simd_lid  [[thread_index_in_simdgroup]]) {

    const uint head_idx = tid.x;
    const uint batch_idx = tid.y;
    if (head_idx >= H || batch_idx >= B) return;

    typedef float U;

    // --- Threadgroup memory: new token (quantized, kept here for SDPA) ---
    threadgroup uint32_t tg_new_packed[MLA_WORDS];     // 32 words = 256 values
    threadgroup U        tg_new_scales[MLA_NGROUPS];   // 4 group scales (float)
    threadgroup U        tg_new_biases[MLA_NGROUPS];   // 4 group biases (float)
    threadgroup U        tg_new_kpe[MLA_RD];           // 64 RoPE values (float)

    // --- Existing threadgroup memory for cross-simdgroup reduction ---
    threadgroup U tg_max[BN];
    threadgroup U tg_sum[BN];
    threadgroup U tg_out[BN * BD];

    const uint cache_base = batch_idx * S_alloc;

    // =================================================================
    // PHASE 1: Quantize new token → threadgroup memory + cache persist
    // Simdgroup 0 handles quantize (32 threads, 8 values each — proven pattern)
    // =================================================================

    if (simd_gid == 0) {
        // --- Quantize new_latent (same pattern as mla_quantize_store) ---
        const device T* lat_ptr = new_latent + batch_idx * MLA_D;
        float vals[8];
        for (uint i = 0; i < 8; i++) {
            vals[i] = static_cast<float>(lat_ptr[simd_lid * 8 + i]);
        }

        // Per-group min/max via simd_shuffle_xor (groups of 8 threads)
        float local_min = vals[0], local_max = vals[0];
        for (uint i = 1; i < 8; i++) {
            local_min = min(local_min, vals[i]);
            local_max = max(local_max, vals[i]);
        }
        for (uint delta = 1; delta <= 4; delta <<= 1) {
            float other_min = simd_shuffle_xor(local_min, static_cast<ushort>(delta));
            float other_max = simd_shuffle_xor(local_max, static_cast<ushort>(delta));
            local_min = min(local_min, other_min);
            local_max = max(local_max, other_max);
        }

        float scale_val = (local_max - local_min) / 15.0f;
        float inv_scale = (scale_val > 0.0f) ? (1.0f / scale_val) : 0.0f;
        float bias_val = local_min;

        // Quantize and pack
        uint packed_word = 0;
        for (uint i = 0; i < 8; i++) {
            float normalized = (vals[i] - bias_val) * inv_scale;
            uint q = static_cast<uint>(clamp(rint(normalized), 0.0f, 15.0f));
            packed_word |= (q << (i * 4));
        }

        // Write to threadgroup memory (source of truth for SDPA)
        tg_new_packed[simd_lid] = packed_word;
        uint group = simd_lid / 8;
        if ((simd_lid & 7) == 0) {
            tg_new_scales[group] = scale_val;
            tg_new_biases[group] = bias_val;
        }

        // Write to cache for persistence — only head 0 writes (all heads compute
        // identical results since latent is shared, avoid redundant writes)
        if (head_idx == 0) {
            cache_packed[(cache_base + S) * MLA_WORDS + simd_lid] = packed_word;
            if ((simd_lid & 7) == 0) {
                cache_scales[(cache_base + S) * MLA_NGROUPS + group] = static_cast<T>(scale_val);
                cache_biases[(cache_base + S) * MLA_NGROUPS + group] = static_cast<T>(bias_val);
            }
        }

        // Copy new_kpe to threadgroup memory + cache (2 values per thread)
        const device T* kpe_ptr = new_kpe + batch_idx * MLA_RD;
        for (uint i = 0; i < 2; i++) {
            uint elem = simd_lid * 2 + i;
            U val = static_cast<U>(kpe_ptr[elem]);
            tg_new_kpe[elem] = val;
            if (head_idx == 0) {
                cache_kpe[(cache_base + S) * MLA_RD + elem] = static_cast<T>(val);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =================================================================
    // PHASE 2: Load queries with scale (same as v1)
    // =================================================================

    thread U q_n[8];
    const device T* q_nope_ptr = q_nope + (batch_idx * H + head_idx) * MLA_D;
    for (uint i = 0; i < 8; i++) {
        q_n[i] = static_cast<U>(attn_scale) * static_cast<U>(q_nope_ptr[simd_lid * 8 + i]);
    }

    thread U q_r[2];
    const device T* q_pe_ptr = q_pe + (batch_idx * H + head_idx) * MLA_RD;
    for (uint i = 0; i < 2; i++) {
        q_r[i] = static_cast<U>(attn_scale) * static_cast<U>(q_pe_ptr[simd_lid * 2 + i]);
    }

    // Output accumulator + online softmax state
    thread U o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    U max_score = -1e20f;
    U sum_exp_score = 0;

    // =================================================================
    // PHASE 3: SDPA loop over positions 0..S (S inclusive = new token)
    // 0..S-1: read from cache (device memory)
    // S: read from threadgroup memory (new token, NOT reread from cache)
    // =================================================================

    const uint total_S = S + 1;

    for (uint s = simd_gid; s < total_S; s += BN) {

        thread U lat[8];
        U rope_partial = 0;

        if (s < S) {
            // --- Existing cache entry: read from device memory ---
            uint word = cache_packed[(cache_base + s) * MLA_WORDS + simd_lid];
            uint group = simd_lid / 8;
            U sc = static_cast<U>(cache_scales[(cache_base + s) * MLA_NGROUPS + group]);
            U bi = static_cast<U>(cache_biases[(cache_base + s) * MLA_NGROUPS + group]);

            for (uint i = 0; i < 8; i++) {
                uint raw = (word >> (i * 4)) & 0xFu;
                lat[i] = static_cast<U>(raw) * sc + bi;
            }

            for (uint i = 0; i < 2; i++) {
                uint elem = simd_lid * 2 + i;
                rope_partial += q_r[i] * static_cast<U>(cache_kpe[(cache_base + s) * MLA_RD + elem]);
            }
        } else {
            // --- New token: read from threadgroup memory ---
            uint word = tg_new_packed[simd_lid];
            uint group = simd_lid / 8;
            U sc = tg_new_scales[group];
            U bi = tg_new_biases[group];

            for (uint i = 0; i < 8; i++) {
                uint raw = (word >> (i * 4)) & 0xFu;
                lat[i] = static_cast<U>(raw) * sc + bi;
            }

            for (uint i = 0; i < 2; i++) {
                uint elem = simd_lid * 2 + i;
                rope_partial += q_r[i] * tg_new_kpe[elem];
            }
        }

        // Nope score via simd_sum
        U nope_partial = 0;
        for (uint i = 0; i < 8; i++) {
            nope_partial += q_n[i] * lat[i];
        }
        U nope_score = simd_sum(nope_partial);
        U rope_score = simd_sum(rope_partial);

        // Online softmax
        U score = nope_score + rope_score;
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        // Value accumulation (reuse dequanted latent)
        for (uint i = 0; i < 8; i++) {
            o[i] = o[i] * factor + exp_score * lat[i];
        }
    }

    // =================================================================
    // PHASE 4: Cross-simdgroup reduction (identical to v1)
    // =================================================================

    if (simd_lid == 0) {
        tg_max[simd_gid] = max_score;
        tg_sum[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    max_score = tg_max[simd_lid];
    U new_max = simd_max(max_score);
    U factor = fast::exp(max_score - new_max);
    sum_exp_score = simd_sum(tg_sum[simd_lid] * factor);

    for (uint i = 0; i < 8; i++) {
        tg_out[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(tg_out[simd_gid * BD + simd_lid] * factor);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write attention output
    if (simd_lid == 0) {
        device T* out_ptr = out + (batch_idx * H + head_idx) * MLA_D;
        for (uint i = 0; i < 8; i++) {
            out_ptr[simd_gid * 8 + i] = static_cast<T>(o[i]);
        }
    }
}

// =============================================================================
// V1 Entry points (kept for backward compatibility)
// =============================================================================

// Entry points
template [[host_name("mla_fused_sdpa_f16")]]
[[kernel]] void mla_fused_sdpa<half>(
    const device half*, const device half*,
    const device uint32_t*, const device half*, const device half*,
    const device half*, device half*,
    const constant uint&, const constant uint&, const constant uint&,
    const constant float&,
    uint3, uint, uint);

template [[host_name("mla_fused_sdpa_bf16")]]
[[kernel]] void mla_fused_sdpa<bfloat>(
    const device bfloat*, const device bfloat*,
    const device uint32_t*, const device bfloat*, const device bfloat*,
    const device bfloat*, device bfloat*,
    const constant uint&, const constant uint&, const constant uint&,
    const constant float&,
    uint3, uint, uint);

// V2 Entry points
template [[host_name("mla_fused_sdpa_v2_f16")]]
[[kernel]] void mla_fused_sdpa_v2<half>(
    const device half*, const device half*,
    device uint32_t*, device half*, device half*, device half*,
    const device half*, const device half*,
    device half*,
    const constant uint&, const constant uint&, const constant uint&,
    const constant uint&, const constant float&,
    uint3, uint, uint);

template [[host_name("mla_fused_sdpa_v2_bf16")]]
[[kernel]] void mla_fused_sdpa_v2<bfloat>(
    const device bfloat*, const device bfloat*,
    device uint32_t*, device bfloat*, device bfloat*, device bfloat*,
    const device bfloat*, const device bfloat*,
    device bfloat*,
    const constant uint&, const constant uint&, const constant uint&,
    const constant uint&, const constant float&,
    uint3, uint, uint);
