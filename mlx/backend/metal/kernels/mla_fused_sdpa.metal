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
