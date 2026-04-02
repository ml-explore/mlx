// Fused INT4 affine quantization for MLA latent cache.
// Replaces mx.quantize multi-dispatch overhead with a single kernel.
// Optimized for MLA dimensions: 256 latent, group_size=64, 4-bit.
//
// One simdgroup (32 threads) per 256-dim vector.
// Each thread handles 8 values. 4 groups of 64 values each.
// Per-group min/max reduction via simd_shuffle_xor (stays within group of 8).

#include <metal_stdlib>
using namespace metal;

constant uint MLA_D       = 256;  // latent dimension
constant uint MLA_GS      = 64;   // quantization group size
constant uint MLA_NGROUPS = 4;    // MLA_D / MLA_GS
constant uint MLA_WORDS   = 32;   // MLA_D / 8 (8 values per uint32 at 4-bit)

template <typename T>
[[kernel]] void mla_quantize_store(
    const device T*        input   [[buffer(0)]],   // [N, 256] fp16 latent vectors
    device uint32_t*       packed  [[buffer(1)]],   // [N, 32] output packed
    device T*              scales  [[buffer(2)]],   // [N, 4] output scales
    device T*              biases  [[buffer(3)]],   // [N, 4] output biases
    const constant uint&   N       [[buffer(4)]],   // total vectors (B * L)
    uint3  tid       [[threadgroup_position_in_grid]],
    uint   simd_lid  [[thread_index_in_simdgroup]]) {

    const uint vec_idx = tid.x;
    if (vec_idx >= N) return;

    // Read 8 values for this thread (256 / 32 = 8)
    const device T* src = input + vec_idx * MLA_D;
    float vals[8];
    for (uint i = 0; i < 8; i++) {
        vals[i] = static_cast<float>(src[simd_lid * 8 + i]);
    }

    // Per-thread local min/max across 8 values
    float local_min = vals[0], local_max = vals[0];
    for (uint i = 1; i < 8; i++) {
        local_min = min(local_min, vals[i]);
        local_max = max(local_max, vals[i]);
    }

    // Reduce min/max across 8 threads in group via simd_shuffle_xor
    // Groups of 8 threads: XOR with 1, 2, 4 stays within group boundaries
    for (uint delta = 1; delta <= 4; delta <<= 1) {
        float other_min = simd_shuffle_xor(local_min, static_cast<ushort>(delta));
        float other_max = simd_shuffle_xor(local_max, static_cast<ushort>(delta));
        local_min = min(local_min, other_min);
        local_max = max(local_max, other_max);
    }

    // All 8 threads in group now have the same min/max
    float scale_val = (local_max - local_min) / 15.0f;
    float inv_scale = (scale_val > 0.0f) ? (1.0f / scale_val) : 0.0f;
    float bias_val = local_min;

    // Quantize and pack 8 values into one uint32
    uint packed_word = 0;
    for (uint i = 0; i < 8; i++) {
        float normalized = (vals[i] - bias_val) * inv_scale;
        uint q = static_cast<uint>(clamp(rint(normalized), 0.0f, 15.0f));
        packed_word |= (q << (i * 4));
    }

    // Write packed word
    packed[vec_idx * MLA_WORDS + simd_lid] = packed_word;

    // Write scale/bias (one thread per group — first thread in each group of 8)
    uint group = simd_lid / 8;
    if ((simd_lid & 7) == 0) {
        scales[vec_idx * MLA_NGROUPS + group] = static_cast<T>(scale_val);
        biases[vec_idx * MLA_NGROUPS + group] = static_cast<T>(bias_val);
    }
}

// Entry points
template [[host_name("mla_quantize_store_f16")]]
[[kernel]] void mla_quantize_store<half>(
    const device half*, device uint32_t*, device half*, device half*,
    const constant uint&,
    uint3, uint);

template [[host_name("mla_quantize_store_bf16")]]
[[kernel]] void mla_quantize_store<bfloat>(
    const device bfloat*, device uint32_t*, device bfloat*, device bfloat*,
    const constant uint&,
    uint3, uint);
