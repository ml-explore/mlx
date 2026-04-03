#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------------
// MLA shared-latent nope score kernel
//
// Computes:
//   scores[b, h, s] = scale * sum_{k=0..255}( q_nope[b,h,k] * dequant(k_latent[b,s,k]) )
//
// Inputs:
//   q_nope      : [B, H, 256]           half or bfloat
//   k_packed    : [B, S, 32]            uint32 packed INT4 (8 vals/word, 32 words total)
//   k_scales    : [B, S, 4]             float32 scale per 64-dim group
//   k_biases    : [B, S, 4]             float32 bias  per 64-dim group
//
// Output:
//   out_scores  : [B, H, S]             float32
//
// Quantization:
//   D = 256, group_size = 64, 4 groups total
//   each 64-dim group uses 8 uint32 words
//   each uint32 packs 8 x 4-bit values
//
// Tiling strategy:
//   - one threadgroup handles one (batch, seq_position, head_tile)
//   - latent for that seq position is dequantized once per 64-dim group
//   - reused across H_TILE heads in the threadgroup
//
// Recommended host dispatch:
//   threadsPerThreadgroup = MTLSizeMake(32, H_TILE, 1)
//   threadgroupsPerGrid   = MTLSizeMake(ceil_div(H, H_TILE), S, B)
//
// Notes:
//   - This is the correct V1 external-extension kernel.
//   - It is intentionally narrow: scores only.
//   - It uses tiled/shared memory and simdgroup reduction.
//   - If this proves a win, the next step is MLX-core/upstream integration.
// -----------------------------------------------------------------------------

constant uint MLA_D                = 256;
constant uint MLA_GROUP_SIZE       = 64;
constant uint MLA_NUM_GROUPS       = 4;
constant uint MLA_WORDS_PER_GROUP  = 8;   // 64 dims / 8 vals per word
constant uint MLA_TOTAL_WORDS      = 32;  // 256 dims / 8 vals per word
constant uint MLA_H_TILE           = 8;   // 8 heads per threadgroup (256 threads total)

// ----------------------------- Helpers ----------------------------------------

inline uint unpack_int4(uint packed_word, uint nibble_idx) {
    return (packed_word >> (nibble_idx * 4)) & 0xFu;
}

inline float dequant_int4_affine(uint q, float scale, float bias) {
    return fma((float)q, scale, bias);
}

// q_nope dtype-generic load helper
template <typename T>
inline float load_q(device const T* q_ptr, uint idx) {
    return float(q_ptr[idx]);
}

// -------------------------- Core kernel body ----------------------------------

template <typename T>
inline void mla_nope_scores_shared_latent_impl(
    device const T*        q_nope,      // [B,H,256]
    device const uint*     k_packed,    // [B,S,32]
    device const half*     k_scales,    // [B,S,4] float16
    device const half*     k_biases,    // [B,S,4] float16
    device float*          out_scores,  // [B,H,S]
    constant uint&         B,
    constant uint&         H,
    constant uint&         S,
    constant float&        score_scale,
    threadgroup float*     k_tile,      // [MLA_GROUP_SIZE] shared memory
    ushort3                tid,         // thread_position_in_threadgroup
    uint3                  tgid         // threadgroup_position_in_grid
) {
    // Threadgroup maps:
    //   tgid.x = head tile index
    //   tgid.y = seq position
    //   tgid.z = batch index
    //
    // Thread position maps:
    //   tid.x = lane within simdgroup [0..31]
    //   tid.y = head row within tile [0..MLA_H_TILE-1]

    const uint lane   = tid.x;               // 0..31
    const uint h_tile = tgid.x;
    const uint s_idx  = tgid.y;
    const uint b_idx  = tgid.z;
    const uint h_idx  = h_tile * MLA_H_TILE + tid.y;

    if (b_idx >= B || s_idx >= S) {
        return;
    }

    float acc = 0.0f;

    // Base pointers for this (b, s)
    const uint packed_base = ((b_idx * S) + s_idx) * MLA_TOTAL_WORDS;   // 32 words
    const uint q_base      = ((b_idx * H) + h_idx) * MLA_D;

    // Process 4 x 64-dim groups
    for (uint g = 0; g < MLA_NUM_GROUPS; ++g) {
        // Dequantize latent once for this sequence position and group.
        // Only tid.y == 0 participates in the load/dequant work.
        if (tid.y == 0) {
            const float scale = float(k_scales[((b_idx * S) + s_idx) * MLA_NUM_GROUPS + g]);
            const float bias  = float(k_biases[((b_idx * S) + s_idx) * MLA_NUM_GROUPS + g]);

            // 32 lanes dequantize 2 values each => 64 values total
            const uint d0 = lane;        // 0..31
            const uint d1 = lane + 32;   // 32..63

            // map dim within group -> packed word + nibble
            const uint word_idx0   = packed_base + g * MLA_WORDS_PER_GROUP + (d0 >> 3);
            const uint nibble_idx0 = d0 & 7;
            const uint word0       = k_packed[word_idx0];
            const uint qv0         = unpack_int4(word0, nibble_idx0);
            k_tile[d0]             = dequant_int4_affine(qv0, scale, bias);

            const uint word_idx1   = packed_base + g * MLA_WORDS_PER_GROUP + (d1 >> 3);
            const uint nibble_idx1 = d1 & 7;
            const uint word1       = k_packed[word_idx1];
            const uint qv1         = unpack_int4(word1, nibble_idx1);
            k_tile[d1]             = dequant_int4_affine(qv1, scale, bias);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each head row computes dot product against the shared 64-dim tile.
        if (h_idx < H) {
            const uint q_group_base = q_base + g * MLA_GROUP_SIZE;

            // 32 lanes each handle 2 q*k products
            const float q0 = load_q(q_nope, q_group_base + lane);
            const float q1 = load_q(q_nope, q_group_base + lane + 32);

            float partial = q0 * k_tile[lane] + q1 * k_tile[lane + 32];

            // Reduce across the 32 lanes for this head row.
            float sum64 = simd_sum(partial);

            // lane 0 writes the per-group contribution into acc
            if (lane == 0) {
                acc += sum64;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final write: one thread per (b,h,s)
    if (h_idx < H && lane == 0) {
        out_scores[((b_idx * H) + h_idx) * S + s_idx] = acc * score_scale;
    }
}

// ---------------------------- Entry points ------------------------------------

// half input
kernel void mla_nope_scores_shared_latent_f16(
    device const half*     q_nope      [[buffer(0)]],
    device const uint*     k_packed    [[buffer(1)]],
    device const half*     k_scales    [[buffer(2)]],
    device const half*     k_biases    [[buffer(3)]],
    device float*          out_scores  [[buffer(4)]],
    constant uint&         B           [[buffer(5)]],
    constant uint&         H           [[buffer(6)]],
    constant uint&         S           [[buffer(7)]],
    constant float&        score_scale [[buffer(8)]],
    ushort3                tid         [[thread_position_in_threadgroup]],
    uint3                  tgid        [[threadgroup_position_in_grid]]
) {
    threadgroup float k_tile[MLA_GROUP_SIZE];
    mla_nope_scores_shared_latent_impl(
        q_nope, k_packed, k_scales, k_biases, out_scores,
        B, H, S, score_scale, k_tile, tid, tgid
    );
}

// bfloat16 input
kernel void mla_nope_scores_shared_latent_bf16(
    device const bfloat*   q_nope      [[buffer(0)]],
    device const uint*     k_packed    [[buffer(1)]],
    device const half*     k_scales    [[buffer(2)]],
    device const half*     k_biases    [[buffer(3)]],
    device float*          out_scores  [[buffer(4)]],
    constant uint&         B           [[buffer(5)]],
    constant uint&         H           [[buffer(6)]],
    constant uint&         S           [[buffer(7)]],
    constant float&        score_scale [[buffer(8)]],
    ushort3                tid         [[thread_position_in_threadgroup]],
    uint3                  tgid        [[threadgroup_position_in_grid]]
) {
    threadgroup float k_tile[MLA_GROUP_SIZE];
    mla_nope_scores_shared_latent_impl(
        q_nope, k_packed, k_scales, k_biases, out_scores,
        B, H, S, score_scale, k_tile, tid, tgid
    );
}
