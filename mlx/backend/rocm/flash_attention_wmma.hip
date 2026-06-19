// WMMA-accelerated flash attention for RDNA 3+ (gfx1100+)
//
// Uses rocwmma 16x16x16 bf16→f32 tiles for Q@K^T and P@V matmuls.
// Implements FlashAttention-2 online softmax.
//
// BLOCK_M=64, BLOCK_N=64, 128 threads (4 waves), each wave owns 16 query rows.
// Shared memory ~50 KB for D=128.

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/dtype_utils.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
#define ROCM_FA_WMMA 1
#include <rocwmma/rocwmma.hpp>
#else
#define ROCM_FA_WMMA 0
#endif

namespace mlx::core {
namespace rocm {

struct FAWmmaParams {
  int B, H, D, qL, kL, gqa_factor;
  float scale;
  int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
};

// Helper: collaborative load from global to shared memory
template <typename T, int STRIDE, int NTHREADS>
__device__ void load_tile(
    T* __restrict__ dst,
    const T* __restrict__ src_base,
    int64_t row_stride,
    int rows,
    int cols,
    int valid_rows,
    int tid) {
  const int total = rows * cols;
  const int per_t = (total + NTHREADS - 1) / NTHREADS;
  for (int i = 0; i < per_t; i++) {
    int idx = i * NTHREADS + tid;
    if (idx < total) {
      int r = idx / cols;
      int c = idx % cols;
      dst[r * STRIDE + c] = (r < valid_rows) ?
          src_base[r * row_stride + c] : static_cast<T>(0.0f);
    }
  }
  // Zero padding columns
  const int pad = STRIDE - cols;
  if (pad > 0) {
    const int total_pad = rows * pad;
    const int per_p = (total_pad + NTHREADS - 1) / NTHREADS;
    for (int i = 0; i < per_p; i++) {
      int idx = i * NTHREADS + tid;
      if (idx < total_pad) {
        int r = idx / pad;
        dst[r * STRIDE + cols + (idx % pad)] = static_cast<T>(0.0f);
      }
    }
  }
}

template <
    typename T,
    bool do_causal,
    int D,
    int BLOCK_M = 64,
    int BLOCK_N = 64>
__global__ void __launch_bounds__(128)
    kernel_sdpa_flash_wmma(
        const T* __restrict__ Q,
        const T* __restrict__ K,
        const T* __restrict__ V,
        T* __restrict__ O,
        const FAWmmaParams params) {
#if ROCM_FA_WMMA
  constexpr int WT = 16; // WMMA tile size
  // Stage Q/K/V one DW-wide head-dim slice at a time so the LDS footprint is
  // capped at the D=128 layout for any head dim (D=256 needs two slices). This
  // keeps the kernel within RDNA's 64 KB LDS budget while still using WMMA.
  constexpr int DW = (D < 128) ? D : 128;
  constexpr int N_DCHUNKS = (D + DW - 1) / DW;
  constexpr int Q_PAD = DW + 4;
  constexpr int KV_PAD = DW + 4;
  constexpr int S_PAD = BLOCK_N + 4;
  constexpr int P_PAD = BLOCK_N + 4;
  constexpr int M_TILES = BLOCK_M / WT;
  constexpr int N_TILES = BLOCK_N / WT;
  constexpr int DW_TILES = DW / WT;
  constexpr int D_TILES = (D + WT - 1) / WT;
  constexpr int NTHREADS = 128;

  const int bid_b = blockIdx.z;
  const int bid_h = blockIdx.x;
  const int bid_kv_h = bid_h / params.gqa_factor;
  const int q_start = blockIdx.y * BLOCK_M;
  const int tid = threadIdx.x;
  const int wave = tid / 32;
  const int lane = tid % 32;

  if (q_start >= params.qL) return;

  // ---- Shared memory layout ----
  // Persistent: m[BLOCK_M], l[BLOCK_M]
  // Tile A:     Q_sh [BLOCK_M][Q_PAD]   (bf16, loaded once)
  // Tile B:     KV_sh[BLOCK_N][KV_PAD]  (bf16, K then P_bf16)
  // Tile C:     S_sh [BLOCK_M][S_PAD]   (f32,  scores then V_bf16)
  extern __shared__ char smem[];
  float* m_arr = reinterpret_cast<float*>(smem);
  float* l_arr = m_arr + BLOCK_M;
  T* Q_sh = reinterpret_cast<T*>(l_arr + BLOCK_M);
  T* KV_sh = Q_sh + BLOCK_M * Q_PAD;
  float* S_sh = reinterpret_cast<float*>(KV_sh + BLOCK_N * KV_PAD);

  // Fragment types
  using frag_a     = rocwmma::fragment<rocwmma::matrix_a,   WT, WT, WT, T, rocwmma::row_major>;
  using frag_b_col = rocwmma::fragment<rocwmma::matrix_b,   WT, WT, WT, T, rocwmma::col_major>;
  using frag_b_row = rocwmma::fragment<rocwmma::matrix_b,   WT, WT, WT, T, rocwmma::row_major>;
  using frag_acc   = rocwmma::fragment<rocwmma::accumulator, WT, WT, WT, float>;

  // Output accumulators: each wave owns D_TILES [16x16] f32 tiles
  frag_acc o_acc[D_TILES];
  for (int d = 0; d < D_TILES; d++)
    rocwmma::fill_fragment(o_acc[d], 0.0f);

  // Init online softmax state
  if (tid < BLOCK_M) {
    m_arr[tid] = -1e30f;
    l_arr[tid] = 0.0f;
  }
  __syncthreads();

  const T* Q_base = Q + bid_b * params.Q_strides[0]
                      + bid_h * params.Q_strides[1]
                      + q_start * params.Q_strides[2];
  const int q_valid = min(BLOCK_M, params.qL - q_start);

  // ---- K/V block loop ----
  for (int k_start = 0; k_start < params.kL; k_start += BLOCK_N) {

    // Causal skip: if entire K block is after all queries' causal limits
    if constexpr (do_causal) {
      int last_q = q_start + min(BLOCK_M, params.qL - q_start) - 1;
      int max_k_allowed = (params.kL - params.qL) + last_q;
      if (k_start > max_k_allowed) break;
    }

    int k_valid = min(BLOCK_N, params.kL - k_start);

    const T* K_base = K + bid_b * params.K_strides[0]
                        + bid_kv_h * params.K_strides[1]
                        + k_start * params.K_strides[2];

    // ---- S = Q @ K^T via WMMA, accumulated over the head dim in DW slices ----
    // Each wave computes S[wave*16 : (wave+1)*16, 0:BLOCK_N]. For D>DW the Q/K
    // slices are staged and consumed one DW-wide chunk at a time so LDS stays
    // bounded; s_acc carries the partial dot products across chunks.
    {
      frag_acc s_acc[N_TILES];
      for (int n = 0; n < N_TILES; n++)
        rocwmma::fill_fragment(s_acc[n], 0.0f);

      for (int dc = 0; dc < N_DCHUNKS; dc++) {
        const int d0 = dc * DW;
        const int dwid = min(DW, D - d0);
        load_tile<T, Q_PAD, NTHREADS>(
            Q_sh, Q_base + d0, params.Q_strides[2], BLOCK_M, dwid, q_valid, tid);
        load_tile<T, KV_PAD, NTHREADS>(
            KV_sh, K_base + d0, params.K_strides[2], BLOCK_N, dwid, k_valid, tid);
        __syncthreads();

        for (int d = 0; d < DW_TILES; d++) {
          frag_a q_frag;
          rocwmma::load_matrix_sync(
              q_frag, Q_sh + wave * WT * Q_PAD + d * WT, Q_PAD);

          for (int n = 0; n < N_TILES; n++) {
            frag_b_col k_frag;
            // col_major load of K[n*16][d*16] gives K^T
            rocwmma::load_matrix_sync(
                k_frag, KV_sh + n * WT * KV_PAD + d * WT, KV_PAD);
            rocwmma::mma_sync(s_acc[n], q_frag, k_frag, s_acc[n]);
          }
        }
        __syncthreads();
      }

      // Scale and store S to shared memory
      for (int n = 0; n < N_TILES; n++) {
        for (int e = 0; e < s_acc[n].num_elements; e++)
          s_acc[n].x[e] *= params.scale;
        rocwmma::store_matrix_sync(
            S_sh + wave * WT * S_PAD + n * WT, s_acc[n], S_PAD,
            rocwmma::mem_row_major);
      }
    }
    __syncthreads();

    // ---- Online softmax (scalar, 64 threads handle 64 rows) ----
    float my_scale_old = 0.0f;
    if (tid < BLOCK_M) {
      int q_idx = q_start + tid;
      bool valid = q_idx < params.qL;
      float old_m = m_arr[tid];
      float old_l = l_arr[tid];

      float new_m = old_m;
      if (valid) {
        for (int j = 0; j < BLOCK_N && (k_start + j) < params.kL; j++) {
          bool use = true;
          if constexpr (do_causal)
            use = (k_start + j) <= (params.kL - params.qL + q_idx);
          if (use)
            new_m = fmaxf(new_m, S_sh[tid * S_PAD + j]);
        }
      }

      float scale_old = (old_m > -1e29f) ? expf(old_m - new_m) : 0.0f;
      float row_sum = 0.0f;

      for (int j = 0; j < BLOCK_N; j++) {
        bool use = valid && (k_start + j) < params.kL;
        if constexpr (do_causal)
          use = use && ((k_start + j) <= (params.kL - params.qL + q_idx));

        if (use) {
          float p = expf(S_sh[tid * S_PAD + j] - new_m);
          S_sh[tid * S_PAD + j] = p;
          row_sum += p;
        } else {
          S_sh[tid * S_PAD + j] = 0.0f;
        }
      }

      m_arr[tid] = new_m;
      l_arr[tid] = old_l * scale_old + row_sum;
      my_scale_old = scale_old;
    }
    // Broadcast scale_old to all lanes in each wave via shared memory
    __shared__ float wave_scale[BLOCK_M];
    if (tid < BLOCK_M) wave_scale[tid] = my_scale_old;
    __syncthreads();

    // ---- Rescale O accumulators by scale_old ----
    // Each wave rescales its 16-row O tiles.
    // Since fragment layout is opaque, store→scale→reload via shared memory.
    {
      // Use KV_sh area as f32 temp (it held K which we no longer need)
      float* o_tmp = reinterpret_cast<float*>(KV_sh);
      constexpr int OT_PAD = WT + 4;

      for (int d = 0; d < D_TILES; d++) {
        rocwmma::store_matrix_sync(
            o_tmp + wave * WT * OT_PAD, o_acc[d], OT_PAD,
            rocwmma::mem_row_major);
        __syncthreads();

        // Scale rows (only 16 threads needed per wave)
        if (lane < WT) {
          int row = wave * WT + lane;
          float s = wave_scale[row];
          for (int c = 0; c < WT; c++)
            o_tmp[wave * WT * OT_PAD + lane * OT_PAD + c] *= s;
        }
        __syncthreads();

        rocwmma::load_matrix_sync(
            o_acc[d], o_tmp + wave * WT * OT_PAD, OT_PAD,
            rocwmma::mem_row_major);
        __syncthreads();
      }
    }

    // ---- Convert P (f32 in S_sh) to bf16 in KV_sh for WMMA P@V ----
    {
      int total = BLOCK_M * BLOCK_N;
      int per_t = (total + NTHREADS - 1) / NTHREADS;
      for (int i = 0; i < per_t; i++) {
        int idx = i * NTHREADS + tid;
        if (idx < total) {
          int r = idx / BLOCK_N;
          int c = idx % BLOCK_N;
          KV_sh[r * P_PAD + c] = static_cast<T>(S_sh[r * S_PAD + c]);
        }
      }
      // Zero padding
      int pad_total = BLOCK_M * 4;
      int pad_per_t = (pad_total + NTHREADS - 1) / NTHREADS;
      for (int i = 0; i < pad_per_t; i++) {
        int idx = i * NTHREADS + tid;
        if (idx < pad_total)
          KV_sh[(idx / 4) * P_PAD + BLOCK_N + (idx % 4)] = static_cast<T>(0.0f);
      }
    }
    __syncthreads();

    // ---- O += P @ V via WMMA, output head dim tiled in DW slices ----
    // P (bf16) stays resident in KV_sh; V is streamed one DW-wide slice into
    // S_sh (the f32 scores are no longer needed) and consumed immediately, so
    // a full D=256 V tile never has to coexist with P in LDS.
    {
      T* P_sh = KV_sh;
      T* V_sh = reinterpret_cast<T*>(S_sh);
      const T* V_base = V + bid_b * params.V_strides[0]
                          + bid_kv_h * params.V_strides[1]
                          + k_start * params.V_strides[2];

      for (int dc = 0; dc < N_DCHUNKS; dc++) {
        const int d0 = dc * DW;
        const int dwid = min(DW, D - d0);
        load_tile<T, KV_PAD, NTHREADS>(
            V_sh, V_base + d0, params.V_strides[2], BLOCK_N, dwid, k_valid, tid);
        __syncthreads();

        for (int dl = 0; dl < DW_TILES; dl++) {
          const int dglobal = dc * DW_TILES + dl;
          if (dglobal >= D_TILES)
            break;
          for (int n = 0; n < N_TILES; n++) {
            frag_a p_frag;
            frag_b_row v_frag;
            rocwmma::load_matrix_sync(
                p_frag, P_sh + wave * WT * P_PAD + n * WT, P_PAD);
            rocwmma::load_matrix_sync(
                v_frag, V_sh + n * WT * KV_PAD + dl * WT, KV_PAD);
            rocwmma::mma_sync(o_acc[dglobal], p_frag, v_frag, o_acc[dglobal]);
          }
        }
        __syncthreads();
      }
    }
    __syncthreads();
  } // end K/V loop

  // ---- Finalize: normalize O and write to global ----
  {
    float* o_tmp = reinterpret_cast<float*>(Q_sh); // Reuse Q_sh
    constexpr int OT_PAD = WT + 4;

    for (int d = 0; d < D_TILES; d++) {
      rocwmma::store_matrix_sync(
          o_tmp + wave * WT * OT_PAD, o_acc[d], OT_PAD,
          rocwmma::mem_row_major);
      __syncthreads();

      if (lane < WT) {
        int row = wave * WT + lane;
        int q_idx = q_start + row;
        if (q_idx < params.qL) {
          float inv_l = (l_arr[row] > 0.0f) ? (1.0f / l_arr[row]) : 0.0f;
          T* dst = O + bid_b * params.O_strides[0]
                     + bid_h * params.O_strides[1]
                     + q_idx * params.O_strides[2]
                     + d * WT;
          float* src = o_tmp + wave * WT * OT_PAD + lane * OT_PAD;
          for (int c = 0; c < WT && (d * WT + c) < D; c++)
            dst[c] = static_cast<T>(src[c] * inv_l);
        }
      }
      __syncthreads();
    }
  }
#endif // ROCM_FA_WMMA
}

} // namespace rocm

// ---- Host interface ----

bool supports_sdpa_flash_wmma(
    const array& q, const array& k, const array& v,
    bool has_arr_mask, bool output_logsumexp) {
  // Host-side check: always enabled when compiled for WMMA-capable targets.
  // The kernel itself guards with #if ROCM_FA_WMMA for device code.
  if (output_logsumexp || has_arr_mask) return false;
  if (q.dtype() != bfloat16 && q.dtype() != float16) return false;
  int D = q.shape(-1);
  if (D != v.shape(-1)) return false;
  return (D == 64 || D == 128 || D == 256);
}

// Shared-memory (LDS) bytes the WMMA flash kernel needs for a given head dim.
// The kernel stages Q/K/V one DW=min(D,128)-wide head-dim slice at a time, so
// the footprint is capped at the D=128 layout regardless of D. Callers compare
// this against the running device's LDS budget before selecting the kernel.
int sdpa_flash_wmma_smem(int D) {
  constexpr int BM = 64, BN = 64;
  const int DW = (D < 128) ? D : 128;
  return 2 * BM * (int)sizeof(float)
       + BM * (DW + 4) * (int)sizeof(uint16_t)
       + BN * (DW + 4) * (int)sizeof(uint16_t)
       + BM * (BN + 4) * (int)sizeof(float);
}

void sdpa_flash_wmma(
    const array& q, const array& k, const array& v,
    float scale, array& o, bool do_causal, Stream s) {
  auto& d = rocm::device(s.device);
  auto& enc = d.get_command_encoder(s);

  int B = q.shape(0), H = q.shape(1), qL = q.shape(2), kL = k.shape(2);
  int D = q.shape(3);

  o.set_data(mlx::core::rocm::malloc_async(o.nbytes(), enc));

  rocm::FAWmmaParams p{};
  p.B = B; p.H = H; p.D = D; p.qL = qL; p.kL = kL;
  p.gqa_factor = H / k.shape(1);
  p.scale = scale;
  for (int i = 0; i < 3; i++) {
    p.Q_strides[i] = q.strides(i);
    p.K_strides[i] = k.strides(i);
    p.V_strides[i] = v.strides(i);
    p.O_strides[i] = o.strides(i);
  }

  constexpr int BM = 64, BN = 64;
  dim3 grid(H, (qL + BM - 1) / BM, B);
  dim3 block(128);

  // Shared memory: m/l + Q + KV + S, with Q/KV sized to one DW-wide slice.
  int smem = sdpa_flash_wmma_smem(D);

  auto launch = [&](auto type_tag, auto causal_tag, auto dim_tag) {
    using DT = decltype(type_tag);
    constexpr bool C = decltype(causal_tag)::value;
    constexpr int DD = decltype(dim_tag)::value;
    enc.launch_kernel([&, p, grid, block, smem](hipStream_t stream) {
      hipLaunchKernelGGL(
          (rocm::kernel_sdpa_flash_wmma<DT, C, DD, BM, BN>),
          grid, block, smem, stream,
          gpu_ptr<const DT>(q), gpu_ptr<const DT>(k),
          gpu_ptr<const DT>(v), gpu_ptr<DT>(o), p);
    });
  };

  auto dispatch_dim = [&](auto tt, auto ct) {
    if (D == 64)  launch(tt, ct, std::integral_constant<int, 64>());
    else if (D == 128) launch(tt, ct, std::integral_constant<int, 128>());
    else if (D == 256) launch(tt, ct, std::integral_constant<int, 256>());
  };

  if (o.dtype() == bfloat16) {
    if (do_causal) dispatch_dim(hip_bfloat16(), std::true_type());
    else dispatch_dim(hip_bfloat16(), std::false_type());
  } else {
    if (do_causal) dispatch_dim(__half(), std::true_type());
    else dispatch_dim(__half(), std::false_type());
  }
}

} // namespace mlx::core
