// FlashAttention-2 backward for ROCm (memory-safe tiled recompute).
//
// Complements flash_attention_wmma.hip forward + LSE emit. Never materializes
// full [B,H,qL,kL] scores — only BM×BN tiles in LDS.
//
// v1: float32 tile GEMMs in LDS + float32 atomic accum for dK/dV (GQA-safe).
// MFMA panel GEMMs can replace the loops later without changing the host API.
//
// Grid: (Hq, ceil(qL/BM), B). Opt-in via MLX_SDPA_FLASH_VJP=1.

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/dtype_utils.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace mlx::core {
namespace rocm {

struct FABwdParams {
  int B, H, Hkv, D, qL, kL, gqa_factor;
  float scale;
  int64_t Q_s[3], K_s[3], V_s[3], dO_s[3];
  int64_t LSE_s[3];
};

} // namespace rocm

namespace {

template <typename T>
__global__ void kernel_sdpa_delta_row(
    const T* __restrict__ O,
    const T* __restrict__ dO,
    float* __restrict__ Delta,
    int B,
    int H,
    int qL,
    int D,
    int64_t Os0,
    int64_t Os1,
    int64_t Os2,
    int64_t dOs0,
    int64_t dOs1,
    int64_t dOs2) {
  (void)B;
  int b = blockIdx.z;
  int h = blockIdx.y;
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= qL)
    return;
  const T* o = O + b * Os0 + h * Os1 + q * Os2;
  const T* g = dO + b * dOs0 + h * dOs1 + q * dOs2;
  float acc = 0.f;
  for (int d = 0; d < D; ++d)
    acc += static_cast<float>(o[d]) * static_cast<float>(g[d]);
  Delta[((size_t)b * H + h) * qL + q] = acc;
}

// FA-2 bwd for fixed D; float LDS tiles; float32 global accum for dQ/dK/dV.
// BM=64, BN=16: more Q-row parallelism, LDS ~62 KB (fits 64 KB devices).
// 64×64 tiles need ~98 KB and fail to launch on MI300X/RDNA3.5.
template <typename T, bool do_causal, int D, int BM>
__global__ void __launch_bounds__(256) kernel_sdpa_flash_bwd_f32acc(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    const float* __restrict__ LSE,
    const float* __restrict__ Delta,
    const T* __restrict__ dO,
    float* __restrict__ dQf,
    float* __restrict__ dKf,
    float* __restrict__ dVf,
    rocm::FABwdParams p) {
  // BM is a template param so the host can shrink the Q-tile for large head
  // dims: LDS = (2*BM + BM*BN + 3*BM*D + 2*BN*D) floats must fit the device
  // budget. D=64->BM=64 (60.5KB), D=128->BM=16 (41KB), D=256->BM=8 (56.6KB).
  // Shrinking BM only adds grid blocks (grid.y = ceil(qL/BM)); the compute is
  // D-agnostic, so this is correctness-preserving.
  constexpr int BN = 16, NTHREADS = 256;
  const int bid_b = blockIdx.z;
  const int bid_h = blockIdx.x;
  const int bid_kv = bid_h / p.gqa_factor;
  const int q_start = blockIdx.y * BM;
  const int tid = threadIdx.x;
  if (q_start >= p.qL)
    return;
  const int q_valid = min(BM, p.qL - q_start);

  // smem layout: lse[BM] delta[BM] S[BM*BN] Q[BM*D] dO[BM*D] K[BN*D] V[BN*D] dQ[BM*D]
  extern __shared__ char smem[];
  float* lse_sh = reinterpret_cast<float*>(smem);
  float* delta_sh = lse_sh + BM;
  float* S_sh = delta_sh + BM;
  float* Q_sh = S_sh + BM * BN;
  float* dO_sh = Q_sh + BM * D;
  float* K_sh = dO_sh + BM * D;
  float* V_sh = K_sh + BN * D;
  float* dQ_sh = V_sh + BN * D;

  if (tid < q_valid) {
    int q = q_start + tid;
    lse_sh[tid] =
        LSE[bid_b * p.LSE_s[0] + bid_h * p.LSE_s[1] + q * p.LSE_s[2]];
    delta_sh[tid] = Delta[((size_t)bid_b * p.H + bid_h) * p.qL + q];
  } else if (tid < BM) {
    lse_sh[tid] = 0.f;
    delta_sh[tid] = 0.f;
  }

  for (int idx = tid; idx < q_valid * D; idx += NTHREADS) {
    int r = idx / D, c = idx % D;
    int q = q_start + r;
    Q_sh[r * D + c] = static_cast<float>(
        Q[bid_b * p.Q_s[0] + bid_h * p.Q_s[1] + q * p.Q_s[2] + c]);
    dO_sh[r * D + c] = static_cast<float>(
        dO[bid_b * p.dO_s[0] + bid_h * p.dO_s[1] + q * p.dO_s[2] + c]);
  }
  for (int idx = tid; idx < BM * D; idx += NTHREADS) {
    int r = idx / D;
    if (r >= q_valid) {
      Q_sh[idx] = 0.f;
      dO_sh[idx] = 0.f;
    }
    dQ_sh[idx] = 0.f;
  }
  __syncthreads();

  for (int k_start = 0; k_start < p.kL; k_start += BN) {
    if constexpr (do_causal) {
      int last_q = q_start + q_valid - 1;
      int max_k = (p.kL - p.qL) + last_q;
      if (k_start > max_k)
        break;
    }
    int k_valid = min(BN, p.kL - k_start);

    for (int idx = tid; idx < k_valid * D; idx += NTHREADS) {
      int r = idx / D, c = idx % D;
      int k = k_start + r;
      K_sh[r * D + c] = static_cast<float>(
          K[bid_b * p.K_s[0] + bid_kv * p.K_s[1] + k * p.K_s[2] + c]);
      V_sh[r * D + c] = static_cast<float>(
          V[bid_b * p.V_s[0] + bid_kv * p.V_s[1] + k * p.V_s[2] + c]);
    }
    for (int idx = tid; idx < BN * D; idx += NTHREADS) {
      if (idx / D >= k_valid) {
        K_sh[idx] = 0.f;
        V_sh[idx] = 0.f;
      }
    }
    __syncthreads();

    // P = exp(scale * Q@K^T - LSE), causal mask
    for (int idx = tid; idx < BM * BN; idx += NTHREADS) {
      int r = idx / BN, c = idx % BN;
      if (r < q_valid && c < k_valid) {
        float acc = 0.f;
#pragma unroll 4
        for (int d = 0; d < D; ++d)
          acc += Q_sh[r * D + d] * K_sh[c * D + d];
        acc *= p.scale;
        int q = q_start + r;
        int k = k_start + c;
        bool use = true;
        if constexpr (do_causal)
          use = (k <= (p.kL - p.qL + q));
        S_sh[r * BN + c] = use ? expf(acc - lse_sh[r]) : 0.f;
      } else {
        S_sh[r * BN + c] = 0.f;
      }
    }
    __syncthreads();

    // dV += P^T @ dO
    for (int idx = tid; idx < k_valid * D; idx += NTHREADS) {
      int c = idx / D, d = idx % D;
      float acc = 0.f;
      for (int r = 0; r < q_valid; ++r)
        acc += S_sh[r * BN + c] * dO_sh[r * D + d];
      atomicAdd(
          &dVf
              [((size_t)bid_b * p.Hkv + bid_kv) * p.kL * D +
               (size_t)(k_start + c) * D + d],
          acc);
    }
    // Barrier BEFORE overwriting S_sh: the dV loop above reads S_sh (=P) across
    // all q-rows for each k-col, and the dS loop below overwrites the same
    // S_sh. Without this sync a thread in dS can clobber S_sh[r,c] before a
    // thread in dV has finished reading it -> corrupted dV. Manifests on wave32
    // (8 wavefronts/block on RDNA vs 4 on CDNA) and compounds across K-tiles
    // (L>=BN), matching the observed L=8-ok / L>=64-diverge on gfx1152/RDNA4.
    __syncthreads();

    // dS = P * (dO @ V^T - Delta)
    for (int idx = tid; idx < BM * BN; idx += NTHREADS) {
      int r = idx / BN, c = idx % BN;
      if (r < q_valid && c < k_valid) {
        float dp = 0.f;
#pragma unroll 4
        for (int d = 0; d < D; ++d)
          dp += dO_sh[r * D + d] * V_sh[c * D + d];
        float P = S_sh[r * BN + c];
        S_sh[r * BN + c] = P * (dp - delta_sh[r]);
      } else {
        S_sh[r * BN + c] = 0.f;
      }
    }
    __syncthreads();

    // dQ += dS @ K * scale
    for (int idx = tid; idx < q_valid * D; idx += NTHREADS) {
      int r = idx / D, d = idx % D;
      float acc = 0.f;
      for (int c = 0; c < k_valid; ++c)
        acc += S_sh[r * BN + c] * K_sh[c * D + d];
      dQ_sh[r * D + d] += acc * p.scale;
    }

    // dK += dS^T @ Q * scale
    for (int idx = tid; idx < k_valid * D; idx += NTHREADS) {
      int c = idx / D, d = idx % D;
      float acc = 0.f;
      for (int r = 0; r < q_valid; ++r)
        acc += S_sh[r * BN + c] * Q_sh[r * D + d];
      atomicAdd(
          &dKf
              [((size_t)bid_b * p.Hkv + bid_kv) * p.kL * D +
               (size_t)(k_start + c) * D + d],
          acc * p.scale);
    }
    __syncthreads();
  }

  for (int idx = tid; idx < q_valid * D; idx += NTHREADS) {
    int r = idx / D, c = idx % D;
    int q = q_start + r;
    dQf[((size_t)bid_b * p.H + bid_h) * p.qL * D + (size_t)q * D + c] =
        dQ_sh[r * D + c];
  }
}

template <typename T>
__global__ void
kernel_cast_f32_to_T(const float* __restrict__ src, T* __restrict__ dst, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = static_cast<T>(src[i]);
}

__global__ void kernel_zero_f32(float* p, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    p[i] = 0.f;
}

} // namespace

// ---- Host interface (still inside namespace mlx::core) ----

// Bytes of dynamic LDS for kernel_sdpa_flash_bwd_f32acc with BM=64, BN=16.
int sdpa_flash_bwd_smem(int D, int BM) {
  constexpr int BN = 16;
  return (2 * BM + BM * BN + 3 * BM * D + 2 * BN * D) * (int)sizeof(float);
}

bool supports_sdpa_flash_wmma_bwd(
    const array& q,
    const array& k,
    const array& v,
    bool has_arr_mask,
    bool has_sinks) {
  if (has_arr_mask || has_sinks)
    return false;
  if (q.dtype() != bfloat16 && q.dtype() != float16)
    return false;
  int D = q.shape(-1);
  if (D != (int)v.shape(-1) || D != (int)k.shape(-1))
    return false;
  // Supported head dims: 64 (lemonseed), 128 (Qwen3.6), 256. BM shrinks with D
  // so LDS stays under the device budget (see kernel_sdpa_flash_bwd_f32acc).
  if (D != 64 && D != 128 && D != 256)
    return false;
  if (q.shape(1) % k.shape(1) != 0)
    return false;
  return true;
}

void sdpa_flash_wmma_bwd(
    const array& q,
    const array& k,
    const array& v,
    const array& o,
    const array& lse,
    const array& d_o,
    float scale,
    bool do_causal,
    array& d_q,
    array& d_k,
    array& d_v,
    Stream s) {
  auto& dev = rocm::device(s.device);
  auto& enc = dev.get_command_encoder(s);

  int B = q.shape(0), H = q.shape(1), qL = q.shape(2), kL = k.shape(2);
  int D = q.shape(3);
  int Hkv = k.shape(1);
  int gqa = H / Hkv;

  d_q.set_data(rocm::malloc_async(d_q.nbytes(), enc));
  d_k.set_data(rocm::malloc_async(d_k.nbytes(), enc));
  d_v.set_data(rocm::malloc_async(d_v.nbytes(), enc));

  size_t n_dq = (size_t)B * H * qL * D;
  size_t n_dk = (size_t)B * Hkv * kL * D;
  size_t n_delta = (size_t)B * H * qL;
  array dQf({(int)n_dq}, float32, nullptr, {});
  array dKf({(int)n_dk}, float32, nullptr, {});
  array dVf({(int)n_dk}, float32, nullptr, {});
  array Delta({(int)n_delta}, float32, nullptr, {});
  dQf.set_data(rocm::malloc_async(dQf.nbytes(), enc));
  dKf.set_data(rocm::malloc_async(dKf.nbytes(), enc));
  dVf.set_data(rocm::malloc_async(dVf.nbytes(), enc));
  Delta.set_data(rocm::malloc_async(Delta.nbytes(), enc));
  enc.add_temporary(dQf);
  enc.add_temporary(dKf);
  enc.add_temporary(dVf);
  enc.add_temporary(Delta);

  // zero f32 accumulators
  {
    dim3 cg((std::max(n_dq, n_dk) + 255) / 256);
    dim3 cb(256);
    enc.add_kernel_node_ex(
        &kernel_zero_f32, cg, cb, 0u, gpu_ptr<float>(dQf), n_dq);
    enc.add_kernel_node_ex(
        &kernel_zero_f32, cg, cb, 0u, gpu_ptr<float>(dKf), n_dk);
    enc.add_kernel_node_ex(
        &kernel_zero_f32, cg, cb, 0u, gpu_ptr<float>(dVf), n_dk);
  }

  rocm::FABwdParams p{};
  p.B = B;
  p.H = H;
  p.Hkv = Hkv;
  p.D = D;
  p.qL = qL;
  p.kL = kL;
  p.gqa_factor = gqa;
  p.scale = scale;
  for (int i = 0; i < 3; i++) {
    p.Q_s[i] = q.strides(i);
    p.K_s[i] = k.strides(i);
    p.V_s[i] = v.strides(i);
    p.dO_s[i] = d_o.strides(i);
  }
  if (lse.ndim() >= 3) {
    p.LSE_s[0] = lse.strides(0);
    p.LSE_s[1] = lse.strides(1);
    p.LSE_s[2] = lse.strides(2);
  } else {
    p.LSE_s[0] = (int64_t)H * qL;
    p.LSE_s[1] = qL;
    p.LSE_s[2] = 1;
  }

  enc.set_input_array(q);
  enc.set_input_array(k);
  enc.set_input_array(v);
  enc.set_input_array(o);
  enc.set_input_array(lse);
  enc.set_input_array(d_o);
  enc.set_output_array(d_q);
  enc.set_output_array(d_k);
  enc.set_output_array(d_v);

  dim3 dgrid((qL + 255) / 256, H, B);
  dim3 dblock(256);
  auto launch_delta = [&](auto tag) {
    using DT = decltype(tag);
    enc.add_kernel_node_ex(
        &kernel_sdpa_delta_row<DT>,
        dgrid,
        dblock,
        0u,
        gpu_ptr<const DT>(o),
        gpu_ptr<const DT>(d_o),
        gpu_ptr<float>(Delta),
        B,
        H,
        qL,
        D,
        o.strides(0),
        o.strides(1),
        o.strides(2),
        d_o.strides(0),
        d_o.strides(1),
        d_o.strides(2));
  };
  if (q.dtype() == bfloat16)
    launch_delta(hip_bfloat16{});
  else
    launch_delta(__half{});

  dim3 block(256);

  auto launch_bwd = [&](auto tag, auto ctag) {
    using DT = decltype(tag);
    constexpr bool C = decltype(ctag)::value;
    // Pick the Q-tile BM per head dim so LDS fits, then launch the matching
    // compile-time instantiation. grid.y and smem both derive from that BM.
    auto go = [&](auto Dc, auto BMc) {
      constexpr int Dv = decltype(Dc)::value;
      constexpr int BMv = decltype(BMc)::value;
      dim3 grid(H, (qL + BMv - 1) / BMv, B);
      int smem = sdpa_flash_bwd_smem(Dv, BMv);
      if (smem > dev.max_shared_memory_per_block()) {
        throw std::runtime_error(
            "[sdpa_flash_wmma_bwd] LDS " + std::to_string(smem) +
            " exceeds device budget " +
            std::to_string(dev.max_shared_memory_per_block()));
      }
      enc.add_kernel_node_ex(
          &kernel_sdpa_flash_bwd_f32acc<DT, C, Dv, BMv>,
          grid,
          block,
          static_cast<uint32_t>(smem),
          gpu_ptr<const DT>(q),
          gpu_ptr<const DT>(k),
          gpu_ptr<const DT>(v),
          gpu_ptr<const float>(lse),
          gpu_ptr<const float>(Delta),
          gpu_ptr<const DT>(d_o),
          gpu_ptr<float>(dQf),
          gpu_ptr<float>(dKf),
          gpu_ptr<float>(dVf),
          p);
    };
    if (D == 64)
      go(std::integral_constant<int, 64>{}, std::integral_constant<int, 64>{});
    else if (D == 128)
      go(std::integral_constant<int, 128>{}, std::integral_constant<int, 16>{});
    else if (D == 256)
      go(std::integral_constant<int, 256>{}, std::integral_constant<int, 8>{});
    else
      throw std::runtime_error(
          "[sdpa_flash_wmma_bwd] unsupported head dim " + std::to_string(D));
  };

  if (q.dtype() == bfloat16) {
    if (do_causal)
      launch_bwd(hip_bfloat16{}, std::true_type{});
    else
      launch_bwd(hip_bfloat16{}, std::false_type{});
  } else {
    if (do_causal)
      launch_bwd(__half{}, std::true_type{});
    else
      launch_bwd(__half{}, std::false_type{});
  }

  auto cast_back = [&](const array& src, array& dst) {
    size_t n = src.size();
    dim3 cg((n + 255) / 256), cb(256);
    if (dst.dtype() == bfloat16) {
      enc.add_kernel_node_ex(
          &kernel_cast_f32_to_T<hip_bfloat16>,
          cg,
          cb,
          0u,
          gpu_ptr<const float>(src),
          gpu_ptr<hip_bfloat16>(dst),
          n);
    } else {
      enc.add_kernel_node_ex(
          &kernel_cast_f32_to_T<__half>,
          cg,
          cb,
          0u,
          gpu_ptr<const float>(src),
          gpu_ptr<__half>(dst),
          n);
    }
  };
  cast_back(dQf, d_q);
  cast_back(dKf, d_k);
  cast_back(dVf, d_v);
}

} // namespace mlx::core
