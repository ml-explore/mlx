// Copyright © 2025 Apple Inc.

// This file is used by both HIP kernel code and host-only C++ code.

#pragma once

// The maximum dimensions of shape/strides passed as kernel parameters.
#define MAX_NDIM 10

// AMD GPU warp (wavefront) size varies by architecture:
// - RDNA (gfx10xx, gfx11xx, gfx12xx, including gfx1152 / 860M): **32**
// - CDNA/GCN (gfx9xx MI-series): **64**
//
// Device code: always use the compiler wavefront size for the offload arch.
// Host code: MLX_HOST_WARP_SIZE from CMake (RDNA→32, CDNA-only→64). Host
// launch dims must still prefer runtime HWInfo::warp_size / Device::warp_size()
// so multi-arch fatbins are correct on every device.
//
// Policy: RDNA = 32. Never launch dim3(64, ...) on RDNA — that produces
// garbage QMV / decode (e.g. "!!!!!!" on gfx1152).
#if defined(__AMDGCN_WAVEFRONT_SIZE__)
// Device code: use the compiler-provided value (32 on RDNA, 64 on CDNA).
#define WARP_SIZE __AMDGCN_WAVEFRONT_SIZE__
#elif defined(__HIP_DEVICE_COMPILE__)
// Device code without wavefront size macro - classify by architecture macros.
// All RDNA generations are wave32.
#if defined(__gfx1010__) || defined(__gfx1011__) || defined(__gfx1012__) || \
    defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || \
    defined(__gfx1033__) || defined(__gfx1034__) || defined(__gfx1035__) || \
    defined(__gfx1036__) || defined(__gfx1100__) || defined(__gfx1101__) || \
    defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || \
    defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__) || \
    defined(__gfx1200__) || defined(__gfx1201__)
#define WARP_SIZE 32 // RDNA
#else
#define WARP_SIZE 64 // CDNA/GCN
#endif
#else
// Host code fallback when CMake did not define MLX_HOST_WARP_SIZE.
// Default **32 (RDNA)** — consumer / multi-arch builds. CDNA-only builds must
// set MLX_HOST_WARP_SIZE=64 via CMake (gfx9-only arch list).
#if defined(MLX_HOST_WARP_SIZE)
#define WARP_SIZE MLX_HOST_WARP_SIZE
#else
#define WARP_SIZE 32 // RDNA default (was 64 — wrong for gfx10/11/12)
#endif
#endif

namespace mlx::core::rocm {

// Configuration constants for ROCm kernels

// Default thread block size
constexpr int kDefaultBlockSize = 256;

// Maximum threads per block (typical for AMD GPUs)
constexpr int kMaxThreadsPerBlock = 1024;

// Warp size (wavefront size) - use the macro for compile-time value
constexpr int kWarpSize = WARP_SIZE;

// Maximum shared memory per block (in bytes)
constexpr int kMaxSharedMemoryPerBlock = 65536;

// Maximum number of dimensions supported
constexpr int kMaxNdim = 8;

// Reduce constants
constexpr int kReduceBlockSize = 256;
constexpr int kReduceMaxBlocks = 1024;

// Copy constants
constexpr int kCopyBlockSize = 256;

// Softmax constants
constexpr int kSoftmaxBlockSize = 256;

// Layer norm constants
constexpr int kLayerNormBlockSize = 256;

// RMS norm constants
constexpr int kRMSNormBlockSize = 256;

// Attention constants
constexpr int kAttentionBlockSize = 256;

// ---- Architecture tier detection and per-arch kernel tuning ----
//
// RocmArchTier provides fine-grained GPU generation identification.
// ArchTuning holds per-arch parameters for kernel dispatch decisions.
// Both are usable from host code and kernel dispatch logic.

enum class RocmArchTier {
  Rdna2, // gfx10xx: RDNA 2, Wave32, no WMMA
  Rdna3, // gfx1100-gfx1103: RDNA 3, Wave32, WMMA, 96KB LDS
  Rdna35, // gfx1150-gfx1152: RDNA 3.5, Wave32, WMMA, 64KB LDS, 32MB IC
  Rdna4, // gfx1200-gfx1201: RDNA 4, Wave32, enhanced WMMA
  Cdna, // gfx9xx: MI-series, Wave64
};

// Hardware capabilities detected at runtime from hipDeviceProp_t.
struct HWInfo {
  RocmArchTier tier;
  int num_cus; // Compute units (multiProcessorCount)
  int simds_per_cu; // SIMDs per CU (2 for RDNA, 4 for CDNA)
  int max_threads_per_cu; // Max resident threads per CU
  int shared_mem_per_cu; // Shared/LDS memory per CU in bytes
  int l2_cache_bytes; // L2/Infinity Cache size
  int warp_size; // Runtime wavefront size (32 RDNA, 64 CDNA). Prefer this for
                 // host launch dims over compile-time WARP_SIZE — multi-arch
                 // fatbins pin MLX_HOST_WARP_SIZE from gfx9 in the arch list
                 // (64) even when running on gfx1152 (32), which yields garbage.
  bool has_native_wmma; // True if arch is on rocWMMA allowlist
                        // (CDNA + RDNA3 dGPU + RDNA3.5 gfx1150–1152 + RDNA4)
  bool is_low_cu_igpu; // num_cus <= 8 (or forced): prefer safer tiles / paths
};

// Per-architecture tuning parameters for quantized matvec and attention
// kernels.
struct ArchTuning {
  // QMV tiled kernel
  int qmv_tile_n; // Output columns per block (L2 reuse)
  // QMV↔GEMM crossover M thresholds
  int qmv_crossover_small; // For K<=2048, N<=2048
  int qmv_crossover_medium; // For K<=4096, N<=4096
  int qmv_crossover_large; // For larger shapes
  // Flash attention
  int fa_block_m; // Queries per flash attention block
  int fa_block_n; // Keys per iteration
};

// Auto-tune based on detected hardware. Adjusts tile sizes based on actual
// CU count to balance occupancy vs L2 reuse.
inline ArchTuning get_arch_tuning(RocmArchTier tier) {
  // Defaults per tier — used when HWInfo isn't available
  switch (tier) {
    case RocmArchTier::Rdna2:
      return ArchTuning{8, 28, 20, 14, 128, 64};
    case RocmArchTier::Rdna3:
      return ArchTuning{16, 36, 24, 16, 64, 64};
    case RocmArchTier::Rdna35:
      // 40 CUs: TILE_N=16 gives best occupancy/reuse balance
      return ArchTuning{16, 36, 24, 16, 64, 64};
    case RocmArchTier::Rdna4:
      return ArchTuning{32, 40, 28, 18, 64, 64};
    case RocmArchTier::Cdna:
    default:
      return ArchTuning{16, 20, 14, 10, 128, 64};
  }
}

// Auto-tune using full hardware info. Adjusts TILE_N based on CU count:
// fewer CUs → smaller tiles so more workgroups stay resident (not thrashing L2).
inline ArchTuning get_arch_tuning(const HWInfo& hw) {
  auto t = get_arch_tuning(hw.tier);

  // TILE_N is bounded by how many column streams L2 holds without evicting the
  // reused X/scales. RDNA 3/3.5 (2 MB L2): 16. RDNA 4 (8 MB L2): 24.
  // Reduced-CU gfx1152 (often 4–8 CUs, 860M / Ryzen AI iGPU): shrink tiles and
  // FA blocks hard — wide tiles oversubscribe L2 and tank occupancy.
  const bool low_cu =
      hw.is_low_cu_igpu || (hw.num_cus > 0 && hw.num_cus <= 8);
  if (hw.tier == RocmArchTier::Rdna3 || hw.tier == RocmArchTier::Rdna35) {
    if (low_cu) {
      t.qmv_tile_n = 4; // was 8 — 8 CUs need even narrower columns
      t.fa_block_m = 32;
      t.fa_block_n = 32;
      t.qmv_crossover_small = 16;
      t.qmv_crossover_medium = 12;
      t.qmv_crossover_large = 8;
    } else if (hw.num_cus > 0 && hw.num_cus <= 16) {
      t.qmv_tile_n = 8;
    } else {
      t.qmv_tile_n = 16;
    }
  } else if (hw.tier == RocmArchTier::Rdna4) {
    if (low_cu || hw.num_cus <= 16) {
      t.qmv_tile_n = 8;
    } else if (hw.l2_cache_bytes >= (6 << 20)) {
      t.qmv_tile_n = 24; // >=6 MB L2 (Navi 48 = 8 MB): wider tile, less waste
    } else {
      t.qmv_tile_n = 16;
    }
  }

  return t;
}

} // namespace mlx::core::rocm
