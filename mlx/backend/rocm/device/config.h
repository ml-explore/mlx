// Copyright © 2025 Apple Inc.

// This file is used by both HIP kernel code and host-only C++ code.

#pragma once

// The maximum dimensions of shape/strides passed as kernel parameters.
#define MAX_NDIM 10

// AMD GPU warp (wavefront) size varies by architecture:
// - CDNA/GCN (gfx9xx and earlier): 64
// - RDNA (gfx10xx, gfx11xx, gfx12xx): 32
//
// The __AMDGCN_WAVEFRONT_SIZE__ macro is defined by the HIP compiler
// based on the target architecture. We use it when available for device code.
//
// IMPORTANT: For host code, we need a consistent value that matches the
// compiled device code. Since we compile for specific architectures via
// CMAKE_HIP_ARCHITECTURES, we need to ensure host and device agree.
//
// For now, we default to 32 (RDNA) since that's the most common consumer GPU.
// If targeting CDNA/GCN architectures, change this to 64.
#if defined(__AMDGCN_WAVEFRONT_SIZE__)
// Device code: use the compiler-provided value
#define WARP_SIZE __AMDGCN_WAVEFRONT_SIZE__
#elif defined(__HIP_DEVICE_COMPILE__)
// Device code without wavefront size macro - check architecture macros
#if defined(__gfx1010__) || defined(__gfx1011__) || defined(__gfx1012__) || \
    defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || \
    defined(__gfx1033__) || defined(__gfx1034__) || defined(__gfx1035__) || \
    defined(__gfx1036__) || defined(__gfx1100__) || defined(__gfx1101__) || \
    defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || \
    defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)
#define WARP_SIZE 32
#else
#define WARP_SIZE 64
#endif
#else
// Host code: use a fixed value that matches the target architecture.
// This MUST match the CMAKE_HIP_ARCHITECTURES setting.
// For RDNA (gfx10xx, gfx11xx, gfx12xx): 32
// For CDNA/GCN (gfx9xx): 64
#define WARP_SIZE 32
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
  bool has_native_wmma; // True if arch is on rocWMMA allowlist
                        // (CDNA1/2/3 + RDNA3 dGPU + gfx1151 + RDNA4)
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
// fewer CUs → larger tiles for more L2 reuse per block.
inline ArchTuning get_arch_tuning(const HWInfo& hw) {
  auto t = get_arch_tuning(hw.tier);

  // TILE_N is bounded by how many column streams L2 holds without evicting the
  // reused X/scales. RDNA 3/3.5 (2 MB L2): 16. RDNA 4 (8 MB L2): 24.
  if (hw.tier == RocmArchTier::Rdna3 || hw.tier == RocmArchTier::Rdna35) {
    t.qmv_tile_n = (hw.num_cus <= 16) ? 8 : 16;
  } else if (hw.tier == RocmArchTier::Rdna4) {
    if (hw.num_cus <= 16) {
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
