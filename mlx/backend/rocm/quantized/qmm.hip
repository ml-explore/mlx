// Copyright © 2025 Apple Inc.

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/gemms/naive_gemm.h"
#include "mlx/backend/rocm/gemms/hipblaslt_gemm.h"
#include "mlx/backend/rocm/gemms/rocblas_gemm.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/quantized/quantized.h"
#include "mlx/backend/rocm/quantized/qmv_tiled_kernel.hip"
#include "mlx/primitives.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <algorithm>
#include <numeric>
// rocWMMA is only supported on CDNA (gfx9xx) and RDNA 3+ (gfx11xx, gfx12xx).
// Guard the include so it doesn't trigger static_assert on RDNA 1/2 (gfx10xx).
// During host compilation __HIP_DEVICE_COMPILE__ is 0 so rocwmma defines
// ROCWMMA_ARCH_HOST and compiles fine.  During device compilation for
// unsupported architectures like gfx1030 the header would static_assert.
#if !defined(__HIP_DEVICE_COMPILE__) || !__HIP_DEVICE_COMPILE__ || \
    defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__) || \
    defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__) || \
    defined(__gfx1200__) || defined(__gfx1201__)
#define ROCM_HAS_WMMA 1
#include <rocwmma/rocwmma.hpp>
#else
#define ROCM_HAS_WMMA 0
#endif
#include <algorithm>
#include <atomic>
#include <cstring>
#include <list>
#include <mutex>
#include <unordered_map>

namespace mlx::core {

namespace rocm {

// Strided 2D row-copy kernel: copies rows from a source with row_stride != cols
// into a contiguous destination.
// src layout: row i starts at src + i * src_row_stride (elements contiguous within row)
// dst layout: row i starts at dst + i * cols (fully contiguous)
//
// When both row strides and cols_bytes are 4-byte aligned, uses uint32_t
// copies (one 4-byte word per thread iteration) for good throughput without
// alignment concerns. Falls back to byte-by-byte for the non-aligned tail.
__global__ void strided_row_copy_kernel(
    const char* __restrict__ src,
    char* __restrict__ dst,
    int64_t num_rows,
    int64_t cols_bytes,
    int64_t src_row_stride_bytes,
    int64_t dst_row_stride_bytes,
    bool use_word_copy) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t grid_stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  if (use_word_copy) {
    // Fast path: 4-byte word copies. All row strides are 4-byte aligned.
    constexpr int64_t WORD = 4;
    int64_t cols_words = cols_bytes / WORD;
    int64_t total_words = num_rows * cols_words;
    for (int64_t i = tid; i < total_words; i += grid_stride) {
      int64_t row = i / cols_words;
      int64_t word_in_row = i % cols_words;
      int64_t src_off = row * src_row_stride_bytes + word_in_row * WORD;
      int64_t dst_off = row * dst_row_stride_bytes + word_in_row * WORD;
      *reinterpret_cast<uint32_t*>(dst + dst_off) =
          *reinterpret_cast<const uint32_t*>(src + src_off);
    }
    // Handle remainder bytes (cols_bytes % 4)
    int64_t remainder_start = cols_words * WORD;
    int64_t remainder_bytes = cols_bytes - remainder_start;
    if (remainder_bytes > 0) {
      for (int64_t i = tid; i < num_rows * remainder_bytes; i += grid_stride) {
        int64_t row = i / remainder_bytes;
        int64_t byte_in_tail = i % remainder_bytes;
        int64_t src_off = row * src_row_stride_bytes + remainder_start + byte_in_tail;
        int64_t dst_off = row * dst_row_stride_bytes + remainder_start + byte_in_tail;
        dst[dst_off] = src[src_off];
      }
    }
  } else {
    // Slow path: byte-by-byte copy for non-aligned strides.
    int64_t total_bytes = num_rows * cols_bytes;
    for (int64_t i = tid; i < total_bytes; i += grid_stride) {
      int64_t row = i / cols_bytes;
      int64_t byte_in_row = i % cols_bytes;
      int64_t src_off = row * src_row_stride_bytes + byte_in_row;
      int64_t dst_off = row * dst_row_stride_bytes + byte_in_row;
      dst[dst_off] = src[src_off];
    }
  }
}

// General strided copy kernel with strides passed as kernel arguments
// (by-value hip_array structs). Avoids device memory allocation +
// hipMemcpyAsync overhead that contiguous_copy_gpu -> copy_general_input
// would incur. Falls back to contiguous_copy_gpu only for ndim > MAX_NDIM.
__global__ void strided_general_copy_kernel(
    const char* __restrict__ src,
    char* __restrict__ dst,
    int64_t total_elems,
    int elem_bytes,
    int ndim,
    hip_array<int64_t, MAX_NDIM> shapes,
    hip_array<int64_t, MAX_NDIM> strides_bytes) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t grid_stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = tid; idx < total_elems; idx += grid_stride) {
    // Convert linear index to strided source offset
    int64_t src_offset = 0;
    int64_t remaining = idx;
    for (int d = ndim - 1; d >= 0; --d) {
      int64_t coord = remaining % shapes[d];
      remaining /= shapes[d];
      src_offset += coord * strides_bytes[d];
    }
    // Copy element bytes -- specialize for common QMM element sizes
    int64_t dst_offset = idx * elem_bytes;
    if (elem_bytes == 2) {
      *reinterpret_cast<uint16_t*>(dst + dst_offset) =
          *reinterpret_cast<const uint16_t*>(src + src_offset);
    } else if (elem_bytes == 4) {
      *reinterpret_cast<uint32_t*>(dst + dst_offset) =
          *reinterpret_cast<const uint32_t*>(src + src_offset);
    } else if (elem_bytes == 1) {
      dst[dst_offset] = src[src_offset];
    } else if (elem_bytes == 8) {
      *reinterpret_cast<uint64_t*>(dst + dst_offset) =
          *reinterpret_cast<const uint64_t*>(src + src_offset);
    } else {
      for (int b = 0; b < elem_bytes; ++b) {
        dst[dst_offset + b] = src[src_offset + b];
      }
    }
  }
}

} // namespace rocm

namespace {

template <typename T>
struct local_type_identity {
  using type = T;
};

// Fast contiguous-copy helper for QMM inputs.
//
// Design goals vs the previous implementation (which called contiguous_copy_gpu
// unconditionally when strides didn't match row-major):
//
//   1. **Already contiguous** -- return immediately (unchanged).
//
//   2. **Inner-contiguous with outer stride gap** -- the most common
//      non-contiguous pattern from `take` / `gather_sort`.  The inner N-1
//      dimensions are packed (stride-1 on the last dim, products match for
//      the rest), but the outermost dimension has a stride larger than the
//      product of inner shapes.  We handle this with a single
//      `strided_row_copy_kernel` launch -- no device memory allocation for
//      shapes/strides, no hipMemcpyAsync.  One kernel dispatch total.
//
//   3. **General non-contiguous** (rare for QMM inputs) -- uses
//      `strided_general_copy_kernel` which takes shapes and strides as
//      kernel arguments (up to QMM_COPY_MAX_DIMS dimensions).  This avoids
//      the 2x allocator::malloc + 2x hipMemcpyAsync that
//      `contiguous_copy_gpu -> copy_general_input` would issue.  One kernel
//      dispatch total.  Falls back to `contiguous_copy_gpu` only for arrays
//      with more than MAX_NDIM (10) dimensions (extremely unlikely for
//      QMM operands).
//
// Net effect: non-contiguous copies go from 5 GPU operations (2 allocs +
// 2 memcpy + 1 kernel) down to 1 kernel launch.
inline array ensure_row_contiguous_matrix(
    const array& x,
    rocm::CommandEncoder& enc,
    const Stream& s) {
  if (x.ndim() == 0) {
    return x;
  }

  // --- Fast path 1: already row-major contiguous ---
  int ndim = x.ndim();
  const auto& strides = x.strides();
  bool row_major_contiguous = true;
  int64_t expected_stride = 1;
  // Track the innermost contiguous dimensions while checking.
  // If we break at dimension i, dimensions [i+1 .. ndim-1] are packed.
  int first_noncontig_dim = -1;
  for (int i = ndim - 1; i >= 0; --i) {
    if (x.shape(i) > 1) {
      if (strides[i] != expected_stride) {
        row_major_contiguous = false;
        first_noncontig_dim = i;
        break;
      }
      expected_stride *= x.shape(i);
    }
  }

  if (row_major_contiguous) {
    return x;
  }

  // Empty arrays don't need copying.
  if (x.size() == 0) {
    return x;
  }

  size_t elem_bytes = x.itemsize();

  // Helper: allocate a contiguous output array and return src/dst pointers.
  // Deferred until we know a copy is actually needed and which path to use.
  auto make_output = [&]() -> array {
    array out(x.shape(), x.dtype(), nullptr, {});
    out.set_data(allocator::malloc(out.nbytes()));
    enc.add_temporary(out);
    return out;
  };

  // --- Fast path 2: inner-contiguous, only outermost dim has a stride gap ---
  // This covers the common case where x comes from take/gather of a [E, K]
  // or [B, M, K] array -- inner dims are packed, outer dim stride > product.
  // We also handle the case where the gap is at any single dimension (not
  // just dim 0) as long as all dimensions below it are packed.
  if (first_noncontig_dim >= 0) {
    // Verify that all dimensions below first_noncontig_dim are packed,
    // and only first_noncontig_dim itself has a non-standard stride.
    // Dimensions above first_noncontig_dim (if any) must also be consistent
    // with first_noncontig_dim's layout.
    bool is_simple_outer_gap = true;
    // Check: first_noncontig_dim's stride must be >= expected_stride
    // (i.e. the inner block is correct, just spaced further apart).
    if (strides[first_noncontig_dim] < expected_stride) {
      is_simple_outer_gap = false;
    }
    // Check dimensions above first_noncontig_dim: their strides must be
    // consistent with first_noncontig_dim's stride * shape products.
    if (is_simple_outer_gap) {
      int64_t outer_expected = strides[first_noncontig_dim] * x.shape(first_noncontig_dim);
      for (int i = first_noncontig_dim - 1; i >= 0; --i) {
        if (x.shape(i) <= 1) continue;
        if (strides[i] != outer_expected) {
          is_simple_outer_gap = false;
          break;
        }
        outer_expected *= x.shape(i);
      }
    }

    if (is_simple_outer_gap && first_noncontig_dim == 0) {
      // Simplest case: only the outermost dim has extra stride.
      // inner_size = product of shapes[1..ndim-1]
      array x_copy = make_output();
      const char* src = reinterpret_cast<const char*>(gpu_ptr<const void>(x));
      char* dst = reinterpret_cast<char*>(gpu_ptr<void>(x_copy));

      int64_t inner_size = 1;
      for (int i = 1; i < ndim; ++i) {
        inner_size *= x.shape(i);
      }
      int64_t num_rows = x.shape(0);
      int64_t cols_bytes = inner_size * static_cast<int64_t>(elem_bytes);
      int64_t src_row_stride_bytes = strides[0] * static_cast<int64_t>(elem_bytes);
      int64_t dst_row_stride_bytes = cols_bytes;
      bool word_copy = (cols_bytes % 4 == 0) &&
                       (src_row_stride_bytes % 4 == 0) &&
                       (dst_row_stride_bytes % 4 == 0);

      int block_size = 256;
      int64_t work_items = word_copy
          ? num_rows * (cols_bytes / 4)
          : num_rows * cols_bytes;
      int num_blocks = static_cast<int>(
          std::min<int64_t>((work_items + block_size - 1) / block_size, 65535));

      enc.launch_kernel([=](hipStream_t stream) {
        hipLaunchKernelGGL(
            rocm::strided_row_copy_kernel,
            dim3(num_blocks), dim3(block_size), 0, stream,
            src, dst,
            num_rows, cols_bytes,
            src_row_stride_bytes, dst_row_stride_bytes,
            word_copy);
      });
      return x_copy;
    }

    if (is_simple_outer_gap) {
      // Gap at an interior dimension. batch_count == 1 is common here.
      int64_t batch_count = 1;
      for (int i = 0; i < first_noncontig_dim; ++i) {
        batch_count *= x.shape(i);
      }
      if (batch_count == 1) {
        array x_copy = make_output();
        const char* src = reinterpret_cast<const char*>(gpu_ptr<const void>(x));
        char* dst = reinterpret_cast<char*>(gpu_ptr<void>(x_copy));

        int64_t inner_size = 1;
        for (int i = first_noncontig_dim + 1; i < ndim; ++i) {
          inner_size *= x.shape(i);
        }
        int64_t slab_rows = x.shape(first_noncontig_dim);
        int64_t cols_bytes = inner_size * static_cast<int64_t>(elem_bytes);
        int64_t src_row_stride_bytes = strides[first_noncontig_dim] * static_cast<int64_t>(elem_bytes);
        int64_t dst_row_stride_bytes = cols_bytes;
        bool word_copy = (cols_bytes % 4 == 0) &&
                         (src_row_stride_bytes % 4 == 0) &&
                         (dst_row_stride_bytes % 4 == 0);

        int block_size = 256;
        int64_t work_items = word_copy
            ? slab_rows * (cols_bytes / 4)
            : slab_rows * cols_bytes;
        int num_blocks = static_cast<int>(
            std::min<int64_t>((work_items + block_size - 1) / block_size, 65535));

        enc.launch_kernel([=](hipStream_t stream) {
          hipLaunchKernelGGL(
              rocm::strided_row_copy_kernel,
              dim3(num_blocks), dim3(block_size), 0, stream,
              src, dst,
              slab_rows, cols_bytes,
              src_row_stride_bytes, dst_row_stride_bytes,
              word_copy);
        });
        return x_copy;
      }
      // batch_count > 1 with interior gap: fall through to general path
    }
  }

  // --- Fast path 3: general non-contiguous, strides as kernel args ---
  // Handles arbitrary stride patterns with up to MAX_NDIM dimensions.
  // Shapes and byte-strides are passed as hip_array structs (by value),
  // so no device memory allocation or hipMemcpyAsync is needed.
  // One kernel launch total.
  if (ndim <= MAX_NDIM) {
    array x_copy = make_output();
    const char* src = reinterpret_cast<const char*>(gpu_ptr<const void>(x));
    char* dst = reinterpret_cast<char*>(gpu_ptr<void>(x_copy));

    int64_t total_elems = x.size();
    int eb = static_cast<int>(elem_bytes);

    int block_size = 256;
    int num_blocks = static_cast<int>(
        std::min<int64_t>((total_elems + block_size - 1) / block_size, 65535));

    // Pack into hip_array structs that can be passed by value to the kernel.
    rocm::hip_array<int64_t, MAX_NDIM> shapes_arg = {};
    rocm::hip_array<int64_t, MAX_NDIM> strides_bytes_arg = {};
    for (int i = 0; i < ndim; ++i) {
      shapes_arg.data_[i] = x.shape(i);
      strides_bytes_arg.data_[i] = strides[i] * static_cast<int64_t>(elem_bytes);
    }

    enc.launch_kernel([=](hipStream_t stream) {
      hipLaunchKernelGGL(
          rocm::strided_general_copy_kernel,
          dim3(num_blocks), dim3(block_size), 0, stream,
          src, dst,
          total_elems, eb, ndim,
          shapes_arg, strides_bytes_arg);
    });
    return x_copy;
  }

  // --- Fallback: ndim > MAX_NDIM (extremely rare for QMM) ---
  // Use the generic copy infrastructure which allocates device buffers
  // for shape/strides arrays (2 allocs + 2 hipMemcpyAsync + 1 kernel).
  array x_copy = contiguous_copy_gpu(x, s);
  enc.add_temporary(x_copy);
  return x_copy;
}

inline int parse_cols_per_block_env(const char* env_name) {
  const char* raw = std::getenv(env_name);
  if (raw == nullptr || *raw == '\0') {
    return 0;
  }

  char* end = nullptr;
  long value = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0') {
    return 0;
  }

  return (value == 4 || value == 8 || value == 16 || value == 32 || value == 64)
      ? static_cast<int>(value)
      : 0;
}

inline int parse_threads_per_col_env(const char* env_name) {
  const char* raw = std::getenv(env_name);
  if (raw == nullptr || *raw == '\0') {
    return 0;
  }

  char* end = nullptr;
  long value = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0') {
    return 0;
  }

  return (value == 16 || value == WARP_SIZE) ? static_cast<int>(value) : 0;
}

inline bool parse_warp_kernel_env(const char* env_name, bool default_value) {
  const char* raw = std::getenv(env_name);
  if (raw == nullptr || *raw == '\0') {
    return default_value;
  }

  if (raw[0] == '0' && raw[1] == '\0') {
    return false;
  }
  if (raw[0] == '1' && raw[1] == '\0') {
    return true;
  }
  return default_value;
}

inline int parse_positive_int_env(const char* env_name, int default_value) {
  const char* raw = std::getenv(env_name);
  if (raw == nullptr || *raw == '\0') {
    return default_value;
  }

  char* end = nullptr;
  long value = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || value <= 0) {
    return default_value;
  }
  return static_cast<int>(value);
}

inline size_t parse_non_negative_size_t_env(
    const char* env_name,
    size_t default_value) {
  const char* raw = std::getenv(env_name);
  if (raw == nullptr || *raw == '\0') {
    return default_value;
  }

  char* end = nullptr;
  unsigned long long value = std::strtoull(raw, &end, 10);
  if (end == raw || *end != '\0') {
    return default_value;
  }
  return static_cast<size_t>(value);
}

inline int parse_non_negative_int_env(const char* env_name, int default_value) {
  const char* raw = std::getenv(env_name);
  if (raw == nullptr || *raw == '\0') {
    return default_value;
  }

  char* end = nullptr;
  long value = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || value < 0) {
    return default_value;
  }
  return static_cast<int>(value);
}

// Check if rocBLAS dequant fast path should be used
// Default ON
inline bool use_rocblas_dequant_path() {
  static bool checked = false;
  static bool enabled = true;
  if (!checked) {
    const char* raw = std::getenv("MLX_ROCM_QMM_DEQUANT_GEMM");
    if (raw != nullptr) {
      enabled = (raw[0] == '1' && raw[1] == '\0');
    }
    checked = true;
  }
  return enabled;
}

inline bool has_only_singleton_batch_dims(const array& x) {
  if (x.ndim() <= 2) {
    return true;
  }
  for (int i = 0; i < x.ndim() - 2; ++i) {
    if (x.shape(i) != 1) {
      return false;
    }
  }
  return true;
}

inline int select_qmv_cols_per_block(int K, int N, int bits) {
  int env_cols = parse_cols_per_block_env("MLX_ROCM_QMV_COLS_PER_BLOCK");
  if (env_cols > 0) {
    return env_cols;
  }

  (void)K;

  if (N < 256) {
    return 4;
  }
  if (K <= 1024) {
    return (N < 1024) ? 8 : 16;
  }
  if (bits == 8) {
    if (N < 1024) {
      return 8;
    }
    if (N < 4096) {
      return 32;
    }
    return 16;
  }
  if (N < 1024) {
    return 8;
  }
  return 16;
}

inline int select_qmv_threads_per_col(int K, int N, int bits, int batch_count) {
  // On RDNA 3.5 (wave32), 16 threads per column gives better occupancy
  // than 32 for most LLM decode shapes. 32 threads only helps for very
  // large K where the extra parallelism in the reduction outweighs the
  // reduced block count.
  int threads_per_col = 16;
  if (WARP_SIZE == 32) {
    bool quant_bits_supported =
        (bits == 2 || bits == 4 || bits == 5 || bits == 6 || bits == 8);
    // On RDNA 3.5 (40 CUs / 20 WGPs), 16 threads/col allows 2 columns
    // per warp, increasing memory-level parallelism for decode. Only use
    // full warp (32) for extreme K where reduction parallelism dominates.
    bool extreme = (batch_count == 1) && (K >= 16384);
    if (quant_bits_supported && extreme) {
      threads_per_col = WARP_SIZE;
    }
  }
  return threads_per_col;
}

// Use shared arch detection from config.h (RocmArchTier, ArchTuning).
// Local aliases for backward compatibility within this file.
using RocmQmvArchTier = rocm::RocmArchTier;

inline rocm::HWInfo detect_rocm_hw_info(rocm::Device& d) {
  static std::mutex hw_mutex;
  static std::unordered_map<int, rocm::HWInfo> hw_cache;

  int hip_device = d.hip_device();
  {
    std::lock_guard<std::mutex> lock(hw_mutex);
    auto it = hw_cache.find(hip_device);
    if (it != hw_cache.end()) {
      return it->second;
    }
  }

  hipDeviceProp_t props{};
  d.make_current();
  hipError_t err = hipGetDeviceProperties(&props, hip_device);

  rocm::HWInfo hw{};
  hw.tier = (WARP_SIZE == 32) ? RocmQmvArchTier::Rdna2 : RocmQmvArchTier::Cdna;

  if (err == hipSuccess) {
    hw.num_cus = props.multiProcessorCount;
    hw.max_threads_per_cu = props.maxThreadsPerMultiProcessor;
    hw.shared_mem_per_cu = props.sharedMemPerBlock;
    hw.l2_cache_bytes = props.l2CacheSize;

    const char* arch_name = props.gcnArchName;
    if (arch_name != nullptr) {
      if (std::strstr(arch_name, "gfx1200") != nullptr ||
          std::strstr(arch_name, "gfx1201") != nullptr) {
        hw.tier = RocmQmvArchTier::Rdna4;
        hw.simds_per_cu = 2;
      } else if (std::strstr(arch_name, "gfx1150") != nullptr ||
                 std::strstr(arch_name, "gfx1151") != nullptr ||
                 std::strstr(arch_name, "gfx1152") != nullptr) {
        hw.tier = RocmQmvArchTier::Rdna35;
        hw.simds_per_cu = 2;
      } else if (std::strstr(arch_name, "gfx11") != nullptr) {
        hw.tier = RocmQmvArchTier::Rdna3;
        hw.simds_per_cu = 2;
      } else if (std::strstr(arch_name, "gfx10") != nullptr) {
        hw.tier = RocmQmvArchTier::Rdna2;
        hw.simds_per_cu = 2;
      } else if (std::strstr(arch_name, "gfx9") != nullptr) {
        hw.tier = RocmQmvArchTier::Cdna;
        hw.simds_per_cu = 4;
      }

      // rocWMMA library arch allowlist (AMD's official support matrix).
      // CDNA1/2/3 use MFMA under rocwmma; RDNA3 dGPU + gfx1151 + RDNA4 use
      // hardware WMMA. Excludes gfx1103/1150/1152 and all gfx10xx (RDNA1/2).
      static const char* const kRocwmmaArches[] = {
          "gfx908",  "gfx90a",  "gfx942",
          "gfx1100", "gfx1101", "gfx1102",
          "gfx1151",
          "gfx1200", "gfx1201",
      };
      for (const char* a : kRocwmmaArches) {
        if (std::strstr(arch_name, a) != nullptr) {
          hw.has_native_wmma = true;
          break;
        }
      }
    }
  }

  {
    std::lock_guard<std::mutex> lock(hw_mutex);
    hw_cache[hip_device] = hw;
  }
  return hw;
}

inline RocmQmvArchTier detect_rocm_qmv_arch_tier(rocm::Device& d) {
  return detect_rocm_hw_info(d).tier;
}

inline int select_qmv_qmm_crossover_m_threshold(
    int K,
    int N,
    int batch_count,
    bool transpose,
    bool can_use_batched_qmv,
    rocm::Device& d) {
  if (!transpose) {
    return 1;
  }
  if ((batch_count > 1) && !can_use_batched_qmv) {
    return 1;
  }

  int small_shape_limit;
  int medium_shape_limit;
  int large_shape_limit;

  auto tuning = rocm::get_arch_tuning(detect_rocm_qmv_arch_tier(d));
  small_shape_limit = tuning.qmv_crossover_small;
  medium_shape_limit = tuning.qmv_crossover_medium;
  large_shape_limit = tuning.qmv_crossover_large;

  if (batch_count > 1 && can_use_batched_qmv) {
    small_shape_limit += 8;
    medium_shape_limit += 6;
    large_shape_limit += 4;
  }

  if (K <= 2048 && N <= 2048) {
    return small_shape_limit;
  }
  if (K <= 4096 && N <= 4096) {
    return medium_shape_limit;
  }
  return large_shape_limit;
}

inline bool should_use_tiny_k_qmv_path(
    int M,
    int N,
    int K,
    int batch_count,
    bool transpose,
    bool can_use_batched_qmv,
    int bits,
    QuantizationMode mode) {
  if (!transpose || can_use_batched_qmv || batch_count != 1) {
    return false;
  }

  bool bits_supported = (bits == 2 || bits == 4 || bits == 8) ||
      (mode == QuantizationMode::Affine && (bits == 5 || bits == 6));
  if (!bits_supported) {
    return false;
  }

  bool tiny_k = (K == 64 || K == 128 || K == 256);
  bool decode_like = (M <= 4);
  bool width_enough = (N >= 512);
  return tiny_k && decode_like && width_enough;
}

inline bool is_aligned_ptr(const void* ptr, size_t align) {
  if (ptr == nullptr || align == 0) {
    return false;
  }
  auto addr = reinterpret_cast<std::uintptr_t>(ptr);
  return (addr % align) == 0;
}

inline bool has_packed_layout_compatibility_for_aligned_qmv(int K, int bits) {
  switch (bits) {
    case 8:
      return (K % 16) == 0;
    case 6:
      return (K % 64) == 0;
    case 4:
      return (K % 32) == 0;
    case 2:
      return (K % 64) == 0;
    default:
      return false;
  }
}

inline bool should_use_alignment_qmv_noshared_path(
    int M,
    int N,
    int K,
    int batch_count,
    bool transpose,
    bool can_use_batched_qmv,
    int bits,
    QuantizationMode mode,
    const void* x_ptr,
    const void* w_ptr,
    const void* scales_ptr,
    const void* biases_ptr,
    bool has_bias) {
  if (!transpose || can_use_batched_qmv || batch_count != 1) {
    return false;
  }

  bool bits_supported = (bits == 2 || bits == 4 || bits == 8) ||
      (mode == QuantizationMode::Affine && bits == 6);
  if (!bits_supported) {
    return false;
  }
  if (!has_packed_layout_compatibility_for_aligned_qmv(K, bits)) {
    return false;
  }

  bool decode_like = (M <= 8);
  bool width_enough = (N >= 1024);
  if (!decode_like || !width_enough) {
    return false;
  }

  bool pointers_aligned = is_aligned_ptr(x_ptr, 16) &&
      is_aligned_ptr(w_ptr, 16) && is_aligned_ptr(scales_ptr, 16);
  if (has_bias) {
    pointers_aligned = pointers_aligned && is_aligned_ptr(biases_ptr, 16);
  }
  return pointers_aligned;
}

inline bool should_use_dequant_gemm_path(
    int M,
    int N,
    int K,
    int batch_count,
    bool non_batched,
    bool transpose,
    bool can_use_batched_qmv,
    rocm::Device& d) {
  int env_threshold =
      parse_positive_int_env("MLX_ROCM_QMM_DEQUANT_M_THRESHOLD", -1);
  if (env_threshold > 0) {
    return M >= env_threshold;
  }

  if (!transpose) {
    return true;
  }

  if (batch_count > 1) {
    if (!can_use_batched_qmv) {
      return true;
    }
  }

  if (!non_batched) {
    return M >= select_qmv_qmm_crossover_m_threshold(
                    K, N, batch_count, transpose, can_use_batched_qmv, d);
  }

  int threshold = select_qmv_qmm_crossover_m_threshold(
      K, N, batch_count, transpose, can_use_batched_qmv, d);

  if (M >= threshold) {
    return true;
  }

  // Favor dequant+GEMM slightly earlier on very large decode-style shapes.
  if (N >= 8192 && K >= 2048) {
    return M >= std::max(8, threshold - 4);
  }
  return false;
}

struct DequantCacheKey {
  std::uintptr_t w_ptr;
  std::uintptr_t scales_ptr;
  std::uintptr_t biases_ptr;
  int group_size;
  int bits;
  int stream_index;
  bool transpose;
  Dtype dtype;

  bool operator==(const DequantCacheKey& other) const {
    return w_ptr == other.w_ptr && scales_ptr == other.scales_ptr &&
        biases_ptr == other.biases_ptr && group_size == other.group_size &&
        bits == other.bits && stream_index == other.stream_index &&
        transpose == other.transpose && dtype == other.dtype;
  }
};

struct DequantCacheKeyHasher {
  size_t operator()(const DequantCacheKey& key) const {
    size_t h = std::hash<std::uintptr_t>{}(key.w_ptr);
    h ^= std::hash<std::uintptr_t>{}(key.scales_ptr) + 0x9e3779b9 + (h << 6) +
        (h >> 2);
    h ^= std::hash<std::uintptr_t>{}(key.biases_ptr) + 0x9e3779b9 + (h << 6) +
        (h >> 2);
    h ^= std::hash<int>{}(key.group_size) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(key.bits) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(key.stream_index) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(static_cast<int>(key.transpose)) + 0x9e3779b9 +
        (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(static_cast<int>(key.dtype.val())) + 0x9e3779b9 +
        (h << 6) + (h >> 2);
    return h;
  }
};

struct DequantCacheEntry {
  array weight;
  array w_source;
  array scales_source;
  std::optional<array> biases_source;
  size_t bytes;
  std::list<DequantCacheKey>::iterator lru_it;
};

inline int dequant_cache_capacity() {
  static int capacity = []() {
    const char* raw = std::getenv("MLX_ROCM_QMM_DEQUANT_CACHE_SIZE");
    if (raw == nullptr || *raw == '\0') {
      return 8;
    }
    char* end = nullptr;
    long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || value < 0) {
      return 8;
    }
    return static_cast<int>(value);
  }();
  return capacity;
}

inline size_t dequant_cache_max_bytes() {
  static size_t max_bytes = parse_non_negative_size_t_env(
      "MLX_ROCM_QMM_DEQUANT_CACHE_MAX_BYTES", 256ULL * 1024ULL * 1024ULL);
  return max_bytes;
}

inline int qmm_gemm_solution_index_f32(bool batched) {
  static int single_index =
      parse_non_negative_int_env("MLX_ROCM_GEMM_F32_SOLUTION_INDEX", 0);
  static int batched_index = parse_non_negative_int_env(
      "MLX_ROCM_GEMM_F32_BATCHED_SOLUTION_INDEX", -1);
  if (!batched) {
    return single_index;
  }
  return batched_index >= 0 ? batched_index : single_index;
}

inline int qmm_gemm_solution_index_bf16(bool batched) {
  static int single_index =
      parse_non_negative_int_env("MLX_ROCM_GEMM_BF16_SOLUTION_INDEX", 0);
  static int batched_index = parse_non_negative_int_env(
      "MLX_ROCM_GEMM_BF16_BATCHED_SOLUTION_INDEX", -1);
  if (!batched) {
    return single_index;
  }
  return batched_index >= 0 ? batched_index : single_index;
}

inline rocblas_operation to_rocblas_op(bool transpose) {
  return transpose ? rocblas_operation_transpose : rocblas_operation_none;
}

void dequant_rocblas_gemm(
    rocm::CommandEncoder& enc,
    bool transpose_a,
    bool transpose_b,
    int M,
    int N,
    int K,
    float alpha,
    const array& a,
    int lda,
    const array& b,
    int ldb,
    float beta,
    array& c,
    int ldc,
    Dtype dtype) {
  const void* a_ptr = gpu_ptr<void>(a);
  const void* b_ptr = gpu_ptr<void>(b);
  void* c_ptr = gpu_ptr<void>(c);

  enc.launch_kernel([&, a_ptr, b_ptr, c_ptr](hipStream_t stream) {
    enc.device().set_rocblas_stream(stream);
    rocblas_handle handle = enc.device().get_rocblas_handle();

    rocblas_operation op_a = to_rocblas_op(transpose_a);
    rocblas_operation op_b = to_rocblas_op(transpose_b);

    switch (dtype) {
      case float32: {
        float alpha_f = alpha;
        float beta_f = beta;
        int solution_index = qmm_gemm_solution_index_f32(false);
        static std::atomic<bool> solution_valid{true};

        if (solution_index > 0 &&
            solution_valid.load(std::memory_order_relaxed)) {
          rocblas_status status = rocblas_gemm_ex(
              handle,
              op_b,
              op_a,
              N,
              M,
              K,
              &alpha_f,
              b_ptr,
              rocblas_datatype_f32_r,
              ldb,
              a_ptr,
              rocblas_datatype_f32_r,
              lda,
              &beta_f,
              c_ptr,
              rocblas_datatype_f32_r,
              ldc,
              c_ptr,
              rocblas_datatype_f32_r,
              ldc,
              rocblas_datatype_f32_r,
              rocblas_gemm_algo_solution_index,
              solution_index,
              0);
          if (status != rocblas_status_success) {
            solution_valid.store(false, std::memory_order_relaxed);
            rocblas_sgemm(
                handle,
                op_b,
                op_a,
                N,
                M,
                K,
                &alpha_f,
                static_cast<const float*>(b_ptr),
                ldb,
                static_cast<const float*>(a_ptr),
                lda,
                &beta_f,
                static_cast<float*>(c_ptr),
                ldc);
          }
        } else {
          rocblas_sgemm(
              handle,
              op_b,
              op_a,
              N,
              M,
              K,
              &alpha_f,
              static_cast<const float*>(b_ptr),
              ldb,
              static_cast<const float*>(a_ptr),
              lda,
              &beta_f,
              static_cast<float*>(c_ptr),
              ldc);
        }
        break;
      }
      case float16: {
        rocblas_half alpha_h, beta_h;
        __half alpha_f16 = static_cast<__half>(alpha);
        __half beta_f16 = static_cast<__half>(beta);
        std::memcpy(&alpha_h, &alpha_f16, sizeof(rocblas_half));
        std::memcpy(&beta_h, &beta_f16, sizeof(rocblas_half));
        rocblas_hgemm(
            handle,
            op_b,
            op_a,
            N,
            M,
            K,
            &alpha_h,
            reinterpret_cast<const rocblas_half*>(b_ptr),
            ldb,
            reinterpret_cast<const rocblas_half*>(a_ptr),
            lda,
            &beta_h,
            reinterpret_cast<rocblas_half*>(c_ptr),
            ldc);
        break;
      }
      case bfloat16: {
        float alpha_f = alpha;
        float beta_f = beta;

        // Try hipBLASLt first for bf16 GEMMs — often faster on RDNA 3.5/CDNA
        if (rocm::is_hipblaslt_available()) {
          try {
            // data_type=0 means "use bfloat16", impl maps internally
            rocm::hipblaslt_gemm_raw(
                stream,
                static_cast<int>(op_b), static_cast<int>(op_a),
                N, M, K,
                &alpha_f, b_ptr, ldb, a_ptr, lda,
                &beta_f, c_ptr, ldc,
                2,   // 2 = bfloat16 (mapped in impl)
                0);  // unused
            break;
          } catch (...) {
            // Fall through to rocBLAS
          }
        }

        int solution_index = qmm_gemm_solution_index_bf16(false);
        static std::atomic<bool> solution_valid{true};

        rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
        if (solution_index > 0 &&
            solution_valid.load(std::memory_order_relaxed)) {
          algo = rocblas_gemm_algo_solution_index;
        } else {
          solution_index = 0;
        }

        rocblas_status status = rocblas_gemm_ex(
            handle,
            op_b,
            op_a,
            N,
            M,
            K,
            &alpha_f,
            b_ptr,
            rocblas_datatype_bf16_r,
            ldb,
            a_ptr,
            rocblas_datatype_bf16_r,
            lda,
            &beta_f,
            c_ptr,
            rocblas_datatype_bf16_r,
            ldc,
            c_ptr,
            rocblas_datatype_bf16_r,
            ldc,
            rocblas_datatype_f32_r,
            algo,
            solution_index,
            0);

        if (status != rocblas_status_success &&
            algo == rocblas_gemm_algo_solution_index) {
          solution_valid.store(false, std::memory_order_relaxed);
          rocblas_gemm_ex(
              handle,
              op_b,
              op_a,
              N,
              M,
              K,
              &alpha_f,
              b_ptr,
              rocblas_datatype_bf16_r,
              ldb,
              a_ptr,
              rocblas_datatype_bf16_r,
              lda,
              &beta_f,
              c_ptr,
              rocblas_datatype_bf16_r,
              ldc,
              c_ptr,
              rocblas_datatype_bf16_r,
              ldc,
              rocblas_datatype_f32_r,
              rocblas_gemm_algo_standard,
              0,
              0);
        }
        break;
      }
      default:
        throw std::runtime_error("Unsupported dtype for rocBLAS GEMM");
    }
  });
}

void dequant_rocblas_gemm_batched(
    rocm::CommandEncoder& enc,
    bool transpose_a,
    bool transpose_b,
    int M,
    int N,
    int K,
    float alpha,
    const array& a,
    int lda,
    int64_t stride_a,
    const array& b,
    int ldb,
    int64_t stride_b,
    float beta,
    array& c,
    int ldc,
    int64_t stride_c,
    int batch_count,
    Dtype dtype) {
  const void* a_ptr = gpu_ptr<void>(a);
  const void* b_ptr = gpu_ptr<void>(b);
  void* c_ptr = gpu_ptr<void>(c);

  enc.launch_kernel([&, a_ptr, b_ptr, c_ptr](hipStream_t stream) {
    enc.device().set_rocblas_stream(stream);
    rocblas_handle handle = enc.device().get_rocblas_handle();

    rocblas_operation op_a = to_rocblas_op(transpose_a);
    rocblas_operation op_b = to_rocblas_op(transpose_b);

    switch (dtype) {
      case float32: {
        float alpha_f = alpha;
        float beta_f = beta;
        int solution_index = qmm_gemm_solution_index_f32(true);
        static std::atomic<bool> solution_valid{true};

        if (solution_index > 0 &&
            solution_valid.load(std::memory_order_relaxed)) {
          rocblas_status status = rocblas_gemm_strided_batched_ex(
              handle,
              op_b,
              op_a,
              N,
              M,
              K,
              &alpha_f,
              b_ptr,
              rocblas_datatype_f32_r,
              ldb,
              stride_b,
              a_ptr,
              rocblas_datatype_f32_r,
              lda,
              stride_a,
              &beta_f,
              c_ptr,
              rocblas_datatype_f32_r,
              ldc,
              stride_c,
              c_ptr,
              rocblas_datatype_f32_r,
              ldc,
              stride_c,
              batch_count,
              rocblas_datatype_f32_r,
              rocblas_gemm_algo_solution_index,
              solution_index,
              0);
          if (status != rocblas_status_success) {
            solution_valid.store(false, std::memory_order_relaxed);
            rocblas_sgemm_strided_batched(
                handle,
                op_b,
                op_a,
                N,
                M,
                K,
                &alpha_f,
                static_cast<const float*>(b_ptr),
                ldb,
                stride_b,
                static_cast<const float*>(a_ptr),
                lda,
                stride_a,
                &beta_f,
                static_cast<float*>(c_ptr),
                ldc,
                stride_c,
                batch_count);
          }
        } else {
          rocblas_sgemm_strided_batched(
              handle,
              op_b,
              op_a,
              N,
              M,
              K,
              &alpha_f,
              static_cast<const float*>(b_ptr),
              ldb,
              stride_b,
              static_cast<const float*>(a_ptr),
              lda,
              stride_a,
              &beta_f,
              static_cast<float*>(c_ptr),
              ldc,
              stride_c,
              batch_count);
        }
        break;
      }
      case float16: {
        rocblas_half alpha_h, beta_h;
        __half alpha_f16 = static_cast<__half>(alpha);
        __half beta_f16 = static_cast<__half>(beta);
        std::memcpy(&alpha_h, &alpha_f16, sizeof(rocblas_half));
        std::memcpy(&beta_h, &beta_f16, sizeof(rocblas_half));
        rocblas_hgemm_strided_batched(
            handle,
            op_b,
            op_a,
            N,
            M,
            K,
            &alpha_h,
            reinterpret_cast<const rocblas_half*>(b_ptr),
            ldb,
            stride_b,
            reinterpret_cast<const rocblas_half*>(a_ptr),
            lda,
            stride_a,
            &beta_h,
            reinterpret_cast<rocblas_half*>(c_ptr),
            ldc,
            stride_c,
            batch_count);
        break;
      }
      case bfloat16: {
        float alpha_f = alpha;
        float beta_f = beta;
        int solution_index = qmm_gemm_solution_index_bf16(true);
        static std::atomic<bool> solution_valid{true};

        rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
        if (solution_index > 0 &&
            solution_valid.load(std::memory_order_relaxed)) {
          algo = rocblas_gemm_algo_solution_index;
        } else {
          solution_index = 0;
        }

        rocblas_status status = rocblas_gemm_strided_batched_ex(
            handle,
            op_b,
            op_a,
            N,
            M,
            K,
            &alpha_f,
            b_ptr,
            rocblas_datatype_bf16_r,
            ldb,
            stride_b,
            a_ptr,
            rocblas_datatype_bf16_r,
            lda,
            stride_a,
            &beta_f,
            c_ptr,
            rocblas_datatype_bf16_r,
            ldc,
            stride_c,
            c_ptr,
            rocblas_datatype_bf16_r,
            ldc,
            stride_c,
            batch_count,
            rocblas_datatype_f32_r,
            algo,
            solution_index,
            0);

        if (status != rocblas_status_success &&
            algo == rocblas_gemm_algo_solution_index) {
          solution_valid.store(false, std::memory_order_relaxed);
          rocblas_gemm_strided_batched_ex(
              handle,
              op_b,
              op_a,
              N,
              M,
              K,
              &alpha_f,
              b_ptr,
              rocblas_datatype_bf16_r,
              ldb,
              stride_b,
              a_ptr,
              rocblas_datatype_bf16_r,
              lda,
              stride_a,
              &beta_f,
              c_ptr,
              rocblas_datatype_bf16_r,
              ldc,
              stride_c,
              c_ptr,
              rocblas_datatype_bf16_r,
              ldc,
              stride_c,
              batch_count,
              rocblas_datatype_f32_r,
              rocblas_gemm_algo_standard,
              0,
              0);
        }
        break;
      }
      default:
        throw std::runtime_error("Unsupported dtype for rocBLAS batched GEMM");
    }
  });
}

} // namespace

namespace rocm {

template <int BITS>
__device__ inline uint8_t
unpack_packed_value(const uint8_t* packed_row, int k, int row_bytes) {
  constexpr uint8_t mask = (1u << BITS) - 1u;
  if constexpr (BITS == 2 || BITS == 4 || BITS == 8) {
    constexpr int pack_factor = 8 / BITS;
    int pack_idx = k / pack_factor;
    int bit_offset = (k % pack_factor) * BITS;
    return (packed_row[pack_idx] >> bit_offset) & mask;
  } else {
    int bit_index = k * BITS;
    int byte_idx = bit_index >> 3;
    int bit_offset = bit_index & 0x7;

    uint32_t window = static_cast<uint32_t>(packed_row[byte_idx]);
    if (byte_idx + 1 < row_bytes) {
      window |= static_cast<uint32_t>(packed_row[byte_idx + 1]) << 8;
    }
    return static_cast<uint8_t>((window >> bit_offset) & mask);
  }
}

template <int BITS>
__device__ inline uint8_t
unpack_packed_value_fast(const uint8_t* packed_row, int k, int row_bytes) {
  if constexpr (BITS == 8) {
    (void)row_bytes;
    return packed_row[k];
  } else if constexpr (BITS == 4) {
    (void)row_bytes;
    uint8_t packed = packed_row[k >> 1];
    return (k & 1) ? (packed >> 4) : (packed & 0xF);
  } else if constexpr (BITS == 2) {
    (void)row_bytes;
    uint8_t packed = packed_row[k >> 2];
    return (packed >> ((k & 0x3) * 2)) & 0x3;
  } else {
    return unpack_packed_value<BITS>(packed_row, k, row_bytes);
  }
}

template <int SUBGROUP_SIZE, typename T>
__device__ __forceinline__ T subgroup_reduce_sum_qmm(T val) {
  static_assert((SUBGROUP_SIZE & (SUBGROUP_SIZE - 1)) == 0);
  static_assert(SUBGROUP_SIZE <= WARP_SIZE);

#pragma unroll
  for (int offset = SUBGROUP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_xor(val, offset);
  }
  return val;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_sum_qmm(T val) {
  return subgroup_reduce_sum_qmm<WARP_SIZE>(val);
}

__device__ inline float fp4_e2m1_to_float(uint8_t val) {
  switch (val & 0xF) {
    case 0x0:
      return 0.0f;
    case 0x1:
      return 0.5f;
    case 0x2:
      return 1.0f;
    case 0x3:
      return 1.5f;
    case 0x4:
      return 2.0f;
    case 0x5:
      return 3.0f;
    case 0x6:
      return 4.0f;
    case 0x7:
      return 6.0f;
    case 0x8:
      return -0.0f;
    case 0x9:
      return -0.5f;
    case 0xA:
      return -1.0f;
    case 0xB:
      return -1.5f;
    case 0xC:
      return -2.0f;
    case 0xD:
      return -3.0f;
    case 0xE:
      return -4.0f;
    case 0xF:
      return -6.0f;
    default:
      return 0.0f;
  }
}

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t val) {
  // Use a simple array lookup or bit manipulation.
  // Actually, MI300 supports hardware fp8 conversion:
  // But we can just use a fast bit manipulation without branches.

  uint32_t sign = (val >> 7) & 0x1;
  uint32_t exp = (val >> 3) & 0xF;
  uint32_t mant = val & 0x7;

  if (exp == 0 && mant == 0) {
    return sign ? -0.0f : 0.0f;
  }

  uint32_t float_exp = exp == 0 ? 0 : exp - 7 + 127;
  // Handle subnormals approximately or cleanly if needed,
  // but for performance, we can just do:
  if (exp == 0) {
    float subnormal = static_cast<float>(mant) * 0.001953125f; // 2^-9
    return sign ? -subnormal : subnormal;
  }

  uint32_t float_mant = mant << 20;
  uint32_t bits = (sign << 31) | (float_exp << 23) | float_mant;
  return __uint_as_float(bits);
}

template <int GROUP_SIZE>
__device__ inline float fp_scale_to_float(uint8_t s) {
  if constexpr (GROUP_SIZE == 16) {
    return fp8_e4m3_to_float(s);
  } else {
    union {
      uint16_t i;
      hip_bfloat16 f;
    } out;
    out.i = (s == 0 ? 0x40 : (static_cast<uint16_t>(s) << 7));
    return static_cast<float>(out.f);
  }
}

template <typename ScaleT, int GROUP_SIZE, bool AFFINE>
__device__ inline float load_scale_value(ScaleT raw) {
  if constexpr (AFFINE) {
    return static_cast<float>(raw);
  } else {
    return fp_scale_to_float<GROUP_SIZE>(static_cast<uint8_t>(raw));
  }
}

template <int BITS, bool AFFINE>
__device__ inline float
dequantize_value(uint8_t quant_val, float scale, float bias) {
  if constexpr (AFFINE) {
    return static_cast<float>(quant_val) * scale + bias;
  } else {
    (void)bias;
    if constexpr (BITS == 8) {
      return fp8_e4m3_to_float(quant_val) * scale;
    } else {
      return fp4_e2m1_to_float(quant_val) * scale;
    }
  }
}

template <
    typename T,
    typename ScaleT,
    int BITS,
    int GROUP_SIZE,
    bool AFFINE,
    int THREADS_PER_COL>
__global__ void __launch_bounds__(1024) qmv_warp_shared_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ out,
    int M,
    int N,
    int K,
    bool has_bias) {
  const int lane = threadIdx.x;
  const int warp_idx = threadIdx.y;
  const int col = blockIdx.x * blockDim.y + warp_idx;
  const int row = blockIdx.y;

  const bool valid = (row < M) && (col < N);

  const int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const int row_bytes = (K * BITS + 7) / 8;

  const T* x_row = (row < M) ? (x + row * K) : nullptr;
  const uint8_t* w_row = valid ? (w + col * row_bytes) : nullptr;
  const ScaleT* scales_row = valid ? (scales + col * num_groups) : nullptr;
  const ScaleT* biases_row =
      (valid && has_bias) ? (biases + col * num_groups) : nullptr;

  float acc = 0.0f;

  // We load a chunk of X into shared memory.
  // We use a chunk size of 1024 elements.
  constexpr int CHUNK_SIZE = 2048;
  __shared__ float shared_x[CHUNK_SIZE];

  for (int chunk_start = 0; chunk_start < K; chunk_start += CHUNK_SIZE) {
    int chunk_end = min(chunk_start + CHUNK_SIZE, K);
    int chunk_len = chunk_end - chunk_start;

    // Collaboratively load X chunk into shared memory
    int tid = warp_idx * blockDim.x + lane;
    for (int i = tid; i < chunk_len; i += blockDim.x * blockDim.y) {
      shared_x[i] = static_cast<float>(x_row[chunk_start + i]);
    }
    __syncthreads();

    if (valid) {
      int g_start = chunk_start / GROUP_SIZE;
      int g_end = (chunk_end + GROUP_SIZE - 1) / GROUP_SIZE;

      for (int g = g_start; g < g_end; ++g) {
        int k_start = max(g * GROUP_SIZE, chunk_start);
        int k_end_g = min((g + 1) * GROUP_SIZE, chunk_end);

        float scale =
            load_scale_value<ScaleT, GROUP_SIZE, AFFINE>(scales_row[g]);
        float bias_val = has_bias ? static_cast<float>(biases_row[g]) : 0.0f;

        if constexpr (AFFINE) {
          float qx_acc0 = 0.0f, qx_acc1 = 0.0f, qx_acc2 = 0.0f, qx_acc3 = 0.0f;
          float x_group_sum = 0.0f;
          float qx_acc = 0.0f;

          if constexpr (BITS == 8) {
            int k_local = lane * 4;
            int step = THREADS_PER_COL * 4;
            for (; k_start + k_local + 3 < k_end_g; k_local += step) {
              int k = k_start + k_local;

              uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_row[k]);
              float w0 = static_cast<float>(w_packed & 0xFF);
              float w1 = static_cast<float>((w_packed >> 8) & 0xFF);
              float w2 = static_cast<float>((w_packed >> 16) & 0xFF);
              float w3 = static_cast<float>((w_packed >> 24) & 0xFF);

              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];

              qx_acc0 = fmaf(x0, w0, qx_acc0);
              qx_acc1 = fmaf(x1, w1, qx_acc1);
              qx_acc2 = fmaf(x2, w2, qx_acc2);
              qx_acc3 = fmaf(x3, w3, qx_acc3);

              if (has_bias)
                x_group_sum += x0 + x1 + x2 + x3;
            }
            qx_acc = qx_acc0 + qx_acc1 + qx_acc2 + qx_acc3;
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              float w_val = static_cast<float>(w_row[k]);
              qx_acc = fmaf(x_val, w_val, qx_acc);
              if (has_bias)
                x_group_sum += x_val;
            }
          } else if constexpr (BITS == 4) {
            int k_local = lane * 8;
            int step = THREADS_PER_COL * 8;
            for (; k_start + k_local + 7 < k_end_g; k_local += step) {
              int k = k_start + k_local;
              uint32_t w_packed =
                  *reinterpret_cast<const uint32_t*>(&w_row[k / 2]);
              float w0 = static_cast<float>(w_packed & 0xF);
              float w1 = static_cast<float>((w_packed >> 4) & 0xF);
              float w2 = static_cast<float>((w_packed >> 8) & 0xF);
              float w3 = static_cast<float>((w_packed >> 12) & 0xF);
              float w4 = static_cast<float>((w_packed >> 16) & 0xF);
              float w5 = static_cast<float>((w_packed >> 20) & 0xF);
              float w6 = static_cast<float>((w_packed >> 24) & 0xF);
              float w7 = static_cast<float>((w_packed >> 28) & 0xF);
              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];
              float x4 = shared_x[k - chunk_start + 4];
              float x5 = shared_x[k - chunk_start + 5];
              float x6 = shared_x[k - chunk_start + 6];
              float x7 = shared_x[k - chunk_start + 7];
              // Four independent accumulators so RDNA4 can dual-issue the FMAs
              // (a single serial qx_acc chain runs at half rate). Mirrors the
              // 8-bit branch above; partial sums are reassociated at group end.
              qx_acc0 = fmaf(x0, w0, qx_acc0);
              qx_acc1 = fmaf(x1, w1, qx_acc1);
              qx_acc2 = fmaf(x2, w2, qx_acc2);
              qx_acc3 = fmaf(x3, w3, qx_acc3);
              qx_acc0 = fmaf(x4, w4, qx_acc0);
              qx_acc1 = fmaf(x5, w5, qx_acc1);
              qx_acc2 = fmaf(x6, w6, qx_acc2);
              qx_acc3 = fmaf(x7, w7, qx_acc3);
              if (has_bias)
                x_group_sum += x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
            }
            qx_acc = qx_acc0 + qx_acc1 + qx_acc2 + qx_acc3;
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(x_val, static_cast<float>(quant_val), qx_acc);
              if (has_bias)
                x_group_sum += x_val;
            }
          } else if constexpr (BITS == 6) {
            // Process 8 weights at a time (48 bits = 6 bytes)
            int k_local = lane * 8;
            int step = THREADS_PER_COL * 8;
            // Need at least 7 bytes of room after byte_idx for safe 8-byte load
            // row_bytes = (K * 6 + 7) / 8, so we need byte_idx + 7 < row_bytes
            int max_safe_k =
                ((row_bytes - 7) * 8) / 6; // Max k where 8-byte load is safe
            for (; k_start + k_local + 7 < k_end_g &&
                 k_start + k_local < max_safe_k;
                 k_local += step) {
              int k = k_start + k_local;
              // 8 weights * 6 bits = 48 bits, starting at bit position k*6
              int byte_idx = (k * 6) / 8;
              int bit_offset = (k * 6) % 8;
              // Safe to load 8 bytes (we checked bounds above)
              uint64_t w_packed;
              memcpy(&w_packed, &w_row[byte_idx], 8);
              w_packed >>= bit_offset;
              // Extract 8 6-bit weights
              float w0 = static_cast<float>(w_packed & 0x3F);
              float w1 = static_cast<float>((w_packed >> 6) & 0x3F);
              float w2 = static_cast<float>((w_packed >> 12) & 0x3F);
              float w3 = static_cast<float>((w_packed >> 18) & 0x3F);
              float w4 = static_cast<float>((w_packed >> 24) & 0x3F);
              float w5 = static_cast<float>((w_packed >> 30) & 0x3F);
              float w6 = static_cast<float>((w_packed >> 36) & 0x3F);
              float w7 = static_cast<float>((w_packed >> 42) & 0x3F);
              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];
              float x4 = shared_x[k - chunk_start + 4];
              float x5 = shared_x[k - chunk_start + 5];
              float x6 = shared_x[k - chunk_start + 6];
              float x7 = shared_x[k - chunk_start + 7];
              qx_acc = fmaf(x0, w0, qx_acc);
              qx_acc = fmaf(x1, w1, qx_acc);
              qx_acc = fmaf(x2, w2, qx_acc);
              qx_acc = fmaf(x3, w3, qx_acc);
              qx_acc = fmaf(x4, w4, qx_acc);
              qx_acc = fmaf(x5, w5, qx_acc);
              qx_acc = fmaf(x6, w6, qx_acc);
              qx_acc = fmaf(x7, w7, qx_acc);
              if (has_bias)
                x_group_sum += x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
            }
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(x_val, static_cast<float>(quant_val), qx_acc);
              if (has_bias)
                x_group_sum += x_val;
            }
          } else {
            for (int k_local = lane; k_start + k_local < k_end_g;
                 k_local += THREADS_PER_COL) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(x_val, static_cast<float>(quant_val), qx_acc);
              if (has_bias)
                x_group_sum += x_val;
            }
          }
          acc += scale * qx_acc;
          if (has_bias)
            acc += bias_val * x_group_sum;
        } else {
          float qx_acc0 = 0.0f, qx_acc1 = 0.0f, qx_acc2 = 0.0f, qx_acc3 = 0.0f;
          float qx_acc = 0.0f;
          if constexpr (BITS == 8) {
            int k_local = lane * 4;
            int step = THREADS_PER_COL * 4;
            for (; k_start + k_local + 3 < k_end_g; k_local += step) {
              int k = k_start + k_local;
              uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_row[k]);
              float w0 = fp8_e4m3_to_float(w_packed & 0xFF);
              float w1 = fp8_e4m3_to_float((w_packed >> 8) & 0xFF);
              float w2 = fp8_e4m3_to_float((w_packed >> 16) & 0xFF);
              float w3 = fp8_e4m3_to_float((w_packed >> 24) & 0xFF);

              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];

              qx_acc0 = fmaf(x0, w0, qx_acc0);
              qx_acc1 = fmaf(x1, w1, qx_acc1);
              qx_acc2 = fmaf(x2, w2, qx_acc2);
              qx_acc3 = fmaf(x3, w3, qx_acc3);
            }
            qx_acc = qx_acc0 + qx_acc1 + qx_acc2 + qx_acc3;
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              float w_val = fp8_e4m3_to_float(w_row[k]);
              qx_acc = fmaf(x_val, w_val, qx_acc);
            }
          } else if constexpr (BITS == 4) {
            for (int k_local = lane; k_start + k_local < k_end_g;
                 k_local += THREADS_PER_COL) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(
                  x_val,
                  dequantize_value<BITS, false>(quant_val, 1.0f, 0.0f),
                  qx_acc);
            }
          } else if constexpr (BITS == 6) {
            int k_local = lane * 8;
            int step = THREADS_PER_COL * 8;
            int max_safe_k = ((row_bytes - 7) * 8) / 6;
            for (; k_start + k_local + 7 < k_end_g &&
                 k_start + k_local < max_safe_k;
                 k_local += step) {
              int k = k_start + k_local;
              int byte_idx = (k * 6) / 8;
              int bit_offset = (k * 6) % 8;
              uint64_t w_packed = 0;
              memcpy(&w_packed, &w_row[byte_idx], 8);
              w_packed >>= bit_offset;
              float w0 =
                  dequantize_value<6, false>(w_packed & 0x3F, 1.0f, 0.0f);
              float w1 = dequantize_value<6, false>(
                  (w_packed >> 6) & 0x3F, 1.0f, 0.0f);
              float w2 = dequantize_value<6, false>(
                  (w_packed >> 12) & 0x3F, 1.0f, 0.0f);
              float w3 = dequantize_value<6, false>(
                  (w_packed >> 18) & 0x3F, 1.0f, 0.0f);
              float w4 = dequantize_value<6, false>(
                  (w_packed >> 24) & 0x3F, 1.0f, 0.0f);
              float w5 = dequantize_value<6, false>(
                  (w_packed >> 30) & 0x3F, 1.0f, 0.0f);
              float w6 = dequantize_value<6, false>(
                  (w_packed >> 36) & 0x3F, 1.0f, 0.0f);
              float w7 = dequantize_value<6, false>(
                  (w_packed >> 42) & 0x3F, 1.0f, 0.0f);
              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];
              float x4 = shared_x[k - chunk_start + 4];
              float x5 = shared_x[k - chunk_start + 5];
              float x6 = shared_x[k - chunk_start + 6];
              float x7 = shared_x[k - chunk_start + 7];
              qx_acc0 = fmaf(x0, w0, qx_acc0);
              qx_acc1 = fmaf(x1, w1, qx_acc1);
              qx_acc2 = fmaf(x2, w2, qx_acc2);
              qx_acc3 = fmaf(x3, w3, qx_acc3);
              qx_acc0 = fmaf(x4, w4, qx_acc0);
              qx_acc1 = fmaf(x5, w5, qx_acc1);
              qx_acc2 = fmaf(x6, w6, qx_acc2);
              qx_acc3 = fmaf(x7, w7, qx_acc3);
            }
            qx_acc = qx_acc0 + qx_acc1 + qx_acc2 + qx_acc3;
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(
                  x_val,
                  dequantize_value<BITS, false>(quant_val, 1.0f, 0.0f),
                  qx_acc);
            }
          } else {
            for (int k_local = lane; k_start + k_local < k_end_g;
                 k_local += THREADS_PER_COL) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(
                  x_val,
                  dequantize_value<BITS, false>(quant_val, 1.0f, 0.0f),
                  qx_acc);
            }
          }
          acc += scale * qx_acc;
        }
      }
    }
    __syncthreads(); // ensure all warps are done before loading next chunk
  }

  acc = subgroup_reduce_sum_qmm<THREADS_PER_COL>(acc);
  if (valid && lane == 0) {
    out[row * N + col] = static_cast<T>(acc);
  }
}

template <
    typename T,
    typename ScaleT,
    int BITS,
    int GROUP_SIZE,
    bool AFFINE,
    int THREADS_PER_COL>
__global__ void __launch_bounds__(1024) qmv_warp_shared_batched_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ out,
    int M,
    int N,
    int K,
    int64_t x_batch_stride,
    int64_t w_batch_stride,
    int64_t sb_batch_stride,
    int64_t out_batch_stride,
    bool has_bias) {
  const int lane = threadIdx.x;
  const int warp_idx = threadIdx.y;
  const int col = blockIdx.x * blockDim.y + warp_idx;
  const int row = blockIdx.y;
  const int batch = blockIdx.z;

  const bool valid = (row < M) && (col < N);

  const int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const int row_bytes = (K * BITS + 7) / 8;

  const T* x_batch_ptr = x + static_cast<int64_t>(batch) * x_batch_stride;
  const uint8_t* w_batch_ptr = w + static_cast<int64_t>(batch) * w_batch_stride;
  const ScaleT* scales_batch_ptr =
      scales + static_cast<int64_t>(batch) * sb_batch_stride;
  const ScaleT* biases_batch_ptr = has_bias
      ? (biases + static_cast<int64_t>(batch) * sb_batch_stride)
      : nullptr;
  T* out_batch_ptr = out + static_cast<int64_t>(batch) * out_batch_stride;

  const T* x_row =
      (row < M) ? (x_batch_ptr + static_cast<int64_t>(row) * K) : nullptr;
  const uint8_t* w_row =
      valid ? (w_batch_ptr + static_cast<int64_t>(col) * row_bytes) : nullptr;
  const ScaleT* scales_row = valid
      ? (scales_batch_ptr + static_cast<int64_t>(col) * num_groups)
      : nullptr;
  const ScaleT* biases_row = (valid && has_bias)
      ? (biases_batch_ptr + static_cast<int64_t>(col) * num_groups)
      : nullptr;

  float acc = 0.0f;

  constexpr int CHUNK_SIZE = 2048;
  __shared__ float shared_x[CHUNK_SIZE];

  for (int chunk_start = 0; chunk_start < K; chunk_start += CHUNK_SIZE) {
    int chunk_end = min(chunk_start + CHUNK_SIZE, K);
    int chunk_len = chunk_end - chunk_start;

    int tid = warp_idx * blockDim.x + lane;
    for (int i = tid; i < chunk_len; i += blockDim.x * blockDim.y) {
      shared_x[i] = static_cast<float>(x_row[chunk_start + i]);
    }
    __syncthreads();

    if (valid) {
      int g_start = chunk_start / GROUP_SIZE;
      int g_end = (chunk_end + GROUP_SIZE - 1) / GROUP_SIZE;

      for (int g = g_start; g < g_end; ++g) {
        int k_start = max(g * GROUP_SIZE, chunk_start);
        int k_end_g = min((g + 1) * GROUP_SIZE, chunk_end);

        float scale =
            load_scale_value<ScaleT, GROUP_SIZE, AFFINE>(scales_row[g]);
        float bias_val = has_bias ? static_cast<float>(biases_row[g]) : 0.0f;

        if constexpr (AFFINE) {
          float qx_acc0 = 0.0f;
          float qx_acc1 = 0.0f;
          float qx_acc2 = 0.0f;
          float qx_acc3 = 0.0f;
          float x_group_sum = 0.0f;
          float qx_acc = 0.0f;

          if constexpr (BITS == 8) {
            int k_local = lane * 4;
            int step = THREADS_PER_COL * 4;
            for (; k_start + k_local + 3 < k_end_g; k_local += step) {
              int k = k_start + k_local;

              uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_row[k]);
              float w0 = static_cast<float>(w_packed & 0xFF);
              float w1 = static_cast<float>((w_packed >> 8) & 0xFF);
              float w2 = static_cast<float>((w_packed >> 16) & 0xFF);
              float w3 = static_cast<float>((w_packed >> 24) & 0xFF);

              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];

              qx_acc0 = fmaf(x0, w0, qx_acc0);
              qx_acc1 = fmaf(x1, w1, qx_acc1);
              qx_acc2 = fmaf(x2, w2, qx_acc2);
              qx_acc3 = fmaf(x3, w3, qx_acc3);

              if (has_bias) {
                x_group_sum += x0 + x1 + x2 + x3;
              }
            }
            qx_acc = qx_acc0 + qx_acc1 + qx_acc2 + qx_acc3;
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              float w_val = static_cast<float>(w_row[k]);
              qx_acc = fmaf(x_val, w_val, qx_acc);
              if (has_bias) {
                x_group_sum += x_val;
              }
            }
          } else if constexpr (BITS == 4) {
            int k_local = lane * 8;
            int step = THREADS_PER_COL * 8;
            for (; k_start + k_local + 7 < k_end_g; k_local += step) {
              int k = k_start + k_local;
              uint32_t w_packed =
                  *reinterpret_cast<const uint32_t*>(&w_row[k / 2]);
              float w0 = static_cast<float>(w_packed & 0xF);
              float w1 = static_cast<float>((w_packed >> 4) & 0xF);
              float w2 = static_cast<float>((w_packed >> 8) & 0xF);
              float w3 = static_cast<float>((w_packed >> 12) & 0xF);
              float w4 = static_cast<float>((w_packed >> 16) & 0xF);
              float w5 = static_cast<float>((w_packed >> 20) & 0xF);
              float w6 = static_cast<float>((w_packed >> 24) & 0xF);
              float w7 = static_cast<float>((w_packed >> 28) & 0xF);
              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];
              float x4 = shared_x[k - chunk_start + 4];
              float x5 = shared_x[k - chunk_start + 5];
              float x6 = shared_x[k - chunk_start + 6];
              float x7 = shared_x[k - chunk_start + 7];
              // Four independent accumulators for RDNA4 dual-issue (mirrors the
              // 8-bit branch); reassociated at group end.
              qx_acc0 = fmaf(x0, w0, qx_acc0);
              qx_acc1 = fmaf(x1, w1, qx_acc1);
              qx_acc2 = fmaf(x2, w2, qx_acc2);
              qx_acc3 = fmaf(x3, w3, qx_acc3);
              qx_acc0 = fmaf(x4, w4, qx_acc0);
              qx_acc1 = fmaf(x5, w5, qx_acc1);
              qx_acc2 = fmaf(x6, w6, qx_acc2);
              qx_acc3 = fmaf(x7, w7, qx_acc3);
              if (has_bias) {
                x_group_sum += x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
              }
            }
            qx_acc = qx_acc0 + qx_acc1 + qx_acc2 + qx_acc3;
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(x_val, static_cast<float>(quant_val), qx_acc);
              if (has_bias) {
                x_group_sum += x_val;
              }
            }
          } else if constexpr (BITS == 6) {
            int k_local = lane * 8;
            int step = THREADS_PER_COL * 8;
            int max_safe_k = ((row_bytes - 7) * 8) / 6;
            for (; k_start + k_local + 7 < k_end_g &&
                 k_start + k_local < max_safe_k;
                 k_local += step) {
              int k = k_start + k_local;
              int byte_idx = (k * 6) / 8;
              int bit_offset = (k * 6) % 8;
              uint64_t w_packed = 0;
              memcpy(&w_packed, &w_row[byte_idx], 8);
              w_packed >>= bit_offset;
              float w0 = static_cast<float>(w_packed & 0x3F);
              float w1 = static_cast<float>((w_packed >> 6) & 0x3F);
              float w2 = static_cast<float>((w_packed >> 12) & 0x3F);
              float w3 = static_cast<float>((w_packed >> 18) & 0x3F);
              float w4 = static_cast<float>((w_packed >> 24) & 0x3F);
              float w5 = static_cast<float>((w_packed >> 30) & 0x3F);
              float w6 = static_cast<float>((w_packed >> 36) & 0x3F);
              float w7 = static_cast<float>((w_packed >> 42) & 0x3F);
              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];
              float x4 = shared_x[k - chunk_start + 4];
              float x5 = shared_x[k - chunk_start + 5];
              float x6 = shared_x[k - chunk_start + 6];
              float x7 = shared_x[k - chunk_start + 7];
              qx_acc = fmaf(x0, w0, qx_acc);
              qx_acc = fmaf(x1, w1, qx_acc);
              qx_acc = fmaf(x2, w2, qx_acc);
              qx_acc = fmaf(x3, w3, qx_acc);
              qx_acc = fmaf(x4, w4, qx_acc);
              qx_acc = fmaf(x5, w5, qx_acc);
              qx_acc = fmaf(x6, w6, qx_acc);
              qx_acc = fmaf(x7, w7, qx_acc);
              if (has_bias) {
                x_group_sum += x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
              }
            }
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(x_val, static_cast<float>(quant_val), qx_acc);
              if (has_bias) {
                x_group_sum += x_val;
              }
            }
          } else {
            for (int k_local = lane; k_start + k_local < k_end_g;
                 k_local += THREADS_PER_COL) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(x_val, static_cast<float>(quant_val), qx_acc);
              if (has_bias) {
                x_group_sum += x_val;
              }
            }
          }
          acc += scale * qx_acc;
          if (has_bias) {
            acc += bias_val * x_group_sum;
          }
        } else {
          float qx_acc0 = 0.0f;
          float qx_acc1 = 0.0f;
          float qx_acc2 = 0.0f;
          float qx_acc3 = 0.0f;
          float qx_acc = 0.0f;
          if constexpr (BITS == 8) {
            int k_local = lane * 4;
            int step = THREADS_PER_COL * 4;
            for (; k_start + k_local + 3 < k_end_g; k_local += step) {
              int k = k_start + k_local;

              uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_row[k]);
              float w0 = fp8_e4m3_to_float(w_packed & 0xFF);
              float w1 = fp8_e4m3_to_float((w_packed >> 8) & 0xFF);
              float w2 = fp8_e4m3_to_float((w_packed >> 16) & 0xFF);
              float w3 = fp8_e4m3_to_float((w_packed >> 24) & 0xFF);

              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];

              qx_acc0 = fmaf(x0, w0, qx_acc0);
              qx_acc1 = fmaf(x1, w1, qx_acc1);
              qx_acc2 = fmaf(x2, w2, qx_acc2);
              qx_acc3 = fmaf(x3, w3, qx_acc3);
            }
            qx_acc = qx_acc0 + qx_acc1 + qx_acc2 + qx_acc3;
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              float w_val = fp8_e4m3_to_float(w_row[k]);
              qx_acc = fmaf(x_val, w_val, qx_acc);
            }
          } else if constexpr (BITS == 4) {
            int k_local = lane * 8;
            int step = THREADS_PER_COL * 8;
            for (; k_start + k_local + 7 < k_end_g; k_local += step) {
              int k = k_start + k_local;
              uint32_t w_packed =
                  *reinterpret_cast<const uint32_t*>(&w_row[k / 2]);
              float w0 = dequantize_value<4, false>(w_packed & 0xF, 1.0f, 0.0f);
              float w1 =
                  dequantize_value<4, false>((w_packed >> 4) & 0xF, 1.0f, 0.0f);
              float w2 =
                  dequantize_value<4, false>((w_packed >> 8) & 0xF, 1.0f, 0.0f);
              float w3 = dequantize_value<4, false>(
                  (w_packed >> 12) & 0xF, 1.0f, 0.0f);
              float w4 = dequantize_value<4, false>(
                  (w_packed >> 16) & 0xF, 1.0f, 0.0f);
              float w5 = dequantize_value<4, false>(
                  (w_packed >> 20) & 0xF, 1.0f, 0.0f);
              float w6 = dequantize_value<4, false>(
                  (w_packed >> 24) & 0xF, 1.0f, 0.0f);
              float w7 = dequantize_value<4, false>(
                  (w_packed >> 28) & 0xF, 1.0f, 0.0f);
              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];
              float x4 = shared_x[k - chunk_start + 4];
              float x5 = shared_x[k - chunk_start + 5];
              float x6 = shared_x[k - chunk_start + 6];
              float x7 = shared_x[k - chunk_start + 7];
              qx_acc = fmaf(x0, w0, qx_acc);
              qx_acc = fmaf(x1, w1, qx_acc);
              qx_acc = fmaf(x2, w2, qx_acc);
              qx_acc = fmaf(x3, w3, qx_acc);
              qx_acc = fmaf(x4, w4, qx_acc);
              qx_acc = fmaf(x5, w5, qx_acc);
              qx_acc = fmaf(x6, w6, qx_acc);
              qx_acc = fmaf(x7, w7, qx_acc);
            }
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(
                  x_val,
                  dequantize_value<BITS, false>(quant_val, 1.0f, 0.0f),
                  qx_acc);
            }
          } else if constexpr (BITS == 6) {
            int k_local = lane * 8;
            int step = THREADS_PER_COL * 8;
            int max_safe_k = ((row_bytes - 7) * 8) / 6;
            for (; k_start + k_local + 7 < k_end_g &&
                 k_start + k_local < max_safe_k;
                 k_local += step) {
              int k = k_start + k_local;
              int byte_idx = (k * 6) / 8;
              int bit_offset = (k * 6) % 8;
              uint64_t w_packed = 0;
              memcpy(&w_packed, &w_row[byte_idx], 8);
              w_packed >>= bit_offset;
              float w0 =
                  dequantize_value<6, false>(w_packed & 0x3F, 1.0f, 0.0f);
              float w1 = dequantize_value<6, false>(
                  (w_packed >> 6) & 0x3F, 1.0f, 0.0f);
              float w2 = dequantize_value<6, false>(
                  (w_packed >> 12) & 0x3F, 1.0f, 0.0f);
              float w3 = dequantize_value<6, false>(
                  (w_packed >> 18) & 0x3F, 1.0f, 0.0f);
              float w4 = dequantize_value<6, false>(
                  (w_packed >> 24) & 0x3F, 1.0f, 0.0f);
              float w5 = dequantize_value<6, false>(
                  (w_packed >> 30) & 0x3F, 1.0f, 0.0f);
              float w6 = dequantize_value<6, false>(
                  (w_packed >> 36) & 0x3F, 1.0f, 0.0f);
              float w7 = dequantize_value<6, false>(
                  (w_packed >> 42) & 0x3F, 1.0f, 0.0f);
              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];
              float x4 = shared_x[k - chunk_start + 4];
              float x5 = shared_x[k - chunk_start + 5];
              float x6 = shared_x[k - chunk_start + 6];
              float x7 = shared_x[k - chunk_start + 7];
              qx_acc = fmaf(x0, w0, qx_acc);
              qx_acc = fmaf(x1, w1, qx_acc);
              qx_acc = fmaf(x2, w2, qx_acc);
              qx_acc = fmaf(x3, w3, qx_acc);
              qx_acc = fmaf(x4, w4, qx_acc);
              qx_acc = fmaf(x5, w5, qx_acc);
              qx_acc = fmaf(x6, w6, qx_acc);
              qx_acc = fmaf(x7, w7, qx_acc);
            }
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(
                  x_val,
                  dequantize_value<BITS, false>(quant_val, 1.0f, 0.0f),
                  qx_acc);
            }
          } else {
            for (int k_local = lane; k_start + k_local < k_end_g;
                 k_local += THREADS_PER_COL) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(
                  x_val,
                  dequantize_value<BITS, false>(quant_val, 1.0f, 0.0f),
                  qx_acc);
            }
          }
          acc += scale * qx_acc;
        }
      }
    }
    __syncthreads();
  }

  acc = subgroup_reduce_sum_qmm<THREADS_PER_COL>(acc);
  if (valid && lane == 0) {
    out_batch_ptr[static_cast<int64_t>(row) * N + col] = static_cast<T>(acc);
  }
}

template <
    typename T,
    typename ScaleT,
    int BITS,
    int GROUP_SIZE,
    bool AFFINE,
    int THREADS_PER_COL>
__global__ void qmv_warp_noshared_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ out,
    int M,
    int N,
    int K,
    bool has_bias) {
  const int lane = threadIdx.x;
  const int col = blockIdx.x * blockDim.y + threadIdx.y;
  const int row = blockIdx.y;

  const bool row_valid = (row < M);
  const bool valid = row_valid && (col < N);

  constexpr int kThreadsPerCol = THREADS_PER_COL;
  const int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const int row_bytes = (K * BITS + 7) / 8;

  const T* x_row = row_valid ? (x + row * K) : nullptr;
  const uint8_t* w_row = valid ? (w + col * row_bytes) : nullptr;
  const ScaleT* scales_row = valid ? (scales + col * num_groups) : nullptr;
  const ScaleT* biases_row =
      (valid && has_bias) ? (biases + col * num_groups) : nullptr;

  float acc = 0.0f;

  for (int g = 0; g < num_groups; ++g) {
    int k_start = g * GROUP_SIZE;
    int k_end = min(k_start + GROUP_SIZE, K);

    if (valid) {
      float scale = load_scale_value<ScaleT, GROUP_SIZE, AFFINE>(scales_row[g]);
      float bias = has_bias ? static_cast<float>(biases_row[g]) : 0.0f;

      if constexpr (AFFINE) {
        float qx_acc0 = 0.0f, qx_acc1 = 0.0f, qx_acc2 = 0.0f, qx_acc3 = 0.0f;
        float x_group_sum = 0.0f;
        float qx_acc = 0.0f;

        if constexpr (BITS == 8) {
          int k_local = lane * 4;
          int step = kThreadsPerCol * 4;
          for (; k_start + k_local + 3 < k_end; k_local += step) {
            int k = k_start + k_local;

            // Read 4 weights at once
            uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_row[k]);
            float w0 = static_cast<float>(w_packed & 0xFF);
            float w1 = static_cast<float>((w_packed >> 8) & 0xFF);
            float w2 = static_cast<float>((w_packed >> 16) & 0xFF);
            float w3 = static_cast<float>((w_packed >> 24) & 0xFF);

            float x0 = static_cast<float>(x_row[k]);
            float x1 = static_cast<float>(x_row[k + 1]);
            float x2 = static_cast<float>(x_row[k + 2]);
            float x3 = static_cast<float>(x_row[k + 3]);

            qx_acc0 = fmaf(x0, w0, qx_acc0);
            qx_acc1 = fmaf(x1, w1, qx_acc1);
            qx_acc2 = fmaf(x2, w2, qx_acc2);
            qx_acc3 = fmaf(x3, w3, qx_acc3);

            if (has_bias) {
              x_group_sum += x0 + x1 + x2 + x3;
            }
          }

          qx_acc = qx_acc0 + qx_acc1 + qx_acc2 + qx_acc3;

          // Tail loop
          for (; k_start + k_local < k_end; k_local++) {
            int k = k_start + k_local;
            float x_val = static_cast<float>(x_row[k]);
            float w_val = static_cast<float>(w_row[k]);
            qx_acc = fmaf(x_val, w_val, qx_acc);
            if (has_bias)
              x_group_sum += x_val;
          }
        } else {
          for (int k_local = lane; k_start + k_local < k_end;
               k_local += kThreadsPerCol) {
            int k = k_start + k_local;
            float x_val = static_cast<float>(x_row[k]);
            uint8_t quant_val =
                unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
            qx_acc = fmaf(x_val, static_cast<float>(quant_val), qx_acc);
            if (has_bias)
              x_group_sum += x_val;
          }
        }

        float group_acc = scale * qx_acc;
        if (has_bias) {
          group_acc = fmaf(bias, x_group_sum, group_acc);
        }
        acc += group_acc;
      } else {
        float qx_acc0 = 0.0f, qx_acc1 = 0.0f, qx_acc2 = 0.0f, qx_acc3 = 0.0f;
        float qx_acc = 0.0f;

        if constexpr (BITS == 8) {
          int k_local = lane * 4;
          int step = kThreadsPerCol * 4;
          for (; k_start + k_local + 3 < k_end; k_local += step) {
            int k = k_start + k_local;

            // Read 4 weights at once
            uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_row[k]);
            float w0 = fp8_e4m3_to_float(w_packed & 0xFF);
            float w1 = fp8_e4m3_to_float((w_packed >> 8) & 0xFF);
            float w2 = fp8_e4m3_to_float((w_packed >> 16) & 0xFF);
            float w3 = fp8_e4m3_to_float((w_packed >> 24) & 0xFF);

            float x0 = static_cast<float>(x_row[k]);
            float x1 = static_cast<float>(x_row[k + 1]);
            float x2 = static_cast<float>(x_row[k + 2]);
            float x3 = static_cast<float>(x_row[k + 3]);

            qx_acc0 = fmaf(x0, w0, qx_acc0);
            qx_acc1 = fmaf(x1, w1, qx_acc1);
            qx_acc2 = fmaf(x2, w2, qx_acc2);
            qx_acc3 = fmaf(x3, w3, qx_acc3);
          }

          qx_acc = qx_acc0 + qx_acc1 + qx_acc2 + qx_acc3;

          for (; k_start + k_local < k_end; k_local++) {
            int k = k_start + k_local;
            float x_val = static_cast<float>(x_row[k]);
            float w_val = fp8_e4m3_to_float(w_row[k]);
            qx_acc = fmaf(x_val, w_val, qx_acc);
          }
          acc += scale * qx_acc;
        } else {
          float qx_acc = 0.0f;
          for (int k_local = lane; k_start + k_local < k_end;
               k_local += kThreadsPerCol) {
            int k = k_start + k_local;
            float x_val = static_cast<float>(x_row[k]);
            uint8_t quant_val =
                unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
            qx_acc = fmaf(
                x_val,
                dequantize_value<BITS, false>(quant_val, 1.0f, 0.0f),
                qx_acc);
          }
          acc += scale * qx_acc;
        }
      }
    }
  }

  acc = subgroup_reduce_sum_qmm<kThreadsPerCol>(acc);
  if (valid && lane == 0) {
    out[row * N + col] = static_cast<T>(acc);
  }
}

template <typename T, typename ScaleT, int BITS, int GROUP_SIZE, bool AFFINE>
__global__ void qmv_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ out,
    int M,
    int N,
    int K,
    bool has_bias) {
  const int row = blockIdx.x;
  const int col = blockIdx.y * blockDim.x + threadIdx.x;

  if (row >= M || col >= N)
    return;

  float acc = 0.0f;
  int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const int row_bytes = (K * BITS + 7) / 8;
  const uint8_t* w_row = w + col * row_bytes;

  for (int g = 0; g < num_groups; ++g) {
    float scale = load_scale_value<ScaleT, GROUP_SIZE, AFFINE>(
        scales[col * num_groups + g]);
    float bias =
        has_bias ? static_cast<float>(biases[col * num_groups + g]) : 0.0f;

    int k_start = g * GROUP_SIZE;
    int k_end = min(k_start + GROUP_SIZE, K);

    float qx_acc = 0.0f;
    if constexpr (BITS == 8) {
      int k = k_start;
      for (; k + 3 < k_end; k += 4) {
        uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_row[k]);
        float w0 = dequantize_value<8, AFFINE>(w_packed & 0xFF, scale, bias);
        float w1 =
            dequantize_value<8, AFFINE>((w_packed >> 8) & 0xFF, scale, bias);
        float w2 =
            dequantize_value<8, AFFINE>((w_packed >> 16) & 0xFF, scale, bias);
        float w3 =
            dequantize_value<8, AFFINE>((w_packed >> 24) & 0xFF, scale, bias);

        qx_acc += static_cast<float>(x[row * K + k]) * w0;
        qx_acc += static_cast<float>(x[row * K + k + 1]) * w1;
        qx_acc += static_cast<float>(x[row * K + k + 2]) * w2;
        qx_acc += static_cast<float>(x[row * K + k + 3]) * w3;
      }
      for (; k < k_end; ++k) {
        float w_val = dequantize_value<8, AFFINE>(w_row[k], scale, bias);
        qx_acc += static_cast<float>(x[row * K + k]) * w_val;
      }
    } else if constexpr (BITS == 6) {
      int k = k_start;
      int max_safe_k = ((row_bytes - 7) * 8) / 6;
      for (; k + 7 < k_end && k < max_safe_k; k += 8) {
        int byte_idx = (k * 6) / 8;
        int bit_offset = (k * 6) % 8;
        uint64_t w_packed = 0;
        memcpy(&w_packed, &w_row[byte_idx], 8);
        w_packed >>= bit_offset;

        float w0 = dequantize_value<6, AFFINE>(w_packed & 0x3F, scale, bias);
        float w1 =
            dequantize_value<6, AFFINE>((w_packed >> 6) & 0x3F, scale, bias);
        float w2 =
            dequantize_value<6, AFFINE>((w_packed >> 12) & 0x3F, scale, bias);
        float w3 =
            dequantize_value<6, AFFINE>((w_packed >> 18) & 0x3F, scale, bias);
        float w4 =
            dequantize_value<6, AFFINE>((w_packed >> 24) & 0x3F, scale, bias);
        float w5 =
            dequantize_value<6, AFFINE>((w_packed >> 30) & 0x3F, scale, bias);
        float w6 =
            dequantize_value<6, AFFINE>((w_packed >> 36) & 0x3F, scale, bias);
        float w7 =
            dequantize_value<6, AFFINE>((w_packed >> 42) & 0x3F, scale, bias);

        qx_acc += static_cast<float>(x[row * K + k]) * w0;
        qx_acc += static_cast<float>(x[row * K + k + 1]) * w1;
        qx_acc += static_cast<float>(x[row * K + k + 2]) * w2;
        qx_acc += static_cast<float>(x[row * K + k + 3]) * w3;
        qx_acc += static_cast<float>(x[row * K + k + 4]) * w4;
        qx_acc += static_cast<float>(x[row * K + k + 5]) * w5;
        qx_acc += static_cast<float>(x[row * K + k + 6]) * w6;
        qx_acc += static_cast<float>(x[row * K + k + 7]) * w7;
      }
      for (; k < k_end; ++k) {
        uint8_t quant_val = unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
        float w_val = dequantize_value<6, AFFINE>(quant_val, scale, bias);
        qx_acc += static_cast<float>(x[row * K + k]) * w_val;
      }
    } else {
      for (int k = k_start; k < k_end; ++k) {
        uint8_t quant_val = unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
        float w_val = dequantize_value<BITS, AFFINE>(quant_val, scale, bias);
        qx_acc += static_cast<float>(x[row * K + k]) * w_val;
      }
    }
    acc += qx_acc;
  }

  out[row * N + col] = static_cast<T>(acc);
}

template <typename T, typename ScaleT, int BITS, int GROUP_SIZE, bool AFFINE>
__global__ void qmv_t_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ out,
    int M,
    int N,
    int K,
    bool has_bias) {
  const int row = blockIdx.x;
  const int col = blockIdx.y * blockDim.x + threadIdx.x;

  if (row >= M || col >= N)
    return;

  float acc = 0.0f;
  int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const int row_bytes = (K * BITS + 7) / 8;
  const uint8_t* w_row = w + col * row_bytes;

  for (int g = 0; g < num_groups; ++g) {
    float scale = load_scale_value<ScaleT, GROUP_SIZE, AFFINE>(
        scales[col * num_groups + g]);
    float bias =
        has_bias ? static_cast<float>(biases[col * num_groups + g]) : 0.0f;

    int k_start = g * GROUP_SIZE;
    int k_end = min(k_start + GROUP_SIZE, K);

    float qx_acc = 0.0f;
    if constexpr (BITS == 8) {
      int k = k_start;
      for (; k + 3 < k_end; k += 4) {
        uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_row[k]);
        float w0 = dequantize_value<8, AFFINE>(w_packed & 0xFF, scale, bias);
        float w1 =
            dequantize_value<8, AFFINE>((w_packed >> 8) & 0xFF, scale, bias);
        float w2 =
            dequantize_value<8, AFFINE>((w_packed >> 16) & 0xFF, scale, bias);
        float w3 =
            dequantize_value<8, AFFINE>((w_packed >> 24) & 0xFF, scale, bias);

        qx_acc += static_cast<float>(x[row * K + k]) * w0;
        qx_acc += static_cast<float>(x[row * K + k + 1]) * w1;
        qx_acc += static_cast<float>(x[row * K + k + 2]) * w2;
        qx_acc += static_cast<float>(x[row * K + k + 3]) * w3;
      }
      for (; k < k_end; ++k) {
        float w_val = dequantize_value<8, AFFINE>(w_row[k], scale, bias);
        qx_acc += static_cast<float>(x[row * K + k]) * w_val;
      }
    } else if constexpr (BITS == 6) {
      int k = k_start;
      int max_safe_k = ((row_bytes - 7) * 8) / 6;
      for (; k + 7 < k_end && k < max_safe_k; k += 8) {
        int byte_idx = (k * 6) / 8;
        int bit_offset = (k * 6) % 8;
        uint64_t w_packed = 0;
        memcpy(&w_packed, &w_row[byte_idx], 8);
        w_packed >>= bit_offset;

        float w0 = dequantize_value<6, AFFINE>(w_packed & 0x3F, scale, bias);
        float w1 =
            dequantize_value<6, AFFINE>((w_packed >> 6) & 0x3F, scale, bias);
        float w2 =
            dequantize_value<6, AFFINE>((w_packed >> 12) & 0x3F, scale, bias);
        float w3 =
            dequantize_value<6, AFFINE>((w_packed >> 18) & 0x3F, scale, bias);
        float w4 =
            dequantize_value<6, AFFINE>((w_packed >> 24) & 0x3F, scale, bias);
        float w5 =
            dequantize_value<6, AFFINE>((w_packed >> 30) & 0x3F, scale, bias);
        float w6 =
            dequantize_value<6, AFFINE>((w_packed >> 36) & 0x3F, scale, bias);
        float w7 =
            dequantize_value<6, AFFINE>((w_packed >> 42) & 0x3F, scale, bias);

        qx_acc += static_cast<float>(x[row * K + k]) * w0;
        qx_acc += static_cast<float>(x[row * K + k + 1]) * w1;
        qx_acc += static_cast<float>(x[row * K + k + 2]) * w2;
        qx_acc += static_cast<float>(x[row * K + k + 3]) * w3;
        qx_acc += static_cast<float>(x[row * K + k + 4]) * w4;
        qx_acc += static_cast<float>(x[row * K + k + 5]) * w5;
        qx_acc += static_cast<float>(x[row * K + k + 6]) * w6;
        qx_acc += static_cast<float>(x[row * K + k + 7]) * w7;
      }
      for (; k < k_end; ++k) {
        uint8_t quant_val = unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
        float w_val = dequantize_value<6, AFFINE>(quant_val, scale, bias);
        qx_acc += static_cast<float>(x[row * K + k]) * w_val;
      }
    } else {
      for (int k = k_start; k < k_end; ++k) {
        uint8_t quant_val = unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
        float w_val = dequantize_value<BITS, AFFINE>(quant_val, scale, bias);
        qx_acc += static_cast<float>(x[row * K + k]) * w_val;
      }
    }
    acc += qx_acc;
  }

  out[row * N + col] = static_cast<T>(acc);
}

} // namespace rocm

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = rocm::device(s.device);
  auto& enc = d.get_command_encoder(s);

  out.set_data(allocator::malloc(out.nbytes()));

  array x = ensure_row_contiguous_matrix(inputs[0], enc, s);
  array w = ensure_row_contiguous_matrix(inputs[1], enc, s);
  array scales = ensure_row_contiguous_matrix(inputs[2], enc, s);
  std::optional<array> biases = std::nullopt;
  bool has_bias = (mode_ == QuantizationMode::Affine) && (inputs.size() == 4);
  if (has_bias) {
    biases = ensure_row_contiguous_matrix(inputs[3], enc, s);
  }

  enc.set_input_array(x);
  enc.set_input_array(w);
  enc.set_input_array(scales);
  if (has_bias)
    enc.set_input_array(biases.value());
  enc.set_output_array(out);

  int K = x.shape(-1);
  int M = out.shape(-2);
  int N = out.shape(-1);

  int64_t matrix_size = static_cast<int64_t>(M) * N;
  int batch_count = static_cast<int>(out.size() / matrix_size);
  int x_batch_count = static_cast<int>(
      x.size() /
      (static_cast<int64_t>(x.shape(-2)) * static_cast<int64_t>(x.shape(-1))));
  int w_batch_count = static_cast<int>(
      w.size() /
      (static_cast<int64_t>(w.shape(-2)) * static_cast<int64_t>(w.shape(-1))));

  bool x_singleton_batch = has_only_singleton_batch_dims(x);
  bool w_singleton_batch = has_only_singleton_batch_dims(w);
  bool non_batched =
      (batch_count == 1) && x_singleton_batch && w_singleton_batch;

  bool bits_supported_by_qmv = (bits_ == 2 || bits_ == 4 || bits_ == 8) ||
      (mode_ == QuantizationMode::Affine && (bits_ == 5 || bits_ == 6));
  bool valid_x_batch = (x_batch_count == 1) || (x_batch_count == batch_count);
  bool valid_w_batch = (w_batch_count == 1) || (w_batch_count == batch_count);
  bool can_use_batched_qmv = transpose_ && bits_supported_by_qmv &&
      (batch_count > 1) && valid_x_batch && valid_w_batch;
  bool force_dequant_gemm = !transpose_ || !bits_supported_by_qmv ||
      ((batch_count > 1) && !can_use_batched_qmv) ||
      (w.ndim() > 2 && !w_singleton_batch && !can_use_batched_qmv);
  bool dequant_gemm_supported_mode = (mode_ == QuantizationMode::Affine);
  bool should_prefer_dequant = should_use_dequant_gemm_path(
      M, N, K, batch_count, non_batched, transpose_, can_use_batched_qmv, d);

  // Dequant + rocBLAS GEMM path
  // Disable with MLX_ROCM_QMM_DEQUANT_GEMM=0 if needed
  if (dequant_gemm_supported_mode && d.is_rocblas_available() &&
      use_rocblas_dequant_path() &&
      (force_dequant_gemm || should_prefer_dequant)) {
    if (!((x_batch_count == 1) || (x_batch_count == batch_count))) {
      throw std::runtime_error(
          "Unsupported x batch shape for dequant GEMM fallback");
    }
    if (!((w_batch_count == 1) || (w_batch_count == batch_count))) {
      throw std::runtime_error(
          "Unsupported w batch shape for dequant GEMM fallback");
    }

    int dequant_rows = transpose_ ? N : K;
    int dequant_cols = transpose_ ? K : N;

    Shape w_dequant_shape = w.shape();
    w_dequant_shape[w_dequant_shape.size() - 2] = dequant_rows;
    w_dequant_shape[w_dequant_shape.size() - 1] = dequant_cols;

    array w_dequant(w_dequant_shape, x.dtype(), nullptr, {});
    bool cache_hit = false;
    int cache_cap = dequant_cache_capacity();
    size_t cache_max_bytes = dequant_cache_max_bytes();
    if (cache_cap > 0 && cache_max_bytes > 0) {
      static std::mutex cache_mutex;
      static std::list<DequantCacheKey> lru;
      static size_t cached_bytes = 0;
      static std::unordered_map<
          DequantCacheKey,
          DequantCacheEntry,
          DequantCacheKeyHasher>
          cache;

      DequantCacheKey key{
          reinterpret_cast<std::uintptr_t>(gpu_ptr<void>(w)),
          reinterpret_cast<std::uintptr_t>(gpu_ptr<void>(scales)),
          has_bias ? reinterpret_cast<std::uintptr_t>(gpu_ptr<void>(*biases))
                   : 0,
          group_size_,
          bits_,
          s.index,
          transpose_,
          x.dtype()};

      {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = cache.find(key);
        if (it != cache.end() && it->second.weight.shape() == w_dequant_shape) {
          lru.splice(lru.begin(), lru, it->second.lru_it);
          w_dequant = it->second.weight;
          cache_hit = true;
        }
      }

      if (!cache_hit) {
        w_dequant.set_data(allocator::malloc(w_dequant.nbytes()));

        if (mode_ == QuantizationMode::Affine) {
          affine_dequantize(
              w, scales, biases, w_dequant, group_size_, bits_, enc, s);
        } else {
          fp_dequantize(w, scales, w_dequant, group_size_, bits_, enc, s);
        }

        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = cache.find(key);
        if (it == cache.end()) {
          size_t entry_bytes = w_dequant.nbytes();
          if (entry_bytes <= cache_max_bytes) {
            lru.push_front(key);
            cache.emplace(
                key,
                DequantCacheEntry{
                    w_dequant,
                    w,
                    scales,
                    has_bias ? std::optional<array>(*biases) : std::nullopt,
                    entry_bytes,
                    lru.begin()});
            cached_bytes += entry_bytes;

            while (static_cast<int>(cache.size()) > cache_cap ||
                   cached_bytes > cache_max_bytes) {
              auto evict = lru.back();
              auto evict_it = cache.find(evict);
              if (evict_it != cache.end()) {
                cached_bytes -= evict_it->second.bytes;
                cache.erase(evict_it);
              }
              lru.pop_back();
            }
          }
        } else {
          size_t entry_bytes = w_dequant.nbytes();
          if (entry_bytes > cache_max_bytes) {
            cached_bytes -= it->second.bytes;
            lru.erase(it->second.lru_it);
            cache.erase(it);
          } else {
            cached_bytes -= it->second.bytes;
            it->second.w_source = w;
            it->second.scales_source = scales;
            it->second.biases_source =
                has_bias ? std::optional<array>(*biases) : std::nullopt;
            it->second.weight = w_dequant;
            it->second.bytes = entry_bytes;
            cached_bytes += it->second.bytes;
            lru.splice(lru.begin(), lru, it->second.lru_it);

            while (static_cast<int>(cache.size()) > cache_cap ||
                   cached_bytes > cache_max_bytes) {
              auto evict = lru.back();
              auto evict_it = cache.find(evict);
              if (evict_it != cache.end()) {
                cached_bytes -= evict_it->second.bytes;
                cache.erase(evict_it);
              }
              lru.pop_back();
            }
          }
        }
      }
    } else {
      w_dequant.set_data(allocator::malloc(w_dequant.nbytes()));

      if (mode_ == QuantizationMode::Affine) {
        affine_dequantize(
            w, scales, biases, w_dequant, group_size_, bits_, enc, s);
      } else {
        fp_dequantize(w, scales, w_dequant, group_size_, bits_, enc, s);
      }
    }

    if (!cache_hit) {
      enc.add_temporary(w_dequant);
    }

    int lda = K;
    int ldb = transpose_ ? K : N;

    if (batch_count == 1 && x_batch_count == 1 && w_batch_count == 1) {
      dequant_rocblas_gemm(
          enc,
          false,
          transpose_,
          M,
          N,
          K,
          1.0f,
          x,
          lda,
          w_dequant,
          ldb,
          0.0f,
          out,
          N,
          x.dtype());
    } else {
      int64_t stride_a =
          (x_batch_count == 1) ? 0 : static_cast<int64_t>(x.shape(-2)) * K;
      int64_t stride_b = (w_batch_count == 1)
          ? 0
          : static_cast<int64_t>(dequant_rows) * dequant_cols;
      int64_t stride_c = static_cast<int64_t>(M) * N;

      dequant_rocblas_gemm_batched(
          enc,
          false,
          transpose_,
          M,
          N,
          K,
          1.0f,
          x,
          lda,
          stride_a,
          w_dequant,
          ldb,
          stride_b,
          0.0f,
          out,
          N,
          stride_c,
          batch_count,
          x.dtype());
    }
    return;
  }

  bool use_fast_qmv = transpose_ && (non_batched || can_use_batched_qmv);
  use_fast_qmv = parse_warp_kernel_env("MLX_ROCM_QMV_USE_WARP", use_fast_qmv);
  if (can_use_batched_qmv) {
    use_fast_qmv = true;
  }
  bool use_tiny_k_qmv = should_use_tiny_k_qmv_path(
      M, N, K, batch_count, transpose_, can_use_batched_qmv, bits_, mode_);

  int block_size = 256;
  dim3 grid(M, (N + block_size - 1) / block_size);

  int fast_threads_per_col =
      select_qmv_threads_per_col(K, N, bits_, batch_count);
  int fast_threads_env =
      parse_threads_per_col_env("MLX_ROCM_QMV_THREADS_PER_COL");
  if (fast_threads_env > 0)
    fast_threads_per_col = fast_threads_env;

  int fast_cols_per_block = select_qmv_cols_per_block(K, N, bits_);
  if (use_tiny_k_qmv) {
    fast_cols_per_block = std::max(fast_cols_per_block, 32);
  }
  if (fast_threads_per_col == 16 && bits_ == 8 && N >= 2048) {
    fast_cols_per_block = std::max(fast_cols_per_block, 64);
  }
  int max_cols_per_block = rocm::kMaxThreadsPerBlock / fast_threads_per_col;
  while (fast_cols_per_block > max_cols_per_block)
    fast_cols_per_block /= 2;
  while (fast_cols_per_block > 1 && (N % fast_cols_per_block) != 0 &&
         fast_cols_per_block > 8) {
    fast_cols_per_block /= 2;
  }

  dim3 fast_block(fast_threads_per_col, fast_cols_per_block);
  dim3 fast_grid((N + fast_cols_per_block - 1) / fast_cols_per_block, M);
  dim3 fast_grid_batched(
      (N + fast_cols_per_block - 1) / fast_cols_per_block, M, batch_count);

  int64_t x_matrix_stride =
      static_cast<int64_t>(x.shape(-2)) * static_cast<int64_t>(x.shape(-1));
  int64_t w_matrix_stride = static_cast<int64_t>(w.shape(-2)) *
      static_cast<int64_t>(w.shape(-1)) *
      static_cast<int64_t>(size_of(w.dtype()));
  int num_groups = (K + group_size_ - 1) / group_size_;
  int64_t sb_matrix_stride =
      static_cast<int64_t>(w.shape(-2)) * static_cast<int64_t>(num_groups);
  int64_t out_matrix_stride = static_cast<int64_t>(M) * N;

  int64_t x_batch_stride = (x_batch_count == 1) ? 0 : x_matrix_stride;
  int64_t w_batch_stride = (w_batch_count == 1) ? 0 : w_matrix_stride;
  int64_t sb_batch_stride = (w_batch_count == 1) ? 0 : sb_matrix_stride;

  const void* x_ptr = gpu_ptr<void>(x);
  const uint8_t* w_ptr = gpu_ptr<const uint8_t>(w);
  const void* scales_ptr = gpu_ptr<void>(scales);
  const void* biases_ptr = has_bias ? gpu_ptr<void>(*biases) : nullptr;
  void* out_ptr = gpu_ptr<void>(out);

  // The noshared variant reads x from global memory redundantly per warp.
  // The shared variant caches x in LDS and is ~15x faster for decode shapes.
  // Always prefer shared unless K is tiny (where LDS overhead isn't worth it).
  bool use_noshared_qmv_variant = use_tiny_k_qmv;

  // L2-optimized tiled QMV with arch-tuned TILE_N.
  // TILE_N is passed as a runtime argument — no template instantiation needed.
  auto hw_info = detect_rocm_hw_info(enc.device());
  auto arch_tuning = rocm::get_arch_tuning(hw_info);
  int tile_n = arch_tuning.qmv_tile_n;
  // Allow env override for benchmarking
  if (auto env = std::getenv("MLX_ROCM_QMV_TILE_N"); env && *env)
    tile_n = std::atoi(env);
  // Ensure N alignment
  while (tile_n > 1 && N % tile_n != 0) tile_n /= 2;

  static bool use_tiled = (std::getenv("MLX_ROCM_QMV_NO_TILED") == nullptr);
  // The tiled QMV kernel (qdequant.hpp pack_factor_u32 = 32/BITS) only packs
  // correctly for power-of-two widths and is only instantiated for 4/8-bit;
  // other widths would match nothing here and leave `out` uninitialized.
  // Restrict to 4/8-bit; 2/5/6-bit fall through to the warp-shared QMV kernel.
  bool tiled_bits_supported = (bits_ == 4 || bits_ == 8);
  if (use_tiled && tiled_bits_supported && use_fast_qmv &&
      !can_use_batched_qmv && tile_n >= 8 &&
      mode_ == QuantizationMode::Affine) {
    enc.launch_kernel([&, x_ptr, w_ptr, scales_ptr, biases_ptr, out_ptr, tile_n](hipStream_t stream) {
      dim3 tiled_block(WARP_SIZE, tile_n);
      const int n_tiles = (N + tile_n - 1) / tile_n;
      // Persistent grid: CU-bounded block count, kernel grid-strides the rest.
      int blocks_per_cu = (hw_info.max_threads_per_cu > 0)
          ? (hw_info.max_threads_per_cu / (tile_n * WARP_SIZE)) : 4;
      if (blocks_per_cu < 1) blocks_per_cu = 1;
      int persistent_y =
          (hw_info.num_cus > 0) ? hw_info.num_cus * blocks_per_cu : n_tiles;
      int grid_y = (n_tiles < persistent_y) ? n_tiles : persistent_y;
      if (grid_y < 1) grid_y = 1;
      dim3 tiled_grid(M, grid_y);

      #define LAUNCH_TILED(T, ScaleT, BITS_V, GS_V) \
        hipLaunchKernelGGL( \
            (rocm::qmv_tiled_kernel<T, ScaleT, BITS_V, GS_V>), \
            tiled_grid, tiled_block, 0, stream, \
            (const T*)x_ptr, (const uint32_t*)w_ptr, \
            (const ScaleT*)scales_ptr, (const ScaleT*)biases_ptr, \
            (T*)out_ptr, M, N, K, has_bias, tile_n, n_tiles)

      if (x.dtype() == bfloat16) {
        if (bits_ == 4) {
          if (group_size_ == 32)  { LAUNCH_TILED(hip_bfloat16, hip_bfloat16, 4, 32); }
          else if (group_size_ == 64)  { LAUNCH_TILED(hip_bfloat16, hip_bfloat16, 4, 64); }
          else if (group_size_ == 128) { LAUNCH_TILED(hip_bfloat16, hip_bfloat16, 4, 128); }
        } else if (bits_ == 8) {
          if (group_size_ == 32)  { LAUNCH_TILED(hip_bfloat16, hip_bfloat16, 8, 32); }
          else if (group_size_ == 64)  { LAUNCH_TILED(hip_bfloat16, hip_bfloat16, 8, 64); }
          else if (group_size_ == 128) { LAUNCH_TILED(hip_bfloat16, hip_bfloat16, 8, 128); }
        }
      } else if (x.dtype() == float16) {
        if (bits_ == 4) {
          if (group_size_ == 32)  { LAUNCH_TILED(__half, __half, 4, 32); }
          else if (group_size_ == 64)  { LAUNCH_TILED(__half, __half, 4, 64); }
          else if (group_size_ == 128) { LAUNCH_TILED(__half, __half, 4, 128); }
        } else if (bits_ == 8) {
          if (group_size_ == 32)  { LAUNCH_TILED(__half, __half, 8, 32); }
          else if (group_size_ == 64)  { LAUNCH_TILED(__half, __half, 8, 64); }
          else if (group_size_ == 128) { LAUNCH_TILED(__half, __half, 8, 128); }
        }
      }
      #undef LAUNCH_TILED
    });
    return;
  }

  // The noshared path used to increase cols_per_block for aligned data.
  // Since we always use the shared variant now, no special grid adjustment needed.

  enc.launch_kernel([&,
                     x_ptr,
                     w_ptr,
                     scales_ptr,
                     biases_ptr,
                     out_ptr,
                     fast_threads_per_col,
                     use_noshared_qmv_variant,
                     x_batch_stride,
                     w_batch_stride,
                     sb_batch_stride,
                     out_matrix_stride](hipStream_t stream) {
    auto launch_qmv =
        [&](auto type_tag, auto scale_tag, auto bits_tag, auto gs_tag) {
          using T = typename decltype(type_tag)::type;
          using ScaleT = typename decltype(scale_tag)::type;
          constexpr int BITS = bits_tag.value;
          constexpr int GROUP_SIZE = gs_tag.value;

          if (mode_ == QuantizationMode::Affine) {
            if (use_fast_qmv) {
              if (can_use_batched_qmv) {
                if (fast_threads_per_col == 16) {
                  hipLaunchKernelGGL(
                      (rocm::qmv_warp_shared_batched_kernel<
                          T,
                          ScaleT,
                          BITS,
                          GROUP_SIZE,
                          true,
                          16>),
                      fast_grid_batched,
                      fast_block,
                      0,
                      stream,
                      (const T*)x_ptr,
                      w_ptr,
                      (const ScaleT*)scales_ptr,
                      (const ScaleT*)biases_ptr,
                      (T*)out_ptr,
                      M,
                      N,
                      K,
                      x_batch_stride,
                      w_batch_stride,
                      sb_batch_stride,
                      out_matrix_stride,
                      has_bias);
                } else {
                  hipLaunchKernelGGL(
                      (rocm::qmv_warp_shared_batched_kernel<
                          T,
                          ScaleT,
                          BITS,
                          GROUP_SIZE,
                          true,
                          WARP_SIZE>),
                      fast_grid_batched,
                      fast_block,
                      0,
                      stream,
                      (const T*)x_ptr,
                      w_ptr,
                      (const ScaleT*)scales_ptr,
                      (const ScaleT*)biases_ptr,
                      (T*)out_ptr,
                      M,
                      N,
                      K,
                      x_batch_stride,
                      w_batch_stride,
                      sb_batch_stride,
                      out_matrix_stride,
                      has_bias);
                }
              } else {
                if (use_noshared_qmv_variant) {
                  if (fast_threads_per_col == 16) {
                    hipLaunchKernelGGL(
                        (rocm::qmv_warp_noshared_kernel<
                            T,
                            ScaleT,
                            BITS,
                            GROUP_SIZE,
                            true,
                            16>),
                        fast_grid,
                        fast_block,
                        0,
                        stream,
                        (const T*)x_ptr,
                        w_ptr,
                        (const ScaleT*)scales_ptr,
                        (const ScaleT*)biases_ptr,
                        (T*)out_ptr,
                        M,
                        N,
                        K,
                        has_bias);
                  } else {
                    hipLaunchKernelGGL(
                        (rocm::qmv_warp_noshared_kernel<
                            T,
                            ScaleT,
                            BITS,
                            GROUP_SIZE,
                            true,
                            WARP_SIZE>),
                        fast_grid,
                        fast_block,
                        0,
                        stream,
                        (const T*)x_ptr,
                        w_ptr,
                        (const ScaleT*)scales_ptr,
                        (const ScaleT*)biases_ptr,
                        (T*)out_ptr,
                        M,
                        N,
                        K,
                        has_bias);
                  }
                } else {
                  if (fast_threads_per_col == 16) {
                    hipLaunchKernelGGL(
                        (rocm::qmv_warp_shared_kernel<
                            T,
                            ScaleT,
                            BITS,
                            GROUP_SIZE,
                            true,
                            16>),
                        fast_grid,
                        fast_block,
                        0,
                        stream,
                        (const T*)x_ptr,
                        w_ptr,
                        (const ScaleT*)scales_ptr,
                        (const ScaleT*)biases_ptr,
                        (T*)out_ptr,
                        M,
                        N,
                        K,
                        has_bias);
                  } else {
                    hipLaunchKernelGGL(
                        (rocm::qmv_warp_shared_kernel<
                            T,
                            ScaleT,
                            BITS,
                            GROUP_SIZE,
                            true,
                            WARP_SIZE>),
                        fast_grid,
                        fast_block,
                        0,
                        stream,
                        (const T*)x_ptr,
                        w_ptr,
                        (const ScaleT*)scales_ptr,
                        (const ScaleT*)biases_ptr,
                        (T*)out_ptr,
                        M,
                        N,
                        K,
                        has_bias);
                  }
                }
              }
            } else if (transpose_) {
              hipLaunchKernelGGL(
                  (rocm::qmv_t_kernel<T, ScaleT, BITS, GROUP_SIZE, true>),
                  grid,
                  dim3(block_size),
                  0,
                  stream,
                  (const T*)x_ptr,
                  w_ptr,
                  (const ScaleT*)scales_ptr,
                  (const ScaleT*)biases_ptr,
                  (T*)out_ptr,
                  M,
                  N,
                  K,
                  has_bias);
            } else {
              hipLaunchKernelGGL(
                  (rocm::qmv_kernel<T, ScaleT, BITS, GROUP_SIZE, true>),
                  grid,
                  dim3(block_size),
                  0,
                  stream,
                  (const T*)x_ptr,
                  w_ptr,
                  (const ScaleT*)scales_ptr,
                  (const ScaleT*)biases_ptr,
                  (T*)out_ptr,
                  M,
                  N,
                  K,
                  has_bias);
            }
          } else {
            if (use_fast_qmv) {
              if (can_use_batched_qmv) {
                if (fast_threads_per_col == 16) {
                  hipLaunchKernelGGL(
                      (rocm::qmv_warp_shared_batched_kernel<
                          T,
                          ScaleT,
                          BITS,
                          GROUP_SIZE,
                          false,
                          16>),
                      fast_grid_batched,
                      fast_block,
                      0,
                      stream,
                      (const T*)x_ptr,
                      w_ptr,
                      (const ScaleT*)scales_ptr,
                      (const ScaleT*)biases_ptr,
                      (T*)out_ptr,
                      M,
                      N,
                      K,
                      x_batch_stride,
                      w_batch_stride,
                      sb_batch_stride,
                      out_matrix_stride,
                      has_bias);
                } else {
                  hipLaunchKernelGGL(
                      (rocm::qmv_warp_shared_batched_kernel<
                          T,
                          ScaleT,
                          BITS,
                          GROUP_SIZE,
                          false,
                          WARP_SIZE>),
                      fast_grid_batched,
                      fast_block,
                      0,
                      stream,
                      (const T*)x_ptr,
                      w_ptr,
                      (const ScaleT*)scales_ptr,
                      (const ScaleT*)biases_ptr,
                      (T*)out_ptr,
                      M,
                      N,
                      K,
                      x_batch_stride,
                      w_batch_stride,
                      sb_batch_stride,
                      out_matrix_stride,
                      has_bias);
                }
              } else {
                if (use_noshared_qmv_variant) {
                  if (fast_threads_per_col == 16) {
                    hipLaunchKernelGGL(
                        (rocm::qmv_warp_noshared_kernel<
                            T,
                            ScaleT,
                            BITS,
                            GROUP_SIZE,
                            false,
                            16>),
                        fast_grid,
                        fast_block,
                        0,
                        stream,
                        (const T*)x_ptr,
                        w_ptr,
                        (const ScaleT*)scales_ptr,
                        (const ScaleT*)biases_ptr,
                        (T*)out_ptr,
                        M,
                        N,
                        K,
                        has_bias);
                  } else {
                    hipLaunchKernelGGL(
                        (rocm::qmv_warp_noshared_kernel<
                            T,
                            ScaleT,
                            BITS,
                            GROUP_SIZE,
                            false,
                            WARP_SIZE>),
                        fast_grid,
                        fast_block,
                        0,
                        stream,
                        (const T*)x_ptr,
                        w_ptr,
                        (const ScaleT*)scales_ptr,
                        (const ScaleT*)biases_ptr,
                        (T*)out_ptr,
                        M,
                        N,
                        K,
                        has_bias);
                  }
                } else {
                  if (fast_threads_per_col == 16) {
                    hipLaunchKernelGGL(
                        (rocm::qmv_warp_shared_kernel<
                            T,
                            ScaleT,
                            BITS,
                            GROUP_SIZE,
                            false,
                            16>),
                        fast_grid,
                        fast_block,
                        0,
                        stream,
                        (const T*)x_ptr,
                        w_ptr,
                        (const ScaleT*)scales_ptr,
                        (const ScaleT*)biases_ptr,
                        (T*)out_ptr,
                        M,
                        N,
                        K,
                        has_bias);
                  } else {
                    hipLaunchKernelGGL(
                        (rocm::qmv_warp_shared_kernel<
                            T,
                            ScaleT,
                            BITS,
                            GROUP_SIZE,
                            false,
                            WARP_SIZE>),
                        fast_grid,
                        fast_block,
                        0,
                        stream,
                        (const T*)x_ptr,
                        w_ptr,
                        (const ScaleT*)scales_ptr,
                        (const ScaleT*)biases_ptr,
                        (T*)out_ptr,
                        M,
                        N,
                        K,
                        has_bias);
                  }
                }
              }
            } else if (transpose_) {
              hipLaunchKernelGGL(
                  (rocm::qmv_t_kernel<T, ScaleT, BITS, GROUP_SIZE, false>),
                  grid,
                  dim3(block_size),
                  0,
                  stream,
                  (const T*)x_ptr,
                  w_ptr,
                  (const ScaleT*)scales_ptr,
                  (const ScaleT*)biases_ptr,
                  (T*)out_ptr,
                  M,
                  N,
                  K,
                  has_bias);
            } else {
              hipLaunchKernelGGL(
                  (rocm::qmv_kernel<T, ScaleT, BITS, GROUP_SIZE, false>),
                  grid,
                  dim3(block_size),
                  0,
                  stream,
                  (const T*)x_ptr,
                  w_ptr,
                  (const ScaleT*)scales_ptr,
                  (const ScaleT*)biases_ptr,
                  (T*)out_ptr,
                  M,
                  N,
                  K,
                  has_bias);
            }
          }
        };

    // Type aliases to avoid template angle brackets in macro args
    using float_id = local_type_identity<float>;
    using half_id = local_type_identity<__half>;
    using bf16_id = local_type_identity<hip_bfloat16>;
    using bits2 = std::integral_constant<int, 2>;
    using bits4 = std::integral_constant<int, 4>;
    using bits5 = std::integral_constant<int, 5>;
    using bits6 = std::integral_constant<int, 6>;
    using bits8 = std::integral_constant<int, 8>;
    using gs32 = std::integral_constant<int, 32>;
    using gs64 = std::integral_constant<int, 64>;
    using gs128 = std::integral_constant<int, 128>;

// Helper macro to dispatch group_size
#define DISPATCH_GROUP_SIZE(type_tag, scale_tag, bits_tag)   \
  do {                                                       \
    switch (group_size_) {                                   \
      case 32:                                               \
        launch_qmv(type_tag, scale_tag, bits_tag, gs32{});   \
        break;                                               \
      case 64:                                               \
        launch_qmv(type_tag, scale_tag, bits_tag, gs64{});   \
        break;                                               \
      case 128:                                              \
        launch_qmv(type_tag, scale_tag, bits_tag, gs128{});  \
        break;                                               \
      default:                                               \
        throw std::runtime_error(                            \
            "Unsupported group_size for QuantizedMatmul: " + \
            std::to_string(group_size_));                    \
    }                                                        \
  } while (0)

    if (x.dtype() == float32) {
      if (bits_ == 8)
        DISPATCH_GROUP_SIZE(float_id{}, float_id{}, bits8{});
      else if (bits_ == 5 && mode_ == QuantizationMode::Affine) {
        DISPATCH_GROUP_SIZE(float_id{}, float_id{}, bits5{});
      } else if (bits_ == 6 && mode_ == QuantizationMode::Affine) {
        DISPATCH_GROUP_SIZE(float_id{}, float_id{}, bits6{});
      } else if (bits_ == 4)
        DISPATCH_GROUP_SIZE(float_id{}, float_id{}, bits4{});
      else if (bits_ == 2)
        DISPATCH_GROUP_SIZE(float_id{}, float_id{}, bits2{});
      else
        throw std::runtime_error(
            "Unsupported bits for QuantizedMatmul float32: " +
            std::to_string(bits_));
    } else if (x.dtype() == float16) {
      if (bits_ == 8)
        DISPATCH_GROUP_SIZE(half_id{}, half_id{}, bits8{});
      else if (bits_ == 5 && mode_ == QuantizationMode::Affine) {
        DISPATCH_GROUP_SIZE(half_id{}, half_id{}, bits5{});
      } else if (bits_ == 6 && mode_ == QuantizationMode::Affine) {
        DISPATCH_GROUP_SIZE(half_id{}, half_id{}, bits6{});
      } else if (bits_ == 4)
        DISPATCH_GROUP_SIZE(half_id{}, half_id{}, bits4{});
      else if (bits_ == 2)
        DISPATCH_GROUP_SIZE(half_id{}, half_id{}, bits2{});
      else
        throw std::runtime_error(
            "Unsupported bits for QuantizedMatmul float16: " +
            std::to_string(bits_));
    } else if (x.dtype() == bfloat16) {
      if (bits_ == 8)
        DISPATCH_GROUP_SIZE(bf16_id{}, bf16_id{}, bits8{});
      else if (bits_ == 5 && mode_ == QuantizationMode::Affine) {
        DISPATCH_GROUP_SIZE(bf16_id{}, bf16_id{}, bits5{});
      } else if (bits_ == 6 && mode_ == QuantizationMode::Affine) {
        DISPATCH_GROUP_SIZE(bf16_id{}, bf16_id{}, bits6{});
      } else if (bits_ == 4)
        DISPATCH_GROUP_SIZE(bf16_id{}, bf16_id{}, bits4{});
      else if (bits_ == 2)
        DISPATCH_GROUP_SIZE(bf16_id{}, bf16_id{}, bits2{});
      else
        throw std::runtime_error(
            "Unsupported bits for QuantizedMatmul bfloat16: " +
            std::to_string(bits_));
    } else {
      throw std::runtime_error("Unsupported dtype for QuantizedMatmul");
    }

#undef DISPATCH_GROUP_SIZE
  });
}

namespace rocm {

// ======================================================================
// GPU-only expert-batched gather QMV for sorted indices.
//
// Grid: (M, ceil(N/cols_per_block), max_unique_experts)
// Each block in z-dimension finds its expert by binary-searching the sorted
// rhs_indices array. No CPU-side run computation needed.
//
// The kernel reads the weight column ONCE per expert and iterates over all
// batch elements assigned to that expert, amortizing weight memory traffic.
// ======================================================================
template <
    typename T,
    typename ScaleT,
    int BITS,
    int GROUP_SIZE,
    bool AFFINE,
    int THREADS_PER_COL>
__global__ void __launch_bounds__(1024) gather_qmv_expert_batched_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    const uint32_t* __restrict__ lhs_indices,
    const uint32_t* __restrict__ rhs_indices,  // SORTED
    T* __restrict__ out,
    int B,
    int M,
    int N,
    int K,
    int E,
    bool has_bias,
    bool implicit_lhs,
    int64_t implicit_x_batch_stride) {
  const int lane = threadIdx.x;
  const int warp_idx = threadIdx.y;
  const int col = blockIdx.y * blockDim.y + warp_idx;
  const int row = blockIdx.x;
  const int expert_slot = blockIdx.z;  // which unique expert this block handles

  if (row >= M || col >= N) return;

  // Find this expert's token range using the expert_slot as a run index.
  // Since rhs_indices is sorted, run boundaries are where values change.
  // We use a parallel scan: all threads cooperate to count unique experts
  // up to expert_slot, then binary-search for the run boundaries.
  //
  // Fast path: lane 0 does a boundary skip using binary search.
  int run_start = 0, run_end = 0;
  uint32_t expert_id = 0;

  if (lane == 0 && warp_idx == 0) {
    // Skip to the expert_slot-th unique expert by jumping over run boundaries.
    // Each boundary is where rhs_indices[i] != rhs_indices[i-1].
    int pos = 0;
    for (int skip = 0; skip < expert_slot && pos < B; ++skip) {
      // Binary search for end of current run (first index where value differs)
      uint32_t cur_val = rhs_indices[pos];
      int lo = pos + 1, hi = B;
      while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (rhs_indices[mid] == cur_val) lo = mid + 1;
        else hi = mid;
      }
      pos = lo;
    }
    if (pos < B) {
      run_start = pos;
      expert_id = rhs_indices[pos];
      // Binary search for end of this expert's run
      int lo = pos + 1, hi = B;
      while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (rhs_indices[mid] == expert_id) lo = mid + 1;
        else hi = mid;
      }
      run_end = lo;
    }
  }

  // Broadcast via shared memory
  __shared__ int s_run_start, s_run_end;
  __shared__ uint32_t s_expert_id;
  if (lane == 0 && warp_idx == 0) {
    s_run_start = run_start;
    s_run_end = run_end;
    s_expert_id = expert_id;
  }
  __syncthreads();
  run_start = s_run_start;
  run_end = s_run_end;
  expert_id = s_expert_id;

  if (run_end <= run_start) return;  // this block has no work
  if (expert_id >= static_cast<uint32_t>(E)) return;

  // Weight pointers for this expert (loaded ONCE, reused for all tokens in run)
  const int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const int row_bytes = (K * BITS + 7) / 8;
  int64_t w_expert_stride = static_cast<int64_t>(N) * row_bytes;
  int64_t sb_expert_stride = static_cast<int64_t>(N) * num_groups;

  const uint8_t* w_row = w + static_cast<int64_t>(expert_id) * w_expert_stride
                        + static_cast<int64_t>(col) * row_bytes;
  const ScaleT* scales_row = scales + static_cast<int64_t>(expert_id) * sb_expert_stride
                            + static_cast<int64_t>(col) * num_groups;
  const ScaleT* biases_row = has_bias
      ? (biases + static_cast<int64_t>(expert_id) * sb_expert_stride
                + static_cast<int64_t>(col) * num_groups)
      : nullptr;

  // Process each batch element in the run
  int64_t x_batch_stride = static_cast<int64_t>(M) * K;
  for (int b = run_start; b < run_end; ++b) {
    uint32_t lhs_idx = implicit_lhs ? 0u : lhs_indices[b];
    int64_t x_offset = implicit_lhs
        ? (static_cast<int64_t>(b) * implicit_x_batch_stride)
        : (static_cast<int64_t>(lhs_idx) * x_batch_stride);
    const T* x_row = x + x_offset + static_cast<int64_t>(row) * K;

    float acc = 0.0f;

    for (int g = 0; g < num_groups; ++g) {
      int k_start = g * GROUP_SIZE;
      int k_end = min(k_start + GROUP_SIZE, K);

      float scale = load_scale_value<ScaleT, GROUP_SIZE, AFFINE>(scales_row[g]);
      float bias_val = has_bias ? static_cast<float>(biases_row[g]) : 0.0f;

      if constexpr (AFFINE) {
        float qx_acc = 0.0f;
        float x_group_sum = 0.0f;

        if constexpr (BITS == 4) {
          int k_local = lane * 8;
          int step = THREADS_PER_COL * 8;
          for (; k_start + k_local + 7 < k_end; k_local += step) {
            int k = k_start + k_local;
            uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_row[k / 2]);
            float w0 = static_cast<float>(w_packed & 0xF);
            float w1 = static_cast<float>((w_packed >> 4) & 0xF);
            float w2 = static_cast<float>((w_packed >> 8) & 0xF);
            float w3 = static_cast<float>((w_packed >> 12) & 0xF);
            float w4 = static_cast<float>((w_packed >> 16) & 0xF);
            float w5 = static_cast<float>((w_packed >> 20) & 0xF);
            float w6 = static_cast<float>((w_packed >> 24) & 0xF);
            float w7 = static_cast<float>((w_packed >> 28) & 0xF);
            float x0 = static_cast<float>(x_row[k]);
            float x1 = static_cast<float>(x_row[k + 1]);
            float x2 = static_cast<float>(x_row[k + 2]);
            float x3 = static_cast<float>(x_row[k + 3]);
            float x4 = static_cast<float>(x_row[k + 4]);
            float x5 = static_cast<float>(x_row[k + 5]);
            float x6 = static_cast<float>(x_row[k + 6]);
            float x7 = static_cast<float>(x_row[k + 7]);
            qx_acc = fmaf(x0, w0, qx_acc);
            qx_acc = fmaf(x1, w1, qx_acc);
            qx_acc = fmaf(x2, w2, qx_acc);
            qx_acc = fmaf(x3, w3, qx_acc);
            qx_acc = fmaf(x4, w4, qx_acc);
            qx_acc = fmaf(x5, w5, qx_acc);
            qx_acc = fmaf(x6, w6, qx_acc);
            qx_acc = fmaf(x7, w7, qx_acc);
            if (has_bias)
              x_group_sum += x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
          }
          for (; k_start + k_local < k_end; k_local++) {
            int k = k_start + k_local;
            float x_val = static_cast<float>(x_row[k]);
            uint8_t quant_val = unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
            qx_acc = fmaf(x_val, static_cast<float>(quant_val), qx_acc);
            if (has_bias) x_group_sum += x_val;
          }
        } else if constexpr (BITS == 8) {
          int k_local = lane * 4;
          int step = THREADS_PER_COL * 4;
          for (; k_start + k_local + 3 < k_end; k_local += step) {
            int k = k_start + k_local;
            uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_row[k]);
            float w0 = static_cast<float>(w_packed & 0xFF);
            float w1 = static_cast<float>((w_packed >> 8) & 0xFF);
            float w2 = static_cast<float>((w_packed >> 16) & 0xFF);
            float w3 = static_cast<float>((w_packed >> 24) & 0xFF);
            float x0 = static_cast<float>(x_row[k]);
            float x1 = static_cast<float>(x_row[k + 1]);
            float x2 = static_cast<float>(x_row[k + 2]);
            float x3 = static_cast<float>(x_row[k + 3]);
            qx_acc = fmaf(x0, w0, qx_acc);
            qx_acc = fmaf(x1, w1, qx_acc);
            qx_acc = fmaf(x2, w2, qx_acc);
            qx_acc = fmaf(x3, w3, qx_acc);
            if (has_bias) x_group_sum += x0 + x1 + x2 + x3;
          }
          for (; k_start + k_local < k_end; k_local++) {
            int k = k_start + k_local;
            float x_val = static_cast<float>(x_row[k]);
            float w_val = static_cast<float>(w_row[k]);
            qx_acc = fmaf(x_val, w_val, qx_acc);
            if (has_bias) x_group_sum += x_val;
          }
        } else {
          for (int k_local = lane; k_start + k_local < k_end; k_local += THREADS_PER_COL) {
            int k = k_start + k_local;
            float x_val = static_cast<float>(x_row[k]);
            uint8_t quant_val = unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
            qx_acc = fmaf(x_val, static_cast<float>(quant_val), qx_acc);
            if (has_bias) x_group_sum += x_val;
          }
        }

        qx_acc = subgroup_reduce_sum_qmm<THREADS_PER_COL>(qx_acc);
        x_group_sum = subgroup_reduce_sum_qmm<THREADS_PER_COL>(x_group_sum);
        acc += scale * qx_acc + bias_val * x_group_sum;
      } else {
        float qx_acc = 0.0f;
        for (int k_local = lane; k_start + k_local < k_end; k_local += THREADS_PER_COL) {
          int k = k_start + k_local;
          float x_val = static_cast<float>(x_row[k]);
          uint8_t quant_val = unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
          qx_acc = fmaf(x_val, dequantize_value<BITS, false>(quant_val, 1.0f, 0.0f), qx_acc);
        }
        acc += scale * subgroup_reduce_sum_qmm<THREADS_PER_COL>(qx_acc);
      }
    }

    if (lane == 0) {
      out[static_cast<int64_t>(b) * M * N + static_cast<int64_t>(row) * N + col] = static_cast<T>(acc);
    }
  }
}

// ======================================================================
// Prefill-optimized gather QMV: groups batch elements by expert.
//
// For sorted rhs_indices, consecutive batch elements hit the same expert.
// This kernel assigns blockIdx.z to contiguous runs of same-expert batches,
// so all rows for one expert share weight reads from global memory.
// Each block handles one column (via warp cooperation) and iterates over
// all M rows for each batch element in the run.
//
// Grid: (num_runs, ceil(N/cols_per_block), max_rows_per_run)
// Where num_runs = number of contiguous expert runs.
// ======================================================================
template <
    typename T,
    typename ScaleT,
    int BITS,
    int GROUP_SIZE,
    bool AFFINE,
    int THREADS_PER_COL>
__global__ void __launch_bounds__(1024) gather_qmv_prefill_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    const uint32_t* __restrict__ lhs_indices,
    const uint32_t* __restrict__ rhs_indices,
    const int* __restrict__ run_starts,   // [num_runs]: start batch idx of each run
    const int* __restrict__ run_lengths,  // [num_runs]: length of each run
    const int* __restrict__ out_perm,     // [B]: sorted batch idx → original batch idx
    T* __restrict__ out,
    int B,
    int M,
    int N,
    int K,
    int E,
    bool has_bias,
    int64_t x_batch_stride) {
  const int lane = threadIdx.x;
  const int warp_idx = threadIdx.y;
  const int col = blockIdx.y * blockDim.y + warp_idx;
  const int run_id = blockIdx.z;
  const int row = blockIdx.x;

  if (row >= M || col >= N) return;

  int run_start = run_starts[run_id];
  int run_len = run_lengths[run_id];

  // All batches in this run have the same expert
  uint32_t rhs_idx = rhs_indices[run_start];
  if (rhs_idx >= static_cast<uint32_t>(E)) return;

  // Weight pointers (same for all batches in run)
  const int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const int row_bytes = (K * BITS + 7) / 8;
  int64_t w_expert_stride = static_cast<int64_t>(N) * row_bytes;
  int64_t sb_expert_stride = static_cast<int64_t>(N) * num_groups;
  int64_t col_w_offset = static_cast<int64_t>(col) * row_bytes;
  int64_t col_sb_offset = static_cast<int64_t>(col) * num_groups;

  const uint8_t* w_row = w + static_cast<int64_t>(rhs_idx) * w_expert_stride + col_w_offset;
  const ScaleT* scales_row = scales + static_cast<int64_t>(rhs_idx) * sb_expert_stride + col_sb_offset;
  const ScaleT* biases_row = has_bias
      ? (biases + static_cast<int64_t>(rhs_idx) * sb_expert_stride + col_sb_offset)
      : nullptr;

  // Process each batch element in the run
  for (int r = 0; r < run_len; ++r) {
    int batch = run_start + r;
    uint32_t lhs_idx = lhs_indices[batch];
    const T* x_row = x + static_cast<int64_t>(lhs_idx) * x_batch_stride + static_cast<int64_t>(row) * K;

    float acc = 0.0f;

    for (int g = 0; g < num_groups; ++g) {
      int k_start = g * GROUP_SIZE;
      int k_end = min(k_start + GROUP_SIZE, K);

      float scale = load_scale_value<ScaleT, GROUP_SIZE, AFFINE>(scales_row[g]);
      float bias_val = has_bias ? static_cast<float>(biases_row[g]) : 0.0f;

      if constexpr (AFFINE) {
        float qx_acc = 0.0f;
        float x_group_sum = 0.0f;

        if constexpr (BITS == 4) {
          int k_local = lane * 8;
          int step = THREADS_PER_COL * 8;
          for (; k_start + k_local + 7 < k_end; k_local += step) {
            int k = k_start + k_local;
            uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_row[k / 2]);
            float w0 = static_cast<float>(w_packed & 0xF);
            float w1 = static_cast<float>((w_packed >> 4) & 0xF);
            float w2 = static_cast<float>((w_packed >> 8) & 0xF);
            float w3 = static_cast<float>((w_packed >> 12) & 0xF);
            float w4 = static_cast<float>((w_packed >> 16) & 0xF);
            float w5 = static_cast<float>((w_packed >> 20) & 0xF);
            float w6 = static_cast<float>((w_packed >> 24) & 0xF);
            float w7 = static_cast<float>((w_packed >> 28) & 0xF);
            float x0 = static_cast<float>(x_row[k]);
            float x1 = static_cast<float>(x_row[k + 1]);
            float x2 = static_cast<float>(x_row[k + 2]);
            float x3 = static_cast<float>(x_row[k + 3]);
            float x4 = static_cast<float>(x_row[k + 4]);
            float x5 = static_cast<float>(x_row[k + 5]);
            float x6 = static_cast<float>(x_row[k + 6]);
            float x7 = static_cast<float>(x_row[k + 7]);
            qx_acc = fmaf(x0, w0, qx_acc);
            qx_acc = fmaf(x1, w1, qx_acc);
            qx_acc = fmaf(x2, w2, qx_acc);
            qx_acc = fmaf(x3, w3, qx_acc);
            qx_acc = fmaf(x4, w4, qx_acc);
            qx_acc = fmaf(x5, w5, qx_acc);
            qx_acc = fmaf(x6, w6, qx_acc);
            qx_acc = fmaf(x7, w7, qx_acc);
            if (has_bias)
              x_group_sum += x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
          }
          for (; k_start + k_local < k_end; k_local++) {
            int k = k_start + k_local;
            float x_val = static_cast<float>(x_row[k]);
            uint8_t quant_val = unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
            qx_acc = fmaf(x_val, static_cast<float>(quant_val), qx_acc);
            if (has_bias) x_group_sum += x_val;
          }
        } else if constexpr (BITS == 8) {
          int k_local = lane * 4;
          int step = THREADS_PER_COL * 4;
          for (; k_start + k_local + 3 < k_end; k_local += step) {
            int k = k_start + k_local;
            uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_row[k]);
            float w0 = static_cast<float>(w_packed & 0xFF);
            float w1 = static_cast<float>((w_packed >> 8) & 0xFF);
            float w2 = static_cast<float>((w_packed >> 16) & 0xFF);
            float w3 = static_cast<float>((w_packed >> 24) & 0xFF);
            float x0 = static_cast<float>(x_row[k]);
            float x1 = static_cast<float>(x_row[k + 1]);
            float x2 = static_cast<float>(x_row[k + 2]);
            float x3 = static_cast<float>(x_row[k + 3]);
            qx_acc = fmaf(x0, w0, qx_acc);
            qx_acc = fmaf(x1, w1, qx_acc);
            qx_acc = fmaf(x2, w2, qx_acc);
            qx_acc = fmaf(x3, w3, qx_acc);
            if (has_bias) x_group_sum += x0 + x1 + x2 + x3;
          }
          for (; k_start + k_local < k_end; k_local++) {
            int k = k_start + k_local;
            float x_val = static_cast<float>(x_row[k]);
            float w_val = static_cast<float>(w_row[k]);
            qx_acc = fmaf(x_val, w_val, qx_acc);
            if (has_bias) x_group_sum += x_val;
          }
        } else {
          for (int k_local = lane; k_start + k_local < k_end; k_local += THREADS_PER_COL) {
            int k = k_start + k_local;
            float x_val = static_cast<float>(x_row[k]);
            uint8_t quant_val = unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
            qx_acc = fmaf(x_val, static_cast<float>(quant_val), qx_acc);
            if (has_bias) x_group_sum += x_val;
          }
        }

        qx_acc = subgroup_reduce_sum_qmm<THREADS_PER_COL>(qx_acc);
        x_group_sum = subgroup_reduce_sum_qmm<THREADS_PER_COL>(x_group_sum);
        acc += scale * qx_acc + bias_val * x_group_sum;
      } else {
        float qx_acc = 0.0f;
        for (int k_local = lane; k_start + k_local < k_end; k_local += THREADS_PER_COL) {
          int k = k_start + k_local;
          float x_val = static_cast<float>(x_row[k]);
          uint8_t quant_val = unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
          qx_acc = fmaf(x_val, dequantize_value<BITS, false>(quant_val, 1.0f, 0.0f), qx_acc);
        }
        acc += scale * subgroup_reduce_sum_qmm<THREADS_PER_COL>(qx_acc);
      }
    }

    if (lane == 0) {
      const int orig_batch = out_perm[batch];
      out[static_cast<int64_t>(orig_batch) * M * N + static_cast<int64_t>(row) * N + col] = static_cast<T>(acc);
    }
  }
}

template <
    typename T,
    typename ScaleT,
    int BITS,
    int GROUP_SIZE,
    bool AFFINE,
    int THREADS_PER_COL>
__global__ void __launch_bounds__(1024) gather_qmv_warp_shared_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    const uint32_t* __restrict__ lhs_indices,
    const uint32_t* __restrict__ rhs_indices,
    const rocm::Shape batch_shape,
    const rocm::Strides lhs_idx_strides,
    const rocm::Strides rhs_idx_strides,
    int batch_ndim,
    T* __restrict__ out,
    int B,
    int M,
    int N,
    int K,
    int E,
    bool has_bias,
    bool implicit_lhs = false,
    int64_t implicit_x_batch_stride = 0) {
  const int lane = threadIdx.x;
  const int warp_idx = threadIdx.y;
  const int col = blockIdx.y * blockDim.y + warp_idx;
  const int row = blockIdx.x;
  const int batch = blockIdx.z;

  if (batch >= B || row >= M) {
    return;
  }

  int64_t rhs_idx_loc = 0;
  int64_t lhs_idx_loc = 0;
  if (batch_ndim == 1) {
    rhs_idx_loc = static_cast<int64_t>(batch) * rhs_idx_strides[0];
    if (!implicit_lhs) {
      lhs_idx_loc = static_cast<int64_t>(batch) * lhs_idx_strides[0];
    }
  } else if (batch_ndim > 1) {
    int64_t elem = static_cast<int64_t>(batch);
    for (int i = batch_ndim - 1; i >= 0; --i) {
      int64_t coord = elem % batch_shape.data_[i];
      rhs_idx_loc += coord * rhs_idx_strides.data_[i];
      if (!implicit_lhs) {
        lhs_idx_loc += coord * lhs_idx_strides.data_[i];
      }
      elem /= batch_shape.data_[i];
    }
  }

  uint32_t lhs_idx = implicit_lhs ? 0u : lhs_indices[lhs_idx_loc];
  uint32_t rhs_idx = rhs_indices[rhs_idx_loc];

  const bool col_valid = col < N;
  const bool expert_valid = rhs_idx < static_cast<uint32_t>(E);
  const bool valid = col_valid && expert_valid;

  const int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const int row_bytes = (K * BITS + 7) / 8;

  int64_t x_batch_stride = static_cast<int64_t>(M) * K;
  int64_t w_batch_stride = static_cast<int64_t>(N) * row_bytes;
  int64_t sb_batch_stride = static_cast<int64_t>(N) * num_groups;
  int64_t col_w_offset = static_cast<int64_t>(col) * row_bytes;
  int64_t col_sb_offset = static_cast<int64_t>(col) * num_groups;

  int64_t x_batch_offset = implicit_lhs
      ? (static_cast<int64_t>(batch) * implicit_x_batch_stride)
      : (static_cast<int64_t>(lhs_idx) * x_batch_stride);
  const T* x_row = x + x_batch_offset + static_cast<int64_t>(row) * K;
  const uint8_t* w_row = valid
      ? (w + static_cast<int64_t>(rhs_idx) * w_batch_stride + col_w_offset)
      : nullptr;
  const ScaleT* scales_row = valid
      ? (scales + static_cast<int64_t>(rhs_idx) * sb_batch_stride +
         col_sb_offset)
      : nullptr;
  const ScaleT* biases_row = (valid && has_bias)
      ? (biases + static_cast<int64_t>(rhs_idx) * sb_batch_stride +
         col_sb_offset)
      : nullptr;

  float acc = 0.0f;

  constexpr int CHUNK_SIZE = 2048;
  __shared__ float shared_x[CHUNK_SIZE];

  for (int chunk_start = 0; chunk_start < K; chunk_start += CHUNK_SIZE) {
    int chunk_end = min(chunk_start + CHUNK_SIZE, K);
    int chunk_len = chunk_end - chunk_start;

    int tid = warp_idx * blockDim.x + lane;
    for (int i = tid; i < chunk_len; i += blockDim.x * blockDim.y) {
      shared_x[i] = static_cast<float>(x_row[chunk_start + i]);
    }
    __syncthreads();

    if (valid) {
      int g_start = chunk_start / GROUP_SIZE;
      int g_end = (chunk_end + GROUP_SIZE - 1) / GROUP_SIZE;

      for (int g = g_start; g < g_end; ++g) {
        int k_start = max(g * GROUP_SIZE, chunk_start);
        int k_end_g = min((g + 1) * GROUP_SIZE, chunk_end);

        float scale =
            load_scale_value<ScaleT, GROUP_SIZE, AFFINE>(scales_row[g]);
        float bias_val = has_bias ? static_cast<float>(biases_row[g]) : 0.0f;

        if constexpr (AFFINE) {
          float qx_acc0 = 0.0f;
          float qx_acc1 = 0.0f;
          float qx_acc2 = 0.0f;
          float qx_acc3 = 0.0f;
          float x_group_sum = 0.0f;
          float qx_acc = 0.0f;

          if constexpr (BITS == 8) {
            int k_local = lane * 4;
            int step = THREADS_PER_COL * 4;
            for (; k_start + k_local + 3 < k_end_g; k_local += step) {
              int k = k_start + k_local;

              uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_row[k]);
              float w0 = static_cast<float>(w_packed & 0xFF);
              float w1 = static_cast<float>((w_packed >> 8) & 0xFF);
              float w2 = static_cast<float>((w_packed >> 16) & 0xFF);
              float w3 = static_cast<float>((w_packed >> 24) & 0xFF);

              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];

              qx_acc0 = fmaf(x0, w0, qx_acc0);
              qx_acc1 = fmaf(x1, w1, qx_acc1);
              qx_acc2 = fmaf(x2, w2, qx_acc2);
              qx_acc3 = fmaf(x3, w3, qx_acc3);

              if (has_bias) {
                x_group_sum += x0 + x1 + x2 + x3;
              }
            }
            qx_acc = qx_acc0 + qx_acc1 + qx_acc2 + qx_acc3;
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              float w_val = static_cast<float>(w_row[k]);
              qx_acc = fmaf(x_val, w_val, qx_acc);
              if (has_bias) {
                x_group_sum += x_val;
              }
            }
          } else if constexpr (BITS == 4) {
            int k_local = lane * 8;
            int step = THREADS_PER_COL * 8;
            for (; k_start + k_local + 7 < k_end_g; k_local += step) {
              int k = k_start + k_local;
              uint32_t w_packed =
                  *reinterpret_cast<const uint32_t*>(&w_row[k / 2]);
              float w0 = static_cast<float>(w_packed & 0xF);
              float w1 = static_cast<float>((w_packed >> 4) & 0xF);
              float w2 = static_cast<float>((w_packed >> 8) & 0xF);
              float w3 = static_cast<float>((w_packed >> 12) & 0xF);
              float w4 = static_cast<float>((w_packed >> 16) & 0xF);
              float w5 = static_cast<float>((w_packed >> 20) & 0xF);
              float w6 = static_cast<float>((w_packed >> 24) & 0xF);
              float w7 = static_cast<float>((w_packed >> 28) & 0xF);
              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];
              float x4 = shared_x[k - chunk_start + 4];
              float x5 = shared_x[k - chunk_start + 5];
              float x6 = shared_x[k - chunk_start + 6];
              float x7 = shared_x[k - chunk_start + 7];
              // Four independent accumulators for RDNA4 dual-issue (mirrors the
              // 8-bit branch); reassociated at group end.
              qx_acc0 = fmaf(x0, w0, qx_acc0);
              qx_acc1 = fmaf(x1, w1, qx_acc1);
              qx_acc2 = fmaf(x2, w2, qx_acc2);
              qx_acc3 = fmaf(x3, w3, qx_acc3);
              qx_acc0 = fmaf(x4, w4, qx_acc0);
              qx_acc1 = fmaf(x5, w5, qx_acc1);
              qx_acc2 = fmaf(x6, w6, qx_acc2);
              qx_acc3 = fmaf(x7, w7, qx_acc3);
              if (has_bias) {
                x_group_sum += x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
              }
            }
            qx_acc = qx_acc0 + qx_acc1 + qx_acc2 + qx_acc3;
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(x_val, static_cast<float>(quant_val), qx_acc);
              if (has_bias) {
                x_group_sum += x_val;
              }
            }
          } else if constexpr (BITS == 6) {
            int k_local = lane * 8;
            int step = THREADS_PER_COL * 8;
            int max_safe_k = ((row_bytes - 7) * 8) / 6;
            for (; k_start + k_local + 7 < k_end_g &&
                 k_start + k_local < max_safe_k;
                 k_local += step) {
              int k = k_start + k_local;
              int byte_idx = (k * 6) / 8;
              int bit_offset = (k * 6) % 8;
              uint64_t w_packed = 0;
              memcpy(&w_packed, &w_row[byte_idx], 8);
              w_packed >>= bit_offset;
              float w0 = static_cast<float>(w_packed & 0x3F);
              float w1 = static_cast<float>((w_packed >> 6) & 0x3F);
              float w2 = static_cast<float>((w_packed >> 12) & 0x3F);
              float w3 = static_cast<float>((w_packed >> 18) & 0x3F);
              float w4 = static_cast<float>((w_packed >> 24) & 0x3F);
              float w5 = static_cast<float>((w_packed >> 30) & 0x3F);
              float w6 = static_cast<float>((w_packed >> 36) & 0x3F);
              float w7 = static_cast<float>((w_packed >> 42) & 0x3F);
              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];
              float x4 = shared_x[k - chunk_start + 4];
              float x5 = shared_x[k - chunk_start + 5];
              float x6 = shared_x[k - chunk_start + 6];
              float x7 = shared_x[k - chunk_start + 7];
              qx_acc = fmaf(x0, w0, qx_acc);
              qx_acc = fmaf(x1, w1, qx_acc);
              qx_acc = fmaf(x2, w2, qx_acc);
              qx_acc = fmaf(x3, w3, qx_acc);
              qx_acc = fmaf(x4, w4, qx_acc);
              qx_acc = fmaf(x5, w5, qx_acc);
              qx_acc = fmaf(x6, w6, qx_acc);
              qx_acc = fmaf(x7, w7, qx_acc);
              if (has_bias) {
                x_group_sum += x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
              }
            }
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(x_val, static_cast<float>(quant_val), qx_acc);
              if (has_bias) {
                x_group_sum += x_val;
              }
            }
          } else {
            for (int k_local = lane; k_start + k_local < k_end_g;
                 k_local += THREADS_PER_COL) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(x_val, static_cast<float>(quant_val), qx_acc);
              if (has_bias) {
                x_group_sum += x_val;
              }
            }
          }
          acc += scale * qx_acc;
          if (has_bias) {
            acc += bias_val * x_group_sum;
          }
        } else {
          float qx_acc0 = 0.0f;
          float qx_acc1 = 0.0f;
          float qx_acc2 = 0.0f;
          float qx_acc3 = 0.0f;
          float qx_acc = 0.0f;
          if constexpr (BITS == 8) {
            int k_local = lane * 4;
            int step = THREADS_PER_COL * 4;
            for (; k_start + k_local + 3 < k_end_g; k_local += step) {
              int k = k_start + k_local;

              uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_row[k]);
              float w0 = fp8_e4m3_to_float(w_packed & 0xFF);
              float w1 = fp8_e4m3_to_float((w_packed >> 8) & 0xFF);
              float w2 = fp8_e4m3_to_float((w_packed >> 16) & 0xFF);
              float w3 = fp8_e4m3_to_float((w_packed >> 24) & 0xFF);

              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];

              qx_acc0 = fmaf(x0, w0, qx_acc0);
              qx_acc1 = fmaf(x1, w1, qx_acc1);
              qx_acc2 = fmaf(x2, w2, qx_acc2);
              qx_acc3 = fmaf(x3, w3, qx_acc3);
            }
            qx_acc = qx_acc0 + qx_acc1 + qx_acc2 + qx_acc3;
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              float w_val = fp8_e4m3_to_float(w_row[k]);
              qx_acc = fmaf(x_val, w_val, qx_acc);
            }
          } else if constexpr (BITS == 4) {
            int k_local = lane * 8;
            int step = THREADS_PER_COL * 8;
            for (; k_start + k_local + 7 < k_end_g; k_local += step) {
              int k = k_start + k_local;
              uint32_t w_packed =
                  *reinterpret_cast<const uint32_t*>(&w_row[k / 2]);
              float w0 = dequantize_value<4, false>(w_packed & 0xF, 1.0f, 0.0f);
              float w1 =
                  dequantize_value<4, false>((w_packed >> 4) & 0xF, 1.0f, 0.0f);
              float w2 =
                  dequantize_value<4, false>((w_packed >> 8) & 0xF, 1.0f, 0.0f);
              float w3 = dequantize_value<4, false>(
                  (w_packed >> 12) & 0xF, 1.0f, 0.0f);
              float w4 = dequantize_value<4, false>(
                  (w_packed >> 16) & 0xF, 1.0f, 0.0f);
              float w5 = dequantize_value<4, false>(
                  (w_packed >> 20) & 0xF, 1.0f, 0.0f);
              float w6 = dequantize_value<4, false>(
                  (w_packed >> 24) & 0xF, 1.0f, 0.0f);
              float w7 = dequantize_value<4, false>(
                  (w_packed >> 28) & 0xF, 1.0f, 0.0f);
              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];
              float x4 = shared_x[k - chunk_start + 4];
              float x5 = shared_x[k - chunk_start + 5];
              float x6 = shared_x[k - chunk_start + 6];
              float x7 = shared_x[k - chunk_start + 7];
              qx_acc = fmaf(x0, w0, qx_acc);
              qx_acc = fmaf(x1, w1, qx_acc);
              qx_acc = fmaf(x2, w2, qx_acc);
              qx_acc = fmaf(x3, w3, qx_acc);
              qx_acc = fmaf(x4, w4, qx_acc);
              qx_acc = fmaf(x5, w5, qx_acc);
              qx_acc = fmaf(x6, w6, qx_acc);
              qx_acc = fmaf(x7, w7, qx_acc);
            }
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(
                  x_val,
                  dequantize_value<BITS, false>(quant_val, 1.0f, 0.0f),
                  qx_acc);
            }
          } else if constexpr (BITS == 6) {
            int k_local = lane * 8;
            int step = THREADS_PER_COL * 8;
            int max_safe_k = ((row_bytes - 7) * 8) / 6;
            for (; k_start + k_local + 7 < k_end_g &&
                 k_start + k_local < max_safe_k;
                 k_local += step) {
              int k = k_start + k_local;
              int byte_idx = (k * 6) / 8;
              int bit_offset = (k * 6) % 8;
              uint64_t w_packed = 0;
              memcpy(&w_packed, &w_row[byte_idx], 8);
              w_packed >>= bit_offset;
              float w0 =
                  dequantize_value<6, false>(w_packed & 0x3F, 1.0f, 0.0f);
              float w1 = dequantize_value<6, false>(
                  (w_packed >> 6) & 0x3F, 1.0f, 0.0f);
              float w2 = dequantize_value<6, false>(
                  (w_packed >> 12) & 0x3F, 1.0f, 0.0f);
              float w3 = dequantize_value<6, false>(
                  (w_packed >> 18) & 0x3F, 1.0f, 0.0f);
              float w4 = dequantize_value<6, false>(
                  (w_packed >> 24) & 0x3F, 1.0f, 0.0f);
              float w5 = dequantize_value<6, false>(
                  (w_packed >> 30) & 0x3F, 1.0f, 0.0f);
              float w6 = dequantize_value<6, false>(
                  (w_packed >> 36) & 0x3F, 1.0f, 0.0f);
              float w7 = dequantize_value<6, false>(
                  (w_packed >> 42) & 0x3F, 1.0f, 0.0f);
              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];
              float x4 = shared_x[k - chunk_start + 4];
              float x5 = shared_x[k - chunk_start + 5];
              float x6 = shared_x[k - chunk_start + 6];
              float x7 = shared_x[k - chunk_start + 7];
              qx_acc = fmaf(x0, w0, qx_acc);
              qx_acc = fmaf(x1, w1, qx_acc);
              qx_acc = fmaf(x2, w2, qx_acc);
              qx_acc = fmaf(x3, w3, qx_acc);
              qx_acc = fmaf(x4, w4, qx_acc);
              qx_acc = fmaf(x5, w5, qx_acc);
              qx_acc = fmaf(x6, w6, qx_acc);
              qx_acc = fmaf(x7, w7, qx_acc);
            }
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(
                  x_val,
                  dequantize_value<BITS, false>(quant_val, 1.0f, 0.0f),
                  qx_acc);
            }
          } else {
            for (int k_local = lane; k_start + k_local < k_end_g;
                 k_local += THREADS_PER_COL) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(
                  x_val,
                  dequantize_value<BITS, false>(quant_val, 1.0f, 0.0f),
                  qx_acc);
            }
          }
          acc += scale * qx_acc;
        }
      }
    }
    __syncthreads();
  }

  acc = subgroup_reduce_sum_qmm<THREADS_PER_COL>(acc);
  if (col_valid && lane == 0) {
    int64_t out_offset = (static_cast<int64_t>(batch) * M + row) * N + col;
    out[out_offset] = expert_valid ? static_cast<T>(acc) : static_cast<T>(0);
  }
}

template <typename T, typename ScaleT, int BITS, int GROUP_SIZE, bool AFFINE>
__global__ void gather_qmv_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    const uint32_t* __restrict__ lhs_indices,
    const uint32_t* __restrict__ rhs_indices,
    const rocm::Shape batch_shape,
    const rocm::Strides lhs_idx_strides,
    const rocm::Strides rhs_idx_strides,
    int batch_ndim,
    T* __restrict__ out,
    int B,
    int M,
    int N,
    int K,
    int E,
    bool has_bias,
    bool implicit_lhs = false,
    int64_t implicit_x_batch_stride = 0) {
  int batch = blockIdx.z;
  int row = blockIdx.x;
  int col = blockIdx.y * blockDim.x + threadIdx.x;
  if (batch >= B || row >= M || col >= N)
    return;
  int64_t lhs_idx_loc = 0;
  int64_t rhs_idx_loc = 0;
  if (batch_ndim == 1) {
    rhs_idx_loc = (int64_t)batch * rhs_idx_strides[0];
    if (!implicit_lhs) {
      lhs_idx_loc = (int64_t)batch * lhs_idx_strides[0];
    }
  } else if (batch_ndim > 1) {
    int64_t elem = (int64_t)batch;
    for (int i = batch_ndim - 1; i >= 0; --i) {
      int64_t coord = elem % batch_shape.data_[i];
      rhs_idx_loc += coord * rhs_idx_strides.data_[i];
      if (!implicit_lhs) {
        lhs_idx_loc += coord * lhs_idx_strides.data_[i];
      }
      elem /= batch_shape.data_[i];
    }
  }
  uint32_t lhs_idx = implicit_lhs ? 0u : lhs_indices[lhs_idx_loc];
  uint32_t rhs_idx = rhs_indices[rhs_idx_loc];
  if (rhs_idx >= static_cast<uint32_t>(E)) {
    out[batch * M * N + row * N + col] = static_cast<T>(0);
    return;
  }

  int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  int row_bytes = (K * BITS + 7) / 8;
  int64_t x_batch_stride = static_cast<int64_t>(M) * K;
  int64_t w_batch_stride = static_cast<int64_t>(N) * row_bytes;
  int64_t sb_batch_stride = static_cast<int64_t>(N) * num_groups;
  int64_t col_w_offset = static_cast<int64_t>(col) * row_bytes;
  int64_t col_sb_offset = static_cast<int64_t>(col) * num_groups;

  int64_t x_batch_offset = implicit_lhs
      ? (static_cast<int64_t>(batch) * implicit_x_batch_stride)
      : (static_cast<int64_t>(lhs_idx) * x_batch_stride);
  const T* x_ptr = x + x_batch_offset + static_cast<int64_t>(row) * K;
  const uint8_t* w_ptr =
      w + static_cast<int64_t>(rhs_idx) * w_batch_stride + col_w_offset;
  const ScaleT* scales_ptr =
      scales + static_cast<int64_t>(rhs_idx) * sb_batch_stride + col_sb_offset;
  const ScaleT* biases_ptr = has_bias
      ? biases + static_cast<int64_t>(rhs_idx) * sb_batch_stride + col_sb_offset
      : nullptr;
  float acc = 0.0f;
  for (int g = 0; g < num_groups; ++g) {
    float scale = load_scale_value<ScaleT, GROUP_SIZE, AFFINE>(scales_ptr[g]);
    float bias = has_bias ? (float)biases_ptr[g] : 0.0f;
    int k_start = g * GROUP_SIZE;
    int k_end = min(k_start + GROUP_SIZE, K);

    if constexpr (BITS == 8) {
      int k = k_start;
      for (; k + 3 < k_end; k += 4) {
        uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_ptr[k]);
        float w0 = dequantize_value<8, AFFINE>(w_packed & 0xFF, scale, bias);
        float w1 =
            dequantize_value<8, AFFINE>((w_packed >> 8) & 0xFF, scale, bias);
        float w2 =
            dequantize_value<8, AFFINE>((w_packed >> 16) & 0xFF, scale, bias);
        float w3 =
            dequantize_value<8, AFFINE>((w_packed >> 24) & 0xFF, scale, bias);

        acc += (float)x_ptr[k] * w0;
        acc += (float)x_ptr[k + 1] * w1;
        acc += (float)x_ptr[k + 2] * w2;
        acc += (float)x_ptr[k + 3] * w3;
      }
      for (; k < k_end; ++k) {
        float w_val = dequantize_value<8, AFFINE>(w_ptr[k], scale, bias);
        acc += (float)x_ptr[k] * w_val;
      }
    } else {
      for (int k = k_start; k < k_end; ++k) {
        uint8_t qv = unpack_packed_value_fast<BITS>(w_ptr, k, row_bytes);
        acc +=
            (float)x_ptr[k] * dequantize_value<BITS, AFFINE>(qv, scale, bias);
      }
    }
  }
  out[batch * M * N + row * N + col] = (T)acc;
}

// ======================================================================
// WMMA-accelerated gather QMV prefill kernel using rocwmma 16x16x16 tiles.
//
// Each wavefront (32 lanes on RDNA 3.5 / gfx1151) computes one 16x16
// output tile.  Weights are dequantized from 4-bit packed format into
// bf16 in shared memory, then loaded into rocwmma fragments for the
// matrix multiply-accumulate.  Accumulation is in float32; the final
// result is converted back to bf16 on store.
//
// Grid:  (ceil(M/16), ceil(N/16), num_runs)
// Block: (32, 1, 1)   -- one wave32 per 16x16 output tile
//
// On architectures without WMMA support (RDNA 1/2) the kernel body is
// an empty stub; dispatch checks prevent it from being launched there.
// ======================================================================
template <typename T, typename ScaleT, int BITS, int GROUP_SIZE, bool AFFINE>
__global__ void __launch_bounds__(32) gather_qmv_wmma_prefill_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    const uint32_t* __restrict__ lhs_indices,
    const uint32_t* __restrict__ rhs_indices,
    const int* __restrict__ run_starts,
    const int* __restrict__ run_lengths,
    const int* __restrict__ out_perm,  // maps sorted batch idx → original batch idx
    T* __restrict__ out,
    int B, int M, int N, int K, int E,
    bool has_bias, int64_t x_batch_stride) {

#if ROCM_HAS_WMMA

  static_assert(BITS == 4, "WMMA prefill kernel only supports 4-bit quantized weights");
  static_assert(AFFINE, "WMMA prefill kernel only supports affine quantization");

  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;

  // Tile coordinates in the output matrix
  const int tile_row = blockIdx.x * WMMA_M;  // starting row of this 16x16 tile
  const int tile_col = blockIdx.y * WMMA_N;  // starting col of this 16x16 tile
  const int run_id   = blockIdx.z;

  // Bounds check -- the dispatch guarantees M and N are multiples of 16,
  // but guard anyway for safety.
  if (tile_row >= M || tile_col >= N) return;

  const int lane = threadIdx.x;  // 0..31

  // Run info
  const int run_start = run_starts[run_id];
  const int run_len   = run_lengths[run_id];

  const uint32_t rhs_idx = rhs_indices[run_start];
  if (rhs_idx >= static_cast<uint32_t>(E)) return;

  // Weight layout constants
  const int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const int row_bytes  = (K * BITS + 7) / 8;  // bytes per weight row (one output col)
  const int64_t w_expert_stride  = static_cast<int64_t>(N) * row_bytes;
  const int64_t sb_expert_stride = static_cast<int64_t>(N) * num_groups;

  // Base pointers for this expert
  const uint8_t* w_expert  = w      + static_cast<int64_t>(rhs_idx) * w_expert_stride;
  const ScaleT*  s_expert  = scales + static_cast<int64_t>(rhs_idx) * sb_expert_stride;
  const ScaleT*  b_expert  = has_bias
      ? (biases + static_cast<int64_t>(rhs_idx) * sb_expert_stride)
      : nullptr;

  // Shared memory for dequantized weight tile [WMMA_K x WMMA_N] in row-major
  // and for x tile [WMMA_M x WMMA_K] in row-major.
  // Total: (16*16 + 16*16) * sizeof(hip_bfloat16) = 1024 bytes
  __shared__ hip_bfloat16 smem_w[WMMA_K * WMMA_N];  // [16][16] row-major
  __shared__ hip_bfloat16 smem_x[WMMA_M * WMMA_K];  // [16][16] row-major

  // Fragment types for bf16 input, f32 accumulation
  using frag_a   = rocwmma::fragment<rocwmma::matrix_a,   WMMA_M, WMMA_N, WMMA_K, hip_bfloat16, rocwmma::row_major>;
  using frag_b   = rocwmma::fragment<rocwmma::matrix_b,   WMMA_M, WMMA_N, WMMA_K, hip_bfloat16, rocwmma::row_major>;
  using frag_acc = rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

  // Process each batch element in the run
  for (int r = 0; r < run_len; ++r) {
    const int batch = run_start + r;
    const uint32_t lhs_idx = lhs_indices[batch];
    const T* x_base = x + static_cast<int64_t>(lhs_idx) * x_batch_stride
                        + static_cast<int64_t>(tile_row) * K;

    // Zero the accumulator for this batch element
    frag_acc acc;
    rocwmma::fill_fragment(acc, 0.0f);

    // Loop over K dimension in chunks of WMMA_K (16)
    for (int k_base = 0; k_base < K; k_base += WMMA_K) {
      // --- Load x tile [WMMA_M x WMMA_K] into shared memory ---
      // 32 lanes load 256 elements (16x16) -> 8 elements per lane
      // Pad with zero for rows beyond M (handles non-16-aligned M)
      #pragma unroll
      for (int i = 0; i < (WMMA_M * WMMA_K + 31) / 32; ++i) {
        int idx = lane + i * 32;
        if (idx < WMMA_M * WMMA_K) {
          int m_local = idx / WMMA_K;
          int k_local = idx % WMMA_K;
          int m_global = tile_row + m_local;
          int k_global = k_base + k_local;
          if (m_global < M && k_global < K) {
            smem_x[idx] = x_base[m_local * K + k_global];
          } else {
            smem_x[idx] = static_cast<hip_bfloat16>(0.0f);
          }
        }
      }

      // --- Dequantize weight tile [WMMA_K x WMMA_N] into shared memory ---
      // Layout: smem_w[k][n] = dequant(w[expert, tile_col + n, k_base + k])
      // w is stored as [N, row_bytes], each row for one output column.
      // We need 16 columns x 16 K values = 256 values, 8 per lane.
      #pragma unroll
      for (int i = 0; i < (WMMA_K * WMMA_N + 31) / 32; ++i) {
        int idx = lane + i * 32;
        if (idx < WMMA_K * WMMA_N) {
          int k_local = idx / WMMA_N;  // row in [K, N]
          int n_local = idx % WMMA_N;  // col in [K, N]
          int k_global = k_base + k_local;
          int n_global = tile_col + n_local;

          if (k_global < K) {
            // Pointer to weight row for output column n_global
            const uint8_t* w_row = w_expert + static_cast<int64_t>(n_global) * row_bytes;

            // Extract 4-bit quantized value
            uint8_t packed = w_row[k_global >> 1];
            uint8_t quant_val = (k_global & 1) ? (packed >> 4) : (packed & 0xF);

            // Dequantize: val = scale * quant_val + bias
            int group_idx = k_global / GROUP_SIZE;
            float scale = static_cast<float>(
                s_expert[static_cast<int64_t>(n_global) * num_groups + group_idx]);
            float bias_val = has_bias
                ? static_cast<float>(
                    b_expert[static_cast<int64_t>(n_global) * num_groups + group_idx])
                : 0.0f;
            float dequant = scale * static_cast<float>(quant_val) + bias_val;
            smem_w[idx] = static_cast<hip_bfloat16>(dequant);
          } else {
            smem_w[idx] = static_cast<hip_bfloat16>(0.0f);
          }
        }
      }

      __syncthreads();

      // --- Load fragments from shared memory and perform MMA ---
      frag_a a_frag;
      frag_b b_frag;

      // Load A from smem_x [WMMA_M x WMMA_K], row-major, ldm = WMMA_K
      rocwmma::load_matrix_sync(a_frag, smem_x, WMMA_K);
      // Load B from smem_w [WMMA_K x WMMA_N], row-major, ldm = WMMA_N
      rocwmma::load_matrix_sync(b_frag, smem_w, WMMA_N);

      // D = A * B + C
      rocwmma::mma_sync(acc, a_frag, b_frag, acc);

      __syncthreads();
    }

    // --- Store the 16x16 result tile ---
    // Store f32 accumulator to shared memory, then convert to bf16 for output.
    __shared__ float smem_out_f32[WMMA_M * WMMA_N];

    rocwmma::store_matrix_sync(smem_out_f32, acc, WMMA_N, rocwmma::mem_row_major);
    __syncthreads();

    // Convert f32 -> bf16 and write to global output (mask out-of-bounds rows)
    // Use out_perm to map sorted batch position back to original output position
    const int orig_batch = out_perm[batch];
    T* out_base = out + static_cast<int64_t>(orig_batch) * M * N
                      + static_cast<int64_t>(tile_row) * N
                      + tile_col;
    #pragma unroll
    for (int i = 0; i < (WMMA_M * WMMA_N + 31) / 32; ++i) {
      int idx = lane + i * 32;
      if (idx < WMMA_M * WMMA_N) {
        int m_local = idx / WMMA_N;
        int n_local = idx % WMMA_N;
        if (tile_row + m_local < M) {
          out_base[m_local * N + n_local] = static_cast<T>(smem_out_f32[idx]);
        }
      }
    }
    __syncthreads();
  }

#endif // ROCM_HAS_WMMA
}

} // namespace rocm

void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = rocm::device(s.device);
  auto& enc = d.get_command_encoder(s);
  out.set_data(allocator::malloc(out.nbytes()));
  array x = ensure_row_contiguous_matrix(inputs[0], enc, s);
  array w = ensure_row_contiguous_matrix(inputs[1], enc, s);
  array scales = ensure_row_contiguous_matrix(inputs[2], enc, s);
  std::optional<array> biases = std::nullopt;
  bool has_bias = (mode_ == QuantizationMode::Affine) && (inputs.size() == 6);
  if (has_bias)
    biases = ensure_row_contiguous_matrix(inputs[3], enc, s);
  const array& lhs_indices = inputs[inputs.size() - 2];
  const array& rhs_indices = inputs[inputs.size() - 1];
  auto [batch_shape, batch_strides] = collapse_contiguous_dims(
      lhs_indices.shape(), {lhs_indices.strides(), rhs_indices.strides()});
  auto batch_shape_param = const_param(batch_shape);
  auto lhs_idx_strides_param = const_param<MAX_NDIM, int64_t>(batch_strides[0]);
  auto rhs_idx_strides_param = const_param<MAX_NDIM, int64_t>(batch_strides[1]);
  int batch_ndim = batch_shape.size();
  enc.set_input_array(x);
  enc.set_input_array(w);
  enc.set_input_array(scales);
  if (has_bias)
    enc.set_input_array(biases.value());
  enc.set_input_array(lhs_indices);
  enc.set_input_array(rhs_indices);
  enc.set_output_array(out);
  int K = x.shape(-1), M = x.shape(-2), N = out.shape(-1),
      B = out.size() / M / N, E = w.size() / w.shape(-1) / w.shape(-2);

  int64_t x_batch_count = x.size() / (static_cast<int64_t>(M) * K);
  bool use_sorted_rhs_schedule = transpose_ && right_sorted_ && (M == 1) &&
      (B >= 16) && (E > 0) && (B / E >= 4) &&
      (x_batch_count == 1 || x_batch_count == B);
  int64_t implicit_x_batch_stride =
      (x_batch_count == 1) ? 0 : static_cast<int64_t>(M) * K;

  int block_size = 256;
  dim3 grid(M, (N + block_size - 1) / block_size, B);

  int fast_threads_per_col = select_qmv_threads_per_col(K, N, bits_, B);
  int fast_threads_env =
      parse_threads_per_col_env("MLX_ROCM_GATHER_QMV_THREADS_PER_COL");
  if (fast_threads_env <= 0) {
    fast_threads_env =
        parse_threads_per_col_env("MLX_ROCM_QMV_THREADS_PER_COL");
  }
  if (fast_threads_env > 0) {
    fast_threads_per_col = fast_threads_env;
  }

  int fast_cols_per_block = select_qmv_cols_per_block(K, N, bits_);
  if (fast_threads_per_col == 16 && bits_ == 8 && N >= 2048) {
    fast_cols_per_block = std::max(fast_cols_per_block, 64);
  }
  int max_cols_per_block = rocm::kMaxThreadsPerBlock / fast_threads_per_col;
  while (fast_cols_per_block > max_cols_per_block) {
    fast_cols_per_block /= 2;
  }
  while (fast_cols_per_block > 1 && (N % fast_cols_per_block) != 0 &&
         fast_cols_per_block > 8) {
    fast_cols_per_block /= 2;
  }

  dim3 fast_block(fast_threads_per_col, fast_cols_per_block);
  dim3 fast_grid(M, (N + fast_cols_per_block - 1) / fast_cols_per_block, B);

  bool bits_supported_by_fast = (bits_ == 2 || bits_ == 4 || bits_ == 8) ||
      (mode_ == QuantizationMode::Affine && (bits_ == 5 || bits_ == 6));
  bool use_fast_gather_qmv = transpose_ && bits_supported_by_fast;
  use_fast_gather_qmv = parse_warp_kernel_env(
      "MLX_ROCM_GATHER_QMV_USE_WARP", use_fast_gather_qmv);
  // ---- Prefill optimization: group by expert for M>1 ----
  // Works with both sorted and unsorted rhs_indices; we sort on CPU.
  // NOTE: MLX's MoE expands tokens to B individual M=1 calls, so M>1 is rare.
  // The WMMA prefill kernel is used when upstream batching produces M>1.
  if (M > 1 && transpose_ && E > 0 && batch_ndim == 1 &&
      mode_ == QuantizationMode::Affine && x.dtype() == bfloat16 &&
      group_size_ == 64 && (bits_ == 4 || bits_ == 8)) {
    // Sort batch elements by expert to form contiguous runs.
    // This allows the kernel to process all tokens for one expert together,
    // sharing weight reads. We create a sorted permutation on CPU.
    const auto* ri_cpu = rhs_indices.data<uint32_t>();
    const auto* li_cpu = lhs_indices.data<uint32_t>();

    // Create sort permutation by expert index
    std::vector<int> perm(B);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](int a, int b) {
      return ri_cpu[a] < ri_cpu[b];
    });

    // Build sorted index arrays and compute runs
    std::vector<uint32_t> sorted_ri(B), sorted_li(B);
    for (int i = 0; i < B; ++i) {
      sorted_ri[i] = ri_cpu[perm[i]];
      sorted_li[i] = li_cpu[perm[i]];
    }

    std::vector<int> run_starts_vec, run_lengths_vec;
    run_starts_vec.reserve(E);
    run_lengths_vec.reserve(E);
    int run_begin = 0;
    for (int b = 1; b <= B; ++b) {
      if (b == B || sorted_ri[b] != sorted_ri[run_begin]) {
        run_starts_vec.push_back(run_begin);
        run_lengths_vec.push_back(b - run_begin);
        run_begin = b;
      }
    }
    int num_runs = static_cast<int>(run_starts_vec.size());

    // Upload sorted indices to GPU
    array sorted_ri_arr({B}, uint32, nullptr, {});
    array sorted_li_arr({B}, uint32, nullptr, {});
    sorted_ri_arr.set_data(allocator::malloc(sorted_ri_arr.nbytes()));
    sorted_li_arr.set_data(allocator::malloc(sorted_li_arr.nbytes()));
    std::memcpy(sorted_ri_arr.data<uint32_t>(), sorted_ri.data(), B * sizeof(uint32_t));
    std::memcpy(sorted_li_arr.data<uint32_t>(), sorted_li.data(), B * sizeof(uint32_t));
    enc.set_input_array(sorted_ri_arr);
    enc.set_input_array(sorted_li_arr);

    // Also need a mapping from sorted position back to original batch index for output
    array perm_arr({B}, int32, nullptr, {});
    perm_arr.set_data(allocator::malloc(perm_arr.nbytes()));
    std::memcpy(perm_arr.data<int>(), perm.data(), B * sizeof(int));
    enc.set_input_array(perm_arr);

    // Upload run info to GPU
    array run_starts_arr({num_runs}, int32, nullptr, {});
    array run_lengths_arr({num_runs}, int32, nullptr, {});
    run_starts_arr.set_data(allocator::malloc(run_starts_arr.nbytes()));
    run_lengths_arr.set_data(allocator::malloc(run_lengths_arr.nbytes()));
    std::memcpy(run_starts_arr.data<int>(), run_starts_vec.data(), num_runs * sizeof(int));
    std::memcpy(run_lengths_arr.data<int>(), run_lengths_vec.data(), num_runs * sizeof(int));
    enc.set_input_array(run_starts_arr);
    enc.set_input_array(run_lengths_arr);

    int64_t x_bs = (x_batch_count == 1) ? 0 : static_cast<int64_t>(M) * K;

    // ---- WMMA path: use 16x16x16 wave matrix multiply when tiles align ----
    // WMMA tiles are 16x16; kernel handles non-aligned M with bounds masking.
    // N must be 16-aligned (typical for transformer hidden dimensions).
    // Gate on the device arch: a multi-arch build can compile this kernel
    // for a target whose gcnArchName isn't on the rocWMMA allowlist
    // (e.g. gfx1030/1103) — dispatching there would crash.
    bool use_wmma = d.has_native_wmma() && (M >= 2) && (N % 16 == 0) &&
        (bits_ == 4);
    use_wmma = parse_warp_kernel_env("MLX_ROCM_GATHER_QMV_USE_WMMA", use_wmma);

    if (use_wmma) {
      // One wave32 per 16x16 output tile
      dim3 wmma_block(32, 1, 1);
      dim3 wmma_grid((M + 15) / 16, (N + 15) / 16, num_runs);
      // Shared memory: smem_w[16*16] + smem_x[16*16] bf16 + smem_out_f32[16*16] f32
      // = 512 + 512 + 1024 = 2048 bytes
      size_t wmma_smem = 0;  // static shared memory, declared in-kernel

      enc.launch_kernel([&](hipStream_t stream) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_wmma_prefill_kernel<hip_bfloat16, hip_bfloat16, 4, 64, true>),
            wmma_grid, wmma_block, wmma_smem, stream,
            gpu_ptr<const hip_bfloat16>(x),
            gpu_ptr<const uint8_t>(w),
            gpu_ptr<const hip_bfloat16>(scales),
            has_bias ? gpu_ptr<const hip_bfloat16>(*biases) : nullptr,
            gpu_ptr<const uint32_t>(sorted_li_arr),
            gpu_ptr<const uint32_t>(sorted_ri_arr),
            gpu_ptr<const int>(run_starts_arr),
            gpu_ptr<const int>(run_lengths_arr),
            gpu_ptr<const int>(perm_arr),
            gpu_ptr<hip_bfloat16>(out),
            B, M, N, K, E, has_bias, x_bs);
      });
      return;
    }

    // ---- Scalar prefill fallback ----
    int fast_threads_per_col_pf = select_qmv_threads_per_col(K, N, bits_, num_runs);
    int fast_cols_per_block_pf = select_qmv_cols_per_block(K, N, bits_);
    int max_cpb = rocm::kMaxThreadsPerBlock / fast_threads_per_col_pf;
    while (fast_cols_per_block_pf > max_cpb) fast_cols_per_block_pf /= 2;
    while (fast_cols_per_block_pf > 1 && (N % fast_cols_per_block_pf) != 0 && fast_cols_per_block_pf > 8)
      fast_cols_per_block_pf /= 2;

    dim3 pf_block(fast_threads_per_col_pf, fast_cols_per_block_pf);
    dim3 pf_grid(M, (N + fast_cols_per_block_pf - 1) / fast_cols_per_block_pf, num_runs);

    enc.launch_kernel([&](hipStream_t stream) {
      auto launch_pf = [&](auto bits_tag) {
        constexpr int BITS = decltype(bits_tag)::value;
        hipLaunchKernelGGL(
            (rocm::gather_qmv_prefill_kernel<hip_bfloat16, hip_bfloat16, BITS, 64, true, 16>),
            pf_grid, pf_block, 0, stream,
            gpu_ptr<const hip_bfloat16>(x),
            gpu_ptr<const uint8_t>(w),
            gpu_ptr<const hip_bfloat16>(scales),
            has_bias ? gpu_ptr<const hip_bfloat16>(*biases) : nullptr,
            gpu_ptr<const uint32_t>(sorted_li_arr),
            gpu_ptr<const uint32_t>(sorted_ri_arr),
            gpu_ptr<const int>(run_starts_arr),
            gpu_ptr<const int>(run_lengths_arr),
            gpu_ptr<const int>(perm_arr),
            gpu_ptr<hip_bfloat16>(out),
            B, M, N, K, E, has_bias, x_bs);
      };
      if (bits_ == 4) launch_pf(std::integral_constant<int, 4>{});
      else launch_pf(std::integral_constant<int, 8>{});
    });
    return;
  }

  const void *x_ptr = gpu_ptr<void>(x), *w_ptr = gpu_ptr<void>(w),
             *scales_ptr = gpu_ptr<void>(scales),
             *biases_ptr = has_bias ? gpu_ptr<void>(*biases) : nullptr;
  const uint32_t *li_ptr = gpu_ptr<const uint32_t>(lhs_indices),
                 *ri_ptr = gpu_ptr<const uint32_t>(rhs_indices);
  void* out_ptr = gpu_ptr<void>(out);

  // GPU-only expert-batched kernel: when indices are sorted, each block finds
  // its expert's token range on-GPU and processes them together. Weight data
  // loaded once per expert column, reused across all tokens for that expert.
  // max_unique_experts = min(B, E) is an upper bound on unique experts.
  // Expert-batched kernel: beneficial when few experts have many tokens each.
  // For high-expert-count models (E=512, top_k=10), most runs have 1-4 tokens,
  // so the per-block run-finding overhead outweighs the shared weight benefit.
  // Enable only when B/E is high enough (e.g., low expert count with long prompt).
  bool use_expert_batched = transpose_ && right_sorted_ && (M == 1) &&
      (B >= 64) && (E > 0) && (E <= 64) && (B / E >= 4) &&
      mode_ == QuantizationMode::Affine &&
      x.dtype() == bfloat16 && group_size_ == 64 && (bits_ == 4 || bits_ == 8);
  use_expert_batched = parse_warp_kernel_env(
      "MLX_ROCM_GATHER_QMV_EXPERT_BATCHED", use_expert_batched);

  if (use_expert_batched) {
    int max_unique_experts = std::min(B, E);
    int eb_threads_per_col = select_qmv_threads_per_col(K, N, bits_, max_unique_experts);
    int eb_cols_per_block = select_qmv_cols_per_block(K, N, bits_);
    int eb_max_cpb = rocm::kMaxThreadsPerBlock / eb_threads_per_col;
    while (eb_cols_per_block > eb_max_cpb) eb_cols_per_block /= 2;
    while (eb_cols_per_block > 1 && (N % eb_cols_per_block) != 0 && eb_cols_per_block > 8)
      eb_cols_per_block /= 2;

    dim3 eb_block(eb_threads_per_col, eb_cols_per_block);
    dim3 eb_grid(M, (N + eb_cols_per_block - 1) / eb_cols_per_block, max_unique_experts);

    enc.launch_kernel([&](hipStream_t stream) {
      auto launch_eb = [&](auto bits_tag) {
        constexpr int BITS = decltype(bits_tag)::value;
        hipLaunchKernelGGL(
            (rocm::gather_qmv_expert_batched_kernel<
                hip_bfloat16, hip_bfloat16, BITS, 64, true, 16>),
            eb_grid, eb_block, 0, stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr, ri_ptr,
            (hip_bfloat16*)out_ptr,
            B, M, N, K, E, has_bias,
            use_sorted_rhs_schedule, implicit_x_batch_stride);
      };
      if (bits_ == 4) launch_eb(std::integral_constant<int, 4>{});
      else launch_eb(std::integral_constant<int, 8>{});
    });
    return;
  }

  // ---- Decode MoE: optional tiled gather-QMV (env-gated A/B) ----
  // The default warp-shared gather kernel launches very few blocks at decode
  // (grid z = B = tokens*top_k, small), starving GPU occupancy. The tiled gather
  // kernel (mirrors the fast 2D qmv_tiled path) also fans out over N tiles, for
  // far more blocks. It reads the word-packed (uint32) weight layout, so it is
  // restricted to 4/8-bit affine like the 2D tiled path, and to a unit-stride 1D
  // batch (the common MoE decode case). Opt in with
  // MLX_ROCM_GATHER_QMV_USE_TILED=1 to benchmark / enable.
  static const bool g_use_tiled_gather =
      (std::getenv("MLX_ROCM_GATHER_QMV_USE_TILED") != nullptr);
  bool gather_tiled_ok = g_use_tiled_gather && transpose_ &&
      mode_ == QuantizationMode::Affine && x.dtype() == bfloat16 &&
      group_size_ == 64 && (bits_ == 4 || bits_ == 8) &&
      batch_ndim == 1 && batch_strides[0].size() == 1 &&
      batch_strides[0][0] == 1 && batch_strides[1][0] == 1;
  int gather_tile_n = 0;
  if (gather_tiled_ok) {
    auto gqmm_hw = detect_rocm_hw_info(enc.device());
    gather_tile_n = rocm::get_arch_tuning(gqmm_hw).qmv_tile_n;
    while (gather_tile_n > 1 && (N % gather_tile_n) != 0) gather_tile_n /= 2;
    if (gather_tile_n < 1) gather_tile_n = 1;
  }

  enc.launch_kernel([&](hipStream_t stream) {
    if (gather_tiled_ok) {
      dim3 gt_grid(M, (N + gather_tile_n - 1) / gather_tile_n, B);
      dim3 gt_block(WARP_SIZE, gather_tile_n);
      int LHS_B = static_cast<int>(x_batch_count);
      auto launch_gt = [&](auto bits_tag) {
        constexpr int BITS = decltype(bits_tag)::value;
        hipLaunchKernelGGL(
            (rocm::gather_qmv_tiled_kernel<hip_bfloat16, hip_bfloat16, BITS, 64>),
            gt_grid, gt_block, 0, stream,
            (const hip_bfloat16*)x_ptr,
            (const uint32_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr, ri_ptr,
            (hip_bfloat16*)out_ptr,
            B, M, N, K, E, LHS_B, has_bias, gather_tile_n);
      };
      if (bits_ == 4) launch_gt(std::integral_constant<int, 4>{});
      else launch_gt(std::integral_constant<int, 8>{});
      return;
    }
    if (use_fast_gather_qmv && mode_ == QuantizationMode::Affine &&
        x.dtype() == bfloat16 && group_size_ == 64 &&
        (bits_ == 4 || bits_ == 6 || bits_ == 8)) {
      auto launch_fast_kernel = [&](auto bits_tag) {
        constexpr int BITS = decltype(bits_tag)::value;
        if (fast_threads_per_col == 16) {
          hipLaunchKernelGGL(
              (rocm::gather_qmv_warp_shared_kernel<
                  hip_bfloat16,
                  hip_bfloat16,
                  BITS,
                  64,
                  true,
                  16>),
              fast_grid,
              fast_block,
              0,
              stream,
              (const hip_bfloat16*)x_ptr,
              (const uint8_t*)w_ptr,
              (const hip_bfloat16*)scales_ptr,
              (const hip_bfloat16*)biases_ptr,
              li_ptr,
              ri_ptr,
              batch_shape_param,
              lhs_idx_strides_param,
              rhs_idx_strides_param,
              batch_ndim,
              (hip_bfloat16*)out_ptr,
              B,
              M,
              N,
              K,
              E,
              has_bias,
              use_sorted_rhs_schedule,
              implicit_x_batch_stride);
        } else {
          hipLaunchKernelGGL(
              (rocm::gather_qmv_warp_shared_kernel<
                  hip_bfloat16,
                  hip_bfloat16,
                  BITS,
                  64,
                  true,
                  WARP_SIZE>),
              fast_grid,
              fast_block,
              0,
              stream,
              (const hip_bfloat16*)x_ptr,
              (const uint8_t*)w_ptr,
              (const hip_bfloat16*)scales_ptr,
              (const hip_bfloat16*)biases_ptr,
              li_ptr,
              ri_ptr,
              batch_shape_param,
              lhs_idx_strides_param,
              rhs_idx_strides_param,
              batch_ndim,
              (hip_bfloat16*)out_ptr,
              B,
              M,
              N,
              K,
              E,
              has_bias,
              use_sorted_rhs_schedule,
              implicit_x_batch_stride);
        }
      };

      if (bits_ == 4) {
        launch_fast_kernel(std::integral_constant<int, 4>{});
      } else if (bits_ == 6) {
        launch_fast_kernel(std::integral_constant<int, 6>{});
      } else {
        launch_fast_kernel(std::integral_constant<int, 8>{});
      }
      return;
    }

#define has_bias has_bias, use_sorted_rhs_schedule, implicit_x_batch_stride

    if (x.dtype() == float32) {
      if (bits_ == 8 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 8, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 8 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 8, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 8 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 8, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 5 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 5, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 5 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 5, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 5 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 5, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 6 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 6, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 6 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 6, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 6 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 6, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 4 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 4, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 4 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 4, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 4 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 4, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 2 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 2, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 2 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 2, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 2 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 2, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else {
        throw std::runtime_error(
            "Unsupported dtype/bits/group_size combination for float32: bits=" +
            std::to_string(bits_) + " gs=" + std::to_string(group_size_));
      }
    } else if (x.dtype() == float16) {
      if (bits_ == 8 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 8, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 8 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 8, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 8 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 8, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 5 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 5, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 5 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 5, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 5 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 5, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 6 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 6, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 6 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 6, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 6 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 6, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 4 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 4, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 4 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 4, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 4 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 4, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 2 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 2, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 2 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 2, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 2 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 2, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else {
        throw std::runtime_error(
            "Unsupported dtype/bits/group_size combination for float16: bits=" +
            std::to_string(bits_) + " gs=" + std::to_string(group_size_));
      }
    } else if (x.dtype() == bfloat16) {
      if (bits_ == 8 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 8, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 8 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 8, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 8 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 8, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 5 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 5, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 5 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 5, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 5 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 5, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 6 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 6, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 6 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 6, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 6 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 6, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 4 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 4, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 4 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 4, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 4 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 4, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 2 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 2, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 2 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 2, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 2 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 2, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else {
        throw std::runtime_error(
            "Unsupported dtype/bits/group_size combination for bfloat16: bits=" +
            std::to_string(bits_) + " gs=" + std::to_string(group_size_));
      }
    }

#undef has_bias
  });
}

} // namespace mlx::core
