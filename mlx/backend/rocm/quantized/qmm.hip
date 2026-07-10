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
#include <hip/hip_fp8.h>
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
    out.set_data(mlx::core::rocm::malloc_async(out.nbytes(), enc));
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

      enc.set_input_array(x);
      enc.set_output_array(x_copy);
      enc.add_kernel_node(
          &rocm::strided_row_copy_kernel,
          dim3(num_blocks), dim3(block_size), 0u,
          src, dst,
          num_rows, cols_bytes,
          src_row_stride_bytes, dst_row_stride_bytes,
          word_copy);
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

        enc.set_input_array(x);
        enc.set_output_array(x_copy);
        enc.add_kernel_node(
            &rocm::strided_row_copy_kernel,
            dim3(num_blocks), dim3(block_size), 0u,
            src, dst,
            slab_rows, cols_bytes,
            src_row_stride_bytes, dst_row_stride_bytes,
            word_copy);
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

    enc.set_input_array(x);
    enc.set_output_array(x_copy);
    enc.add_kernel_node(
        &rocm::strided_general_copy_kernel,
        dim3(num_blocks), dim3(block_size), 0u,
        src, dst,
        total_elems, eb, ndim,
        shapes_arg, strides_bytes_arg);
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

      static const char* const kWmmaArches[] = {
          "gfx908",  "gfx90a",  "gfx942",
          "gfx1100", "gfx1101", "gfx1102",
          "gfx1151", "gfx1152",
          "gfx1200", "gfx1201",
      };
      for (const char* a : kWmmaArches) {
        if (std::strstr(arch_name, a) != nullptr) {
          hw.has_native_wmma = true;
          break;
        }
      }
    }
    fprintf(stderr, "[mlx-rocm] %s native_wmma=%d cus=%d\n",
            arch_name ? arch_name : "?", (int)hw.has_native_wmma, hw.num_cus);
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

// --- fp8 (e4m3) GEMM path for RDNA4 -----------------------------------------
// The half-precision activation and dequantized weight are cast to e4m3 with a
// per-tensor absmax scale, multiplied on fp8 matrix cores (~1.5-2.4x bf16), and
// descaled back to bf16 by hipBLASLt. fp8 buffers are raw bytes (no MLX dtype).

constexpr float kE4M3Max = 448.0f;

template <typename T>
__global__ void fp8_absmax_kernel(
    const T* __restrict__ src, size_t n, float* __restrict__ amax) {
  __shared__ float sm[256];
  float local = 0.0f;
  for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
       j += static_cast<size_t>(gridDim.x) * blockDim.x) {
    local = fmaxf(local, fabsf(static_cast<float>(src[j])));
  }
  sm[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + s]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    // Non-negative floats compare monotonically as ints.
    atomicMax(reinterpret_cast<int*>(amax), __float_as_int(sm[0]));
  }
}

// Writes the descale factor (amax/448) hipBLASLt multiplies back into the
// output. Guards the all-zero case.
__global__ void fp8_descale_kernel(
    const float* __restrict__ amax, float* __restrict__ descale) {
  float a = *amax;
  *descale = (a > 0.0f) ? (a / kE4M3Max) : 1.0f;
}

template <typename T>
__global__ void fp8_cast_kernel(
    const T* __restrict__ src,
    size_t n,
    const float* __restrict__ amax,
    __hip_fp8_e4m3* __restrict__ dst) {
  float a = *amax;
  float inv = (a > 0.0f) ? (kE4M3Max / a) : 0.0f;
  for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
       j += static_cast<size_t>(gridDim.x) * blockDim.x) {
    dst[j] = __hip_fp8_e4m3(static_cast<float>(src[j]) * inv);
  }
}

template <typename T>
void fp8_quantize(
    hipStream_t stream,
    const T* src,
    size_t n,
    float* amax,
    float* descale,
    __hip_fp8_e4m3* dst) {
  int threads = 256;
  int blocks = static_cast<int>(std::min<size_t>((n + threads - 1) / threads, 4096));
  fp8_absmax_kernel<T><<<blocks, threads, 0, stream>>>(src, n, amax);
  fp8_descale_kernel<<<1, 1, 0, stream>>>(amax, descale);
  fp8_cast_kernel<T><<<blocks, threads, 0, stream>>>(src, n, amax, dst);
}

// Per-tensor absmax of the dequantized affine weight, computed from the quant
// params alone (dequant values are linear in q, so the extrema are at q=0 and
// q=qmax). Reads only scales/biases (group_size x fewer elements than the
// weight) — no full-weight pass.
template <typename ScaleT, int BITS>
__global__ void weight_absmax_kernel(
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    int num_groups,
    float* __restrict__ absmax) {
  __shared__ float sm[256];
  float local = 0.0f;
  constexpr int qmax = (1 << BITS) - 1;
  for (int g = blockIdx.x * blockDim.x + threadIdx.x; g < num_groups;
       g += static_cast<int>(gridDim.x) * blockDim.x) {
    float s = static_cast<float>(scales[g]);
    float b = biases ? static_cast<float>(biases[g]) : 0.0f;
    local = fmaxf(local, fmaxf(fabsf(b), fabsf(s * qmax + b)));
  }
  sm[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + s]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicMax(reinterpret_cast<int*>(absmax), __float_as_int(sm[0]));
  }
}

// Dequantize packed affine weights straight to e4m3 (no bf16 intermediate),
// scaled by 448/absmax. No ROCm/HIP library consumes packed quant, so this is
// the necessary hand-rolled step.
template <typename ScaleT, int BITS>
__global__ void dequant_to_e4m3_kernel(
    const uint8_t* __restrict__ input,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    __hip_fp8_e4m3* __restrict__ output,
    int num_groups,
    int group_size,
    const float* __restrict__ absmax) {
  int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (group_idx >= num_groups)
    return;
  float inv = (*absmax > 0.0f) ? (kE4M3Max / *absmax) : 0.0f;
  float scale = static_cast<float>(scales[group_idx]);
  float bias = biases ? static_cast<float>(biases[group_idx]) : 0.0f;
  int input_base = group_idx * (group_size * BITS / 8);
  __hip_fp8_e4m3* group_output = output + group_idx * group_size;
  constexpr uint8_t mask = static_cast<uint8_t>((1u << BITS) - 1u);
  for (int i = 0; i < group_size; ++i) {
    int bit_index = i * BITS;
    int byte_idx = input_base + (bit_index >> 3);
    int bit_offset = bit_index & 0x7;
    uint32_t packed = static_cast<uint32_t>(input[byte_idx]);
    if (bit_offset + BITS > 8) {
      packed |= static_cast<uint32_t>(input[byte_idx + 1]) << 8;
    }
    int qv = static_cast<int>((packed >> bit_offset) & mask);
    float dq = static_cast<float>(qv) * scale + bias;
    group_output[i] = __hip_fp8_e4m3(dq * inv);
  }
}

// out[M,N] = x[M,K] @ w^T via e4m3 hipBLASLt. The weight is dequantized
// straight from packed quant to e4m3; the activation is cast to e4m3; hipBLASLt
// descales both back to bf16. transpose mirrors the dequant_rocblas_gemm
// convention (transpose_a=false, transpose_b=transpose).
void dequant_fp8_gemm(
    rocm::CommandEncoder& enc,
    bool transpose,
    int M,
    int N,
    int K,
    const array& x,
    const array& wq,
    const array& scales,
    const std::optional<array>& biases,
    int group_size,
    int bits,
    array& out,
    int dq_rows,
    int dq_cols) {
  array x_fp8(Shape{M, K}, uint8, nullptr, {});
  x_fp8.set_data(mlx::core::rocm::malloc_async(x_fp8.nbytes(), enc));
  array w_fp8(Shape{dq_rows, dq_cols}, uint8, nullptr, {});
  w_fp8.set_data(mlx::core::rocm::malloc_async(w_fp8.nbytes(), enc));
  array scratch(Shape{4}, float32, nullptr, {});
  scratch.set_data(mlx::core::rocm::malloc_async(scratch.nbytes(), enc));
  enc.add_temporary(x_fp8);
  enc.add_temporary(w_fp8);
  enc.add_temporary(scratch);

  enc.set_input_array(x);
  enc.set_input_array(wq);
  enc.set_input_array(scales);
  if (biases)
    enc.set_input_array(*biases);
  enc.set_output_array(out);

  rocblas_operation op_a = to_rocblas_op(false);
  rocblas_operation op_b = to_rocblas_op(transpose);
  int lda = K;
  int ldb = transpose ? K : N;
  Dtype xdt = x.dtype();
  Dtype sdt = scales.dtype();
  bool has_bias = biases.has_value();
  size_t nw = static_cast<size_t>(dq_rows) * dq_cols;
  int wgroups = static_cast<int>(nw / group_size);

  enc.launch_kernel([=, &enc](hipStream_t stream) {
    const void* xp = gpu_ptr<const void>(x);
    const void* wqp = gpu_ptr<const void>(wq);
    const void* sp = gpu_ptr<const void>(scales);
    const void* bp = has_bias ? gpu_ptr<const void>(*biases) : nullptr;
    void* op = gpu_ptr<void>(out);
    float* amax_x = reinterpret_cast<float*>(gpu_ptr<void>(scratch));
    float* desc_x = amax_x + 1;
    float* amax_w = amax_x + 2;
    float* desc_w = amax_x + 3;
    auto* xf = reinterpret_cast<__hip_fp8_e4m3*>(gpu_ptr<void>(x_fp8));
    auto* wf = reinterpret_cast<__hip_fp8_e4m3*>(gpu_ptr<void>(w_fp8));
    size_t nx = static_cast<size_t>(M) * K;
    (void)hipMemsetAsync(amax_x, 0, sizeof(float), stream);
    (void)hipMemsetAsync(amax_w, 0, sizeof(float), stream);

    if (xdt == bfloat16) {
      fp8_quantize<__hip_bfloat16>(
          stream, static_cast<const __hip_bfloat16*>(xp), nx, amax_x, desc_x, xf);
    } else {
      fp8_quantize<__half>(
          stream, static_cast<const __half*>(xp), nx, amax_x, desc_x, xf);
    }

    int threads = 256;
    int blocks = std::max(1, (wgroups + threads - 1) / threads);
    int absblocks = std::min(blocks, 4096);
#define LAUNCH_DEQUANT_E4M3(ScaleT, BITS)                                   \
  do {                                                                      \
    weight_absmax_kernel<ScaleT, BITS><<<absblocks, threads, 0, stream>>>(  \
        static_cast<const ScaleT*>(sp),                                     \
        static_cast<const ScaleT*>(bp),                                     \
        wgroups,                                                            \
        amax_w);                                                            \
    fp8_descale_kernel<<<1, 1, 0, stream>>>(amax_w, desc_w);                \
    dequant_to_e4m3_kernel<ScaleT, BITS><<<blocks, threads, 0, stream>>>(   \
        static_cast<const uint8_t*>(wqp),                                   \
        static_cast<const ScaleT*>(sp),                                     \
        static_cast<const ScaleT*>(bp),                                     \
        wf,                                                                 \
        wgroups,                                                            \
        group_size,                                                         \
        amax_w);                                                            \
  } while (0)
#define DISPATCH_BITS_E4M3(ScaleT)                  \
  switch (bits) {                                   \
    case 2:                                         \
      LAUNCH_DEQUANT_E4M3(ScaleT, 2);               \
      break;                                        \
    case 4:                                         \
      LAUNCH_DEQUANT_E4M3(ScaleT, 4);               \
      break;                                        \
    case 5:                                         \
      LAUNCH_DEQUANT_E4M3(ScaleT, 5);               \
      break;                                        \
    case 6:                                         \
      LAUNCH_DEQUANT_E4M3(ScaleT, 6);               \
      break;                                        \
    case 8:                                         \
      LAUNCH_DEQUANT_E4M3(ScaleT, 8);               \
      break;                                        \
    default:                                        \
      break;                                        \
  }
    if (sdt == bfloat16) {
      DISPATCH_BITS_E4M3(__hip_bfloat16);
    } else if (sdt == float16) {
      DISPATCH_BITS_E4M3(__half);
    } else {
      DISPATCH_BITS_E4M3(float);
    }
#undef DISPATCH_BITS_E4M3
#undef LAUNCH_DEQUANT_E4M3

    // Column-major swap: A<-w, B<-x, M<->N (same as the bf16 dequant path).
    rocm::hipblaslt_gemm_fp8_raw(
        stream,
        static_cast<int>(op_b),
        static_cast<int>(op_a),
        N,
        M,
        K,
        wf,
        ldb,
        xf,
        lda,
        op,
        N,
        desc_w,
        desc_x);
  });
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

    // Prefer hipBLASLt for all supported dtypes. rocBLAS/Tensile ships
    // incomplete gfx1201 coverage (missing large-tile kernels such as
    // MT128x128x32) and faults on large prefill GEMMs; hipBLASLt covers them.
    if (rocm::is_hipblaslt_available() &&
        (dtype == float32 || dtype == float16 || dtype == bfloat16)) {
      int dt_hint = (dtype == float16) ? 1 : (dtype == bfloat16) ? 2 : 3;
      float alpha_f = alpha;
      float beta_f = beta;
      try {
        rocm::hipblaslt_gemm_raw(
            stream,
            static_cast<int>(op_b), static_cast<int>(op_a),
            N, M, K,
            &alpha_f, b_ptr, ldb, a_ptr, lda,
            &beta_f, c_ptr, ldc,
            dt_hint, 0);
        return;
      } catch (...) {
        // Fall through to rocBLAS below.
      }
    }

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

// ---------------------------------------------------------------------------
// int8 integer-dot QMV (4-bit affine). Activation is quantized to int8
// (per-group scale dx + exact group-sum Sx) in a prepass; weight nibbles are
// fed straight to v_dot4_i32_iu8. Affine dequant: scale*dx*<q_int> + bias*Sx.
// Gated by MLX_QMV_IDOT. Mirrors llama.cpp mmvq (integer dot, not matrix core).
// ---------------------------------------------------------------------------
__device__ __forceinline__ int sudot4_iu8(int wcodes, int xcodes, int acc) {
  int _s = acc;
#pragma unroll
  for (int _i = 0; _i < 4; ++_i) {
    unsigned int _w = (unsigned int)((wcodes >> (8 * _i)) & 0xFF);
    int _x = (int)(signed char)((xcodes >> (8 * _i)) & 0xFF);
    _s += (int)_w * _x;
  }
  return _s;
}

template <typename T, int GROUP_SIZE>
__global__ void quantize_act_q8_kernel(
    const T* __restrict__ x,
    int32_t* __restrict__ xpack,
    float* __restrict__ dx,
    float* __restrict__ Sx,
    int M,
    int K) {
  int row = blockIdx.x;
  if (row >= M)
    return;
  int ng = K / GROUP_SIZE;
  int nwords = K / 8;
  const T* xr = x + static_cast<size_t>(row) * K;
  int tid = threadIdx.x, nt = blockDim.x;
  for (int g = tid; g < ng; g += nt) {
    float mx = 0.0f, sm = 0.0f;
    int base = g * GROUP_SIZE;
    for (int j = 0; j < GROUP_SIZE; ++j) {
      float v = static_cast<float>(xr[base + j]);
      mx = fmaxf(mx, fabsf(v));
      sm += v;
    }
    float d = mx / 127.0f;
    if (d == 0.0f)
      d = 1.0f;
    dx[static_cast<size_t>(row) * ng + g] = d;
    Sx[static_cast<size_t>(row) * ng + g] = sm;
  }
  __syncthreads();
  for (int wi = tid; wi < nwords; wi += nt) {
    int base = wi * 8;
    int g = base / GROUP_SIZE;
    float inv = 1.0f / dx[static_cast<size_t>(row) * ng + g];
    int8_t q[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int qq = __float2int_rn(static_cast<float>(xr[base + j]) * inv);
      qq = qq < -127 ? -127 : (qq > 127 ? 127 : qq);
      q[j] = static_cast<int8_t>(qq);
    }
    int even = static_cast<uint8_t>(q[0]) | (static_cast<uint8_t>(q[2]) << 8) |
        (static_cast<uint8_t>(q[4]) << 16) | (static_cast<uint8_t>(q[6]) << 24);
    int odd = static_cast<uint8_t>(q[1]) | (static_cast<uint8_t>(q[3]) << 8) |
        (static_cast<uint8_t>(q[5]) << 16) | (static_cast<uint8_t>(q[7]) << 24);
    size_t o = (static_cast<size_t>(row) * nwords + wi) * 2;
    xpack[o] = even;
    xpack[o + 1] = odd;
  }
}

template <typename T, typename ScaleT, int GROUP_SIZE>
__global__ void qmv_idot_kernel(
    const int32_t* __restrict__ xpack,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    const float* __restrict__ dx,
    const float* __restrict__ Sx,
    T* __restrict__ out,
    int M,
    int N,
    int K,
    bool has_bias) {
  int lane = threadIdx.x;
  int col = blockIdx.x * blockDim.y + threadIdx.y;
  int row = blockIdx.y;
  if (row >= M || col >= N)
    return;
  int ng = K / GROUP_SIZE;
  int nwords = K / 8;
  int row_bytes = (K + 1) / 2;
  const uint32_t* wr =
      reinterpret_cast<const uint32_t*>(w + static_cast<size_t>(col) * row_bytes);
  const int32_t* xr = xpack + static_cast<size_t>(row) * nwords * 2;
  float acc = 0.0f;
  for (int wi = lane; wi < nwords; wi += 32) {
    uint32_t vui = wr[wi];
    int ew = vui & 0x0F0F0F0F;
    int ow = (vui >> 4) & 0x0F0F0F0F;
    int q = sudot4_iu8(ew, xr[2 * wi], 0);
    q = sudot4_iu8(ow, xr[2 * wi + 1], q);
    int g = (wi * 8) / GROUP_SIZE;
    float scale =
        load_scale_value<ScaleT, GROUP_SIZE, true>(scales[static_cast<size_t>(col) * ng + g]);
    acc += scale * dx[static_cast<size_t>(row) * ng + g] * static_cast<float>(q);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    acc += __shfl_down(acc, o);
  float bsum = 0.0f;
  if (has_bias) {
    for (int g = lane; g < ng; g += 32)
      bsum += static_cast<float>(biases[static_cast<size_t>(col) * ng + g]) *
          Sx[static_cast<size_t>(row) * ng + g];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      bsum += __shfl_down(bsum, o);
  }
  if (lane == 0)
    out[static_cast<size_t>(row) * N + col] = static_cast<T>(acc + bsum);
}

// Dense wide-load QMV (4-bit affine, decode). One warp per output column, flat
// full-K loop with 128-bit (uint4 = 32 nibbles) weight loads. Microbench
// sustains ~175 GB/s on real 8MB matmuls vs qmv_tiled's ~108 (kernel-bound, not
// size/interleaving). Activation from global (L2-cached). Default on; opt out
// with MLX_QMV_NO_WIDE. Requires K % 32 == 0; GROUP_SIZE in {32,64,128}.
template <typename T, typename ScaleT, int GROUP_SIZE, int BITS>
__global__ void __launch_bounds__(1024) qmv_wide_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ out,
    int M,
    int N,
    int K,
    bool has_bias) {
  constexpr int WPER = 128 / BITS;
  int lane = threadIdx.x;
  int col = blockIdx.x * blockDim.y + threadIdx.y;
  int row = blockIdx.y;
  if (row >= M || col >= N)
    return;
  int ng = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  int row_bytes = (K * BITS) / 8;
  const uint8_t* wr = w + static_cast<size_t>(col) * row_bytes;
  const ScaleT* scr = scales + static_cast<size_t>(col) * ng;
  const ScaleT* bcr = has_bias ? (biases + static_cast<size_t>(col) * ng) : nullptr;
  const T* xr = x + static_cast<size_t>(row) * K;
  float acc = 0.0f;
  for (int k = lane * WPER; k < K; k += 32 * WPER) {
    int g = k / GROUP_SIZE;
    float s = load_scale_value<ScaleT, GROUP_SIZE, true>(scr[g]);
    float b = has_bias ? static_cast<float>(bcr[g]) : 0.0f;
    uint4 wv = *reinterpret_cast<const uint4*>(&wr[(k * BITS) / 8]);
    const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv);
    float la = 0.0f, lx = 0.0f;
    if constexpr (BITS == 4) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        uint32_t wd = wp[q];
        int kb = k + q * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float xv = static_cast<float>(xr[kb + j]);
          la += xv * static_cast<float>((wd >> (4 * j)) & 0xF);
          if (has_bias)
            lx += xv;
        }
      }
    } else {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        uint32_t wd = wp[q];
        int kb = k + q * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float xv = static_cast<float>(xr[kb + j]);
          la += xv * static_cast<float>((wd >> (8 * j)) & 0xFF);
          if (has_bias)
            lx += xv;
        }
      }
    }
    acc += s * la + b * lx;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    acc += __shfl_down(acc, o);
  if (lane == 0)
    out[static_cast<size_t>(row) * N + col] = static_cast<T>(acc);
}

// Dense fused WMMA quantized GEMM (q4/q6/q8 affine), graph-addable.
//   out[M,N] = x[M,K] @ dequant(w)[K,N], w stored [N, K*BITS/32] packed (transpose).
// Unpacks packed weights in-kernel (128-bit loads, bit-level) — no bf16 weight
// materialization (vs the dequant+rocBLAS path), so it both runs as a single
// add_kernel_node graph node (no graph fragmentation) and avoids the dequant
// memory traffic. 4 warps/block share the gathered x tile; grid-strides over M.
template <typename T, typename ScaleT, int BITS, int GROUP_SIZE, bool HAS_BIAS>
__global__ void __launch_bounds__(128) qmm_wmma_dense_kernel(
    const T* __restrict__ x,
    const uint32_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ out,
    int M, int N, int K) {
#if ROCM_HAS_WMMA
  constexpr int WM = 16, WN = 16, WK = 16, MT = 4, NT = 2, NWARP = 4;
  constexpr int BM = WM * MT;          // 64 token rows / block
  constexpr int BN_W = WN * NT;        // 32 cols / warp
  constexpr int BN = BN_W * NWARP;     // 128 cols / block
  constexpr int BK = GROUP_SIZE;       // one quant group per K-tile (=64)
  constexpr int CH = BK / 8;           // uint4 chunks per gathered x row (bf16)

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int n0 = blockIdx.y * BN, wn0 = n0 + warp * BN_W;
  if (n0 >= N) return;
  const int Kp = K * BITS / 32;        // uint32 per weight row
  const int Kg = K / GROUP_SIZE;       // quant groups per row

  __shared__ hip_bfloat16 sX[BM][BK];
  __shared__ hip_bfloat16 sB[BK][BN];
  __shared__ float sC[BM][BN];
  using frag_a   = rocwmma::fragment<rocwmma::matrix_a,   WM, WN, WK, hip_bfloat16, rocwmma::row_major>;
  using frag_b   = rocwmma::fragment<rocwmma::matrix_b,   WM, WN, WK, hip_bfloat16, rocwmma::row_major>;
  using frag_acc = rocwmma::fragment<rocwmma::accumulator, WM, WN, WK, float>;

  for (int m0 = blockIdx.x * BM; m0 < M; m0 += gridDim.x * BM) {
    frag_acc cf[MT][NT];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
      #pragma unroll
      for (int t = 0; t < NT; ++t) rocwmma::fill_fragment(cf[mt][t], 0.0f);

    for (int kk = 0; kk < K; kk += BK) {
      // gather x tile: BM contiguous rows x BK cols, 128-bit loads
      for (int idx = tid; idx < BM * CH; idx += NWARP * 32) {
        const int r = idx / CH, ch = idx % CH, row = m0 + r;
        uint4* dst = reinterpret_cast<uint4*>(&sX[r][ch * 8]);
        if (row < M)
          *dst = *reinterpret_cast<const uint4*>(&x[static_cast<int64_t>(row) * K + kk + ch * 8]);
        else
          *dst = uint4{0, 0, 0, 0};
      }
      // dequantize this warp's 32-col slab (one column per lane), BITS-generic
      {
        const int col = warp * BN_W + lane, n = wn0 + lane;
        if (n < N) {
          const int bg = kk / GROUP_SIZE;
          const float scl = static_cast<float>(scales[n * Kg + bg]);
          const float bia = HAS_BIAS ? static_cast<float>(biases[n * Kg + bg]) : 0.0f;
          const uint32_t* wbase = &w[static_cast<int64_t>(n) * Kp + (kk * BITS) / 32];
          if constexpr (BITS == 6) {
            uint32_t wl[BK * 6 / 32];
            #pragma unroll
            for (int u = 0; u < (BK * 6 / 128); ++u)
              reinterpret_cast<uint4*>(wl)[u] =
                  *reinterpret_cast<const uint4*>(wbase + u * 4);
            #pragma unroll
            for (int k = 0; k < BK; ++k) {
              const int b = k * 6, idx2 = b >> 5, off = b & 31;
              const uint64_t w64 = uint64_t(wl[idx2]) |
                  (uint64_t((idx2 + 1 < BK * 6 / 32) ? wl[idx2 + 1] : 0u) << 32);
              const uint32_t q = uint32_t(w64 >> off) & 0x3Fu;
              sB[k][col] = static_cast<hip_bfloat16>(float(q) * scl + bia);
            }
          } else {
            constexpr int WPER = 32 / BITS;
            constexpr int U4S = (BK * BITS) / 128;
            constexpr uint32_t QM = (1u << BITS) - 1u;
            #pragma unroll
            for (int u = 0; u < U4S; ++u) {
              const uint4 wv = *reinterpret_cast<const uint4*>(wbase + u * 4);
              const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv);
              #pragma unroll
              for (int q = 0; q < 4; ++q) {
                const uint32_t wd = wp[q];
                #pragma unroll
                for (int j = 0; j < WPER; ++j) {
                  const int k = (u * 4 + q) * WPER + j;
                  sB[k][col] = static_cast<hip_bfloat16>(
                      float((wd >> (BITS * j)) & QM) * scl + bia);
                }
              }
            }
          }
        } else {
          #pragma unroll
          for (int k = 0; k < BK; ++k) sB[k][col] = static_cast<hip_bfloat16>(0.0f);
        }
      }
      __syncthreads();
      #pragma unroll
      for (int ks = 0; ks < BK; ks += WK) {
        frag_a af[MT];
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          rocwmma::load_matrix_sync(af[mt], &sX[mt * WM][ks], BK);
        #pragma unroll
        for (int t = 0; t < NT; ++t) {
          frag_b bf;
          rocwmma::load_matrix_sync(bf, &sB[ks][warp * BN_W + t * WN], BN);
          #pragma unroll
          for (int mt = 0; mt < MT; ++mt)
            rocwmma::mma_sync(cf[mt][t], af[mt], bf, cf[mt][t]);
        }
      }
      __syncthreads();
    }
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
      #pragma unroll
      for (int t = 0; t < NT; ++t)
        rocwmma::store_matrix_sync(&sC[mt * WM][warp * BN_W + t * WN], cf[mt][t], BN,
                                   rocwmma::mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < BM * BN; idx += NWARP * 32) {
      const int r = idx / BN, c = idx % BN, row = m0 + r;
      if (row < M && (n0 + c) < N)
        out[static_cast<int64_t>(row) * N + n0 + c] = static_cast<T>(sC[r][c]);
    }
    __syncthreads();
  }
#endif // ROCM_HAS_WMMA
}

} // namespace rocm

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = rocm::device(s.device);
  auto& enc = d.get_command_encoder(s);

  out.set_data(mlx::core::rocm::malloc_async(out.nbytes(), enc));

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

  // Fused WMMA quantized GEMM (q4/q6/q8): graph-addable single node, no dequant
  // materialization. Default ON where the device has native WMMA (opt out with
  // MLX_ROCM_WMMA_QMM=0); dispatch below falls to rocBLAS when WMMA is absent or
  // the shape doesn't qualify. Single (non-batched) bf16 affine prefill.
  {
    static const bool wmma_qmm_off = [] {
      const char* e = std::getenv("MLX_ROCM_WMMA_QMM");
      return e && e[0] == '0';
    }();
    const bool want_wmma = !wmma_qmm_off;
    if (want_wmma && transpose_ && mode_ == QuantizationMode::Affine &&
        x.dtype() == bfloat16 && scales.dtype() == bfloat16 &&
        (!biases.has_value() || biases->dtype() == bfloat16) &&
        batch_count == 1 && x_batch_count == 1 && w_batch_count == 1 && M > 1 &&
        (N % 16 == 0) && group_size_ == 64 && (K % 64 == 0) &&
        (bits_ == 4 || bits_ == 6 || bits_ == 8) && d.has_native_wmma()) {
      dim3 block(128, 1, 1);
      dim3 grid((M + 63) / 64, (N + 127) / 128, 1);
      const bool hb = biases.has_value();
      auto launch = [&](auto bits_tag, auto hb_tag) {
        constexpr int BITS = decltype(bits_tag)::value;
        constexpr bool HB = decltype(hb_tag)::value;
        enc.add_kernel_node(
            &rocm::qmm_wmma_dense_kernel<hip_bfloat16, hip_bfloat16, BITS, 64, HB>,
            grid, block, 0u,
            gpu_ptr<const hip_bfloat16>(x),
            gpu_ptr<const uint32_t>(w),
            gpu_ptr<const hip_bfloat16>(scales),
            static_cast<const hip_bfloat16*>(
                hb ? gpu_ptr<const hip_bfloat16>(*biases) : nullptr),
            gpu_ptr<hip_bfloat16>(out),
            M, N, K);
      };
      auto by_bias = [&](auto bits_tag) {
        if (hb) launch(bits_tag, std::true_type{});
        else launch(bits_tag, std::false_type{});
      };
      if (bits_ == 4) by_bias(std::integral_constant<int, 4>{});
      else if (bits_ == 6) by_bias(std::integral_constant<int, 6>{});
      else by_bias(std::integral_constant<int, 8>{});
      return;
    }
  }

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

    // fp8 e4m3 path (RDNA4 prefill): dequantize the weight straight to e4m3 and
    // cast the activation, then run the GEMM on fp8 matrix cores. Capability-
    // gated — devices without e4m3 kernels stay on the bf16 dequant path below.
    if ((mode_ == QuantizationMode::Affine) && (x.dtype() == bfloat16) &&
        (batch_count == 1) && (x_batch_count == 1) && (w_batch_count == 1) &&
        (M >= 64) && rocm::device_has_fp8_gemm(d.hip_device())) {
      dequant_fp8_gemm(
          enc,
          transpose_,
          M,
          N,
          K,
          x,
          w,
          scales,
          biases,
          group_size_,
          bits_,
          out,
          dequant_rows,
          dequant_cols);
      return;
    }

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
        w_dequant.set_data(mlx::core::rocm::malloc_async(w_dequant.nbytes(), enc));

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
      w_dequant.set_data(mlx::core::rocm::malloc_async(w_dequant.nbytes(), enc));

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

  // Full-wave tiled 6-bit QMV; MLX_ROCM_QMV_6BIT_SLOW reverts to warp_shared.
  static bool use_6bit_tiled =
      (std::getenv("MLX_ROCM_QMV_6BIT_SLOW") == nullptr);
  // GROUP_SIZE must be a multiple of 16 (the per-lane value count) so each
  // lane's 16 weights fall in a single group → one scale/bias per lane, as the
  // tiled accumulation assumes. 32/64/128 all satisfy this.
  bool gs6_supported =
      (group_size_ == 32 || group_size_ == 64 || group_size_ == 128);
  bool x6_dtype_supported =
      (x.dtype() == bfloat16 || x.dtype() == float16);
  if (use_6bit_tiled && use_tiled && bits_ == 6 && (K % 64) == 0 &&
      gs6_supported && x6_dtype_supported && use_fast_qmv &&
      !can_use_batched_qmv && tile_n >= 8 &&
      mode_ == QuantizationMode::Affine) {
    {
      dim3 tiled_block(WARP_SIZE, tile_n);
      const int n_tiles = (N + tile_n - 1) / tile_n;
      int blocks_per_cu = (hw_info.max_threads_per_cu > 0)
          ? (hw_info.max_threads_per_cu / (tile_n * WARP_SIZE)) : 4;
      if (blocks_per_cu < 1) blocks_per_cu = 1;
      int persistent_y =
          (hw_info.num_cus > 0) ? hw_info.num_cus * blocks_per_cu : n_tiles;
      int grid_y = (n_tiles < persistent_y) ? n_tiles : persistent_y;
      if (grid_y < 1) grid_y = 1;
      dim3 tiled_grid(M, grid_y);

      #define LAUNCH_TILED_6BIT(T, ScaleT, GS_V) \
        enc.add_kernel_node( \
            &rocm::qmv_tiled_6bit_kernel<T, ScaleT, GS_V>, \
            tiled_grid, tiled_block, 0u, \
            (const T*)x_ptr, (const uint8_t*)w_ptr, \
            (const ScaleT*)scales_ptr, (const ScaleT*)biases_ptr, \
            (T*)out_ptr, M, N, K, has_bias, tile_n, n_tiles)

      if (x.dtype() == bfloat16) {
        if (group_size_ == 32)  { LAUNCH_TILED_6BIT(hip_bfloat16, hip_bfloat16, 32); }
        else if (group_size_ == 64)  { LAUNCH_TILED_6BIT(hip_bfloat16, hip_bfloat16, 64); }
        else if (group_size_ == 128) { LAUNCH_TILED_6BIT(hip_bfloat16, hip_bfloat16, 128); }
      } else if (x.dtype() == float16) {
        if (group_size_ == 32)  { LAUNCH_TILED_6BIT(__half, __half, 32); }
        else if (group_size_ == 64)  { LAUNCH_TILED_6BIT(__half, __half, 64); }
        else if (group_size_ == 128) { LAUNCH_TILED_6BIT(__half, __half, 128); }
      }
      #undef LAUNCH_TILED_6BIT
    }
    return;
  }

  // Dense wide-load (128-bit / uint4) QMV decode path (4-bit affine). DEFAULT ON
  // (opt out with MLX_QMV_NO_WIDE): +14-20% on dense decode, never regresses.
  // Replaces qmv_tiled for decode; ~175 vs ~108 GB/s.
  static bool use_wide_qmv = (std::getenv("MLX_QMV_NO_WIDE") == nullptr);
  bool wide_ok = use_wide_qmv && (bits_ == 4 || bits_ == 8) &&
      mode_ == QuantizationMode::Affine &&
      has_bias && transpose_ && batch_count == 1 && non_batched && M >= 1 &&
      M <= 8 && (K % 32 == 0) &&
      (group_size_ == 32 || group_size_ == 64 || group_size_ == 128) &&
      (x.dtype() == bfloat16 || x.dtype() == float16);
  if (wide_ok) {
    int wcols = 8;
    dim3 wblock(32, wcols);
    dim3 wgrid((N + wcols - 1) / wcols, M);
#define LAUNCH_QMV_WIDE(TT, ST, GS, BV)                                      \
  enc.add_kernel_node(                                                       \
      &rocm::qmv_wide_kernel<TT, ST, GS, BV>, wgrid, wblock, 0u,             \
      (const TT*)x_ptr, w_ptr, (const ST*)scales_ptr, (const ST*)biases_ptr, \
      (TT*)out_ptr, M, N, K, has_bias)
#define LAUNCH_QMV_WIDE_GS(TT, ST, BV)                                       \
  do {                                                                       \
    if (group_size_ == 32) { LAUNCH_QMV_WIDE(TT, ST, 32, BV); }             \
    else if (group_size_ == 64) { LAUNCH_QMV_WIDE(TT, ST, 64, BV); }        \
    else { LAUNCH_QMV_WIDE(TT, ST, 128, BV); }                              \
  } while (0)
    if (x.dtype() == bfloat16) {
      if (bits_ == 4) { LAUNCH_QMV_WIDE_GS(hip_bfloat16, hip_bfloat16, 4); }
      else { LAUNCH_QMV_WIDE_GS(hip_bfloat16, hip_bfloat16, 8); }
    } else {
      if (bits_ == 4) { LAUNCH_QMV_WIDE_GS(__half, __half, 4); }
      else { LAUNCH_QMV_WIDE_GS(__half, __half, 8); }
    }
#undef LAUNCH_QMV_WIDE_GS
#undef LAUNCH_QMV_WIDE
    return;
  }

  // int8 integer-dot QMV (4-bit affine, small-M decode). Quantize activation to
  // int8 (per-group scale+sum) in a prepass, then v_dot4_i32_iu8. Gated by
  // MLX_QMV_IDOT so it's an opt-in A/B against the tiled path.
  static bool use_idot_qmv = (std::getenv("MLX_QMV_IDOT") != nullptr);
  if (use_idot_qmv && bits_ == 4 && mode_ == QuantizationMode::Affine &&
      has_bias && transpose_ && batch_count == 1 && non_batched && M >= 1 &&
      M <= 8 && (K % 8 == 0) && (K % group_size_ == 0) &&
      (group_size_ == 32 || group_size_ == 64 || group_size_ == 128) &&
      (x.dtype() == bfloat16 || x.dtype() == float16)) {
    int ng = K / group_size_;
    int nwords = K / 8;
    array xpack({M, nwords * 2}, int32, nullptr, {});
    xpack.set_data(mlx::core::rocm::malloc_async(xpack.nbytes(), enc));
    array dxq({M, ng}, float32, nullptr, {});
    dxq.set_data(mlx::core::rocm::malloc_async(dxq.nbytes(), enc));
    array sxq({M, ng}, float32, nullptr, {});
    sxq.set_data(mlx::core::rocm::malloc_async(sxq.nbytes(), enc));
    enc.set_output_array(xpack);
    enc.set_output_array(dxq);
    enc.set_output_array(sxq);
    int32_t* xpack_ptr = gpu_ptr<int32_t>(xpack);
    float* dxq_ptr = gpu_ptr<float>(dxq);
    float* sxq_ptr = gpu_ptr<float>(sxq);

    dim3 q_grid(M), q_block(256);
    dim3 mv_block(32, 8);
    dim3 mv_grid((N + 7) / 8, M);

#define LAUNCH_IDOT(TT, ST, GS)                                            \
  enc.add_kernel_node(                                                     \
      &rocm::quantize_act_q8_kernel<TT, GS>, q_grid, q_block, 0u,          \
      (const TT*)x_ptr, xpack_ptr, dxq_ptr, sxq_ptr, M, K);                \
  enc.add_kernel_node(                                                     \
      &rocm::qmv_idot_kernel<TT, ST, GS>, mv_grid, mv_block, 0u,           \
      (const int32_t*)xpack_ptr, w_ptr, (const ST*)scales_ptr,             \
      (const ST*)biases_ptr, (const float*)dxq_ptr, (const float*)sxq_ptr, \
      (TT*)out_ptr, M, N, K, has_bias)

    if (x.dtype() == bfloat16) {
      if (group_size_ == 32) { LAUNCH_IDOT(hip_bfloat16, hip_bfloat16, 32); }
      else if (group_size_ == 64) { LAUNCH_IDOT(hip_bfloat16, hip_bfloat16, 64); }
      else { LAUNCH_IDOT(hip_bfloat16, hip_bfloat16, 128); }
    } else {
      if (group_size_ == 32) { LAUNCH_IDOT(__half, __half, 32); }
      else if (group_size_ == 64) { LAUNCH_IDOT(__half, __half, 64); }
      else { LAUNCH_IDOT(__half, __half, 128); }
    }
#undef LAUNCH_IDOT
    return;
  }

  // The tiled QMV kernel (qdequant.hpp pack_factor_u32 = 32/BITS) only packs
  // correctly for power-of-two widths and is only instantiated for 4/8-bit;
  // other widths would match nothing here and leave `out` uninitialized.
  // Restrict to 4/8-bit; 2/5/6-bit fall through to the warp-shared QMV kernel.
  bool tiled_bits_supported = (bits_ == 4 || bits_ == 8);
  if (use_tiled && tiled_bits_supported && use_fast_qmv &&
      !can_use_batched_qmv && tile_n >= 8 &&
      mode_ == QuantizationMode::Affine) {
    {
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
        enc.add_kernel_node( \
            &rocm::qmv_tiled_kernel<T, ScaleT, BITS_V, GS_V>, \
            tiled_grid, tiled_block, 0u, \
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
    }
    return;
  }

  // The noshared path used to increase cols_per_block for aligned data.
  // Since we always use the shared variant now, no special grid adjustment needed.

  {
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
                  enc.add_kernel_node(
                      &rocm::qmv_warp_shared_batched_kernel<
                          T,
                          ScaleT,
                          BITS,
                          GROUP_SIZE,
                          true,
                          16>,
                      fast_grid_batched,
                      fast_block,
                      0u,
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
                  enc.add_kernel_node(
                      &rocm::qmv_warp_shared_batched_kernel<
                          T,
                          ScaleT,
                          BITS,
                          GROUP_SIZE,
                          true,
                          WARP_SIZE>,
                      fast_grid_batched,
                      fast_block,
                      0u,
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
                    enc.add_kernel_node(
                        &rocm::qmv_warp_noshared_kernel<
                            T,
                            ScaleT,
                            BITS,
                            GROUP_SIZE,
                            true,
                            16>,
                        fast_grid,
                        fast_block,
                        0u,
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
                    enc.add_kernel_node(
                        &rocm::qmv_warp_noshared_kernel<
                            T,
                            ScaleT,
                            BITS,
                            GROUP_SIZE,
                            true,
                            WARP_SIZE>,
                        fast_grid,
                        fast_block,
                        0u,
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
                    enc.add_kernel_node(
                        &rocm::qmv_warp_shared_kernel<
                            T,
                            ScaleT,
                            BITS,
                            GROUP_SIZE,
                            true,
                            16>,
                        fast_grid,
                        fast_block,
                        0u,
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
                    enc.add_kernel_node(
                        &rocm::qmv_warp_shared_kernel<
                            T,
                            ScaleT,
                            BITS,
                            GROUP_SIZE,
                            true,
                            WARP_SIZE>,
                        fast_grid,
                        fast_block,
                        0u,
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
              enc.add_kernel_node(
                  &rocm::qmv_t_kernel<T, ScaleT, BITS, GROUP_SIZE, true>,
                  grid,
                  dim3(block_size),
                  0u,
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
              enc.add_kernel_node(
                  &rocm::qmv_kernel<T, ScaleT, BITS, GROUP_SIZE, true>,
                  grid,
                  dim3(block_size),
                  0u,
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
                  enc.add_kernel_node(
                      &rocm::qmv_warp_shared_batched_kernel<
                          T,
                          ScaleT,
                          BITS,
                          GROUP_SIZE,
                          false,
                          16>,
                      fast_grid_batched,
                      fast_block,
                      0u,
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
                  enc.add_kernel_node(
                      &rocm::qmv_warp_shared_batched_kernel<
                          T,
                          ScaleT,
                          BITS,
                          GROUP_SIZE,
                          false,
                          WARP_SIZE>,
                      fast_grid_batched,
                      fast_block,
                      0u,
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
                    enc.add_kernel_node(
                        &rocm::qmv_warp_noshared_kernel<
                            T,
                            ScaleT,
                            BITS,
                            GROUP_SIZE,
                            false,
                            16>,
                        fast_grid,
                        fast_block,
                        0u,
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
                    enc.add_kernel_node(
                        &rocm::qmv_warp_noshared_kernel<
                            T,
                            ScaleT,
                            BITS,
                            GROUP_SIZE,
                            false,
                            WARP_SIZE>,
                        fast_grid,
                        fast_block,
                        0u,
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
                    enc.add_kernel_node(
                        &rocm::qmv_warp_shared_kernel<
                            T,
                            ScaleT,
                            BITS,
                            GROUP_SIZE,
                            false,
                            16>,
                        fast_grid,
                        fast_block,
                        0u,
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
                    enc.add_kernel_node(
                        &rocm::qmv_warp_shared_kernel<
                            T,
                            ScaleT,
                            BITS,
                            GROUP_SIZE,
                            false,
                            WARP_SIZE>,
                        fast_grid,
                        fast_block,
                        0u,
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
              enc.add_kernel_node(
                  &rocm::qmv_t_kernel<T, ScaleT, BITS, GROUP_SIZE, false>,
                  grid,
                  dim3(block_size),
                  0u,
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
              enc.add_kernel_node(
                  &rocm::qmv_kernel<T, ScaleT, BITS, GROUP_SIZE, false>,
                  grid,
                  dim3(block_size),
                  0u,
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
  }
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

// int8 integer-dot gather expert QMV (4-bit affine, decode). Activation already
// quantized to int8 (xpack even/odd + per-group scale dxq + sum sxq) by a
// prepass; expert weight nibbles fed to v_dot4_i32_iu8. Same run-find as the
// fp32 expert kernel; only the inner matmul changes. THREADS_PER_COL = 32.
template <typename T, typename ScaleT, int GROUP_SIZE>
__global__ void __launch_bounds__(1024) gather_qmv_idot_expert_batched_kernel(
    const int32_t* __restrict__ xpack,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    const float* __restrict__ dxq,
    const float* __restrict__ sxq,
    const uint32_t* __restrict__ lhs_indices,
    const uint32_t* __restrict__ rhs_indices,
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
  const int expert_slot = blockIdx.z;
  if (row >= M || col >= N)
    return;

  int run_start = 0, run_end = 0;
  uint32_t expert_id = 0;
  if (lane == 0 && warp_idx == 0) {
    int pos = 0;
    for (int skip = 0; skip < expert_slot && pos < B; ++skip) {
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
      int lo = pos + 1, hi = B;
      while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (rhs_indices[mid] == expert_id) lo = mid + 1;
        else hi = mid;
      }
      run_end = lo;
    }
  }
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
  if (run_end <= run_start)
    return;
  if (expert_id >= static_cast<uint32_t>(E))
    return;

  const int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const int row_bytes = (K * 4 + 7) / 8;
  const int wpg = GROUP_SIZE / 8;
  const int nwords = K / 8;
  int64_t w_expert_stride = static_cast<int64_t>(N) * row_bytes;
  int64_t sb_expert_stride = static_cast<int64_t>(N) * num_groups;
  const uint32_t* wr = reinterpret_cast<const uint32_t*>(
      w + static_cast<int64_t>(expert_id) * w_expert_stride +
      static_cast<int64_t>(col) * row_bytes);
  const ScaleT* scales_row = scales +
      static_cast<int64_t>(expert_id) * sb_expert_stride +
      static_cast<int64_t>(col) * num_groups;
  const ScaleT* biases_row = has_bias
      ? (biases + static_cast<int64_t>(expert_id) * sb_expert_stride +
         static_cast<int64_t>(col) * num_groups)
      : nullptr;

  int64_t x_batch_stride = static_cast<int64_t>(M) * K;
  for (int b = run_start; b < run_end; ++b) {
    uint32_t lhs_idx = implicit_lhs ? 0u : lhs_indices[b];
    int64_t x_offset = implicit_lhs
        ? (static_cast<int64_t>(b) * implicit_x_batch_stride)
        : (static_cast<int64_t>(lhs_idx) * x_batch_stride);
    int64_t arow = x_offset / K + row;
    const int32_t* xpack_row = xpack + arow * nwords * 2;
    const float* dxr = dxq + arow * num_groups;
    const float* sxr = sxq + arow * num_groups;

    float acc = 0.0f;
    for (int g = 0; g < num_groups; ++g) {
      float scale =
          load_scale_value<ScaleT, GROUP_SIZE, true>(scales_row[g]);
      float bias_val = has_bias ? static_cast<float>(biases_row[g]) : 0.0f;
      int w0 = g * wpg, w1 = w0 + wpg;
      int qsum = 0;
      for (int wi = w0 + lane; wi < w1; wi += 32) {
        uint32_t vui = wr[wi];
        int ew = vui & 0x0F0F0F0F;
        int ow = (vui >> 4) & 0x0F0F0F0F;
        qsum = sudot4_iu8(ew, xpack_row[2 * wi], qsum);
        qsum = sudot4_iu8(ow, xpack_row[2 * wi + 1], qsum);
      }
      float qf = subgroup_reduce_sum_qmm<32>(static_cast<float>(qsum));
      acc += scale * dxr[g] * qf + bias_val * sxr[g];
    }
    if (lane == 0)
      out[static_cast<int64_t>(b) * M * N + static_cast<int64_t>(row) * N +
          col] = static_cast<T>(acc);
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

// int8 integer-dot decode gather QMV (4-bit affine). One block per routing
// entry (batch=blockIdx.z); activation already quantized to int8 by a prepass
// (xpack even/odd + per-group scale dxq + sum sxq). 32 lanes per column.
template <typename T, typename ScaleT, int GROUP_SIZE>
__global__ void __launch_bounds__(1024) gather_qmv_idot_warp_shared_kernel(
    const int32_t* __restrict__ xpack,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    const float* __restrict__ dxq,
    const float* __restrict__ sxq,
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
    bool implicit_lhs,
    int64_t implicit_x_batch_stride) {
  const int lane = threadIdx.x;
  const int warp_idx = threadIdx.y;
  const int col = blockIdx.y * blockDim.y + warp_idx;
  const int row = blockIdx.x;
  const int batch = blockIdx.z;
  if (batch >= B || row >= M)
    return;

  int64_t rhs_idx_loc = 0, lhs_idx_loc = 0;
  if (batch_ndim == 1) {
    rhs_idx_loc = static_cast<int64_t>(batch) * rhs_idx_strides[0];
    if (!implicit_lhs)
      lhs_idx_loc = static_cast<int64_t>(batch) * lhs_idx_strides[0];
  } else if (batch_ndim > 1) {
    int64_t elem = batch;
    for (int i = batch_ndim - 1; i >= 0; --i) {
      int64_t coord = elem % batch_shape.data_[i];
      rhs_idx_loc += coord * rhs_idx_strides.data_[i];
      if (!implicit_lhs)
        lhs_idx_loc += coord * lhs_idx_strides.data_[i];
      elem /= batch_shape.data_[i];
    }
  }
  uint32_t lhs_idx = implicit_lhs ? 0u : lhs_indices[lhs_idx_loc];
  uint32_t rhs_idx = rhs_indices[rhs_idx_loc];

  const bool col_valid = col < N;
  const bool expert_valid = rhs_idx < static_cast<uint32_t>(E);

  const int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const int row_bytes = (K * 4 + 7) / 8;
  const int wpg = GROUP_SIZE / 8;
  const int nwords = K / 8;

  float acc = 0.0f;
  if (col_valid && expert_valid) {
    int64_t w_batch_stride = static_cast<int64_t>(N) * row_bytes;
    int64_t sb_batch_stride = static_cast<int64_t>(N) * num_groups;
    const uint32_t* wr = reinterpret_cast<const uint32_t*>(
        w + static_cast<int64_t>(rhs_idx) * w_batch_stride +
        static_cast<int64_t>(col) * row_bytes);
    const ScaleT* scales_row = scales +
        static_cast<int64_t>(rhs_idx) * sb_batch_stride +
        static_cast<int64_t>(col) * num_groups;
    const ScaleT* biases_row = has_bias
        ? (biases + static_cast<int64_t>(rhs_idx) * sb_batch_stride +
           static_cast<int64_t>(col) * num_groups)
        : nullptr;

    int64_t x_batch_offset = implicit_lhs
        ? (static_cast<int64_t>(batch) * implicit_x_batch_stride)
        : (static_cast<int64_t>(lhs_idx) * static_cast<int64_t>(M) * K);
    int64_t arow = x_batch_offset / K + row;
    const int32_t* xpack_row = xpack + arow * nwords * 2;
    const float* dxr = dxq + arow * num_groups;
    const float* sxr = sxq + arow * num_groups;

    for (int g = 0; g < num_groups; ++g) {
      float scale =
          load_scale_value<ScaleT, GROUP_SIZE, true>(scales_row[g]);
      float bias_val = has_bias ? static_cast<float>(biases_row[g]) : 0.0f;
      int w0 = g * wpg, w1 = w0 + wpg;
      int qsum = 0;
      for (int wi = w0 + lane; wi < w1; wi += 32) {
        uint32_t vui = wr[wi];
        int ew = vui & 0x0F0F0F0F;
        int ow = (vui >> 4) & 0x0F0F0F0F;
        qsum = sudot4_iu8(ew, xpack_row[2 * wi], qsum);
        qsum = sudot4_iu8(ow, xpack_row[2 * wi + 1], qsum);
      }
      float qf = subgroup_reduce_sum_qmm<32>(static_cast<float>(qsum));
      acc += scale * dxr[g] * qf + bias_val * sxr[g];
    }
  }
  if (col_valid && lane == 0) {
    int64_t out_offset = (static_cast<int64_t>(batch) * M + row) * N + col;
    out[out_offset] = expert_valid ? static_cast<T>(acc) : static_cast<T>(0);
  }
}

// Wide-load gather QMV (4-bit affine, decode). Flat per-(batch,col) warp with
// 128-bit (uint4 = 32 nibbles) weight loads instead of the per-group 32-bit
// loop, to raise effective gather bandwidth (microbench: R9700 77->398 GB/s).
// Activation read from global (tiny, L2-cached). Per-group affine applied
// inline. Requires K % 32 == 0; GROUP_SIZE in {32,64,128}. Default on; opt out
// with MLX_QMV_NO_WIDE.
template <typename T, typename ScaleT, int GROUP_SIZE>
__global__ void __launch_bounds__(1024) gather_qmv_wide_kernel(
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
    bool implicit_lhs,
    int64_t implicit_x_batch_stride) {
  int lane = threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int row = blockIdx.x;
  int batch = blockIdx.z;
  if (batch >= B || row >= M)
    return;

  int64_t rhs_loc = 0, lhs_loc = 0;
  if (batch_ndim == 1) {
    rhs_loc = static_cast<int64_t>(batch) * rhs_idx_strides[0];
    if (!implicit_lhs)
      lhs_loc = static_cast<int64_t>(batch) * lhs_idx_strides[0];
  } else if (batch_ndim > 1) {
    int64_t e = batch;
    for (int i = batch_ndim - 1; i >= 0; --i) {
      int64_t c = e % batch_shape.data_[i];
      rhs_loc += c * rhs_idx_strides.data_[i];
      if (!implicit_lhs)
        lhs_loc += c * lhs_idx_strides.data_[i];
      e /= batch_shape.data_[i];
    }
  }
  uint32_t lhs_idx = implicit_lhs ? 0u : lhs_indices[lhs_loc];
  uint32_t rhs_idx = rhs_indices[rhs_loc];
  bool col_valid = col < N;
  bool expert_valid = rhs_idx < static_cast<uint32_t>(E);

  int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  int row_bytes = (K * 4 + 7) / 8;
  float acc = 0.0f;
  if (col_valid && expert_valid) {
    int64_t w_bs = static_cast<int64_t>(N) * row_bytes;
    int64_t sb_bs = static_cast<int64_t>(N) * num_groups;
    const uint8_t* wr = w + static_cast<int64_t>(rhs_idx) * w_bs +
        static_cast<int64_t>(col) * row_bytes;
    const ScaleT* scr = scales + static_cast<int64_t>(rhs_idx) * sb_bs +
        static_cast<int64_t>(col) * num_groups;
    const ScaleT* bcr = has_bias
        ? (biases + static_cast<int64_t>(rhs_idx) * sb_bs +
           static_cast<int64_t>(col) * num_groups)
        : nullptr;
    int64_t x_off = implicit_lhs
        ? (static_cast<int64_t>(batch) * implicit_x_batch_stride)
        : (static_cast<int64_t>(lhs_idx) * static_cast<int64_t>(M) * K);
    const T* xr = x + x_off + static_cast<int64_t>(row) * K;

    for (int k = lane * 32; k < K; k += 32 * 32) {
      int g = k / GROUP_SIZE;
      float s = load_scale_value<ScaleT, GROUP_SIZE, true>(scr[g]);
      float b = has_bias ? static_cast<float>(bcr[g]) : 0.0f;
      uint4 wv = *reinterpret_cast<const uint4*>(&wr[k / 2]);
      const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv);
      float la = 0.0f, lx = 0.0f;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        uint32_t wd = wp[q];
        int kb = k + q * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float xv = static_cast<float>(xr[kb + j]);
          int nib = (wd >> (4 * j)) & 0xF;
          la += xv * static_cast<float>(nib);
          if (has_bias)
            lx += xv;
        }
      }
      acc += s * la + b * lx;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    acc += __shfl_down(acc, o);
  if (col_valid && lane == 0)
    out[(static_cast<int64_t>(batch) * M + row) * N + col] =
        expert_valid ? static_cast<T>(acc) : static_cast<T>(0);
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
__global__ void __launch_bounds__(128) gather_qmv_wmma_prefill_kernel(
    const T* __restrict__ x,
    const uint32_t* __restrict__ w,
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

  // Register-tiled segmented GEMM, 4 warps/block. The 4 warps SHARE one gathered
  // sX tile (each warp owns a 32-col slab of a 128-col block tile), so the X gather
  // happens once per 4 N-tiles instead of once per N-tile -> 4x less X traffic and
  // 4 resident warps/WGP for latency hiding. 128-bit weight/x loads, nibble unpack.
  constexpr int WM = 16, WN = 16, WK = 16;
  constexpr int MT = 4, NT = 2;            // per-warp tile: BM=64 rows, BN_W=32 cols
  constexpr int NWARP = 4;
  constexpr int BM = WM * MT;              // 64 token rows / block
  constexpr int BN_W = WN * NT;            // 32 cols / warp
  constexpr int BN = BN_W * NWARP;         // 128 cols / block
  constexpr int BK = GROUP_SIZE;           // one quant group per K-tile (=64)
  constexpr int U4S = (BK * BITS) / 128;   // 128-bit (uint4) weight loads per tile
  constexpr int CH = BK / 8;               // uint4 chunks per gathered x row (bf16)

  const int tid  = threadIdx.x;            // 0..127
  const int warp = tid >> 5;               // 0..3
  const int lane = tid & 31;               // 0..31
  const int run_id   = blockIdx.z;
  const int run_start = run_starts[run_id];
  const int run_len   = run_lengths[run_id];
  const int n0   = blockIdx.y * BN;        // first output column (block)
  const int wn0  = n0 + warp * BN_W;       // first output column (this warp)
  if (run_len == 0 || n0 >= N) return;

  // GPU grouping launches one run per expert, so run_id IS the expert index.
  const uint32_t rhs_idx = static_cast<uint32_t>(run_id);
  (void)rhs_indices;
  if (rhs_idx >= static_cast<uint32_t>(E)) return;

  const int Kp = K * BITS / 32;            // uint32 per weight row
  const int Kg = K / GROUP_SIZE;           // quant groups per row
  const uint32_t* Wb = w      + static_cast<int64_t>(rhs_idx) * N * Kp;
  const ScaleT*   Sb = scales + static_cast<int64_t>(rhs_idx) * N * Kg;
  const ScaleT*   Bb = has_bias
      ? (biases + static_cast<int64_t>(rhs_idx) * N * Kg) : nullptr;

  __shared__ hip_bfloat16 sX[BM][BK];      // gathered x rows (shared by all 4 warps)
  __shared__ hip_bfloat16 sB[BK][BN];      // dequantized weight tile (128 cols)
  __shared__ float sC[BM][BN];

  using frag_a   = rocwmma::fragment<rocwmma::matrix_a,   WM, WN, WK, hip_bfloat16, rocwmma::row_major>;
  using frag_b   = rocwmma::fragment<rocwmma::matrix_b,   WM, WN, WK, hip_bfloat16, rocwmma::row_major>;
  using frag_acc = rocwmma::fragment<rocwmma::accumulator, WM, WN, WK, float>;

  // Grid-stride over the run's token-tiles: grid.x is a bounded occupancy knob
  // (not correctness), so the dispatch needs no max_run_len readback/sync and
  // launches no excess empty blocks. A block covers tiles row0, row0+stride, ...
  for (int row0 = blockIdx.x * BM; row0 < run_len; row0 += gridDim.x * BM) {
  frag_acc cf[MT][NT];
  #pragma unroll
  for (int mt = 0; mt < MT; ++mt)
    #pragma unroll
    for (int t = 0; t < NT; ++t) rocwmma::fill_fragment(cf[mt][t], 0.0f);

  for (int kk = 0; kk < K; kk += BK) {
    // gather x tile once for the whole block: BM rows x BK cols, all 128 threads
    for (int idx = tid; idx < BM * CH; idx += NWARP * 32) {
      const int r = idx / CH, ch = idx % CH, br = row0 + r;
      uint4* dst = reinterpret_cast<uint4*>(&sX[r][ch * 8]);
      if (br < run_len) {
        const uint32_t tok = lhs_indices[run_start + br];
        *dst = *reinterpret_cast<const uint4*>(
            &x[static_cast<int64_t>(tok) * x_batch_stride + kk + ch * 8]);
      } else {
        *dst = uint4{0, 0, 0, 0};
      }
    }
    // dequantize this warp's 32-col slab of the weight tile, 128-bit loads.
    // Columns past N (last block when N%128!=0) are zero-filled, not read.
    {
      const int col = warp * BN_W + lane, n = wn0 + lane;
      if (n < N) {
        const uint32_t* wb = &Wb[static_cast<int64_t>(n) * Kp + (kk * BITS) / 32];
        const int bg = kk / GROUP_SIZE;
        const float scl = static_cast<float>(Sb[n * Kg + bg]);
        const float bia = has_bias ? static_cast<float>(Bb[n * Kg + bg]) : 0.0f;
        #pragma unroll
        for (int u = 0; u < U4S; ++u) {
          const uint4 wv = *reinterpret_cast<const uint4*>(wb + u * 4);
          const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv);
          #pragma unroll
          for (int q = 0; q < 4; ++q) {
            const uint32_t wd = wp[q];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
              const int k = (u * 4 + q) * 8 + j;
              sB[k][col] = static_cast<hip_bfloat16>(
                  float((wd >> (4 * j)) & 0xF) * scl + bia);
            }
          }
        }
      } else {
        #pragma unroll
        for (int k = 0; k < BK; ++k) sB[k][col] = static_cast<hip_bfloat16>(0.0f);
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ks = 0; ks < BK; ks += WK) {
      frag_a af[MT];
      #pragma unroll
      for (int mt = 0; mt < MT; ++mt)
        rocwmma::load_matrix_sync(af[mt], &sX[mt * WM][ks], BK);
      #pragma unroll
      for (int t = 0; t < NT; ++t) {
        frag_b bf;
        rocwmma::load_matrix_sync(bf, &sB[ks][warp * BN_W + t * WN], BN);
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          rocwmma::mma_sync(cf[mt][t], af[mt], bf, cf[mt][t]);
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int mt = 0; mt < MT; ++mt)
    #pragma unroll
    for (int t = 0; t < NT; ++t)
      rocwmma::store_matrix_sync(&sC[mt * WM][warp * BN_W + t * WN], cf[mt][t], BN,
                                 rocwmma::mem_row_major);
  __syncthreads();
  for (int idx = tid; idx < BM * BN; idx += NWARP * 32) {
    const int r = idx / BN, c = idx % BN, br = row0 + r;
    if (br < run_len && (n0 + c) < N) {
      const int orig_batch = out_perm[run_start + br];
      out[static_cast<int64_t>(orig_batch) * M * N + n0 + c] =
          static_cast<T>(sC[r][c]);
    }
  }
  __syncthreads();  // all threads done reading sC before next tile overwrites smem
  }  // grid-stride row0 loop

#endif // ROCM_HAS_WMMA
}

// ---- GPU-side expert grouping (counting sort; keys = expert in [0,E)) ----
// Replaces the per-call host sort + device sync. All on-device, no host round
// trip. Runs map 1:1 to experts (run_id == expert), so the GEMM reads run_id as
// its expert and the launch needs no host-side run counts.
__global__ void moe_group_zero_kernel(int* a, int* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { a[i] = 0; b[i] = 0; }
}
__global__ void moe_group_count_kernel(
    const uint32_t* __restrict__ rhs, int B, int E, int* __restrict__ counts) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < B) { uint32_t e = rhs[b]; if (e < static_cast<uint32_t>(E)) atomicAdd(&counts[e], 1); }
}
__global__ void moe_group_scan_kernel(
    const int* __restrict__ counts, int E,
    int* __restrict__ run_starts, int* __restrict__ run_lengths,
    int* __restrict__ max_run_len) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int acc = 0, mx = 0;
    for (int e = 0; e < E; ++e) {
      run_starts[e] = acc;
      run_lengths[e] = counts[e];
      acc += counts[e];
      if (counts[e] > mx) mx = counts[e];
    }
    *max_run_len = mx;
  }
}
__global__ void moe_group_scatter_kernel(
    const uint32_t* __restrict__ rhs, const uint32_t* __restrict__ lhs, int B, int E,
    const int* __restrict__ run_starts, int* __restrict__ cursors,
    uint32_t* __restrict__ sorted_li, int* __restrict__ perm) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < B) {
    uint32_t e = rhs[b];
    if (e < static_cast<uint32_t>(E)) {
      int pos = run_starts[e] + atomicAdd(&cursors[e], 1);
      sorted_li[pos] = lhs[b];
      perm[pos] = b;
    }
  }
}

} // namespace rocm

void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = rocm::device(s.device);
  auto& enc = d.get_command_encoder(s);
  out.set_data(mlx::core::rocm::malloc_async(out.nbytes(), enc));
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
  // Grouped (segmented) WMMA prefill: MLX feeds M==1 with B = tokens*top_k batch
  // elements. Sort by expert into runs and pack each run's tokens into the WMMA
  // M dimension (matrix-matrix), instead of the old per-token M=1 QMV. Gate on B
  // being large enough that runs fill tiles (prefill, not decode).
  static const bool kGroupedPrefillEnabled = [] {
    const char* e = std::getenv("MLX_ROCM_GROUPED_PREFILL");
    return e && std::atoi(e) != 0;  // opt-in; WIP, regresses until kernel+GPU-grouping land
  }();
  static const int kGroupedPrefillMinB = [] {
    const char* e = std::getenv("MLX_ROCM_GROUPED_PREFILL_MIN_B");
    return e ? std::atoi(e) : 64;
  }();
  if (kGroupedPrefillEnabled && M == 1 && B >= kGroupedPrefillMinB && transpose_ &&
      E > 0 && batch_ndim == 1 &&
      mode_ == QuantizationMode::Affine && x.dtype() == bfloat16 &&
      group_size_ == 64 && bits_ == 4) {
    // WMMA arch gate (must hold for the grouped path to be worthwhile).
    bool use_wmma = d.has_native_wmma() && (N % 16 == 0) && (bits_ == 4);
    use_wmma = parse_warp_kernel_env("MLX_ROCM_GATHER_QMV_USE_WMMA", use_wmma);
    if (use_wmma) {
      // ---- GPU-side counting-sort grouping (no host round trip) ----
      // counts/cursors/run_starts/run_lengths are per-expert (E); sorted_li/perm
      // are per-batch-elem (B). Runs map 1:1 to experts.
      array counts_arr({E}, int32, nullptr, {});
      array cursors_arr({E}, int32, nullptr, {});
      array run_starts_arr({E}, int32, nullptr, {});
      array run_lengths_arr({E}, int32, nullptr, {});
      array sorted_li_arr({B}, uint32, nullptr, {});
      array perm_arr({B}, int32, nullptr, {});
      array max_rl_arr({1}, int32, nullptr, {});
      counts_arr.set_data(mlx::core::rocm::malloc_async(counts_arr.nbytes(), enc));
      cursors_arr.set_data(mlx::core::rocm::malloc_async(cursors_arr.nbytes(), enc));
      run_starts_arr.set_data(mlx::core::rocm::malloc_async(run_starts_arr.nbytes(), enc));
      run_lengths_arr.set_data(mlx::core::rocm::malloc_async(run_lengths_arr.nbytes(), enc));
      sorted_li_arr.set_data(mlx::core::rocm::malloc_async(sorted_li_arr.nbytes(), enc));
      perm_arr.set_data(mlx::core::rocm::malloc_async(perm_arr.nbytes(), enc));
      max_rl_arr.set_data(mlx::core::rocm::malloc_async(max_rl_arr.nbytes(), enc));
      enc.set_input_array(counts_arr);
      enc.set_input_array(cursors_arr);
      enc.set_input_array(run_starts_arr);
      enc.set_input_array(run_lengths_arr);
      enc.set_input_array(sorted_li_arr);
      enc.set_input_array(perm_arr);
      enc.set_input_array(max_rl_arr);

      const int gthreads = 256;
      enc.add_kernel_node(
          &rocm::moe_group_zero_kernel, dim3((E + gthreads - 1) / gthreads), dim3(gthreads), 0u,
          gpu_ptr<int>(counts_arr), gpu_ptr<int>(cursors_arr), E);
      enc.add_kernel_node(
          &rocm::moe_group_count_kernel, dim3((B + gthreads - 1) / gthreads), dim3(gthreads), 0u,
          gpu_ptr<const uint32_t>(rhs_indices), B, E, gpu_ptr<int>(counts_arr));
      enc.add_kernel_node(
          &rocm::moe_group_scan_kernel, dim3(1), dim3(1), 0u,
          gpu_ptr<const int>(counts_arr), E,
          gpu_ptr<int>(run_starts_arr), gpu_ptr<int>(run_lengths_arr),
          gpu_ptr<int>(max_rl_arr));
      enc.add_kernel_node(
          &rocm::moe_group_scatter_kernel, dim3((B + gthreads - 1) / gthreads), dim3(gthreads), 0u,
          gpu_ptr<const uint32_t>(rhs_indices), gpu_ptr<const uint32_t>(lhs_indices), B, E,
          gpu_ptr<const int>(run_starts_arr), gpu_ptr<int>(cursors_arr),
          gpu_ptr<uint32_t>(sorted_li_arr), gpu_ptr<int>(perm_arr));

      int64_t x_bs = (x_batch_count == 1) ? 0 : static_cast<int64_t>(M) * K;

      // Bound grid.x to the largest run's token tiles (not B) so ~98% of x-tiles
      // aren't empty. Eager: read back max_run_len (one 4-byte DtoH). Under graph
      // capture a host sync is illegal, so fall back to the data-independent
      // worst case (ceil(B/64)); empty tiles past a run return immediately.
      // grid.x is a bounded occupancy knob: the kernel grid-strides over each
      // run's token-tiles, so any run length is covered regardless of grid.x.
      // Size it to ~4x the average run's tiles — enough parallelism for moderate
      // routing imbalance, far fewer empty blocks than ceil(B/64), and crucially
      // NO max_run_len readback (that needed a full pipeline-draining
      // enc.synchronize() per gather_qmm call, which tanked small-input PP/s).
      static const int gridx_mult = [] {
        const char* e = std::getenv("MLX_GRIDX_MULT");
        return e ? std::max(1, std::atoi(e)) : 4;
      }();
      const int avg_tiles = ((B + E - 1) / E + 63) / 64;
      int grid_x = std::max(1, std::min((B + 63) / 64, avg_tiles * gridx_mult));
      dim3 wmma_block(128, 1, 1);          // 4 warps
      dim3 wmma_grid(grid_x, (N + 127) / 128, E);
      enc.add_kernel_node_ex(
          &rocm::gather_qmv_wmma_prefill_kernel<hip_bfloat16, hip_bfloat16, 4, 64, true>,
          wmma_grid, wmma_block, 0u,
          gpu_ptr<const hip_bfloat16>(x),
          gpu_ptr<const uint32_t>(w),
          gpu_ptr<const hip_bfloat16>(scales),
          static_cast<const hip_bfloat16*>(has_bias ? gpu_ptr<const hip_bfloat16>(*biases) : nullptr),
          gpu_ptr<const uint32_t>(sorted_li_arr),
          gpu_ptr<const uint32_t>(rhs_indices),
          gpu_ptr<const int>(run_starts_arr),
          gpu_ptr<const int>(run_lengths_arr),
          gpu_ptr<const int>(perm_arr),
          gpu_ptr<hip_bfloat16>(out),
          B, M, N, K, E, has_bias, x_bs);
      return;
    }
    // !use_wmma: fall through to the standard (non-grouped) dispatch below.
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

    {
      auto launch_eb = [&](auto bits_tag) {
        constexpr int BITS = decltype(bits_tag)::value;
        enc.add_kernel_node(
            &rocm::gather_qmv_expert_batched_kernel<
                hip_bfloat16, hip_bfloat16, BITS, 64, true, 16>,
            eb_grid, eb_block, 0u,
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
    }
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

  // Full-wave tiled 6-bit gather; MLX_ROCM_QMV_6BIT_SLOW reverts to warp_shared.
  static const bool g_use_6bit_tiled_gather =
      (std::getenv("MLX_ROCM_QMV_6BIT_SLOW") == nullptr);
  bool g_gs6_supported =
      (group_size_ == 32 || group_size_ == 64 || group_size_ == 128);
  bool gather_tiled_6bit_ok = g_use_6bit_tiled_gather && transpose_ &&
      mode_ == QuantizationMode::Affine && x.dtype() == bfloat16 &&
      bits_ == 6 && (K % 64) == 0 && g_gs6_supported &&
      !use_sorted_rhs_schedule &&
      batch_ndim == 1 && batch_strides[0].size() == 1 &&
      batch_strides[0][0] == 1 && batch_strides[1][0] == 1;

  int gather_tile_n = 0;
  if (gather_tiled_ok || gather_tiled_6bit_ok) {
    auto gqmm_hw = detect_rocm_hw_info(enc.device());
    gather_tile_n = rocm::get_arch_tuning(gqmm_hw).qmv_tile_n;
    while (gather_tile_n > 1 && (N % gather_tile_n) != 0) gather_tile_n /= 2;
    if (gather_tile_n < 1) gather_tile_n = 1;
  }
  if (gather_tiled_6bit_ok && gather_tile_n < 8) gather_tiled_6bit_ok = false;

  {
    if (gather_tiled_6bit_ok) {
      dim3 gt_grid(M, (N + gather_tile_n - 1) / gather_tile_n, B);
      dim3 gt_block(WARP_SIZE, gather_tile_n);
      int LHS_B = static_cast<int>(x_batch_count);
      #define LAUNCH_GATHER_TILED_6BIT(GS_V) \
        enc.add_kernel_node( \
            &rocm::gather_qmv_tiled_6bit_kernel<hip_bfloat16, hip_bfloat16, GS_V>, \
            gt_grid, gt_block, 0u, \
            (const hip_bfloat16*)x_ptr, (const uint8_t*)w_ptr, \
            (const hip_bfloat16*)scales_ptr, (const hip_bfloat16*)biases_ptr, \
            li_ptr, ri_ptr, (hip_bfloat16*)out_ptr, \
            B, M, N, K, E, LHS_B, has_bias, gather_tile_n)
      if (group_size_ == 32) { LAUNCH_GATHER_TILED_6BIT(32); }
      else if (group_size_ == 64) { LAUNCH_GATHER_TILED_6BIT(64); }
      else { LAUNCH_GATHER_TILED_6BIT(128); }
      #undef LAUNCH_GATHER_TILED_6BIT
      return;
    }
    if (gather_tiled_ok) {
      dim3 gt_grid(M, (N + gather_tile_n - 1) / gather_tile_n, B);
      dim3 gt_block(WARP_SIZE, gather_tile_n);
      int LHS_B = static_cast<int>(x_batch_count);
      auto launch_gt = [&](auto bits_tag) {
        constexpr int BITS = decltype(bits_tag)::value;
        enc.add_kernel_node(
            &rocm::gather_qmv_tiled_kernel<hip_bfloat16, hip_bfloat16, BITS, 64>,
            gt_grid, gt_block, 0u,
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
    // Wide-load (128-bit / uint4) gather decode path (4-bit affine), gated by
    // MLX_QMV_WIDE. Raises gather weight-read bandwidth (microbench R9700
    // 77->398 GB/s). Flat per-(batch,col) warp; requires K % 32 == 0.
    static bool use_wide_gather = (std::getenv("MLX_QMV_NO_WIDE") == nullptr);
    if (use_wide_gather && use_fast_gather_qmv &&
        mode_ == QuantizationMode::Affine && x.dtype() == bfloat16 &&
        bits_ == 4 && (K % 32 == 0) && has_bias &&
        (group_size_ == 32 || group_size_ == 64 || group_size_ == 128)) {
      int wcols = 8;
      dim3 wblock(32, wcols);
      dim3 wgrid(M, (N + wcols - 1) / wcols, B);
#define LAUNCH_WIDE(GS)                                                       \
  enc.add_kernel_node(                                                        \
      &rocm::gather_qmv_wide_kernel<hip_bfloat16, hip_bfloat16, GS>, wgrid,   \
      wblock, 0u, (const hip_bfloat16*)x_ptr, (const uint8_t*)w_ptr,          \
      (const hip_bfloat16*)scales_ptr, (const hip_bfloat16*)biases_ptr,       \
      li_ptr, ri_ptr, batch_shape_param, lhs_idx_strides_param,               \
      rhs_idx_strides_param, batch_ndim, (hip_bfloat16*)out_ptr, B, M, N, K,  \
      E, has_bias, use_sorted_rhs_schedule, implicit_x_batch_stride)
      if (group_size_ == 32) { LAUNCH_WIDE(32); }
      else if (group_size_ == 64) { LAUNCH_WIDE(64); }
      else { LAUNCH_WIDE(128); }
#undef LAUNCH_WIDE
      return;
    }

    // int8 integer-dot decode gather path (4-bit affine). Experimental and OFF
    // by default (separate from MLX_QMV_IDOT): measured net-NEGATIVE on MoE decode
    // because the expert gather is latency/launch bound, not issue bound, so the
    // per-call activation-quant prepass overhead dominates. Kept behind its own
    // flag for A/B; the dense QuantizedMatmul idot path remains a win.
    static bool use_idot_gather =
        (std::getenv("MLX_QMV_IDOT_GATHER") != nullptr);
    if (use_idot_gather && use_fast_gather_qmv &&
        mode_ == QuantizationMode::Affine && x.dtype() == bfloat16 &&
        group_size_ == 64 && bits_ == 4 && (K % 8 == 0) && has_bias) {
      int ng = K / 64;
      int nwords = K / 8;
      int x_rows = static_cast<int>(x.size() / K);
      array xpack({x_rows, nwords * 2}, int32, nullptr, {});
      xpack.set_data(mlx::core::rocm::malloc_async(xpack.nbytes(), enc));
      array dxq({x_rows, ng}, float32, nullptr, {});
      dxq.set_data(mlx::core::rocm::malloc_async(dxq.nbytes(), enc));
      array sxq({x_rows, ng}, float32, nullptr, {});
      sxq.set_data(mlx::core::rocm::malloc_async(sxq.nbytes(), enc));
      enc.set_output_array(xpack);
      enc.set_output_array(dxq);
      enc.set_output_array(sxq);
      int32_t* xpack_ptr = gpu_ptr<int32_t>(xpack);
      float* dxq_ptr = gpu_ptr<float>(dxq);
      float* sxq_ptr = gpu_ptr<float>(sxq);

      enc.add_kernel_node(
          &rocm::quantize_act_q8_kernel<hip_bfloat16, 64>,
          dim3(x_rows), dim3(256), 0u,
          (const hip_bfloat16*)x_ptr, xpack_ptr, dxq_ptr, sxq_ptr, x_rows, K);

      int idot_cols = 8;
      dim3 idot_block(32, idot_cols);
      dim3 idot_grid(M, (N + idot_cols - 1) / idot_cols, B);
      enc.add_kernel_node(
          &rocm::gather_qmv_idot_warp_shared_kernel<hip_bfloat16, hip_bfloat16, 64>,
          idot_grid, idot_block, 0u,
          (const int32_t*)xpack_ptr, (const uint8_t*)w_ptr,
          (const hip_bfloat16*)scales_ptr, (const hip_bfloat16*)biases_ptr,
          (const float*)dxq_ptr, (const float*)sxq_ptr,
          li_ptr, ri_ptr, batch_shape_param, lhs_idx_strides_param,
          rhs_idx_strides_param, batch_ndim, (hip_bfloat16*)out_ptr,
          B, M, N, K, E, has_bias, use_sorted_rhs_schedule,
          implicit_x_batch_stride);
      return;
    }

    if (use_fast_gather_qmv && mode_ == QuantizationMode::Affine &&
        x.dtype() == bfloat16 && group_size_ == 64 &&
        (bits_ == 4 || bits_ == 6 || bits_ == 8)) {
      auto launch_fast_kernel = [&](auto bits_tag) {
        constexpr int BITS = decltype(bits_tag)::value;
        if (fast_threads_per_col == 16) {
          enc.add_kernel_node(
              &rocm::gather_qmv_warp_shared_kernel<
                  hip_bfloat16,
                  hip_bfloat16,
                  BITS,
                  64,
                  true,
                  16>,
              fast_grid,
              fast_block,
              0u,
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
          enc.add_kernel_node(
              &rocm::gather_qmv_warp_shared_kernel<
                  hip_bfloat16,
                  hip_bfloat16,
                  BITS,
                  64,
                  true,
                  WARP_SIZE>,
              fast_grid,
              fast_block,
              0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<float, float, 8, 32, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<float, float, 8, 64, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<float, float, 8, 128, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<float, float, 5, 32, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<float, float, 5, 64, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<float, float, 5, 128, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<float, float, 6, 32, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<float, float, 6, 64, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<float, float, 6, 128, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<float, float, 4, 32, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<float, float, 4, 64, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<float, float, 4, 128, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<float, float, 2, 32, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<float, float, 2, 64, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<float, float, 2, 128, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<__half, __half, 8, 32, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<__half, __half, 8, 64, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<__half, __half, 8, 128, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<__half, __half, 5, 32, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<__half, __half, 5, 64, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<__half, __half, 5, 128, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<__half, __half, 6, 32, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<__half, __half, 6, 64, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<__half, __half, 6, 128, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<__half, __half, 4, 32, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<__half, __half, 4, 64, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<__half, __half, 4, 128, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<__half, __half, 2, 32, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<__half, __half, 2, 64, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<__half, __half, 2, 128, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 8, 32, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 8, 64, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 8, 128, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 5, 32, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 5, 64, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 5, 128, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 6, 32, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 6, 64, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 6, 128, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 4, 32, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 4, 64, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 4, 128, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 2, 32, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 2, 64, true>,
            grid,
            dim3(block_size),
            0u,
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
        enc.add_kernel_node(
            &rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 2, 128, true>,
            grid,
            dim3(block_size),
            0u,
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
  }
}

} // namespace mlx::core
