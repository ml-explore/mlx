// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/utils.h"
#include "mlx/device.h"
#include "mlx/memory.h"
#include "mlx/utils.h"

#include <hip/hip_runtime.h>
#include <unistd.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace mlx::core {

namespace rocm {

constexpr int page_size = 16384;

// Check if ROCm device is available
static bool rocm_available() {
  static int available = -1;
  if (available < 0) {
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    available = (err == hipSuccess && device_count > 0) ? 1 : 0;
  }
  return available == 1;
}

// Check if managed memory (HMM) is supported on this device.
static bool managed_memory_supported() {
  static int supported = -1;
  if (supported < 0) {
    if (!rocm_available()) {
      supported = 0;
    } else {
      void* test_ptr = nullptr;
      hipError_t err = hipMallocManaged(&test_ptr, 64);
      if (err == hipSuccess) {
        (void)hipFree(test_ptr);
        supported = 1;
      } else {
        supported = 0;
      }
    }
  }
  return supported == 1;
}

static bool is_integrated() {
  static int integrated = -1;
  if (integrated < 0) {
    if (!rocm_available()) {
      integrated = 0;
    } else {
      int device = 0;
      (void)hipGetDevice(&device);
      hipDeviceProp_t props;
      hipError_t err = hipGetDeviceProperties(&props, device);
      integrated = (err == hipSuccess && props.integrated == 1) ? 1 : 0;
    }
  }
  return integrated == 1;
}

static bool device_is_integrated(int dev) {
  static int cache[16] = {-1, -1, -1, -1, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1};
  if (dev < 0 || dev >= 16)
    return false;
  if (cache[dev] < 0) {
    hipDeviceProp_t p;
    cache[dev] =
        (hipGetDeviceProperties(&p, dev) == hipSuccess && p.integrated == 1) ? 1
                                                                             : 0;
  }
  return cache[dev] == 1;
}

static bool use_finegrained() {
  if (const char* e = std::getenv("MLX_ROCM_FINEGRAINED"))
    return std::atoi(e) != 0;
  return true;
}

// CUDA-aligned: use hipMallocAsync when the device has a mem pool (default ON).
// MLX_ROCM_NO_ASYNC_POOL=1 forces BufferCache + hipMalloc only.
static bool use_async_pool() {
  static const bool off = std::getenv("MLX_ROCM_NO_ASYNC_POOL") != nullptr;
  return !off;
}

static int alloc_device_tag() {
  return use_finegrained() ? -1 : 0;
}

inline void* rocm_unified_malloc(size_t size, bool& is_managed) {
  void* data = nullptr;
  hipError_t err;
  // Bind the alloc to the MLX-selected GPU. set_default_device(gpu,N) only sets
  // MLX bookkeeping; it never calls hipSetDevice. Without this, allocations made
  // OUTSIDE the eval path — notably the slab warmup at allocator construction —
  // land on whatever device is current (device 0 at startup), so the model's
  // small/intermediate tensors live on the APU while weights live on the dGPU.
  // A dGPU kernel then reads APU memory across the (TB5) link and hangs. Use raw
  // hipSetDevice (NOT device().make_current(), whose Device construction + device-
  // flags loop faults against device-0's already-created context).
  {
    mlx::core::Device dd = mlx::core::default_device();
    if (dd.type == mlx::core::Device::gpu) {
      int cur = -1;
      if (hipGetDevice(&cur) == hipSuccess && cur != dd.index)
        (void)hipSetDevice(dd.index);
    }
  }
  if (size > (16ull << 20) && std::getenv("MLX_ALLOC_DEBUG")) {
    int d = -1;
    (void)hipGetDevice(&d);
    fprintf(stderr, "[alloc] %zu MB curdev=%d defdev=%d finegrained=%d\n",
            size >> 20, d, mlx::core::default_device().index,
            (int)use_finegrained());
  }
  if (use_finegrained()) {
    // Integrated APU: unified LPDDR5, host-coherent. One pointer feeds kernels
    // (gpu_ptr) and the CPU (raw_ptr) — no host shadow, coherent at sync points.
    err = hipExtMallocWithFlags(&data, size, hipDeviceMallocFinegrained);
  } else {
    // Discrete GPU: coarse-grained VRAM (no coherency requirement). CPU access
    // goes through the pinned host shadow (ensure_host_shadow/flush_host_shadow).
    err = hipMalloc(&data, size);
  }
  if (err == hipSuccess) {
    is_managed = true;
    return data;
  }
  // Discrete-GPU training (MI300X): falling back to hipMallocManaged spills
  // into GTT (system RAM) once HBM is exhausted — silent thrash / SEGV later.
  // Default OFF on discrete GPUs; set MLX_ROCM_ALLOW_MANAGED_FALLBACK=1 to
  // re-enable (APU/HMM platforms). Explicit NO_MANAGED still wins.
  {
    int dev = 0;
    (void)hipGetDevice(&dev);
    const bool allow = std::getenv("MLX_ROCM_ALLOW_MANAGED_FALLBACK") != nullptr;
    const bool forbid = std::getenv("MLX_ROCM_NO_MANAGED_FALLBACK") != nullptr ||
        (!allow && !device_is_integrated(dev));
    if (forbid) {
      std::ostringstream oss;
      oss << "hipMalloc failed (device VRAM exhausted); managed/GTT fallback "
             "disabled: "
          << hipGetErrorString(err) << " size=" << size << ".";
      throw std::runtime_error(oss.str());
    }
  }
  // Fallbacks for platforms without fine-grained device memory.
  if (managed_memory_supported()) {
    err = hipMallocManaged(&data, size);
    is_managed = true;
  } else {
    err = hipHostMalloc(&data, size, hipHostMallocDefault);
    is_managed = false;
  }
  if (err != hipSuccess) {
    std::ostringstream oss;
    oss << "hipMalloc (unified) failed: " << hipGetErrorString(err) << ".";
    throw std::runtime_error(oss.str());
  }
  return data;
}

inline void rocm_unified_free(void* data, bool is_managed) {
  if (is_managed) {
    (void)hipFree(data);
  } else {
    (void)hipHostFree(data);
  }
}

// Apply memory hints for the managed-memory fallback path. Fine-grained device
// memory (the primary path) is already VRAM-resident, so these are no-ops there
// (errors swallowed); they only matter if rocm_unified_malloc fell back to HMM.
static void apply_slab_hints(void* data, size_t size) {
  if (!rocm_available())
    return;
  int device = 0;
  (void)hipGetDevice(&device);
  // Managed/SVM hints apply only to integrated (APU) memory. On discrete GPUs
  // they fail (hsa_amd_svm_attributes_set) and corrupt the HIP runtime.
  if (!device_is_integrated(device))
    return;
  (void)hipMemAdvise(data, size, hipMemAdviseSetAccessedBy, device);
  (void)hipMemPrefetchAsync(data, size, device, nullptr);
}

// ---------------------------------------------------------------------------
// SizeClassPool
// ---------------------------------------------------------------------------

void SizeClassPool::init(size_t block_size, size_t slab_page_size) {
  block_size_ = block_size;
  slab_page_size_ = slab_page_size;
}

SizeClassPool::~SizeClassPool() {
  for (size_t i = 0; i < backing_pages_.size(); i++) {
    rocm_unified_free(backing_pages_[i], is_managed_);
    delete[] block_arrays_[i];
  }
}

bool SizeClassPool::grow() {
  if (!rocm_available() || block_size_ == 0)
    return false;

  void* data = nullptr;
  try {
    data = rocm_unified_malloc(slab_page_size_, is_managed_);
  } catch (...) {
    return false;
  }

  // Apply memory hints for GPU access
  apply_slab_hints(data, slab_page_size_);

  size_t num_blocks = slab_page_size_ / block_size_;
  auto* blocks = new Block[num_blocks];

  // Chain blocks into the free list
  for (size_t i = 0; i < num_blocks; i++) {
    blocks[i].next = (i + 1 < num_blocks) ? &blocks[i + 1] : next_free_;
  }
  next_free_ = &blocks[0];

  backing_pages_.push_back(data);
  block_arrays_.push_back(blocks);
  blocks_per_page_.push_back(num_blocks);
  free_count_ += num_blocks;
  total_blocks_ += num_blocks;

  return true;
}

RocmBuffer* SizeClassPool::malloc() {
  if (next_free_ == nullptr)
    return nullptr;

  Block* b = next_free_;
  next_free_ = next_free_->next;
  free_count_--;

  // Fast path: single page (common case after warmup)
  if (block_arrays_.size() == 1) {
    size_t idx = static_cast<size_t>(b - block_arrays_[0]);
    b->buf.data = static_cast<char*>(backing_pages_[0]) + idx * block_size_;
    b->buf.size = block_size_;
    b->buf.is_managed = is_managed_;
    b->buf.device = alloc_device_tag();
    b->buf.host_shadow = nullptr;
    b->buf.host_dirty = false;
    return &b->buf;
  }

  // Multi-page: find which backing page this block belongs to
  for (size_t page = 0; page < block_arrays_.size(); page++) {
    Block* base = block_arrays_[page];
    size_t count = blocks_per_page_[page];
    if (b >= base && b < base + count) {
      size_t idx = static_cast<size_t>(b - base);
      b->buf.data =
          static_cast<char*>(backing_pages_[page]) + idx * block_size_;
      b->buf.size = block_size_;
      b->buf.is_managed = is_managed_;
      b->buf.device = alloc_device_tag();
      b->buf.host_shadow = nullptr;
      b->buf.host_dirty = false;
      return &b->buf;
    }
  }

  return nullptr;
}

void SizeClassPool::free(RocmBuffer* buf) {
  auto* b = reinterpret_cast<Block*>(buf);
  b->next = next_free_;
  next_free_ = b;
  free_count_++;
}

bool SizeClassPool::in_pool(RocmBuffer* buf) const {
  if (block_arrays_.empty())
    return false;
  auto* b = reinterpret_cast<const Block*>(buf);

  // Fast path: single page
  if (block_arrays_.size() == 1) {
    return b >= block_arrays_[0] && b < block_arrays_[0] + blocks_per_page_[0];
  }

  for (size_t page = 0; page < block_arrays_.size(); page++) {
    if (b >= block_arrays_[page] &&
        b < block_arrays_[page] + blocks_per_page_[page]) {
      return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// SlabAllocator
// ---------------------------------------------------------------------------

// Slab page sizes per tier (indexed by size class)
static constexpr size_t kSlabPageSizes[SlabAllocator::kNumSizeClasses] = {
    64 * 1024, // 8B blocks
    64 * 1024, // 16B
    64 * 1024, // 32B
    64 * 1024, // 64B
    64 * 1024, // 128B
    256 * 1024, // 256B
    256 * 1024, // 512B
    1024 * 1024, // 1KB
    1024 * 1024, // 2KB
    1024 * 1024, // 4KB
    1024 * 1024, // 8KB
    1024 * 1024, // 16KB
    2 * 1024 * 1024, // 32KB
    4 * 1024 * 1024, // 64KB
    8 * 1024 * 1024, // 128KB
    16 * 1024 * 1024, // 256KB
    32 * 1024 * 1024, // 512KB
    64 * 1024 * 1024, // 1MB
};

// Whether to pre-allocate each tier at startup
static constexpr bool kPreallocate[SlabAllocator::kNumSizeClasses] = {
    true,
    true,
    true,
    true,
    true, // 8B-128B
    true,
    true, // 256B-512B
    true,
    true,
    true,
    true,
    true, // 1KB-16KB
    false,
    false,
    false,
    false,
    false,
    false, // 32KB-1MB: on demand
};

SlabAllocator::SlabAllocator() {
  for (int i = 0; i < kNumSizeClasses; i++) {
    size_t block_size = static_cast<size_t>(1)
        << (i + 3); // 2^3=8 through 2^20=1MB
    pools_[i].init(block_size, kSlabPageSizes[i]);
  }
}

int SlabAllocator::size_class_index(size_t size) {
  if (size == 0 || size > kMaxSlabSize)
    return -1;
  if (size <= 8)
    return 0;
  // ceil(log2(size)) - 3, computed via bit manipulation
  int bits = 64 - __builtin_clzll(size - 1); // ceil(log2(size))
  return bits - 3;
}

size_t SlabAllocator::round_to_size_class(size_t size) {
  if (size <= 8)
    return 8;
  if (size > kMaxSlabSize)
    return size;
  // Round up to next power of 2
  return static_cast<size_t>(1) << (64 - __builtin_clzll(size - 1));
}

void SlabAllocator::warmup() {
  if (!rocm_available())
    return;
  for (int i = 0; i < kNumSizeClasses; i++) {
    if (kPreallocate[i]) {
      pools_[i].grow();
    }
  }
}

RocmBuffer* SlabAllocator::malloc(size_t size) {
  int idx = size_class_index(size);
  if (idx < 0)
    return nullptr;
  return pools_[idx].malloc();
}

void SlabAllocator::free(RocmBuffer* buf) {
  // O(1) dispatch: use buf->size to find the correct pool
  int idx = size_class_index(buf->size);
  if (idx >= 0 && pools_[idx].initialized()) {
    pools_[idx].free(buf);
  }
}

bool SlabAllocator::in_pool(RocmBuffer* buf) const {
  // O(1) dispatch: size determines the pool, then verify membership
  int idx = size_class_index(buf->size);
  if (idx >= 0 && pools_[idx].initialized()) {
    return pools_[idx].in_pool(buf);
  }
  return false;
}

bool SlabAllocator::grow(size_t size) {
  int idx = size_class_index(size);
  if (idx < 0)
    return false;
  return pools_[idx].grow();
}

size_t SlabAllocator::total_allocated() const {
  size_t total = 0;
  for (int i = 0; i < kNumSizeClasses; i++) {
    total += pools_[i].total_allocated();
  }
  return total;
}

size_t SlabAllocator::free_memory() const {
  size_t total = 0;
  for (int i = 0; i < kNumSizeClasses; i++) {
    total += pools_[i].free_memory();
  }
  return total;
}

// ---------------------------------------------------------------------------
// RocmAllocator — CUDA-aligned BufferCache + hipMallocAsync (see cuda/allocator.cpp)
// ---------------------------------------------------------------------------

// BufferCache free_ callback: free device memory + delete shell (CUDA free_cuda_buffer).
// Must not hold mutex_ when calling hipFreeAsync if we also take pool_mutex from
// another order — free_device only uses HIP APIs.
static void free_rocm_buffer_cb(RocmBuffer* buf);

RocmAllocator::RocmAllocator()
    : buffer_cache_(
          page_size,
          [](RocmBuffer* buf) { return buf->size; },
          [](RocmBuffer* buf) { free_rocm_buffer_cb(buf); }),
      memory_limit_(0),
      free_limit_(0),
      max_pool_size_(0),
      active_memory_(0),
      peak_memory_(0) {
  if (!rocm_available()) {
    return;
  }

  size_t free, total;
  hipError_t err = hipMemGetInfo(&free, &total);
  if (err == hipSuccess) {
    int dev = 0;
    (void)hipGetDevice(&dev);
    // Integrated APU: unified memory is shared with the CPU/system, so keep a
    // conservative cap. Discrete GPU: it is dedicated VRAM — use almost all of
    // it. The old 0.8 cap stranded ~6GB on a 32GB card, so once the working set
    // crossed 0.8*total every allocation evicted the buffer cache, and on a
    // discrete GPU each eviction is a blocking hipFree (waits on GPU drain) —
    // which stalls decode. Leave only a small reserve for driver/fragmentation.
    if (device_is_integrated(dev)) {
      // The APU's managed/fine-grained allocations live in the large unified
      // pool (system RAM / GTT), but hipMemGetInfo reports only the tiny
      // device-visible VRAM carveout. Sizing the cache to that carveout makes
      // the allocator evict on nearly every allocation, and each eviction is a
      // blocking hipFree that deadlocks under heavy async load (MTP). Size the
      // limit to system RAM, which is what the unified pool actually draws from.
      size_t sys_ram = static_cast<size_t>(sysconf(_SC_PHYS_PAGES)) *
          static_cast<size_t>(sysconf(_SC_PAGE_SIZE));
      memory_limit_ = std::max(
          static_cast<size_t>(total * 0.8), static_cast<size_t>(sys_ram * 0.8));
    } else {
      size_t reserve = 2ull << 30; // 2 GB driver/TTM headroom (was 512MB — tight)
      memory_limit_ = (total > reserve) ? (total - reserve) : total;
    }
    total_memory_ = total;
    free_limit_ = (total > memory_limit_) ? (total - memory_limit_) : 0;
    // CUDA: max_pool_size_ = memory_limit_ (host BufferCache free-list).
    max_pool_size_ = memory_limit_;
  }

  // Per-device hipMemPool + dedicated free stream for the async pool path.
  if (use_async_pool()) {
    int n = 0;
    (void)hipGetDeviceCount(&n);
    mem_pools_.resize(n, nullptr);
    free_streams_.resize(n, nullptr);
    int saved = 0;
    (void)hipGetDevice(&saved);
    for (int i = 0; i < n; ++i) {
      int supported = 0;
      (void)hipDeviceGetAttribute(
          &supported, hipDeviceAttributeMemoryPoolsSupported, i);
      if (!supported)
        continue;
      // gfx1201 (RDNA4): the ROCm async pool's stream-ordered reuse
      // (hipMallocAsync) hands back blocks whose prior owner's work hasn't
      // drained — regardless of our free discipline (confirmed: synchronizing the
      // stream before every free does NOT help; only avoiding hipMallocAsync
      // does). It produces intermittent garbage at long-context prefill. Leave
      // this device on the unified/slab path (malloc_async falls back when the
      // pool is null). Override with MLX_ROCM_FORCE_ASYNC_POOL=1.
      {
        hipDeviceProp_t props{};
        if (hipGetDeviceProperties(&props, i) == hipSuccess &&
            std::string(props.gcnArchName).find("gfx1201") != std::string::npos &&
            !std::getenv("MLX_ROCM_FORCE_ASYNC_POOL")) {
          continue;
        }
      }
      (void)hipSetDevice(i);
      hipMemPool_t pool = nullptr;
      if (hipDeviceGetDefaultMemPool(&pool, i) == hipSuccess) {
        mem_pools_[i] = pool;
        // Allow the driver to return idle pool memory to the device free list.
        // Default retention can pin nearly all HBM after a fat step → GTT spill.
        // 0 = release aggressively when the pool has unused reserved blocks.
        uint64_t threshold = 0;
        (void)hipMemPoolSetAttribute(
            pool, hipMemPoolAttrReleaseThreshold, &threshold);
        hipStream_t s = nullptr;
        if (hipStreamCreateWithFlags(&s, hipStreamNonBlocking) == hipSuccess)
          free_streams_[i] = s;
      }
    }
    (void)hipSetDevice(saved);
  }

  // Pre-allocate slab pages for common allocation sizes
  slab_allocator_.warmup();

}


// CUDA free_async equivalent: free device storage only (do not delete shell).
void RocmAllocator::free_device(RocmBuffer& buf, void* stream_v) {
  if (buf.host_shadow) {
    (void)hipHostFree(buf.host_shadow);
    buf.host_shadow = nullptr;
  }
  if (!buf.data) {
    return;
  }
  // device == -1 or is_managed with no stream pool → blocking free / unified.
  if (buf.device < 0 || buf.is_managed) {
    if (buf.device == -1) {
      rocm_unified_free(buf.data, buf.is_managed);
    } else {
      (void)hipFree(buf.data);
    }
    buf.data = nullptr;
    return;
  }
  // Stream-ordered pool buffer (CUDA: cudaFreeAsync).
  hipStream_t stream = static_cast<hipStream_t>(stream_v);
  if (!stream) {
    stream = static_cast<hipStream_t>(buf.alloc_stream);
  }
  if (!stream && buf.device >= 0 &&
      buf.device < static_cast<int>(free_streams_.size())) {
    stream = static_cast<hipStream_t>(free_streams_[buf.device]);
  }
  if (stream && buf.device >= 0 &&
      buf.device < static_cast<int>(mem_pools_.size()) && mem_pools_[buf.device]) {
    (void)hipFreeAsync(buf.data, stream);
  } else {
    (void)hipFree(buf.data);
  }
  buf.data = nullptr;
}

// CUDA free_cuda_buffer: free device mem + delete shell.
void RocmAllocator::free_rocm_buffer(RocmBuffer* buf) {
  if (!buf) {
    return;
  }
  if (slab_allocator_.in_pool(buf)) {
    slab_allocator_.free(buf);
    return;
  }
  free_device(*buf, buf->alloc_stream);
  delete buf;
}

static void free_rocm_buffer_cb(RocmBuffer* buf) {
  // Called from BufferCache under mutex_ while releasing. CUDA does the same
  // (free_cuda_buffer under lock). free_device only touches HIP.
  allocator().free_rocm_buffer(buf);
}

// CUDA: Buffer malloc(size) { return malloc_async(size, -1, nullptr); }
// ROCm: keep tiny slab path for ≤1MB, else CUDA path.
Buffer RocmAllocator::malloc(size_t size) {
  if (!rocm_available()) {
    throw std::runtime_error(
        "Cannot allocate ROCm memory: no ROCm-capable device detected. "
        "Please use CPU backend instead.");
  }
  if (decode_arena_.active && size > 0) {
    std::lock_guard lock(mutex_);
    if (RocmBuffer* b = arena_alloc(size)) {
      return Buffer{b};
    }
  }
  // Small sizes: ROCm slab freelist (CUDA uses 8-byte scalar pool).
  if (size > 0 && size <= SlabAllocator::kMaxSlabSize) {
    std::unique_lock lock(mutex_);
    size_t rounded = SlabAllocator::round_to_size_class(size);
    RocmBuffer* buf = slab_allocator_.malloc(rounded);
    if (!buf && slab_allocator_.grow(rounded)) {
      buf = slab_allocator_.malloc(rounded);
    }
    if (buf) {
      active_memory_ += rounded;
      peak_memory_ = std::max(active_memory_, peak_memory_);
      return Buffer{buf};
    }
  }
  // CUDA path: stream-less → device -1 (unified / hipMalloc).
  return malloc_async(size, /*device=*/-1, /*stream=*/nullptr);
}

// CUDA malloc_async — mirrored (hip names).
Buffer RocmAllocator::malloc_async(size_t size, int device, void* stream_v) {
  if (decode_arena_.active && size > 0) {
    std::lock_guard lock(mutex_);
    if (RocmBuffer* b = arena_alloc(size)) {
      return Buffer{b};
    }
  }

  if (size == 0) {
    return Buffer{new RocmBuffer{nullptr, 0, true, -1, nullptr, false, nullptr}};
  }

  hipStream_t stream = static_cast<hipStream_t>(stream_v);

  // CUDA size rounding.
  if (size < static_cast<size_t>(page_size)) {
    size_t p = 8;
    while (p < size)
      p <<= 1;
    size = p;
  } else {
    size = page_size * ((size + page_size - 1) / page_size);
  }

  // CUDA: tiny or no stream → unified path (device = -1).
  if (size <= 8 || stream == nullptr || !use_async_pool()) {
    device = -1;
  }
  if (device >= 0 &&
      (device >= static_cast<int>(mem_pools_.size()) || !mem_pools_[device])) {
    device = -1;
  }

  std::unique_lock lock(mutex_);
  RocmBuffer* buf = buffer_cache_.reuse_from_cache(size);
  if (!buf) {
    int64_t mem_to_free = static_cast<int64_t>(get_active_memory()) +
        static_cast<int64_t>(get_cache_memory()) + static_cast<int64_t>(size) -
        static_cast<int64_t>(memory_limit_);
    if (mem_to_free > 0) {
      buffer_cache_.release_cached_buffers(static_cast<size_t>(mem_to_free));
    }
    lock.unlock();

    void* data = nullptr;
    bool is_managed = true;
    if (device == -1) {
      data = rocm_unified_malloc(size, is_managed);
      // Discrete hipMalloc sets is_managed=true in rocm_unified_malloc; device
      // tag: APU uses -1, dGPU uses 0 with is_managed true (not stream-pool).
      int dev_tag = alloc_device_tag();
      buf = new RocmBuffer{
          data, size, is_managed, dev_tag, nullptr, false, nullptr};
    } else {
      hipError_t err = hipMallocAsync(&data, size, stream);
      if (err != hipSuccess || !data) {
        (void)hipGetLastError();
        // Fallback blocking alloc (CUDA throws; we fall back then throw).
        data = nullptr;
        err = hipMalloc(&data, size);
        if (err != hipSuccess || !data) {
          (void)hipGetLastError();
          throw std::runtime_error(
              "[malloc_async] Unable to allocate " + std::to_string(size) +
              " bytes.");
        }
        buf = new RocmBuffer{
            data, size, /*is_managed=*/true, device, nullptr, false, nullptr};
      } else {
        // Stream-pool buffer (CUDA device >= 0).
        buf = new RocmBuffer{
            data,
            size,
            /*is_managed=*/false,
            device,
            nullptr,
            false,
            stream};
      }
    }
    lock.lock();

    // CUDA: if mempool reserved is huge, release host cache.
    if (get_cache_memory() > 0 && device >= 0 &&
        device < static_cast<int>(mem_pools_.size()) && mem_pools_[device]) {
      size_t used = 0;
      if (hipMemPoolGetAttribute(
              static_cast<hipMemPool_t>(mem_pools_[device]),
              hipMemPoolAttrReservedMemCurrent,
              &used) == hipSuccess &&
          used > (total_memory_ - free_limit_)) {
        buffer_cache_.release_cached_buffers(
            free_limit_ ? free_limit_ : (size_t(1) << 30));
      }
    }
  } else {
    // Cache hit: re-bind stream for later free_device (CUDA does not store
    // stream; we do for ROCm free_async ordering when cache-evicted).
    if (device >= 0 && stream) {
      buf->alloc_stream = stream;
      buf->device = device;
    }
  }

  active_memory_ += buf->size;
  peak_memory_ = std::max(active_memory_, peak_memory_);
  if (get_cache_memory() > max_pool_size_) {
    buffer_cache_.release_cached_buffers(get_cache_memory() - max_pool_size_);
  }
  return Buffer{buf};
}

static std::mutex g_deferred_mutex;
// Frees deferred during graph build, tagged with the graph generation (chunk)
// active when the buffer was freed. A generation's buffers are reclaimed once
// that chunk's launch has completed — NOT hoarded until the per-token
// synchronize, which ballooned graph-mode memory to a whole token's working set.
static std::vector<std::pair<uint64_t, Buffer>> g_deferred_frees;
static std::atomic<uint64_t> g_graph_gen{1};
static std::atomic<size_t> g_deferred_bytes{0};

size_t graph_deferred_bytes() {
  return g_deferred_bytes.load(std::memory_order_relaxed);
}

uint64_t graph_current_gen() {
  return g_graph_gen.load(std::memory_order_relaxed);
}
void graph_advance_gen() {
  g_graph_gen.fetch_add(1, std::memory_order_relaxed);
}

// Reclaim every deferred buffer from a completed generation (<= gen). Launches
// retire in stream order, so once chunk `gen` is done, all earlier chunks are
// too — freeing <= gen never races an in-flight node.
void free_graph_generation(uint64_t gen) {
  std::vector<Buffer> to_free;
  {
    std::lock_guard<std::mutex> lk(g_deferred_mutex);
    std::vector<std::pair<uint64_t, Buffer>> keep;
    keep.reserve(g_deferred_frees.size());
    for (auto& p : g_deferred_frees) {
      if (p.first <= gen) {
        to_free.push_back(p.second);
      } else {
        keep.push_back(p);
      }
    }
    g_deferred_frees.swap(keep);
  }
  // Diagnostic (MLX_GRAPH_POISON_FREE): instead of releasing, overwrite the
  // buffer with a sentinel and LEAK it (keep it mapped). If a later kernel reads
  // a buffer we freed too early, the output turns to garbage (not a crash) and
  // the buffer stays mapped — so the fault is a read-after-free we can bisect,
  // rather than an unmapped-page segfault.
  static const bool poison = std::getenv("MLX_GRAPH_POISON_FREE") != nullptr;
  for (auto b : to_free) {
    auto* buf = static_cast<RocmBuffer*>(b.ptr());
    if (buf) {
      g_deferred_bytes.fetch_sub(buf->size, std::memory_order_relaxed);
    }
    if (poison) {
      if (buf && buf->data) {
        (void)hipMemset(buf->data, 0x7F, buf->size);
      }
      continue; // leak: keep it mapped with sentinel contents
    }
    allocator().free(b, /*force=*/true);
  }
}

void flush_graph_deferred_frees() {
  free_graph_generation(~uint64_t(0));
}

// Reclaim a completed generation's STREAM-ORDERED pool buffers via hipFreeAsync
// on their alloc (= generation) stream. Called inline at commit() right after the
// chunk's hipGraphLaunch, so the free is queued AFTER the launch on the same
// stream and retires after the graph that reads the buffer — no blocking pipeline
// drain. Unified/slab buffers (not stream-ordered) are left deferred for the
// synchronize flush / the sync+flush cap backstop.
void free_graph_generation_async(uint64_t gen) {
  std::vector<Buffer> to_free;
  {
    std::lock_guard<std::mutex> lk(g_deferred_mutex);
    std::vector<std::pair<uint64_t, Buffer>> keep;
    keep.reserve(g_deferred_frees.size());
    for (auto& p : g_deferred_frees) {
      auto* buf = static_cast<RocmBuffer*>(p.second.ptr());
      bool stream_ordered = buf && buf->device >= 0 && !buf->is_managed &&
          buf->alloc_stream != nullptr;
      if (p.first <= gen && stream_ordered) {
        g_deferred_bytes.fetch_sub(buf->size, std::memory_order_relaxed);
        to_free.push_back(p.second);
      } else {
        keep.push_back(p);
      }
    }
    g_deferred_frees.swap(keep);
  }
  for (auto b : to_free) {
    // force=true -> free_async -> hipFreeAsync(buf->data, buf->alloc_stream).
    allocator().free(b, /*force=*/true);
  }
}

void RocmAllocator::free(Buffer buffer, bool force) {
  auto* buf = static_cast<RocmBuffer*>(buffer.ptr());
  if (!buf) {
    return;
  }
  if (buf->size == 0) {
    delete buf;
    return;
  }
  if (buf->data && decode_arena_.contains(buf->data)) {
    return;
  }
  if (buf->alloc_stream == reinterpret_cast<void*>(static_cast<uintptr_t>(1))) {
    std::lock_guard lock(mutex_);
    if (active_memory_ >= buf->size)
      active_memory_ -= buf->size;
    delete buf;
    return;
  }
  static const bool nodefer = std::getenv("MLX_GRAPH_NODEFER") != nullptr;
  if (!force && !nodefer && graph_active()) {
    g_deferred_bytes.fetch_add(buf->size, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_deferred_mutex);
    g_deferred_frees.push_back(
        {g_graph_gen.load(std::memory_order_relaxed), buffer});
    return;
  }
  std::unique_lock lock(mutex_);
  active_memory_ -= buf->size;
  if (slab_allocator_.in_pool(buf)) {
    slab_allocator_.free(buf);
    return;
  }
  // CUDA: BufferCache first.
  if (get_cache_memory() < max_pool_size_) {
    buffer_cache_.recycle_to_cache(buf);
    return;
  }
  free_rocm_buffer(buf);
}


Buffer RocmAllocator::make_buffer(void* ptr, size_t size) {
  // Alien wrap (Metal/CUDA parity). Caller owns lifetime via release().
  if (!ptr || size == 0) {
    return Buffer{nullptr};
  }
  auto* rb = new RocmBuffer{
      ptr,
      size,
      /*is_managed=*/true, // not stream-pool; release() only deletes shell
      alloc_device_tag(),
      nullptr,
      false,
      nullptr};
  // Mark as alien: free/release must NOT hipFree the user pointer.
  // We encode alien by is_managed=true and alloc_stream = (void*)1 sentinel
  // checked in release(); free() of alien is a no-op on device mem.
  rb->alloc_stream = reinterpret_cast<void*>(static_cast<uintptr_t>(1));
  std::lock_guard lock(mutex_);
  active_memory_ += size;
  peak_memory_ = std::max(active_memory_, peak_memory_);
  return Buffer{rb};
}

void RocmAllocator::release(Buffer buffer) {
  auto* buf = static_cast<RocmBuffer*>(buffer.ptr());
  if (!buf) {
    return;
  }
  {
    std::lock_guard lock(mutex_);
    if (active_memory_ >= buf->size)
      active_memory_ -= buf->size;
  }
  // Alien: only free the shell. Owned buffers: real free.
  const bool alien =
      buf->alloc_stream == reinterpret_cast<void*>(static_cast<uintptr_t>(1));
  if (alien) {
    delete buf;
    return;
  }
  free(buffer, /*force=*/true);
}

size_t RocmAllocator::size(Buffer buffer) const {
  auto* buf = static_cast<RocmBuffer*>(buffer.ptr());
  if (!buf) {
    return 0;
  }
  return buf->size;
}

// --- Deterministic decode arena ---------------------------------------------
// Bump-allocate `size` from the arena. Caller holds mutex_. Returns nullptr on
// overflow (caller falls back to the normal pool, keeping correctness; the
// token is then non-deterministic and the engine must rebuild rather than
// relaunch). Wrappers live in a deque so their RocmBuffer* stay stable and are
// recycled across token resets — allocation #N reuses wrapper #N at the same
// arena offset, hence the same device address every token.
RocmBuffer* RocmAllocator::arena_alloc(size_t size) {
  auto& a = decode_arena_;
  if (!a.active || !a.base || size == 0) {
    return nullptr;
  }
  size_t aligned = (size + 255) & ~size_t(255);
  if (a.offset + aligned > a.capacity) {
    a.overflowed = true;
    return nullptr;
  }
  RocmBuffer* rb;
  if (a.next_wrapper < a.wrappers.size()) {
    rb = &a.wrappers[a.next_wrapper];
  } else {
    a.wrappers.emplace_back();
    rb = &a.wrappers.back();
  }
  a.next_wrapper++;
  rb->data = static_cast<char*>(a.base) + a.offset;
  rb->size = size;
  rb->is_managed = true;   // unified backing; raw_ptr serves CPU reads directly
  rb->device = -1;         // unified (APU): no host shadow
  rb->host_shadow = nullptr;
  rb->host_dirty = false;
  rb->alloc_stream = nullptr;
  a.offset += aligned;
  a.high_water = std::max(a.high_water, a.offset);
  return rb;
}

bool RocmAllocator::decode_arena_begin(size_t capacity, int device,
                                       void* stream) {
  (void)device;
  (void)stream;
  std::lock_guard lock(mutex_);
  if (!decode_arena_.base || decode_arena_.capacity < capacity) {
    if (decode_arena_.base) {
      (void)hipFree(decode_arena_.base);
      active_memory_ -= decode_arena_.capacity;
      decode_arena_.base = nullptr;
      decode_arena_.capacity = 0;
    }
    void* p = nullptr;
    // Use the same backing as the normal allocator: fine-grained device memory
    // on the APU (host-coherent, NO managed-migration overhead) and proper VRAM
    // on a dGPU. hipMallocManaged here makes the captured forward pay GPU page-
    // fault/migration costs every relaunch, which erased the graph's launch-
    // batching win. rocm_unified_malloc throws on hard failure.
    bool arena_managed = false;
    p = rocm_unified_malloc(capacity, arena_managed);
    if (!p) {
      (void)hipGetLastError();
      return false;
    }
    decode_arena_.base = p;
    decode_arena_.capacity = capacity;
    active_memory_ += capacity;
    peak_memory_ = std::max(active_memory_, peak_memory_);
  }
  decode_arena_.offset = 0;
  decode_arena_.next_wrapper = 0;
  decode_arena_.overflowed = false;
  decode_arena_.high_water = 0;
  decode_arena_.active = true;
  return true;
}

void RocmAllocator::decode_arena_reset() {
  std::lock_guard lock(mutex_);
  decode_arena_.offset = 0;
  decode_arena_.next_wrapper = 0;
  decode_arena_.overflowed = false;
}

void RocmAllocator::decode_arena_freeze_floor() {
  std::lock_guard lock(mutex_);
  decode_arena_.floor_offset = decode_arena_.offset;
  decode_arena_.floor_wrapper = decode_arena_.next_wrapper;
}

void RocmAllocator::decode_arena_reset_to_floor() {
  std::lock_guard lock(mutex_);
  // Preserve [0, floor_offset) (the recorded exec's baked buffers) and their
  // wrappers; new (sampling) allocations resume above the floor.
  decode_arena_.offset = decode_arena_.floor_offset;
  decode_arena_.next_wrapper = decode_arena_.floor_wrapper;
  decode_arena_.overflowed = false;
}

void RocmAllocator::decode_arena_end() {
  std::lock_guard lock(mutex_);
  decode_arena_.active = false;
}


void RocmAllocator::ensure_host_shadow(RocmBuffer& buf) {
  // Integrated APU buffers are already host-coherent — never reached.
  if (buf.device == -1) {
    return;
  }
  // Allocate the pinned host mirror once, then refresh it from VRAM. The VRAM
  // copy in buf.data is KEPT (no hipFree, device stays != -1) so gpu_ptr()
  // keeps feeding kernels the resident device pointer; only CPU reads see the
  // host mirror. No per-weight VRAM doubling / migration.
  if (buf.host_shadow == nullptr) {
    hipError_t err =
        hipHostMalloc(&buf.host_shadow, buf.size, hipHostMallocDefault);
    if (err != hipSuccess) {
      buf.host_shadow = nullptr;
      std::ostringstream oss;
      oss << "hipHostMalloc (host shadow) failed: " << hipGetErrorString(err)
          << ".";
      throw std::runtime_error(oss.str());
    }
  }
  // Refresh from VRAM only when the shadow is NOT already the authoritative copy
  // (i.e. no un-flushed CPU writes pending) — otherwise we'd clobber them.
  if (!buf.host_dirty) {
    hipError_t err =
        hipMemcpy(buf.host_shadow, buf.data, buf.size, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
      std::ostringstream oss;
      oss << "hipMemcpy (host shadow) failed: " << hipGetErrorString(err) << ".";
      throw std::runtime_error(oss.str());
    }
  }
}

void RocmAllocator::flush_host_shadow(RocmBuffer& buf) {
  if (buf.host_shadow == nullptr || !buf.host_dirty) {
    return;
  }
  (void)hipMemcpy(buf.data, buf.host_shadow, buf.size, hipMemcpyHostToDevice);
  buf.host_dirty = false;
}

size_t RocmAllocator::get_active_memory() const {
  return active_memory_;
}

size_t RocmAllocator::get_peak_memory() const {
  return peak_memory_;
}

void RocmAllocator::reset_peak_memory() {
  std::lock_guard lock(mutex_);
  peak_memory_ = 0;
}

size_t RocmAllocator::get_memory_limit() {
  return memory_limit_;
}

size_t RocmAllocator::set_memory_limit(size_t limit) {
  std::lock_guard lock(mutex_);
  std::swap(limit, memory_limit_);
  free_limit_ = (total_memory_ > memory_limit_) ? (total_memory_ - memory_limit_) : 0;
  return limit;
}

size_t RocmAllocator::get_cache_memory() const {
  return buffer_cache_.cache_size();
}

size_t RocmAllocator::set_cache_limit(size_t limit) {
  std::lock_guard lk(mutex_);
  std::swap(limit, max_pool_size_);
  return limit;
}

void RocmAllocator::clear_cache() {
  std::lock_guard lk(mutex_);
  buffer_cache_.clear();
}



RocmAllocator& allocator() {
  static RocmAllocator* allocator_ = new RocmAllocator;
  return *allocator_;
}

Buffer malloc_async(size_t size, CommandEncoder& encoder) {
  return allocator().malloc_async(
      size,
      encoder.device().hip_device(),
      static_cast<hipStream_t>(encoder.stream()));
}

} // namespace rocm

namespace allocator {

Allocator& allocator() {
  return rocm::allocator();
}

void* Buffer::raw_ptr() {
  if (!ptr_) {
    return nullptr;
  }
  auto& cbuf = *static_cast<rocm::RocmBuffer*>(ptr_);

  if (cbuf.device == -1) {
    // Unified memory on iGPU: fine-grained coherent memory means CPU sees
    // GPU writes without explicit sync. Only sync if the stream has pending
    // work (hipStreamQuery returns hipErrorNotReady when busy).
    if (hipStreamQuery(nullptr) != hipSuccess) {
      (void)hipStreamSynchronize(nullptr);
    }
  } else {
    // Discrete GPU: serve CPU access from the pinned host mirror (fresh D2H),
    // keeping the VRAM copy authoritative. Synchronize the device first so the
    // producing kernel has finished before the D2H read — a lighter null-stream
    // query is NOT sufficient (the value may be produced on a non-default stream)
    // and reading early returns stale zeros (crashes / garbage).
    (void)hipDeviceSynchronize();
    rocm::allocator().ensure_host_shadow(cbuf);
    return cbuf.host_shadow;
  }
  return cbuf.data;
}


// Alien (externally-owned) buffers can be wrapped without a copy, matching
// the cuda/metal/no_gpu backends (this was missing for ROCm).
bool can_reuse_alien_buffer(void* /*ptr*/) {
  return true;
}
} // namespace allocator

size_t get_active_memory() {
  return rocm::allocator().get_active_memory();
}
size_t get_peak_memory() {
  return rocm::allocator().get_peak_memory();
}
void reset_peak_memory() {
  return rocm::allocator().reset_peak_memory();
}
size_t set_memory_limit(size_t limit) {
  return rocm::allocator().set_memory_limit(limit);
}
size_t get_memory_limit() {
  return rocm::allocator().get_memory_limit();
}
size_t get_cache_memory() {
  return rocm::allocator().get_cache_memory();
}
size_t set_cache_limit(size_t limit) {
  return rocm::allocator().set_cache_limit(limit);
}
void clear_cache() {
  rocm::allocator().clear_cache();
}

// Not supported in ROCm.
size_t set_wired_limit(size_t) {
  return 0;
}

// --- Decode-arena bridge (called from the engine's graph-decode loop) -------
bool decode_arena_begin(size_t capacity, int device, void* stream) {
  return rocm::allocator().decode_arena_begin(capacity, device, stream);
}
void decode_arena_reset() {
  rocm::allocator().decode_arena_reset();
}
void decode_arena_freeze_floor() {
  rocm::allocator().decode_arena_freeze_floor();
}
void decode_arena_reset_to_floor() {
  rocm::allocator().decode_arena_reset_to_floor();
}
void decode_arena_end() {
  rocm::allocator().decode_arena_end();
}
bool decode_arena_active() {
  return rocm::allocator().decode_arena_active();
}
size_t decode_arena_high_water() {
  return rocm::allocator().decode_arena_high_water();
}
bool decode_arena_overflowed() {
  return rocm::allocator().decode_arena_overflowed();
}

} // namespace mlx::core
