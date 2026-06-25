// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/utils.h"
#include "mlx/device.h"
#include "mlx/memory.h"
#include "mlx/utils.h"

#include <hip/hip_runtime.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <mutex>
#include <sstream>

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

// CUDA-style stream-ordered device pool (hipMallocAsync/hipFreeAsync). Always
// on where the device supports memory pools; allocations fall back to the
// unified path only for pool-less devices or stream-less requests.
static bool use_async_pool() {
  // MLX_ROCM_NO_ASYNC_POOL: fall back to the unified/slab path. The ROCm async
  // pool (hipMallocAsync) faults inside the driver under the HIP-graph workload
  // on gfx1201 (R9700) during async decode — this isolates that.
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
// RocmAllocator
// ---------------------------------------------------------------------------

RocmAllocator::RocmAllocator()
    : buffer_cache_(
          page_size,
          [](RocmBuffer* buf) { return buf->size; },
          [this](RocmBuffer* buf) { rocm_free(buf); }),
      memory_limit_(0),
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
      size_t reserve = 512ull << 20; // 512 MB driver/TTM headroom
      memory_limit_ = (total > reserve) ? (total - reserve) : total;
    }
    max_pool_size_ = memory_limit_;
    total_memory_ = total;
    free_limit_ = (total > memory_limit_) ? (total - memory_limit_) : 0;
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
      (void)hipSetDevice(i);
      hipMemPool_t pool = nullptr;
      if (hipDeviceGetDefaultMemPool(&pool, i) == hipSuccess) {
        mem_pools_[i] = pool;
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

// Unified-path frees deferred out of free() so they never run a blocking
// hipFree on the completion-worker thread (which self-deadlocks). Drained by
// malloc on the eval thread, where a blocking hipFree is safe. Pool buffers
// don't use this — they free non-blocking via hipFreeAsync.
static std::mutex g_pending_free_mutex;
static std::vector<RocmBuffer*> g_pending_frees;

Buffer RocmAllocator::malloc(size_t size) {
  if (!rocm_available()) {
    throw std::runtime_error(
        "Cannot allocate ROCm memory: no ROCm-capable device detected. "
        "Please use CPU backend instead.");
  }

  // Deterministic decode arena: serve transient decode allocations from the
  // rewound bump region so addresses replay identically each token. On overflow
  // arena_alloc returns nullptr and we fall through to the normal path.
  if (decode_arena_.active) {
    std::lock_guard lock(mutex_);
    if (RocmBuffer* b = arena_alloc(size)) {
      return Buffer{b};
    }
  }

  // Drain deferred unified frees on this (eval) thread, outside any lock.
  {
    std::vector<RocmBuffer*> to_free;
    {
      std::lock_guard<std::mutex> lk(g_pending_free_mutex);
      to_free.swap(g_pending_frees);
    }
    for (auto* b : to_free) {
      rocm_free(b);
    }
  }

  auto orig_size = size;
  std::unique_lock lock(mutex_);

  // Round size to appropriate boundary
  if (size <= SlabAllocator::kMaxSlabSize) {
    size = SlabAllocator::round_to_size_class(size);

    // Try slab allocator (O(1) free-list pop)
    RocmBuffer* buf = slab_allocator_.malloc(size);
    if (buf) {
      active_memory_ += size;
      peak_memory_ = std::max(active_memory_, peak_memory_);
      return Buffer{buf};
    }

    // Pool exhausted — grow (holds lock during HIP alloc, acceptable for rare
    // path)
    if (slab_allocator_.grow(size)) {
      buf = slab_allocator_.malloc(size);
      if (buf) {
        active_memory_ += size;
        peak_memory_ = std::max(active_memory_, peak_memory_);
        return Buffer{buf};
      }
    }

    // Slab growth failed — fall through to BufferCache
    // Slab growth failed — fall through to BufferCache
  } else {
    // Large allocation: page-align
    size = page_size * ((size + page_size - 1) / page_size);
  }

  // Stream-less allocations (model load, KV, non-wired primitives) use unified
  // memory + the BufferCache. The wired primitives route their outputs through
  // malloc_async (the pool) instead; this path is the safe fallback.
  RocmBuffer* buf = buffer_cache_.reuse_from_cache(size);
  if (!buf) {
    int64_t mem_to_free =
        get_active_memory() + get_cache_memory() + size - memory_limit_;
    if (mem_to_free > 0) {
      buffer_cache_.release_cached_buffers(mem_to_free);
    }
    lock.unlock();
    bool is_managed = false;
    void* data = rocm_unified_malloc(size, is_managed);
    buf = new RocmBuffer{data, size, is_managed, alloc_device_tag(), nullptr, false, nullptr};
    lock.lock();
  }
  active_memory_ += size;
  peak_memory_ = std::max(active_memory_, peak_memory_);
  if (get_cache_memory() > max_pool_size_) {
    buffer_cache_.release_cached_buffers(get_cache_memory() - max_pool_size_);
  }
  return Buffer{buf};
}

// Serializes ROCm async-pool alloc/free across the eval and worker threads.
static std::mutex& pool_mutex() {
  static std::mutex m;
  return m;
}

Buffer RocmAllocator::malloc_async(size_t size, int device, void* stream_v) {
  // Deterministic decode arena takes priority over the stream-ordered pool so
  // per-token activation addresses replay identically (see arena_alloc).
  if (decode_arena_.active && size > 0) {
    std::lock_guard lock(mutex_);
    if (RocmBuffer* b = arena_alloc(size)) {
      return Buffer{b};
    }
  }

  hipStream_t stream = static_cast<hipStream_t>(stream_v);
  // Fall back to the unified path unless the pool is usable for this request.
  if (!use_async_pool() || stream == nullptr || device < 0 ||
      device >= static_cast<int>(mem_pools_.size()) ||
      mem_pools_[device] == nullptr || size == 0 ||
      size <= SlabAllocator::kMaxSlabSize) {
    return malloc(size);
  }

  size = page_size * ((size + page_size - 1) / page_size);

  // Bypass our BufferCache entirely: the hipMemPool already manages reuse and
  // retention (ReleaseThreshold=MAX). Layering our own eviction on top causes
  // hipFreeAsync storms that starve the HSA handler pool and wedge. Let the GPU
  // manage its own memory — alloc straight from the pool.
  void* data = nullptr;
  hipError_t err;
  {
    // Serialize pool ops: the eval thread allocates here while the worker thread
    // frees (free_async -> hipFreeAsync) concurrently; the ROCm async pool faults
    // under concurrent alloc+free on this path. g_pool_mtx orders them.
    std::lock_guard<std::mutex> plk(pool_mutex());
    err = hipMallocAsync(&data, size, stream);
  }
  if (err != hipSuccess || !data) {
    (void)hipGetLastError();
    return malloc(size); // pool exhausted: fall back to unified
  }
  // is_managed=false marks this as a stream-ordered pool buffer (freed via
  // hipFreeAsync); device>=0 routes CPU access through the host shadow.
  RocmBuffer* buf = new RocmBuffer{data, size, false, device, nullptr, false, stream};
  std::lock_guard lock(mutex_);
  active_memory_ += buf->size;
  peak_memory_ = std::max(active_memory_, peak_memory_);
  return Buffer{buf};
}

void RocmAllocator::free_async(RocmBuffer* buf, void* stream_v) {
  hipStream_t stream = static_cast<hipStream_t>(stream_v);
  // Free on the buffer's own alloc/eval stream so the free retires in order
  // behind its last use and the pool reclaims it (a separate idle free-stream
  // never executes during a forward, so the pool can't reuse and VRAM grows).
  if (!stream)
    stream = static_cast<hipStream_t>(buf->alloc_stream);
  if (!stream && buf->device >= 0 &&
      buf->device < static_cast<int>(free_streams_.size())) {
    stream = static_cast<hipStream_t>(free_streams_[buf->device]);
  }
  if (buf->host_shadow) {
    (void)hipHostFree(buf->host_shadow);
    buf->host_shadow = nullptr;
  }
  {
    std::lock_guard<std::mutex> plk(pool_mutex());
    if (stream) {
      (void)hipFreeAsync(buf->data, stream);
    } else {
      (void)hipFree(buf->data);
    }
  }
  delete buf;
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

  // Decode-arena buffers are owned by the arena (whole region rewound per
  // token); individual frees are no-ops. The backing region is released only by
  // the allocator teardown. Identify by pointer-range so RocmBuffer stays POD.
  if (decode_arena_.contains(buf->data)) {
    return;
  }

  // Defer frees while a graph is being built so its nodes' buffers stay valid
  // until the chunk launches. Tag with the current generation so it can be
  // reclaimed as soon as that chunk completes (see free_graph_generation).
  // force=true skips this (used by the deferred-free flush itself).
  // MLX_GRAPH_NODEFER: skip deferral entirely and rely on add_temporary (which
  // holds every graph input's array::Data until its chunk's completion handler)
  // — testing whether defer-all is redundant for the auto-batch path.
  static const bool nodefer = std::getenv("MLX_GRAPH_NODEFER") != nullptr;
  if (!force && !nodefer && graph_active()) {
    g_deferred_bytes.fetch_add(buf->size, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_deferred_mutex);
    g_deferred_frees.push_back({g_graph_gen.load(std::memory_order_relaxed),
                                buffer});
    return;
  }

  std::unique_lock lock(mutex_);
  active_memory_ -= buf->size;

  // Slab-allocated buffers go back to the slab free list
  if (slab_allocator_.in_pool(buf)) {
    slab_allocator_.free(buf);
    return;
  }

  // Stream-ordered pool buffer (the common case): return it straight to the
  // hipMemPool via hipFreeAsync on its own stream. The pool owns reuse/retention.
  if (buf->device >= 0 && !buf->is_managed) {
    free_async(buf, nullptr);
    return;
  }

  // Unified buffer (model load / KV / non-wired primitives). Recycle to the
  // BufferCache, or defer the blocking hipFree off the worker thread.
  if (get_cache_memory() < max_pool_size_) {
    buffer_cache_.recycle_to_cache(buf);
  } else {
    std::lock_guard<std::mutex> lk(g_pending_free_mutex);
    g_pending_frees.push_back(buf);
  }
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
    // Unified backing so the APU CPU (logits sampling) reads it directly.
    if (hipMallocManaged(&p, capacity) != hipSuccess || !p) {
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

void RocmAllocator::decode_arena_end() {
  std::lock_guard lock(mutex_);
  decode_arena_.active = false;
}

void RocmAllocator::rocm_free(RocmBuffer* buf) {
  // Stream-ordered pool buffer: free non-blocking via hipFreeAsync.
  if (buf->device >= 0 && !buf->is_managed) {
    free_async(buf, nullptr);
    return;
  }
  if (buf->host_shadow) {
    (void)hipHostFree(buf->host_shadow);
    buf->host_shadow = nullptr;
  }
  if (buf->device == -1) {
    rocm_unified_free(buf->data, buf->is_managed);
  } else {
    (void)hipFree(buf->data);
  }
  delete buf;
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
  return limit;
}

size_t RocmAllocator::get_cache_memory() const {
  // Only report BufferCache size. Slab free memory is infrastructure,
  // not cache — including it inflates the count and causes premature
  // eviction of large buffers from the BufferCache.
  return buffer_cache_.cache_size();
}

size_t RocmAllocator::set_cache_limit(size_t limit) {
  std::lock_guard lk(mutex_);
  std::swap(limit, max_pool_size_);
  // Trim the reuse pool down to the new cap NOW, while the caller is at an idle
  // point (e.g. just after warmup). Otherwise the trim happens lazily on the
  // next malloc — i.e. during the first forward — and its blocking hipFree
  // (which on a discrete GPU implicitly synchronizes the device and can force a
  // TTM eviction) wedges the command queue mid-pass.
  if (get_cache_memory() > max_pool_size_) {
    buffer_cache_.release_cached_buffers(get_cache_memory() - max_pool_size_);
  }
  return limit;
}

void RocmAllocator::clear_cache() {
  // The hipMemPool owns reuse/retention for pool buffers; releasing memory means
  // trimming it. Drain the device first so trimmed blocks have no outstanding
  // work, then drain deferred unified frees on this (safe) thread. Do NOT
  // blocking-clear the unified BufferCache under pool handler pressure — those
  // buffers are bounded by max_pool_size_ and reused.
  (void)hipDeviceSynchronize();
  for (void* p : mem_pools_) {
    if (p)
      (void)hipMemPoolTrimTo(static_cast<hipMemPool_t>(p), 0);
  }
  std::vector<RocmBuffer*> to_free;
  {
    std::lock_guard<std::mutex> lk(g_pending_free_mutex);
    to_free.swap(g_pending_frees);
  }
  for (auto* b : to_free) {
    rocm_free(b);
  }
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
