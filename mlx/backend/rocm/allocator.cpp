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

inline void* rocm_unified_malloc(size_t size, bool& is_managed) {
  void* data = nullptr;
  hipError_t err;
  if (size > (256ull << 20) && std::getenv("MLX_ALLOC_DEBUG")) {
    int d = -1;
    (void)hipGetDevice(&d);
    fprintf(stderr, "[alloc] %zu MB on hip device %d\n", size >> 20, d);
  }
  // Fine-grained device memory is the right primitive on BOTH targets:
  //  - Integrated APU (gfx1151): allocates from unified LPDDR5, host-coherent.
  //  - Discrete RDNA4 (gfx1201): allocates VRAM-RESIDENT memory that is also
  //    mapped into the host address space over the PCIe BAR (ReBAR). One pointer
  //    feeds kernels at full VRAM bandwidth (gpu_ptr) and the CPU directly
  //    (raw_ptr) — no host shadow, no migration, coherent at sync points.
  // Measured on gfx1201: a 1 GiB fine-grained alloc consumes the full 1074 MB of
  // VRAM and is CPU read/writable, whereas hipMallocManaged migrates only ~150
  // MB to the device and streams the rest over PCIe (~11 GB/s).
  err = hipExtMallocWithFlags(&data, size, hipDeviceMallocFinegrained);
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
  // Hint: GPU is the primary accessor.
  (void)hipMemAdvise(data, size, hipMemAdviseSetAccessedBy, device);
  // Prefetch to GPU to avoid cold-start page faults.
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
    b->buf.device = -1;
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
      b->buf.device = -1;
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
    memory_limit_ = total * 0.8;
    max_pool_size_ = memory_limit_;
  }

  // Pre-allocate slab pages for common allocation sizes
  slab_allocator_.warmup();
}

Buffer RocmAllocator::malloc(size_t size) {
  if (!rocm_available()) {
    throw std::runtime_error(
        "Cannot allocate ROCm memory: no ROCm-capable device detected. "
        "Please use CPU backend instead.");
  }

  // Arena fast path: deterministic bump allocation for HIP Graph capture
  if (arena_.active()) {
    RocmBuffer* buf = arena_.malloc(size);
    if (buf)
      return Buffer{buf};
    // Arena exhausted — fall through to normal path
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
  } else {
    // Large allocation: page-align
    size = page_size * ((size + page_size - 1) / page_size);
  }

  // Try BufferCache
  RocmBuffer* buf = buffer_cache_.reuse_from_cache(size);
  if (!buf) {
    // Memory pressure: try to reclaim cache
    int64_t mem_to_free =
        get_active_memory() + get_cache_memory() + size - memory_limit_;
    if (mem_to_free > 0) {
      buffer_cache_.release_cached_buffers(mem_to_free);
    }

    lock.unlock();
    // Both the integrated APU and the discrete RDNA4 GPU use fine-grained device
    // memory with device == -1: the allocation is VRAM-resident (full bandwidth
    // for kernels via gpu_ptr) and host-coherent over the BAR (CPU access via
    // raw_ptr returns the same pointer). No host shadow, no migration.
    bool is_managed = false;
    void* data = rocm_unified_malloc(size, is_managed);
    buf = new RocmBuffer{data, size, is_managed, -1, nullptr, false};
    lock.lock();
  }
  active_memory_ += size;
  peak_memory_ = std::max(active_memory_, peak_memory_);

  // Maintain cache below limit
  if (get_cache_memory() > max_pool_size_) {
    buffer_cache_.release_cached_buffers(get_cache_memory() - max_pool_size_);
  }
  return Buffer{buf};
}

static std::mutex g_deferred_mutex;
static std::vector<Buffer> g_deferred_frees;

void flush_graph_deferred_frees() {
  std::vector<Buffer> to_free;
  {
    std::lock_guard<std::mutex> lk(g_deferred_mutex);
    to_free.swap(g_deferred_frees);
  }
  for (auto b : to_free) {
    allocator().free(b);
  }
}

void RocmAllocator::free(Buffer buffer) {
  auto* buf = static_cast<RocmBuffer*>(buffer.ptr());
  if (!buf) {
    return;
  }

  // Defer all frees while a captured graph is alive so its baked buffer
  // addresses stay valid through replay.
  if (graph_active()) {
    std::lock_guard<std::mutex> lk(g_deferred_mutex);
    g_deferred_frees.push_back(buffer);
    return;
  }

  // Arena fast path: no-op (memory freed in bulk on arena.end())
  if (arena_.active()) {
    arena_.free(buf);
    return;
  }

  std::unique_lock lock(mutex_);
  active_memory_ -= buf->size;

  // Slab-allocated buffers go back to the slab free list
  if (slab_allocator_.in_pool(buf)) {
    slab_allocator_.free(buf);
    return;
  }

  // Large buffers go to the BufferCache
  if (get_cache_memory() < max_pool_size_) {
    buffer_cache_.recycle_to_cache(buf);
  } else {
    rocm_free(buf);
  }
}

size_t RocmAllocator::size(Buffer buffer) const {
  auto* buf = static_cast<RocmBuffer*>(buffer.ptr());
  if (!buf) {
    return 0;
  }
  return buf->size;
}

void RocmAllocator::rocm_free(RocmBuffer* buf) {
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
  return limit;
}

void RocmAllocator::clear_cache() {
  std::lock_guard lk(mutex_);
  buffer_cache_.clear();
}

// ---------------------------------------------------------------------------
// DecodeArena implementation
// ---------------------------------------------------------------------------

DecodeArena::~DecodeArena() {
  end();
}

bool DecodeArena::begin(size_t capacity_bytes) {
  if (base_)
    end();

  // Align capacity to page boundary
  capacity_bytes = (capacity_bytes + 4095) & ~size_t(4095);

  bool managed = false;
  void* data = nullptr;
  try {
    data = rocm_unified_malloc(capacity_bytes, managed);
  } catch (...) {
    return false;
  }

  base_ = data;
  capacity_ = capacity_bytes;
  offset_ = 0;
  is_managed_ = managed;
  desc_index_ = 0;
  descriptors_.clear();
  descriptors_.reserve(512); // Typical decode step has ~300 allocations
  return true;
}

void DecodeArena::reset() {
  offset_ = 0;
  desc_index_ = 0;
}

void DecodeArena::end() {
  if (!base_)
    return;
  rocm_unified_free(base_, is_managed_);
  base_ = nullptr;
  capacity_ = 0;
  offset_ = 0;
  descriptors_.clear();
  desc_index_ = 0;
}

RocmBuffer* DecodeArena::malloc(size_t size) {
  if (!base_)
    return nullptr;

  // Align to 256 bytes for GPU access patterns
  size_t aligned = (size + 255) & ~size_t(255);
  if (offset_ + aligned > capacity_)
    return nullptr;

  void* ptr = static_cast<char*>(base_) + offset_;
  offset_ += aligned;

  // Reuse or create a RocmBuffer descriptor
  if (desc_index_ < descriptors_.size()) {
    auto& d = descriptors_[desc_index_];
    d.data = ptr;
    d.size = size;
    d.host_shadow = nullptr;
    d.host_dirty = false;
    desc_index_++;
    return &d;
  }

  // Fully initialize host_shadow/host_dirty: gpu_ptr() reads host_dirty, so an
  // uninitialized value could spuriously trigger a flush of a garbage pointer.
  descriptors_.push_back(RocmBuffer{ptr, size, is_managed_, -1, nullptr, false});
  desc_index_++;
  return &descriptors_.back();
}

RocmAllocator& allocator() {
  static RocmAllocator* allocator_ = new RocmAllocator;
  return *allocator_;
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
    // Discrete GPU: serve the CPU access from the pinned host mirror; keep the
    // VRAM copy resident. Mark dirty so any CPU write is flushed back to VRAM by
    // the next gpu_ptr(). Kernels still get VRAM via gpu_ptr().
    (void)hipDeviceSynchronize();
    rocm::allocator().ensure_host_shadow(cbuf);
    cbuf.host_dirty = true;
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

} // namespace mlx::core
