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

// Stream-ordered hipMemPool (hipMallocAsync / hipFreeAsync).
//
// DEFAULT OFF on discrete GPUs. hipMemPool retention is what pinned ~192GB HBM
// and spilled 10GB+ into GTT on MI300X train while a 1.5B model only needs
// ~15GB weights+Adam + ~30–70GB live acts. Metal uses host BufferCache + real
// free; CUDA's pool is better behaved. ROCm pool kept reserved pages forever.
//
//   MLX_ROCM_FORCE_ASYNC_POOL=1  opt-in async (decode experiments only)
//   MLX_ROCM_NO_ASYNC_POOL=1     force off (default behavior on dGPU)
static bool use_async_pool() {
  static const int mode = [] {
    if (std::getenv("MLX_ROCM_FORCE_ASYNC_POOL"))
      return 1;
    if (std::getenv("MLX_ROCM_NO_ASYNC_POOL"))
      return 0;
    // Discrete default OFF; integrated may use async if present.
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess)
      return 0;
    return device_is_integrated(dev) ? 1 : 0;
  }();
  return mode != 0;
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
// ManagedDevicePool
// ---------------------------------------------------------------------------

namespace {

constexpr size_t kManagedAlign = 256;
// Prefer modest slabs so fully-free slabs can hipFree back to the driver
// often (a single 8GB slab with one live page cannot shrink).
constexpr size_t kManagedMinSlab = 64ull << 20; // 64 MiB
constexpr size_t kManagedMaxSlab = 256ull << 20; // 256 MiB

size_t managed_align(size_t n) {
  return (n + kManagedAlign - 1) & ~(kManagedAlign - 1);
}

} // namespace

ManagedDevicePool::~ManagedDevicePool() {
  for (auto& s : slabs_) {
    if (s.base)
      (void)hipFree(s.base);
  }
  slabs_.clear();
  free_by_addr_.clear();
  free_by_size_.clear();
  live_.clear();
}

bool ManagedDevicePool::owns(const void* ptr) const {
  if (!ptr)
    return false;
  auto p = static_cast<const char*>(ptr);
  for (const auto& s : slabs_) {
    if (!s.base)
      continue;
    auto b = static_cast<const char*>(s.base);
    if (p >= b && p < b + s.capacity)
      return true;
  }
  return false;
}

void ManagedDevicePool::insert_free(void* ptr, size_t size, size_t slab_idx) {
  if (!ptr || size == 0)
    return;
  // Coalesce with next free block if adjacent.
  auto next = free_by_addr_.lower_bound(ptr);
  if (next != free_by_addr_.end()) {
    char* end = static_cast<char*>(ptr) + size;
    if (end == static_cast<char*>(next->first) &&
        next->second.slab_idx == slab_idx) {
      // absorb next
      auto range = free_by_size_.equal_range(next->second.size);
      for (auto it = range.first; it != range.second; ++it) {
        if (it->second == next->first) {
          free_by_size_.erase(it);
          break;
        }
      }
      size += next->second.size;
      free_list_bytes_ -= next->second.size;
      free_by_addr_.erase(next);
    }
  }
  // Coalesce with previous free block if adjacent.
  auto prev = free_by_addr_.lower_bound(ptr);
  if (prev != free_by_addr_.begin()) {
    --prev;
    char* prev_end =
        static_cast<char*>(prev->second.ptr) + prev->second.size;
    if (prev_end == ptr && prev->second.slab_idx == slab_idx) {
      auto range = free_by_size_.equal_range(prev->second.size);
      for (auto it = range.first; it != range.second; ++it) {
        if (it->second == prev->first) {
          free_by_size_.erase(it);
          break;
        }
      }
      size += prev->second.size;
      free_list_bytes_ -= prev->second.size;
      ptr = prev->second.ptr;
      free_by_addr_.erase(prev);
    }
  }
  FreeBlock fb{ptr, size, slab_idx};
  free_by_addr_[ptr] = fb;
  free_by_size_.emplace(size, ptr);
  free_list_bytes_ += size;
}

bool ManagedDevicePool::take_fit(size_t size, FreeBlock& out) {
  auto it = free_by_size_.lower_bound(size);
  if (it == free_by_size_.end())
    return false;
  // Prefer near fit (≤2×) when available; else first large enough.
  auto best = it;
  for (auto j = it; j != free_by_size_.end() && j->first <= size * 2 + (1u << 20);
       ++j) {
    best = j;
    break; // lower_bound already smallest >= size
  }
  void* ptr = best->second;
  free_by_size_.erase(best);
  auto ait = free_by_addr_.find(ptr);
  if (ait == free_by_addr_.end())
    return false;
  free_list_bytes_ -= ait->second.size;
  out = ait->second;
  free_by_addr_.erase(ait);
  return true;
}

bool ManagedDevicePool::grow_slab(size_t need, int device) {
  size_t cap = need;
  if (cap < kManagedMinSlab)
    cap = kManagedMinSlab;
  if (cap > kManagedMaxSlab && need <= kManagedMaxSlab)
    cap = kManagedMaxSlab;
  // Requests larger than max slab get an exact-sized dedicated slab.
  if (need > kManagedMaxSlab)
    cap = managed_align(need);
  cap = managed_align(cap);

  void* base = nullptr;
  hipError_t err = hipMalloc(&base, cap);
  if (err != hipSuccess || !base) {
    (void)hipGetLastError();
    // Retry exact need.
    cap = managed_align(need);
    err = hipMalloc(&base, cap);
    if (err != hipSuccess || !base) {
      (void)hipGetLastError();
      return false;
    }
  }
  Slab s;
  s.base = base;
  s.capacity = cap;
  s.live_bytes = 0;
  s.device = device;
  slabs_.push_back(s);
  reserved_bytes_ += cap;
  insert_free(base, cap, slabs_.size() - 1);
  return true;
}

RocmBuffer* ManagedDevicePool::malloc(size_t size, int device) {
  if (size == 0) {
    return new RocmBuffer{nullptr, 0, true, device, nullptr, false, nullptr};
  }
  size = managed_align(size);
  FreeBlock blk;
  if (!take_fit(size, blk)) {
    if (!grow_slab(size, device))
      return nullptr;
    if (!take_fit(size, blk))
      return nullptr;
  }
  void* ptr = blk.ptr;
  size_t rem = blk.size - size;
  if (rem >= kManagedAlign) {
    insert_free(static_cast<char*>(ptr) + size, rem, blk.slab_idx);
  } else {
    // Keep remainder attached (slight waste).
    size = blk.size;
  }
  slabs_[blk.slab_idx].live_bytes += size;
  live_[ptr] = LiveBlock{size, blk.slab_idx};
  live_bytes_ += size;
  // is_managed=true → not hipMallocAsync stream-pool; device>=0 → VRAM.
  // free goes through ManagedDevicePool via owns().
  return new RocmBuffer{
      ptr, size, /*is_managed=*/true, device, nullptr, false, nullptr};
}

void ManagedDevicePool::free(RocmBuffer* buf) {
  if (!buf) {
    return;
  }
  if (buf->host_shadow) {
    (void)hipHostFree(buf->host_shadow);
    buf->host_shadow = nullptr;
  }
  void* ptr = buf->data;
  if (ptr) {
    auto it = live_.find(ptr);
    if (it != live_.end()) {
      size_t sz = it->second.size;
      size_t si = it->second.slab_idx;
      live_bytes_ -= sz;
      if (si < slabs_.size()) {
        if (slabs_[si].live_bytes >= sz)
          slabs_[si].live_bytes -= sz;
        else
          slabs_[si].live_bytes = 0;
      }
      live_.erase(it);
      insert_free(ptr, sz, si);
      // Shrink: if this slab is fully free, return it to the driver.
      if (si < slabs_.size() && slabs_[si].live_bytes == 0 && slabs_[si].base) {
        // Remove all free blocks belonging to this slab.
        for (auto fit = free_by_addr_.begin(); fit != free_by_addr_.end();) {
          if (fit->second.slab_idx == si) {
            free_list_bytes_ -= fit->second.size;
            auto range = free_by_size_.equal_range(fit->second.size);
            for (auto sit = range.first; sit != range.second; ++sit) {
              if (sit->second == fit->first) {
                free_by_size_.erase(sit);
                break;
              }
            }
            fit = free_by_addr_.erase(fit);
          } else {
            ++fit;
          }
        }
        reserved_bytes_ -= slabs_[si].capacity;
        (void)hipFree(slabs_[si].base);
        slabs_[si].base = nullptr;
        slabs_[si].capacity = 0;
      }
    } else {
      // Not tracked as live — still try hipFree if outside pool (shouldn't).
      (void)hipFree(ptr);
    }
  }
  buf->data = nullptr;
  delete buf;
}

size_t ManagedDevicePool::shrink() {
  size_t freed = 0;
  for (size_t si = 0; si < slabs_.size(); ++si) {
    if (!slabs_[si].base || slabs_[si].live_bytes > 0)
      continue;
    for (auto fit = free_by_addr_.begin(); fit != free_by_addr_.end();) {
      if (fit->second.slab_idx == si) {
        free_list_bytes_ -= fit->second.size;
        auto range = free_by_size_.equal_range(fit->second.size);
        for (auto sit = range.first; sit != range.second; ++sit) {
          if (sit->second == fit->first) {
            free_by_size_.erase(sit);
            break;
          }
        }
        fit = free_by_addr_.erase(fit);
      } else {
        ++fit;
      }
    }
    freed += slabs_[si].capacity;
    reserved_bytes_ -= slabs_[si].capacity;
    (void)hipFree(slabs_[si].base);
    slabs_[si].base = nullptr;
    slabs_[si].capacity = 0;
  }
  return freed;
}

// ---------------------------------------------------------------------------
// RocmAllocator
// ---------------------------------------------------------------------------

// Pending device frees (BufferCache eviction + deferred unified frees).
// Declared before RocmAllocator ctor so the cache free_ lambda can queue.
static std::mutex g_pending_free_mutex;
static std::vector<RocmBuffer*> g_pending_frees;

// Device frees deferred while BufferCache mutates under mutex_ (avoids
// mutex_ + pool_mutex lock-order inversion with malloc_async).
static void queue_device_free(RocmBuffer* buf) {
  std::lock_guard<std::mutex> lk(g_pending_free_mutex);
  g_pending_frees.push_back(buf);
}

static void drain_device_frees() {
  std::vector<RocmBuffer*> to_free;
  {
    std::lock_guard<std::mutex> lk(g_pending_free_mutex);
    to_free.swap(g_pending_frees);
  }
  for (auto* b : to_free) {
    if (!b)
      continue;
    // Already removed from active_memory_. Just release device memory + shell.
    auto& alloc = allocator();
    if (alloc.free_if_managed_pool(b))
      continue;
    if (b->device >= 0 && !b->is_managed) {
      alloc.free_async(b, b->alloc_stream);
    } else {
      if (b->host_shadow) {
        (void)hipHostFree(b->host_shadow);
        b->host_shadow = nullptr;
      }
      if (b->data) {
        if (b->device == -1)
          rocm_unified_free(b->data, b->is_managed);
        else
          (void)hipFree(b->data);
        b->data = nullptr;
      }
      delete b;
    }
  }
}

bool RocmAllocator::free_if_managed_pool(RocmBuffer* buf) {
  if (!buf || !buf->data)
    return false;
  std::lock_guard lock(mutex_);
  if (!managed_pool_.owns(buf->data))
    return false;
  managed_pool_.free(buf);
  return true;
}

RocmAllocator::RocmAllocator()
    : buffer_cache_(
          page_size,
          [](RocmBuffer* buf) { return buf->size; },
          // Queue only — real free runs in drain_device_frees() without mutex_.
          [](RocmBuffer* buf) { queue_device_free(buf); }),
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
      size_t reserve = 2ull << 30; // 2 GB driver/TTM headroom (was 512MB — tight)
      memory_limit_ = (total > reserve) ? (total - reserve) : total;
    }
    total_memory_ = total;
    free_limit_ = (total > memory_limit_) ? (total - memory_limit_) : 0;
    // Host reuse cache: big enough for MoE pack temps (few GB), tiny vs 192GB.
    // Overridden by mx.set_cache_limit / LEMONSEED_CACHE_LIMIT_GB.
    max_pool_size_ = 8ull << 30;
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

  // Managed grow/shrink pool default ON for discrete GPUs.
  // MLX_ROCM_MANAGED_POOL=0 disables; =1 forces on (even integrated).
  {
    const char* e = std::getenv("MLX_ROCM_MANAGED_POOL");
    int dev = 0;
    (void)hipGetDevice(&dev);
    if (e && (e[0] == '0' || e[0] == 'f' || e[0] == 'F' || e[0] == 'n'))
      use_managed_pool_ = false;
    else if (e && (e[0] == '1' || e[0] == 't' || e[0] == 'T' || e[0] == 'y'))
      use_managed_pool_ = true;
    else
      use_managed_pool_ = !device_is_integrated(dev);
  }
}

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

  // Drain deferred device frees on this (eval) thread, outside any lock.
  drain_device_frees();

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

  // Large discrete allocs: single ManagedDevicePool (grow free-list, shrink
  // empty slabs). Model weights + activations share one manager; free when
  // last array ref drops returns the block without hipMemPool pin / GTT.
  if (use_managed_pool_ && size > SlabAllocator::kMaxSlabSize &&
      alloc_device_tag() >= 0) {
    int64_t mem_to_free =
        get_active_memory() + get_cache_memory() + size - memory_limit_;
    if (mem_to_free > 0) {
      buffer_cache_.release_cached_buffers(static_cast<size_t>(mem_to_free));
    }
    RocmBuffer* mbuf = managed_pool_.malloc(size, alloc_device_tag());
    if (mbuf) {
      active_memory_ += mbuf->size;
      peak_memory_ = std::max(active_memory_, peak_memory_);
      return Buffer{mbuf};
    }
    // Fall through to unified if grow failed.
  }

  // Stream-less / small / managed-pool-off: BufferCache + hipMalloc.
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

  drain_device_frees();

  if (size == 0) {
    return Buffer{new RocmBuffer{
        nullptr, 0, true, -1, nullptr, false, nullptr}};
  }

  hipStream_t stream = static_cast<hipStream_t>(stream_v);

  // Prefer ManagedDevicePool for large discrete allocs (default). Falls back
  // to malloc() which also uses the managed pool. hipMallocAsync only if
  // FORCE_ASYNC_POOL and managed pool off.
  if (use_managed_pool_ || !use_async_pool() || stream == nullptr ||
      device < 0 || device >= static_cast<int>(mem_pools_.size()) ||
      mem_pools_[device] == nullptr || size <= SlabAllocator::kMaxSlabSize) {
    return malloc(size);
  }

  size = page_size * ((size + page_size - 1) / page_size);

  // Soft limit: reclaim unified BufferCache before asking the pool for more.
  {
    std::lock_guard lock(mutex_);
    int64_t over = static_cast<int64_t>(get_active_memory()) +
        static_cast<int64_t>(get_cache_memory()) + static_cast<int64_t>(size) -
        static_cast<int64_t>(memory_limit_);
    if (over > 0 && get_cache_memory() > 0) {
      buffer_cache_.release_cached_buffers(static_cast<size_t>(over));
    }
  }
  drain_device_frees();

  // Keep HBM headroom so the driver does not migrate into GTT.
  {
    size_t free_b = 0, total_b = 0;
    if (hipMemGetInfo(&free_b, &total_b) == hipSuccess) {
      constexpr size_t kHeadroom = 4ull << 30;
      if (free_b < kHeadroom || free_b < size + (1ull << 30)) {
        {
          std::lock_guard lock(mutex_);
          if (get_cache_memory() > 0)
            buffer_cache_.release_cached_buffers(get_cache_memory());
        }
        drain_device_frees();
        (void)hipMemPoolTrimTo(
            static_cast<hipMemPool_t>(mem_pools_[device]), kHeadroom);
      }
    }
  }

  void* data = nullptr;
  hipError_t err;
  {
    std::lock_guard<std::mutex> plk(pool_mutex());
    err = hipMallocAsync(&data, size, stream);
  }
  if (err != hipSuccess || !data) {
    (void)hipGetLastError();
    (void)hipMemPoolTrimTo(static_cast<hipMemPool_t>(mem_pools_[device]), 0);
    std::lock_guard<std::mutex> plk(pool_mutex());
    err = hipMallocAsync(&data, size, stream);
  }
  if (err != hipSuccess || !data) {
    (void)hipGetLastError();
    // Blocking hipMalloc (no managed/GTT on discrete).
    return malloc(size);
  }

  // is_managed=false marks stream-pool buffer; alloc_stream is REQUIRED for
  // safe free_async (see free_async — no idle free_stream fallback).
  RocmBuffer* buf =
      new RocmBuffer{data, size, false, device, nullptr, false, stream};
  std::lock_guard lock(mutex_);
  active_memory_ += buf->size;
  peak_memory_ = std::max(active_memory_, peak_memory_);
  return Buffer{buf};
}

void RocmAllocator::free_async(RocmBuffer* buf, void* stream_v) {
  // ONLY for stream-ordered pool buffers (is_managed=false, device>=0).
  // Unified/hipMalloc frees go through drain_device_frees / rocm_free.
  if (!buf) {
    return;
  }
  // Alien shell from make_buffer.
  if (buf->alloc_stream == reinterpret_cast<void*>(static_cast<uintptr_t>(1))) {
    delete buf;
    return;
  }

  // Resolve free stream. NEVER use free_streams_[dev] (idle non-blocking
  // stream that does not wait for compute) — that let hipMemPool reclaim
  // blocks while kernels still ran (HIP 700 IMA). CUDA's free_stream is
  // joined via the mempool; ROCm's was not used correctly in MLX.
  hipStream_t stream = static_cast<hipStream_t>(stream_v);
  if (!stream) {
    stream = static_cast<hipStream_t>(buf->alloc_stream);
  }

  if (buf->host_shadow) {
    (void)hipHostFree(buf->host_shadow);
    buf->host_shadow = nullptr;
  }

  if (buf->data) {
    if (stream) {
      std::lock_guard<std::mutex> plk(pool_mutex());
      hipError_t e = hipFreeAsync(buf->data, stream);
      if (e != hipSuccess) {
        (void)hipGetLastError();
        (void)hipDeviceSynchronize();
        (void)hipFree(buf->data);
      }
    } else {
      // No stream to order against — must drain before blocking free.
      (void)hipDeviceSynchronize();
      (void)hipFree(buf->data);
    }
    buf->data = nullptr;
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

  // Managed grow/shrink pool: last ref dropped → free list (coalesced); empty
  // slabs hipFree'd. This is the path the user asked for.
  if (buf->data && managed_pool_.owns(buf->data)) {
    managed_pool_.free(buf);
    return;
  }

  // Stream-ordered hipMallocAsync buffer: free_async on alloc stream only.
  if (buf->device >= 0 && !buf->is_managed) {
    lock.unlock();
    free_async(buf, nullptr);
    return;
  }

  // Other hipMalloc / unified: BufferCache until max_pool_size_, else free.
  if (get_cache_memory() < max_pool_size_) {
    buffer_cache_.recycle_to_cache(buf);
    return;
  }
  queue_device_free(buf);
  lock.unlock();
  drain_device_frees();
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
  // Keep free_limit_ / cache cap coherent with the new limit (CUDA updates
  // free_limit indirectly via total-memory bookkeeping; we were leaving a
  // stale 512MB free_limit after mx.set_memory_limit(150GB)).
  if (total_memory_ > memory_limit_) {
    free_limit_ = total_memory_ - memory_limit_;
  } else {
    free_limit_ = 0;
  }
  // Cap the unified BufferCache at the memory limit so recycle does not hold
  // a second full working set in VRAM on top of active allocations.
  if (max_pool_size_ > memory_limit_) {
    max_pool_size_ = memory_limit_;
  }
  return limit;
}

size_t RocmAllocator::get_cache_memory() const {
  // Only report BufferCache size. Slab free memory is infrastructure,
  // not cache — including it inflates the count and causes premature
  // eviction of large buffers from the BufferCache.
  return buffer_cache_.cache_size();
}

size_t RocmAllocator::set_cache_limit(size_t limit) {
  {
    std::lock_guard lk(mutex_);
    std::swap(limit, max_pool_size_);
    // Trim the reuse pool down to the new cap NOW (queues device frees).
    if (get_cache_memory() > max_pool_size_) {
      buffer_cache_.release_cached_buffers(get_cache_memory() - max_pool_size_);
    }
  }
  drain_device_frees();
  return limit;
}

void RocmAllocator::clear_cache() {
  // Drop BufferCache + shrink managed pool empty slabs + trim hipMemPool.
  {
    std::lock_guard lock(mutex_);
    buffer_cache_.clear();
    managed_pool_.shrink();
  }
  drain_device_frees();
  (void)hipDeviceSynchronize();
  for (void* p : mem_pools_) {
    if (p)
      (void)hipMemPoolTrimTo(static_cast<hipMemPool_t>(p), 0);
  }
  drain_device_frees();
  std::vector<RocmBuffer*> to_free;
  {
    std::lock_guard<std::mutex> lk(g_pending_free_mutex);
    to_free.swap(g_pending_frees);
  }
  for (auto* b : to_free) {
    free_async(b, b && b->device >= 0 && !b->is_managed ? b->alloc_stream : nullptr);
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
