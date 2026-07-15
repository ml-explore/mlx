// Copyright © 2025 Apple Inc.
//
// ROCm memory management — structured to match mlx/backend/cuda/allocator.cpp.
// hip* replaces cuda*; discrete-GPU host access uses host_shadow instead of
// CUDA's move_to_unified_memory. No custom grow/shrink HBM pool, no multi-tier
// slab freelist — only BufferCache + hipMallocAsync + 8-byte SmallSizePool.

#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/utils.h"
#include "mlx/device.h"
#include "mlx/memory.h"
#include "mlx/utils.h"

#include <hip/hip_runtime.h>
#include <unistd.h>

#include <atomic>
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

// Any allocations smaller than this will try to use the small pool (CUDA).
constexpr int small_block_size = 8;

// The small pool size in bytes. Multiple of page_size and small_block_size.
constexpr int small_pool_size = 4 * page_size;

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

static bool rocm_available() {
  static int available = -1;
  if (available < 0) {
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    available = (err == hipSuccess && device_count > 0) ? 1 : 0;
  }
  return available == 1;
}

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

// Fine-grained = host-coherent device mem (APU). On discrete (MI300X) default
// OFF — coarse hipMalloc VRAM. Override with MLX_ROCM_FINEGRAINED=1/0.
static bool use_finegrained() {
  if (const char* e = std::getenv("MLX_ROCM_FINEGRAINED"))
    return std::atoi(e) != 0;
  int dev = 0;
  if (hipGetDevice(&dev) == hipSuccess && !device_is_integrated(dev))
    return false;
  return true;
}

// Stream-ordered hipMallocAsync. Default OFF on ROCm: the host BufferCache is
// the CUDA-style recycle path; hipMemPool + hipFreeAsync has caused gfx page
// faults / IMA 700 on MI300X when buffers are retagged across streams.
// Set MLX_ROCM_USE_ASYNC_POOL=1 to enable (CUDA-like device pools).
// MLX_ROCM_NO_ASYNC_POOL=1 forces off (legacy name).
static bool use_async_pool() {
  static const bool forced_off =
      std::getenv("MLX_ROCM_NO_ASYNC_POOL") != nullptr;
  static const bool forced_on =
      std::getenv("MLX_ROCM_USE_ASYNC_POOL") != nullptr ||
      std::getenv("MLX_ROCM_FORCE_ASYNC_POOL") != nullptr;
  if (forced_off)
    return false;
  if (forced_on)
    return true;
  return false; // safe default on ROCm
}

static void ensure_mlx_device_current() {
  mlx::core::Device dd = mlx::core::default_device();
  if (dd.type == mlx::core::Device::gpu) {
    int cur = -1;
    if (hipGetDevice(&cur) == hipSuccess && cur != dd.index)
      (void)hipSetDevice(dd.index);
  }
}

// CUDA unified_malloc: managed if supported else host pinned.
// ROCm discrete training: prefer real VRAM (hipMalloc) so we never spill GTT.
// APU: fine-grained coherent. Managed only as explicit fallback.
inline void* unified_malloc(size_t size, bool& is_managed) {
  void* data = nullptr;
  hipError_t err;
  ensure_mlx_device_current();

  if (use_finegrained()) {
    err = hipExtMallocWithFlags(&data, size, hipDeviceMallocFinegrained);
    if (err == hipSuccess) {
      is_managed = true;
      return data;
    }
  } else {
    err = hipMalloc(&data, size);
    if (err == hipSuccess) {
      is_managed = true;
      return data;
    }
  }

  // Discrete: do not fall back to managed/GTT unless opted in (CUDA managed
  // on discrete is different; on ROCm it spills system RAM → thrash).
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

inline void unified_free(void* data, bool is_managed) {
  if (is_managed) {
    (void)hipFree(data);
  } else {
    (void)hipHostFree(data);
  }
}

// ---------------------------------------------------------------------------
// SmallSizePool — CUDA-identical structure
// ---------------------------------------------------------------------------

SmallSizePool::SmallSizePool() {
  if (!rocm_available()) {
    return;
  }
  auto num_blocks = small_pool_size / small_block_size;
  buffer_ = new Block[num_blocks];
  next_free_ = buffer_;

  data_ = unified_malloc(small_pool_size, data_managed_);

  auto curr = next_free_;
  for (size_t i = 1; i < static_cast<size_t>(num_blocks); ++i) {
    curr->next = buffer_ + i;
    curr = curr->next;
  }
  curr->next = nullptr;
}

SmallSizePool::~SmallSizePool() {
  if (data_) {
    unified_free(data_, data_managed_);
  }
  delete[] buffer_;
}

RocmBuffer* SmallSizePool::malloc() {
  if (next_free_ == nullptr) {
    return nullptr;
  }
  Block* b = next_free_;
  uint64_t i = static_cast<uint64_t>(next_free_ - buffer_);
  next_free_ = next_free_->next;
  b->buf.data = static_cast<char*>(data_) + i * small_block_size;
  b->buf.size = small_block_size;
  b->buf.device = -1;
  b->buf.is_managed = data_managed_;
  b->buf.host_shadow = nullptr;
  b->buf.host_dirty = false;
  b->buf.alloc_stream = nullptr;
  return &b->buf;
}

void SmallSizePool::free(RocmBuffer* buf) {
  auto* b = reinterpret_cast<Block*>(buf);
  b->next = next_free_;
  next_free_ = b;
}

bool SmallSizePool::in_pool(RocmBuffer* buf) {
  if (!buffer_) {
    return false;
  }
  constexpr int num_blocks = (small_pool_size / small_block_size);
  auto* b = reinterpret_cast<Block*>(buf);
  int64_t block_num = b - buffer_;
  return block_num >= 0 && block_num < num_blocks;
}

// ---------------------------------------------------------------------------
// RocmAllocator — CUDA control flow
// ---------------------------------------------------------------------------

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

  size_t free = 0;
  hipError_t err = hipMemGetInfo(&free, &total_memory_);
  if (err != hipSuccess) {
    return;
  }

  // CUDA: memory_limit_ = total * 0.95; max_pool_size_ = memory_limit_.
  int dev = 0;
  (void)hipGetDevice(&dev);
  if (device_is_integrated(dev)) {
    // APU unified pool draws from system RAM; hipMemGetInfo is a tiny carveout.
    size_t sys_ram = static_cast<size_t>(sysconf(_SC_PHYS_PAGES)) *
        static_cast<size_t>(sysconf(_SC_PAGE_SIZE));
    memory_limit_ = std::max(
        static_cast<size_t>(total_memory_ * 0.8),
        static_cast<size_t>(sys_ram * 0.8));
  } else {
    memory_limit_ = static_cast<size_t>(total_memory_ * 0.95);
  }
  free_limit_ = total_memory_ > memory_limit_ ? total_memory_ - memory_limit_ : 0;
  max_pool_size_ = memory_limit_;

  // CUDA: per-device default mem pool + free stream when pools supported.
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
      if (!supported) {
        continue;
      }
      // gfx1201: ROCm async pool stream-order bugs (see history). Leave null
      // so malloc_async falls back to unified (device=-1). Override:
      // MLX_ROCM_FORCE_ASYNC_POOL=1.
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
        // CUDA does not set ReleaseThreshold. On ROCm the default can pin nearly
        // all HBM after fat steps → GTT spill. Release aggressively (0).
        uint64_t threshold = 0;
        (void)hipMemPoolSetAttribute(
            pool, hipMemPoolAttrReleaseThreshold, &threshold);
        hipStream_t s = nullptr;
        if (hipStreamCreateWithFlags(&s, hipStreamNonBlocking) == hipSuccess) {
          free_streams_[i] = s;
        }
      }
    }
    (void)hipSetDevice(saved);
  }
  // scalar_pool_ constructs itself (CUDA: member default-init).
}

// CUDA free_async
void RocmAllocator::free_async(RocmBuffer& buf, void* stream_v) {
  if (buf.host_shadow) {
    (void)hipHostFree(buf.host_shadow);
    buf.host_shadow = nullptr;
  }
  if (!buf.data) {
    return;
  }
  if (buf.device == -1) {
    unified_free(buf.data, buf.is_managed);
  } else if (
      buf.device >= 0 && buf.device < static_cast<int>(mem_pools_.size()) &&
      mem_pools_[buf.device]) {
    hipStream_t stream = static_cast<hipStream_t>(stream_v);
    if (!stream) {
      stream = static_cast<hipStream_t>(buf.alloc_stream);
    }
    if (!stream && buf.device < static_cast<int>(free_streams_.size())) {
      stream = static_cast<hipStream_t>(free_streams_[buf.device]);
    }
    if (stream) {
      (void)hipFreeAsync(buf.data, stream);
    } else {
      (void)hipFree(buf.data);
    }
  } else {
    (void)hipFree(buf.data);
  }
  buf.data = nullptr;
}

// CUDA free_cuda_buffer (must be called with mutex_ for scalar pool)
void RocmAllocator::free_rocm_buffer(RocmBuffer* buf) {
  if (!buf) {
    return;
  }
  if (scalar_pool_.in_pool(buf)) {
    scalar_pool_.free(buf);
  } else {
    free_async(*buf);
    delete buf;
  }
}

static void free_rocm_buffer_cb(RocmBuffer* buf) {
  allocator().free_rocm_buffer(buf);
}

// CUDA: Buffer malloc(size) { return malloc_async(size, -1, nullptr); }
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
  return malloc_async(size, /*device=*/-1, /*stream=*/nullptr);
}

// CUDA malloc_async — mirrored with hip names.
Buffer RocmAllocator::malloc_async(size_t size, int device, void* stream_v) {
  if (decode_arena_.active && size > 0) {
    std::lock_guard lock(mutex_);
    if (RocmBuffer* b = arena_alloc(size)) {
      return Buffer{b};
    }
  }

  if (size == 0) {
    return Buffer{new RocmBuffer{
        nullptr, 0, /*is_managed=*/true, /*device=*/-1, nullptr, false, nullptr}};
  }

  hipStream_t stream = static_cast<hipStream_t>(stream_v);

  // CUDA size rounding.
  if (size <= static_cast<size_t>(small_block_size)) {
    size = 8;
  } else if (size < static_cast<size_t>(page_size)) {
    size = static_cast<size_t>(next_power_of_2(static_cast<int>(size)));
  } else {
    size = page_size * ((size + page_size - 1) / page_size);
  }

  // CUDA: tiny or no stream → unified path (device = -1).
  if (size <= static_cast<size_t>(small_block_size) || stream == nullptr ||
      !use_async_pool()) {
    device = -1;
  }
  if (device >= 0 &&
      (device >= static_cast<int>(mem_pools_.size()) || !mem_pools_[device])) {
    device = -1;
  }

  // Find available buffer from cache.
  std::unique_lock lock(mutex_);
  RocmBuffer* buf = buffer_cache_.reuse_from_cache(size);
  if (!buf) {
    // If we have a lot of memory pressure try to reclaim from the cache.
    int64_t mem_to_free = static_cast<int64_t>(get_active_memory()) +
        static_cast<int64_t>(get_cache_memory()) + static_cast<int64_t>(size) -
        static_cast<int64_t>(memory_limit_);
    if (mem_to_free > 0) {
      buffer_cache_.release_cached_buffers(static_cast<size_t>(mem_to_free));
    }

    // Try the scalar pool first (CUDA).
    if (size <= static_cast<size_t>(small_block_size)) {
      buf = scalar_pool_.malloc();
    }
    lock.unlock();
    if (!buf) {
      void* data = nullptr;
      bool is_managed = true;
      if (device == -1) {
        data = unified_malloc(size, is_managed);
        buf = new RocmBuffer{
            data, size, is_managed, /*device=*/-1, nullptr, false, nullptr};
      } else {
        (void)hipSetDevice(device);
        hipError_t err = hipMallocAsync(&data, size, stream);
        if (err != hipSuccess || !data) {
          (void)hipGetLastError();
          std::ostringstream msg;
          msg << "[malloc_async] Unable to allocate " << size << " bytes: "
              << hipGetErrorString(err);
          throw std::runtime_error(msg.str());
        }
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

    // CUDA: if any mem pool reserved is huge, release host cache.
    if (get_cache_memory() > 0) {
      for (size_t i = 0; i < mem_pools_.size(); ++i) {
        auto* p = static_cast<hipMemPool_t>(mem_pools_[i]);
        if (!p) {
          continue;
        }
        size_t used = 0;
        if (hipMemPoolGetAttribute(
                p, hipMemPoolAttrReservedMemCurrent, &used) == hipSuccess &&
            used > (total_memory_ - free_limit_)) {
          buffer_cache_.release_cached_buffers(free_limit_);
          break;
        }
      }
    }
  } else {
    // Cache hit. CUDA: if cached buffer is on a different device, move to
    // unified. Never retag device=-1 (hipMalloc) as a stream-pool buffer —
    // that made free_async call hipFreeAsync on hipMalloc memory → page faults.
    if (buf->device >= 0 && device >= 0 && buf->device != device) {
      // Cross-device: drop to unified for safety (sync copy).
      void* data = nullptr;
      bool is_managed = true;
      lock.unlock();
      data = unified_malloc(buf->size, is_managed);
      (void)hipMemcpy(data, buf->data, buf->size, hipMemcpyDeviceToDevice);
      free_async(*buf);
      buf->data = data;
      buf->is_managed = is_managed;
      buf->device = -1;
      buf->alloc_stream = nullptr;
      lock.lock();
    } else if (buf->device >= 0 && stream && !buf->is_managed) {
      // Same stream-pool buffer: remember last stream for free_async ordering.
      buf->alloc_stream = stream;
    }
    // device==-1 buffers stay device==-1 regardless of request.
  }

  active_memory_ += buf->size;
  peak_memory_ = std::max(active_memory_, peak_memory_);

  // Maintain the cache below the requested limit.
  if (get_cache_memory() > max_pool_size_) {
    buffer_cache_.release_cached_buffers(get_cache_memory() - max_pool_size_);
  }
  return Buffer{buf};
}

// ---------------------------------------------------------------------------
// Graph deferred free (HIP graph capture — not in CUDA backend; required so
// graph-referenced buffers stay live until the generation retires).
// ---------------------------------------------------------------------------

static std::mutex g_deferred_mutex;
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
      continue;
    }
    allocator().free(b, /*force=*/true);
  }
}

void flush_graph_deferred_frees() {
  free_graph_generation(~uint64_t(0));
}

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
    allocator().free(b, /*force=*/true);
  }
}

// CUDA free + ROCm graph defer / arena / alien sentinels.
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
  // Alien sentinel (make_buffer): shell only.
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
  // Always recycle first. CUDA frees immediately when cache is at
  // max_pool_size_; on ROCm that hipFree path is catastrophic (100–500 ms
  // stalls in the bwd→Adam gap while the freelist is full of same-sized
  // blocks the next step needs). Recycle this buffer (MRU), then trim the
  // LRU tail if over the cap — same bound, but the just-freed sizes stay
  // available for the next malloc hit.
  buffer_cache_.recycle_to_cache(buf);
  if (get_cache_memory() > max_pool_size_) {
    buffer_cache_.release_cached_buffers(get_cache_memory() - max_pool_size_);
  }
}

Buffer RocmAllocator::make_buffer(void* ptr, size_t size) {
  if (!ptr || size == 0) {
    return Buffer{nullptr};
  }
  auto* rb = new RocmBuffer{
      ptr,
      size,
      /*is_managed=*/true,
      /*device=*/-1,
      nullptr,
      false,
      reinterpret_cast<void*>(static_cast<uintptr_t>(1))}; // alien sentinel
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

// --- Decode arena (graph decode only; not used by CUDA or train default) ----

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
  rb->is_managed = true;
  rb->device = -1;
  rb->host_shadow = nullptr;
  rb->host_dirty = false;
  rb->alloc_stream = nullptr;
  a.offset += aligned;
  a.high_water = std::max(a.high_water, a.offset);
  return rb;
}

bool RocmAllocator::decode_arena_begin(
    size_t capacity,
    int device,
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
    bool arena_managed = false;
    p = unified_malloc(capacity, arena_managed);
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
  decode_arena_.offset = decode_arena_.floor_offset;
  decode_arena_.next_wrapper = decode_arena_.floor_wrapper;
  decode_arena_.overflowed = false;
}

void RocmAllocator::decode_arena_end() {
  std::lock_guard lock(mutex_);
  decode_arena_.active = false;
}

void RocmAllocator::ensure_host_shadow(RocmBuffer& buf) {
  if (buf.device == -1) {
    return;
  }
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
  free_limit_ = total_memory_ > memory_limit_ ? total_memory_ - memory_limit_ : 0;
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
  // CUDA: heap-allocate so destructor does not run at exit (cache leak OK).
  static auto* allocator_ = new RocmAllocator();
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
    if (hipStreamQuery(nullptr) != hipSuccess) {
      (void)hipStreamSynchronize(nullptr);
    }
  } else {
    // Discrete: host shadow (CUDA: move_to_unified_memory).
    (void)hipDeviceSynchronize();
    rocm::allocator().ensure_host_shadow(cbuf);
    return cbuf.host_shadow;
  }
  return cbuf.data;
}

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

size_t set_wired_limit(size_t) {
  return 0;
}

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
