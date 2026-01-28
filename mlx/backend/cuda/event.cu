// Copyright © 2024 Apple Inc.

#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/event.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/event.h"
#include "mlx/scheduler.h"

#include <map>
#include <vector>

#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace cu {

///////////////////////////////////////////////////////////////////////////////
// CudaEvent implementations
///////////////////////////////////////////////////////////////////////////////

namespace {

// Manage cached cudaEvent_t objects.
class CudaEventPool {
 public:
  CudaEventHandle create(Device& d, int flags) {
    if (!on_creation_thread()) {
      return CudaEventHandle(d, flags);
    }
    auto& cache = cache_for(d, flags);
    if (cache.empty()) {
      return CudaEventHandle(d, flags);
    } else {
      CudaEventHandle ret = std::move(cache.back());
      cache.pop_back();
      return ret;
    }
  }

  void release(CudaEventHandle event) {
    if (!on_creation_thread()) {
      // Event will be destroyed directly instead of getting moved to cache.
      return;
    }
    cache_for(event.device, event.flags).push_back(std::move(event));
  }

 private:
  std::vector<CudaEventHandle>& cache_for(Device& d, int flags) {
    return cache_[d.cuda_device()][flags];
  }

  bool on_creation_thread() {
    return std::this_thread::get_id() == thread_id_;
  }

  // The CudaEvent may be created and destroyed on different threads (for
  // example when waiting on GPU work in CPU stream), we don't want to make
  // the cache thread-safe as it adds overhead, so we just skip cache when
  // using events in worker threads.
  std::thread::id thread_id_{std::this_thread::get_id()};

  // {device: {flags: [events]}}
  std::map<int, std::map<int, std::vector<CudaEventHandle>>> cache_;
};

CudaEventPool& cuda_event_pool() {
  static CudaEventPool pool;
  return pool;
}

} // namespace

CudaEventHandle::CudaEventHandle(Device& d, int flags)
    : device(d), flags(flags) {
  device.make_current();
  CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&handle_, flags));
  assert(handle_ != nullptr);
}

CudaEvent::CudaEvent(Device& d, int flags)
    : event_(cuda_event_pool().create(d, flags)) {}

CudaEvent::~CudaEvent() {
  cuda_event_pool().release(std::move(event_));
}

void CudaEvent::wait() {
  nvtx3::scoped_range r("cu::CudaEvent::wait");
  event_.device.make_current();
  cudaEventSynchronize(event_);
}

void CudaEvent::wait(cudaStream_t stream) {
  event_.device.make_current();
  cudaStreamWaitEvent(stream, event_);
}

void CudaEvent::record(cudaStream_t stream) {
  event_.device.make_current();
  cudaEventRecord(event_, stream);
}

bool CudaEvent::completed() const {
  // Note: cudaEventQuery can be safely called from any device.
  return cudaEventQuery(event_) == cudaSuccess;
}

// static
void CudaEvent::init_pool() {
  cuda_event_pool();
}

// Wraps CudaEvent with a few features:
// 1. The class can be copied.
// 2. Make wait/record work with CPU streams.
// 3. Add checks for waiting on un-recorded event.
class CopyableCudaEvent {
 public:
  explicit CopyableCudaEvent(Device& d)
      : event_(
            std::make_shared<CudaEvent>(
                d,
                cudaEventDisableTiming | cudaEventBlockingSync)) {}

  void wait() {
    event_->wait();
  }

  void wait(Stream s) {
    if (s.device == mlx::core::Device::cpu) {
      scheduler::enqueue(s, [*this]() mutable {
        check_recorded();
        event_->wait();
      });
    } else {
      check_recorded();
      auto& encoder = cu::get_command_encoder(s);
      encoder.commit();
      event_->wait(encoder.stream());
    }
  }

  void record(Stream s) {
    if (s.device == mlx::core::Device::cpu) {
      throw std::runtime_error("CudaEvent can not wait on CPU stream.");
    } else {
      auto& encoder = cu::get_command_encoder(s);
      encoder.commit();
      event_->record(encoder.stream());
      recorded_ = true;
    }
  }

  bool is_signaled() const {
    return recorded_ && event_->completed();
  }

 private:
  void check_recorded() const {
    if (!recorded_) {
      throw std::runtime_error(
          "Should not wait on a CudaEvent before recording.");
    }
  }

  std::shared_ptr<CudaEvent> event_;
  bool recorded_{false};
};

///////////////////////////////////////////////////////////////////////////////
// AtomicEvent implementations
///////////////////////////////////////////////////////////////////////////////

__host__ __device__ void event_wait(uint32_t* ptr, uint32_t value) {
  cuda::atomic_ref<uint32_t> ac(*ptr);
  uint32_t current;
  while ((current = ac.load()) < value) {
    ac.wait(current);
  }
}

__host__ __device__ void event_signal(uint32_t* ptr, uint32_t value) {
  cuda::atomic_ref<uint32_t> ac(*ptr);
  ac.store(value);
  ac.notify_all();
}

__global__ void event_wait_kernel(uint32_t* ptr, uint32_t value) {
  event_wait(ptr, value);
}

__global__ void event_signal_kernel(uint32_t* ptr, uint32_t value) {
  __threadfence_system();
  event_signal(ptr, value);
  __threadfence_system();
}

auto check_gpu_coherency() {
  static auto coherency = []() {
    int device_count = gpu::device_count();
    bool concurrent_managed_access = true;
    bool host_native_atomic = true;
    for (int i = 0; i < device_count; ++i) {
      auto& d = cu::device(i);
      concurrent_managed_access &= d.concurrent_managed_access();
      host_native_atomic &= d.host_native_atomic();
    }
    return std::make_tuple(concurrent_managed_access, host_native_atomic);
  }();
  return coherency;
}

AtomicEvent::AtomicEvent() {
  void* buf;
  cudaError_t (*cuda_free)(void*);
  // There are 3 kinds of systems we are implementing for:
  // 1. concurrentManagedAccess == true
  //    => use cuda::atom_ref on managed memory
  // 2. hostNativeAtomicSupported == true
  //    => use cuda::atom_ref on pinned host memory
  // 2. no hardware cpu/gpu coherency
  //    => use cuda::atom_ref on device memory
  auto [concurrent_managed_access, host_native_atomic] = check_gpu_coherency();
  if (concurrent_managed_access) {
    CHECK_CUDA_ERROR(cudaMallocManaged(&buf, sizeof(uint32_t)));
    cuda_free = cudaFree;
    coherent_ = true;
  } else if (host_native_atomic) {
    CHECK_CUDA_ERROR(cudaMallocHost(&buf, sizeof(uint32_t)));
    cuda_free = cudaFreeHost;
    coherent_ = true;
  } else {
    CHECK_CUDA_ERROR(cudaMalloc(&buf, sizeof(uint32_t)));
    cuda_free = cudaFree;
    coherent_ = false;
  }
  buf_ = std::shared_ptr<void>(
      buf, [cuda_free](void* buf) { CHECK_CUDA_ERROR(cuda_free(buf)); });
  if (coherent_) {
    *ptr() = 0;
  } else {
    CHECK_CUDA_ERROR(cudaMemset(buf, 0, sizeof(uint32_t)));
  }
}

void AtomicEvent::wait(uint32_t value) {
  nvtx3::scoped_range r("cu::AtomicEvent::wait");
  if (coherent_) {
    event_wait(ptr(), value);
  } else {
    while (!is_signaled(value)) {
      std::this_thread::yield();
    }
  }
}

void AtomicEvent::wait(cudaStream_t stream, uint32_t value) {
  event_wait_kernel<<<1, 1, 0, stream>>>(ptr(), value);
}

void AtomicEvent::wait(Stream s, uint32_t value) {
  nvtx3::scoped_range r("cu::AtomicEvent::wait(s)");
  if (s.device == mlx::core::Device::cpu) {
    scheduler::enqueue(s, [*this, value]() mutable { wait(value); });
  } else {
    auto& encoder = get_command_encoder(s);
    encoder.commit();
    wait(encoder.stream(), value);
    encoder.add_completed_handler([buf = buf_]() {});
  }
}

void AtomicEvent::signal(uint32_t value) {
  nvtx3::scoped_range r("cu::AtomicEvent::signal");
  if (coherent_) {
    event_signal(ptr(), value);
  } else {
    signal(signal_stream(), value);
  }
}

void AtomicEvent::signal(cudaStream_t stream, uint32_t value) {
  event_signal_kernel<<<1, 1, 0, stream>>>(ptr(), value);
}

void AtomicEvent::signal(Stream s, uint32_t value) {
  nvtx3::scoped_range r("cu::AtomicEvent::signal(s)");
  if (s.device == mlx::core::Device::cpu) {
    // Signal through a GPU stream so the atomic is updated in GPU - updating
    // the atomic in CPU sometimes does not get GPU notified.
    scheduler::enqueue(
        s, [*this, value]() mutable { signal(signal_stream(), value); });
  } else {
    auto& encoder = get_command_encoder(s);
    encoder.commit();
    signal(encoder.stream(), value);
    encoder.add_completed_handler([buf = buf_]() {});
  }
}

bool AtomicEvent::is_signaled(uint32_t val) const {
  return value() >= val;
}

uint32_t AtomicEvent::value() const {
  nvtx3::scoped_range r("cu::AtomicEvent::value");
  if (coherent_) {
    cuda::atomic_ref<uint32_t> ac(*ptr());
    return ac.load();
  } else {
    uint32_t val;
    CHECK_CUDA_ERROR(
        cudaMemcpy(&val, ptr(), sizeof(uint32_t), cudaMemcpyDeviceToHost));
    return val;
  }
}

const CudaStream& AtomicEvent::signal_stream() {
  static CudaStream stream(device(0));
  return stream;
}

} // namespace cu

///////////////////////////////////////////////////////////////////////////////
// Event implementations
///////////////////////////////////////////////////////////////////////////////

namespace {

struct EventImpl {
  // CudaEvent is preferred when possible because it is fast, however we have
  // to fallback to AtomicEvent in following cases:
  // 1. the event is used to wait/signal a cpu stream;
  // 2. signal value other than 1 has been specified.
  std::unique_ptr<cu::CopyableCudaEvent> cuda;
  std::unique_ptr<cu::AtomicEvent> atomic;

  bool is_created() const {
    return cuda || atomic;
  }

  void ensure_created(Stream s, uint64_t signal_value) {
    if (is_created()) {
      return;
    }
    if (s.device == mlx::core::Device::cpu || signal_value > 1) {
      nvtx3::mark("Using slow AtomicEvent");
      atomic = std::make_unique<cu::AtomicEvent>();
    } else {
      cuda = std::make_unique<cu::CopyableCudaEvent>(cu::device(s.device));
    }
  }
};

} // namespace

Event::Event(Stream s) : stream_(s) {
  event_ = std::shared_ptr<void>(
      new EventImpl(), [](void* ptr) { delete static_cast<EventImpl*>(ptr); });
}

void Event::wait() {
  auto* event = static_cast<EventImpl*>(event_.get());
  assert(event->is_created());
  if (event->cuda) {
    assert(value() == 1);
    event->cuda->wait();
  } else {
    event->atomic->wait(value());
  }
  CHECK_CUDA_ERROR(cudaPeekAtLastError());
}

void Event::wait(Stream s) {
  auto* event = static_cast<EventImpl*>(event_.get());
  assert(event->is_created());
  if (event->cuda) {
    assert(value() == 1);
    event->cuda->wait(s);
  } else {
    event->atomic->wait(s, value());
  }
}

void Event::signal(Stream s) {
  auto* event = static_cast<EventImpl*>(event_.get());
  event->ensure_created(s, value());
  if (event->cuda) {
    assert(value() == 1);
    event->cuda->record(s);
  } else {
    event->atomic->signal(s, value());
  }
}

bool Event::is_signaled() const {
  auto* event = static_cast<EventImpl*>(event_.get());
  if (!event->is_created()) {
    return false;
  }
  if (event->cuda) {
    assert(value() == 1);
    return event->cuda->is_signaled();
  } else {
    return event->atomic->is_signaled(value());
  }
}

} // namespace mlx::core
