// Copyright © 2023-2026 Apple Inc.

#include "mlx/scheduler.h"
#include "mlx/backend/cpu/eval.h"
#include "mlx/backend/gpu/eval.h"

namespace mlx::core {

void synchronize(Stream s) {
  if (s.device == mlx::core::Device::cpu) {
    auto p = std::make_shared<std::promise<void>>();
    std::future<void> f = p->get_future();
    scheduler::enqueue(s, [p = std::move(p)]() { p->set_value(); });
    f.wait();
  } else {
    gpu::synchronize(s);
  }
}

void synchronize(ThreadLocalStream s) {
  synchronize(stream_from_thread_local_stream(s));
}

void synchronize() {
  synchronize(default_stream(default_device()));
}

void clear_streams() {
  cpu::clear_streams();
  gpu::clear_streams();
}

namespace scheduler {

Scheduler::Scheduler() {
  gpu::init();
}

Scheduler::~Scheduler() = default;

void Scheduler::enqueue(Stream s, std::function<void()> task) {
  StreamThread* st = nullptr;
  {
    std::shared_lock lock(threads_mtx_);
    auto it = threads_.find(s.index);
    if (it != threads_.end()) {
      st = it->second.get();
    }
  }
  if (!st) {
    std::unique_lock lock(threads_mtx_);
    auto it = threads_.find(s.index);
    if (it == threads_.end()) {
      it = threads_.emplace(s.index, std::make_unique<StreamThread>()).first;
    }
    st = it->second.get();
  }
  st->enqueue(std::move(task));
}

// Leak the scheduler singleton on all platforms. During static destruction,
// worker threads may still be executing JIT-compiled code that has been
// unmapped, causing SIGSEGV (macOS/Linux) or join() deadlocks (Windows/MSVC
// CRT).
// The OS reclaims all resources at process exit anyway.
Scheduler& scheduler() {
  static Scheduler* scheduler = new Scheduler;
  return *scheduler;
}

} // namespace scheduler
} // namespace mlx::core
