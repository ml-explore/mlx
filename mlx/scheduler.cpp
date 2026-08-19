// Copyright © 2023-2026 Apple Inc.

#include <future>
#include <thread>

#include "mlx/backend/cpu/eval.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/compile_impl.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

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
  detail::compile_clear_cache(detail::compile_cache());
  cpu::clear_streams();
  gpu::clear_streams();
}

namespace scheduler {

struct StreamThread {
  std::mutex mtx;
  std::queue<std::function<void()>> q;
  std::condition_variable cond;
  bool stop;
  std::thread thread;
  Error error;

  StreamThread() : stop(false), thread(&StreamThread::thread_fn, this) {}

  ~StreamThread() {
    {
      std::lock_guard<std::mutex> lk(mtx);
      stop = true;
    }
    cond.notify_one();
    thread.join();
  }

  void thread_fn() {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lk(mtx);
        cond.wait(lk, [this] { return !this->q.empty() || this->stop; });
        if (q.empty() && stop) {
          return;
        }
        task = std::move(q.front());
        q.pop();
      }

      task();
    }
  }

  void enqueue(std::function<void()> f) {
    if (is_main_thread()) {
      error.check();
    }
    {
      std::lock_guard<std::mutex> lk(mtx);
      if (stop) {
        throw std::runtime_error(
            "Cannot enqueue work after stream is stopped.");
      }
      q.emplace(std::move(f));
    }
    cond.notify_one();
  }
};

Scheduler::Scheduler() {
  is_main_thread();
  gpu::init();
}

Scheduler::~Scheduler() = default;

void Scheduler::enqueue(Stream s, std::function<void()> task) {
  auto& st = get_thread(s);
  st.enqueue([&st, task = std::move(task)]() mutable {
    try {
      task();
    } catch (const std::exception& error) {
      // Set error to stream only when no error happended before, to preserve
      // the earliest error.
      if (!st.error.valid()) {
        st.error.set_message(std::make_shared<std::string>(error.what()));
      }
    }
  });
}

void Scheduler::wait_event(
    Stream s,
    Event event,
    std::function<void(Event&)> task) {
  assert(s.device == Device::cpu);
  auto& st = get_thread(s);
  st.enqueue([&st, event = std::move(event), task = std::move(task)]() mutable {
    task(event);
    // Poison current stream if the waited event has error.
    st.error.store_if_valid(event.load_error());
  });
}

void Scheduler::signal_event(
    Stream s,
    Event event,
    std::function<void(Event&)> task) {
  assert(s.device == Device::cpu);
  auto& st = get_thread(s);
  st.enqueue([&st, event = std::move(event), task = std::move(task)]() mutable {
    // Poison the signal event if current stream has error.
    if (st.error.valid()) {
      event.set_error(st.error);
    }
    task(event);
  });
}

StreamThread& Scheduler::get_thread(Stream s) {
  {
    std::shared_lock lock(threads_mtx_);
    auto it = threads_.find(s.index);
    if (it != threads_.end()) {
      return *it->second.get();
    }
  }
  std::unique_lock lock(threads_mtx_);
  auto it = threads_.find(s.index);
  if (it == threads_.end()) {
    it = threads_.emplace(s.index, std::make_unique<StreamThread>()).first;
  }
  return *it->second.get();
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
