// Copyright © 2023-2026 Apple Inc.

#include <future>
#include <thread>

#ifndef _WIN32
#include <pthread.h>
#endif

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
    scheduler::check_error(s);
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

void Scheduler::prepare_fork() {
  threads_mtx_.lock();
  mtx.lock();
}

void Scheduler::after_fork_in_parent() {
  mtx.unlock();
  threads_mtx_.unlock();
}

void Scheduler::after_fork_in_child() {
  // Only the forking thread exists in the child, so the stream threads are
  // gone. They cannot be joined, so leak them and start new ones on demand.
  for (auto& [index, st] : threads_) {
    st.release();
  }
  threads_.clear();
  n_active_tasks_ = 0;
  new (&completion_cv) std::condition_variable;
  new (&mtx) std::mutex;
  new (&threads_mtx_) std::shared_mutex;
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

void Scheduler::check_error(Stream s) {
  get_thread(s).error.check();
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
// It is not a function local static, because a fork while another thread is
// constructing it would leave the child waiting on that static's guard.
namespace {
std::atomic<Scheduler*> scheduler_instance{nullptr};
std::mutex scheduler_init_mtx;
} // namespace

Scheduler& scheduler() {
  auto* s = scheduler_instance.load(std::memory_order_acquire);
  if (!s) {
    std::lock_guard lk(scheduler_init_mtx);
    s = scheduler_instance.load(std::memory_order_relaxed);
    if (!s) {
      s = new Scheduler;
      scheduler_instance.store(s, std::memory_order_release);
    }
  }
  return *s;
}

#ifndef _WIN32
namespace {

void prepare_fork() {
  // Wait for a scheduler under construction and hold off new ones.
  scheduler_init_mtx.lock();
  if (auto* s = scheduler_instance.load(std::memory_order_relaxed)) {
    s->prepare_fork();
  }
}

void after_fork_in_parent() {
  if (auto* s = scheduler_instance.load(std::memory_order_relaxed)) {
    s->after_fork_in_parent();
  }
  scheduler_init_mtx.unlock();
}

void after_fork_in_child() {
  if (auto* s = scheduler_instance.load(std::memory_order_relaxed)) {
    s->after_fork_in_child();
  }
  new (&scheduler_init_mtx) std::mutex;
}

// Registered at load time. Registering from the constructor would wait on the
// fork lock while the scheduler is only half built.
[[maybe_unused]] const int fork_handlers =
    pthread_atfork(prepare_fork, after_fork_in_parent, after_fork_in_child);

} // namespace
#endif

} // namespace scheduler
} // namespace mlx::core
