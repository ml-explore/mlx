// Copyright © 2023 Apple Inc.

#pragma once

#include <atomic>
#include <future>
#include <queue>
#include <shared_mutex>
#include <thread>
#include <unordered_map>

#include "mlx/api.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/device.h"
#include "mlx/stream.h"

namespace mlx::core::scheduler {

struct StreamThread {
  std::mutex mtx;
  std::queue<std::function<void()>> q;
  std::condition_variable cond;
  bool stop;
  std::thread thread;

  // Errors raised from async GPU completion handlers can't safely
  // propagate as C++ exceptions through Metal's dispatch infrastructure
  // (an exception unwinding through Objective-C frames hits
  // _objc_terminate -> abort()). The handler captures the message
  // here and the next user-thread eval()/finalize() call re-throws
  // it on a thread the language runtime can handle.
  //
  // To address the state-safety concern raised in upstream issue
  // #2670 ("we don't have guarantees on the state being in a
  // reasonable condition if there is an exception during eval"),
  // the stream is also POISONED on capture. Once poisoned, every
  // subsequent user-thread entry refuses to queue more work and
  // throws — the caller must explicitly reset the stream
  // (mx.clear_streams()) to resume. This ensures no further
  // operations execute against potentially-corrupt encoder state.
  std::mutex error_mtx;
  std::string captured_error;
  bool poisoned{false};

  void capture_error(std::string msg) {
    std::lock_guard<std::mutex> lk(error_mtx);
    if (captured_error.empty()) {
      captured_error = std::move(msg);
    }
    poisoned = true;
  }

  // Returns the captured error string (and clears it) on the first
  // call after poisoning. Subsequent calls return a generic
  // "stream poisoned" message until the stream is explicitly reset
  // via reset_error().
  std::string take_error() {
    std::lock_guard<std::mutex> lk(error_mtx);
    if (!poisoned) {
      return {};
    }
    if (!captured_error.empty()) {
      auto out = std::move(captured_error);
      captured_error.clear();
      return out;
    }
    return "[METAL] Stream is in error state from a prior failure. "
           "Call mx.clear_streams() (or destroy this Stream) before "
           "queuing more work.";
  }

  void reset_error() {
    std::lock_guard<std::mutex> lk(error_mtx);
    captured_error.clear();
    poisoned = false;
  }

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

class MLX_API Scheduler {
 public:
  Scheduler();
  ~Scheduler();

  // Not copyable or moveable
  Scheduler(const Scheduler&) = delete;
  Scheduler(Scheduler&&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;
  Scheduler& operator=(Scheduler&&) = delete;

  void enqueue(Stream s, std::function<void()> task);

  void notify_new_task(const Stream& stream) {
    {
      std::lock_guard<std::mutex> lk(mtx);
      n_active_tasks_++;
    }
    completion_cv.notify_all();
  }

  void notify_task_completion(const Stream& stream) {
    {
      std::lock_guard<std::mutex> lk(mtx);
      n_active_tasks_--;
    }
    completion_cv.notify_all();
  }

  // Capture an error from an async callback for the given stream's
  // worker thread. Safe to call from Metal completion handlers — never
  // throws, never blocks on user threads. Silently no-ops if the
  // stream's thread doesn't exist (already torn down).
  void capture_error(const Stream& stream, std::string msg) {
    std::shared_lock lock(threads_mtx_);
    auto it = threads_.find(stream.index);
    if (it != threads_.end()) {
      it->second->capture_error(std::move(msg));
    }
  }

  // Take + clear the captured error for a stream. Returns empty string
  // when no error is pending. Stream stays poisoned until reset_error.
  std::string take_error(const Stream& stream) {
    std::shared_lock lock(threads_mtx_);
    auto it = threads_.find(stream.index);
    if (it != threads_.end()) {
      return it->second->take_error();
    }
    return {};
  }

  // Clear the poisoned flag for a stream. Called by mx.clear_streams()
  // (and any future reset API) so the stream is usable again. The
  // caller is expected to also have torn down the underlying GPU
  // encoder state before clearing the flag.
  void reset_error(const Stream& stream) {
    std::shared_lock lock(threads_mtx_);
    auto it = threads_.find(stream.index);
    if (it != threads_.end()) {
      it->second->reset_error();
    }
  }

  void reset_all_errors() {
    std::shared_lock lock(threads_mtx_);
    for (auto& [_, st] : threads_) {
      st->reset_error();
    }
  }

  int n_active_tasks() const {
    return n_active_tasks_;
  }

  void wait_for_one() {
    std::unique_lock<std::mutex> lk(mtx);
    int n_tasks_old = n_active_tasks();
    if (n_tasks_old > 1) {
      completion_cv.wait(lk, [this, n_tasks_old] {
        return this->n_active_tasks() < n_tasks_old;
      });
    }
  }

 private:
  friend Stream mlx::core::new_stream(Device d);

  int n_active_tasks_{0};
  std::unordered_map<int, std::unique_ptr<StreamThread>> threads_;
  std::shared_mutex threads_mtx_;
  std::condition_variable completion_cv;
  std::mutex mtx;
};

MLX_API Scheduler& scheduler();

template <typename F>
void enqueue(const Stream& stream, F&& f) {
  scheduler().enqueue(stream, std::forward<F>(f));
}

inline int n_active_tasks() {
  return scheduler().n_active_tasks();
}

inline void notify_new_task(const Stream& stream) {
  scheduler().notify_new_task(stream);
}

inline void notify_task_completion(const Stream& stream) {
  scheduler().notify_task_completion(stream);
}

inline void capture_error(const Stream& stream, std::string msg) {
  scheduler().capture_error(stream, std::move(msg));
}

inline std::string take_error(const Stream& stream) {
  return scheduler().take_error(stream);
}

inline void reset_error(const Stream& stream) {
  scheduler().reset_error(stream);
}

inline void reset_all_errors() {
  scheduler().reset_all_errors();
}

inline void wait_for_one() {
  scheduler().wait_for_one();
}

} // namespace mlx::core::scheduler
