// Copyright © 2023 Apple Inc.

#pragma once

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <queue>
#include <shared_mutex>
#include <thread>
#include <unordered_map>

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

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
#if defined(__APPLE__)
    // exo-mlx-tune: env-gated QoS pin for stream worker threads.
    // Diagnosed via JACCL_TRACE_PROGRESS=1: rank 0 (the API/master
    // host) sees asymmetric busy-poll stalls (17M+ poll_iters)
    // during MTP verify all_reduces while rank 1 completes in 2
    // poll_iters. The comm-stream worker thread is getting
    // descheduled by the macOS scheduler — purely C++ busy-poll,
    // no Python re-entry. Pinning the thread to a higher QoS
    // keeps it on a P-core under contention.
    //
    // Off by default (safety: USER_INTERACTIVE has misbehaved on
    // some cluster states — opt in deliberately). Set
    // MLX_STREAM_QOS=user_initiated|user_interactive|default|utility
    // to enable. Once-per-process getenv() — cheap.
    static const int qos_class = [] {
      const char* v = std::getenv("MLX_STREAM_QOS");
      if (v == nullptr) return -1;
      if (std::strcmp(v, "user_interactive") == 0) return (int)QOS_CLASS_USER_INTERACTIVE;
      if (std::strcmp(v, "user_initiated") == 0) return (int)QOS_CLASS_USER_INITIATED;
      if (std::strcmp(v, "default") == 0) return (int)QOS_CLASS_DEFAULT;
      if (std::strcmp(v, "utility") == 0) return (int)QOS_CLASS_UTILITY;
      if (std::strcmp(v, "off") == 0) return -1;
      return -1;
    }();
    if (qos_class != -1) {
      pthread_set_qos_class_self_np((qos_class_t)qos_class, 0);
    }
#endif
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

inline void wait_for_one() {
  scheduler().wait_for_one();
}

} // namespace mlx::core::scheduler
