// Copyright © 2023 Apple Inc.

#pragma once

#include <atomic>
#include <future>
#include <queue>
#include <thread>
#include <unordered_map>

#include "mlx/backend/metal/metal.h"
#include "mlx/device.h"
#include "mlx/stream.h"

namespace mlx::core::scheduler {

struct StreamThread {
  std::mutex mtx;
  std::queue<std::function<void()>> q;
  std::condition_variable cond;
  bool stop;
  Stream stream;
  std::thread thread;

  StreamThread(Stream stream)
      : stop(false), stream(stream), thread(&StreamThread::thread_fn, this) {}

  ~StreamThread() {
    {
      std::unique_lock<std::mutex> lk(mtx);
      stop = true;
    }
    cond.notify_one();
    thread.join();
  }

  void thread_fn() {
    bool initialized = false;
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

      // thread_fn may be called from a static initializer and we cannot
      // call metal-cpp until all static initializers have completed. waiting
      // for a task to arrive means that user code is running so metal-cpp
      // can safely be called.
      if (!initialized) {
        initialized = true;
        metal::new_stream(stream);
      }

      task();
    }
  }

  template <typename F>
  void enqueue(F&& f) {
    {
      std::unique_lock<std::mutex> lk(mtx);
      if (stop) {
        throw std::runtime_error(
            "Cannot enqueue work after stream is stopped.");
      }
      q.emplace(std::forward<F>(f));
    }
    cond.notify_one();
  }
};

class Scheduler {
 public:
  Scheduler() : n_active_tasks_(0) {
    if (metal::is_available()) {
      default_streams_.insert({Device::gpu, new_stream(Device::gpu)});
    }
    default_streams_.insert({Device::cpu, new_stream(Device::cpu)});
  }

  // Not copyable or moveable
  Scheduler(const Scheduler&) = delete;
  Scheduler(Scheduler&&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;
  Scheduler& operator=(Scheduler&&) = delete;

  Stream new_stream(const Device& d) {
    auto stream = Stream(streams_.size(), d);
    streams_.push_back(new StreamThread{stream});
    return stream;
  }

  template <typename F>
  void enqueue(const Stream& stream, F&& f);

  Stream get_default_stream(const Device& d) {
    return default_streams_.at(d.type);
  }

  void set_default_stream(const Stream& s) {
    default_streams_.at(s.device.type) = s;
  }

  void notify_new_task(const Stream& stream) {
    {
      std::unique_lock<std::mutex> lk(mtx);
      n_active_tasks_++;
    }
    completion_cv.notify_all();
  }

  void notify_task_completion(const Stream& stream) {
    {
      std::unique_lock<std::mutex> lk(mtx);
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
        return this->n_active_tasks() != n_tasks_old;
      });
    }
  }

  ~Scheduler() {
    for (auto s : streams_) {
      delete s;
    }
  }

 private:
  int n_active_tasks_;
  std::vector<StreamThread*> streams_;
  std::unordered_map<Device::DeviceType, Stream> default_streams_;
  std::condition_variable completion_cv;
  std::mutex mtx;
};

template <typename F>
void Scheduler::enqueue(const Stream& stream, F&& f) {
  streams_[stream.index]->enqueue(std::forward<F>(f));
}

Scheduler& scheduler();

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
