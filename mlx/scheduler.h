// Copyright © 2023 Apple Inc.

#pragma once

#include <atomic>
#include <future>
#include <queue>
#include <thread>
#include <unordered_map>

#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/metal_impl.h"
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

  template <typename F>
  void enqueue(F&& f) {
    {
      std::lock_guard<std::mutex> lk(mtx);
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
    streams_.emplace_back(streams_.size(), d);
    if (d == Device::gpu) {
      threads_.push_back(nullptr);
      metal::new_stream(streams_.back());
    } else {
      threads_.push_back(new StreamThread{});
    }
    return streams_.back();
  }

  template <typename F>
  void enqueue(const Stream& stream, F&& f);

  Stream get_default_stream(const Device& d) const {
    return default_streams_.at(d.type);
  }
  Stream get_stream(int index) const {
    return streams_.at(index);
  }

  void set_default_stream(const Stream& s) {
    default_streams_.at(s.device.type) = s;
  }

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
        return this->n_active_tasks() != n_tasks_old;
      });
    }
  }

  ~Scheduler() {
    for (auto s : streams_) {
      synchronize(s);
    }
    for (auto t : threads_) {
      if (t != nullptr) {
        delete t;
      }
    }
  }

 private:
  int n_active_tasks_;
  std::vector<StreamThread*> threads_;
  std::vector<Stream> streams_;
  std::unordered_map<Device::DeviceType, Stream> default_streams_;
  std::condition_variable completion_cv;
  std::mutex mtx;
};

template <typename F>
void Scheduler::enqueue(const Stream& stream, F&& f) {
  threads_[stream.index]->enqueue(std::forward<F>(f));
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
