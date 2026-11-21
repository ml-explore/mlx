// Copyright © 2023 Apple Inc.

#pragma once

#include <atomic>
#include <queue>
#include <set>
#include <shared_mutex>
#include <unordered_map>

#include "mlx/api.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/device.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mlx::core::scheduler {

class StreamThread;

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
  void wait_event(Stream s, Event event, std::function<void(Event&)> task);
  void signal_event(Stream s, Event event, std::function<void(Event&)> task);
  void check_error(Stream s);

  void notify_new_task(const Stream& stream) {
    n_active_tasks_.fetch_add(1);
  }

  void notify_task_completion(const Stream& stream) {
    n_active_tasks_.fetch_sub(1);
    n_active_tasks_.notify_all();
  }

  // Block the caller when too many tasks or too much memory is in flight.
  // Commits the GPU streams in open_streams first so that tasks can finish.
  void wait_for_capacity(const std::set<Stream>& open_streams);

 private:
  friend Stream mlx::core::new_stream(Device d);

  StreamThread& get_thread(Stream s);

  std::atomic<int> n_active_tasks_{0};
  std::unordered_map<int, std::unique_ptr<StreamThread>> threads_;
  std::shared_mutex threads_mtx_;
};

MLX_API Scheduler& scheduler();

template <typename F>
inline void enqueue(Stream s, F&& f) {
  scheduler().enqueue(s, std::forward<F>(f));
}

// Like enqueue but the task is used for processing the passed event.
template <typename F>
inline void wait_event(Stream s, Event event, F&& f) {
  scheduler().wait_event(s, std::move(event), std::forward<F>(f));
}

template <typename F>
inline void signal_event(Stream s, Event event, F&& f) {
  scheduler().signal_event(s, std::move(event), std::forward<F>(f));
}

// Throw and clear the error stored in the stream, if any.
inline void check_error(Stream s) {
  scheduler().check_error(s);
}

inline void notify_new_task(const Stream& stream) {
  scheduler().notify_new_task(stream);
}

inline void notify_task_completion(const Stream& stream) {
  scheduler().notify_task_completion(stream);
}

inline void wait_for_capacity(const std::set<Stream>& open_streams) {
  scheduler().wait_for_capacity(open_streams);
}

} // namespace mlx::core::scheduler
