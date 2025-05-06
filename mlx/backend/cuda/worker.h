// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/event.h"
#include "mlx/backend/cuda/utils.h"

#include <functional>
#include <map>
#include <mutex>
#include <thread>

namespace mlx::core::cu {

// Run tasks in worker thread, synchronized with cuda stream.
class Worker {
 public:
  Worker();
  ~Worker();

  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  // Add a pending |task| that will run when consumed or commited.
  void add_task(std::function<void()> task);

  // Run pending tasks immediately in current thread.
  void consume_in_this_thread();

  // Put pending tasks in a batch.
  void end_batch();

  // Inform worker thread to run current batches now.
  void commit();

  // Inform worker thread to run current batches after kernels in |stream|
  // finish running.
  void commit(cudaStream_t stream);

  // Return how many batches have been added but not committed yet.
  size_t uncommited_batches() const {
    return uncommited_batches_;
  }

 private:
  void thread_fn();

  uint64_t batch_{0};
  size_t uncommited_batches_{0};

  // Cuda stream and event for signaling kernel completion.
  CudaStream signal_stream_;
  CudaEvent signal_event_;

  // Worker thread.
  SharedEvent worker_event_;
  std::thread worker_;
  std::mutex worker_mutex_;
  bool stop_{false};

  // Tasks are put in |pending_tasks_| first, and then moved to
  // |worker_tasks_| when end_batch() is called.
  using Tasks = std::vector<std::function<void()>>;
  Tasks pending_tasks_;
  std::map<uint64_t, Tasks> worker_tasks_;
};

} // namespace mlx::core::cu
