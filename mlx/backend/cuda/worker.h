// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/event.h"

#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>

namespace mlx::core::cu {

// Run tasks in worker thread, synchronized with cuda stream.
class Worker {
 public:
  explicit Worker(Device& d);
  ~Worker();

  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  // Add a pending |task| that will run when consumed or commited.
  void add_task(std::function<void()> task);

  // Inform worker thread to run current batches after kernels in |stream|
  // finish running.
  void commit(cudaStream_t stream);

 private:
  using Tasks = std::vector<std::function<void()>>;

  // State shared with the detached worker thread, which may outlive the Worker
  // and free the state after the CUDA runtime is gone.
  struct State {
    static void signal(void* data);
    void thread_fn();

    std::mutex mtx;
    std::condition_variable cond;
    uint64_t signaled_batch{0};
    bool stop{false};
    std::map<uint64_t, Tasks> worker_tasks;
  };

  uint64_t committed_batch_{0};

  // Cuda stream and event for signaling kernel completion.
  CudaStream signal_stream_;
  CudaEvent signal_event_;

  // Tasks are put in |pending_tasks_| first, and then moved to
  // |state_->worker_tasks| when commit() is called.
  Tasks pending_tasks_;
  std::shared_ptr<State> state_;
};

} // namespace mlx::core::cu
