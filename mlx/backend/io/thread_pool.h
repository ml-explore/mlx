// Copyright © 2024 Apple Inc.

#pragma once

#include <future>
#include <queue>
#include <unordered_set>

#include "mlx/array.h"

namespace mlx::core::io::detail {

class ThreadPool {
 public:
  explicit ThreadPool(int workers);
  ~ThreadPool();

  ThreadPool(ThreadPool&&) = delete;
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  std::future<void> enqueue(
      std::function<void()> task,
      const std::vector<array>& inputs,
      const std::vector<array>& outputs);
  std::future<void> barrier(
      const std::vector<int>& worker_ids,
      std::function<void()> on_barrier);
  std::future<void> barrier(const std::vector<int>& worker_ids);
  std::future<void> barrier(std::function<void()> on_barrier);
  std::future<void> barrier();

 private:
  std::future<void> enqueue(std::function<void()> task, int worker_idx);
  void add_outputs_to_worker(const std::vector<array>& outputs, int worker_idx);
  std::function<void()> remove_outputs_when_done(
      std::function<void()> task,
      const std::vector<array>& outputs,
      int worker_idx);
  void worker(int idx);

  std::vector<std::queue<std::packaged_task<void()>>> task_queues_;
  std::vector<std::mutex> queue_mutexes_;
  std::vector<std::condition_variable> queue_cvs_;
  std::vector<std::mutex> set_mutexes_;
  std::vector<std::unordered_set<std::uintptr_t>> output_sets_;
  bool stop_;
  std::vector<std::thread> workers_;
};

} // namespace mlx::core::io::detail
