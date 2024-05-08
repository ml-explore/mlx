// Copyright © 2024 Apple Inc.

#include <numeric>

#include "mlx/backend/io/thread_pool.h"

namespace mlx::core::io::detail {

ThreadPool::ThreadPool(int workers) : stop_(false) {
  for (int i = 0; i < workers; i++) {
    workers_.emplace_back(&ThreadPool::worker, this, i);
  }
}

ThreadPool::~ThreadPool() {
  stop_ = true;
  for (auto& cv : queue_cvs_) {
    cv.notify_one();
  }

  for (auto& t : workers_) {
    if (t.joinable()) {
      t.join();
    }
  }
}

std::future<void> ThreadPool::enqueue(
    std::function<void()> task,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs) {
  std::vector<int> barriers;
  for (int i = 0; i < output_sets_.size(); i++) {
    std::lock_guard<std::mutex> lock(set_mutexes_[i]);

    for (auto& a : inputs) {
      if (output_sets_[i].find(a.buffer().ptr()) != output_sets_[i].end()) {
        barriers.push_back(i);
        break;
      }
    }
  }

  // Case 1: Barriers is empty so try to add it to the smallest queue
  if (barriers.empty()) {
    auto min_queue = std::min_element(
        task_queues_.begin(), task_queues_.end(), [](auto& left, auto& right) {
          return left.size() < right.size();
        });
    int worker_idx = std::distance(task_queues_.begin(), min_queue);

    add_outputs_to_worker(outputs, worker_idx);
    return enqueue(
        remove_outputs_when_done(std::move(task), outputs, worker_idx),
        worker_idx);
  }

  // Case 2: Barriers has only one queue so put that into that queue
  if (barriers.size() == 1) {
    int worker_idx = barriers[0];
    add_outputs_to_worker(outputs, worker_idx);
    return enqueue(
        remove_outputs_when_done(std::move(task), outputs, worker_idx),
        worker_idx);
  }

  // Case 3: We need to add a barrier before our task and add it to the
  // smallest queue of the barriers.
  auto min_queue = std::min_element(
      barriers.begin(), barriers.end(), [this](auto left, auto right) {
        return task_queues_[left].size() < task_queues_[right].size();
      });
  int worker_idx = *min_queue;
  barriers.erase(min_queue);
  std::shared_future<void> queue_barrier =
      barrier(barriers); // We shouldn't need shared future here
  add_outputs_to_worker(outputs, worker_idx);
  return enqueue(
      remove_outputs_when_done(
          [queue_barrier = std::move(queue_barrier),
           og_task = std::move(task)]() {
            queue_barrier.wait();
            og_task();
          },
          outputs,
          worker_idx),
      worker_idx);
}

std::future<void> ThreadPool::enqueue(
    std::function<void()> task,
    int worker_idx) {
  std::packaged_task<void()> pt(std::move(task));
  std::future<void> result = pt.get_future();
  {
    std::lock_guard<std::mutex> lock(queue_mutexes_[worker_idx]);
    task_queues_[worker_idx].emplace(std::move(pt));
  }
  queue_cvs_[worker_idx].notify_one();
  return result;
}

void ThreadPool::add_outputs_to_worker(
    const std::vector<array>& outputs,
    int worker_idx) {
  if (outputs.size() == 0) {
    return;
  }

  std::lock_guard<std::mutex> lock(set_mutexes_[worker_idx]);
  for (auto& a : outputs) {
    output_sets_[worker_idx].insert(a.buffer().ptr());
  }
}

std::function<void()> ThreadPool::remove_outputs_when_done(
    std::function<void()> task,
    const std::vector<array>& outputs,
    int worker_idx) {
  if (outputs.size() == 0) {
    return task;
  }

  std::vector<const void*> output_buffers;
  for (auto& a : outputs) {
    output_buffers.push_back(a.buffer().ptr());
  }

  return [og_task = std::move(task),
          buffers = std::move(output_buffers),
          worker_idx,
          this]() {
    og_task();
    {
      std::lock_guard<std::mutex> lock(set_mutexes_[worker_idx]);
      for (auto b : buffers) {
        output_sets_[worker_idx].erase(b);
      }
    }
  };
}

std::future<void> ThreadPool::barrier(
    const std::vector<int>& worker_ids,
    std::function<void()> on_barrier) {
  auto workers = std::make_shared<std::atomic<int>>(worker_ids.size());
  auto promise = std::make_shared<std::promise<void>>();
  auto future = promise->get_future();

  for (auto idx : worker_ids) {
    enqueue(
        [workers, promise, on_barrier = std::move(on_barrier)]() {
          (*workers)--;
          if (*workers <= 0) {
            on_barrier();
            promise->set_value();
          }
        },
        idx);
  }

  return future;
}

std::future<void> ThreadPool::barrier(const std::vector<int>& worker_ids) {
  auto noop = []() {};
  return barrier(worker_ids, std::move(noop));
}

std::future<void> ThreadPool::barrier(std::function<void()> on_barrier) {
  std::vector<int> worker_ids(workers_.size());
  std::iota(worker_ids.begin(), worker_ids.end(), 0);
  return barrier(worker_ids, std::move(on_barrier));
}

std::future<void> ThreadPool::barrier() {
  auto noop = []() {};
  return barrier(std::move(noop));
}

void ThreadPool::worker(int idx) {
  while (true) {
    std::packaged_task<void()> task;
    {
      std::unique_lock<std::mutex> lock(queue_mutexes_[idx]);
      queue_cvs_[idx].wait(
          lock, [this, idx]() { return stop_ || !task_queues_[idx].empty(); });
      if (task_queues_[idx].empty()) {
        if (stop_) {
          break;
        } else {
          continue;
        }
      }
      task = std::move(task_queues_[idx].front());
      task_queues_[idx].pop();
    }
    task();
  }
}

} // namespace mlx::core::io::detail
