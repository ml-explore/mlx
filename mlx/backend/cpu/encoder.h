// Copyright © 2025 Apple Inc.

#pragma once

#include <unordered_map>

#include "mlx/array.h"
#include "mlx/scheduler.h"
#include "mlx/threadpool.h"

namespace mlx::core::cpu {

// Number of dispatches per scheduler task
constexpr int DISPATCHES_PER_TASK = 10;

MLX_API ThreadPool& thread_pool();
MLX_API size_t thread_pool_size();

struct MLX_API CommandEncoder {
  CommandEncoder(Stream stream) : stream_(stream) {}

  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;
  CommandEncoder(CommandEncoder&&) = delete;
  CommandEncoder& operator=(CommandEncoder&&) = delete;

  void set_input_array(const array& a) {}
  void set_output_array(array& a) {}

  // Hold onto a temporary until any already scheduled tasks which use it as
  // an input are complete.
  void add_temporary(array arr) {
    temporaries_.push_back(std::move(arr));
  }

  void add_temporaries(std::vector<array> arrays) {
    temporaries_.insert(
        temporaries_.end(),
        std::make_move_iterator(arrays.begin()),
        std::make_move_iterator(arrays.end()));
  }

  std::vector<array>& temporaries() {
    return temporaries_;
  }

  template <class F, class... Args>
  void dispatch(F&& f, Args&&... args) {
    num_ops_ = (num_ops_ + 1) % DISPATCHES_PER_TASK;
    auto task = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
    if (num_ops_ == 0) {
      scheduler::notify_new_task(stream_);
      auto task_wrap = [s = stream_, task = std::move(task)]() mutable {
        task();
        scheduler::notify_task_completion(s);
      };
      scheduler::enqueue(stream_, std::move(task_wrap));
    } else {
      scheduler::enqueue(stream_, std::move(task));
    }
  }

  template <class F>
  void dispatch_parallel(size_t n, F&& f, size_t grain_size = 32768) {
    dispatch([n, grain_size, f = std::forward<F>(f)]() mutable {
      if (n == 0) {
        return;
      }

      if (grain_size == 0) {
        grain_size = 1;
      }

      auto num_threads = thread_pool_size();

      if (n <= grain_size || num_threads <= 1) {
        f(0, n);
        return;
      }

      // grain_size is minimum amount of range work to assign to a thread-pool
      // task. Small ranges run on stream worker to avoid overhead.
      size_t num_tasks = std::min(num_threads, n / grain_size);

      if (num_tasks <= 1) {
        f(0, n);
        return;
      }

      size_t chunk_size = n / num_tasks + (n % num_tasks != 0);

      std::vector<std::future<void>> futures;
      futures.reserve(num_tasks);

      for (size_t start = 0; start < n; start += chunk_size) {
        size_t end = std::min(start + chunk_size, n);
        futures.emplace_back(
            thread_pool().enqueue([start, end, &f]() { f(start, end); }));
      }

      for (auto& future : futures) {
        future.get();
      }
    });
  }

 private:
  Stream stream_;
  std::vector<array> temporaries_;
  int num_ops_{0};
};

MLX_API CommandEncoder& get_command_encoder(Stream s);

std::unordered_map<int, CommandEncoder>& get_command_encoders();
std::unordered_map<int, CommandEncoder>& get_global_command_encoders();

} // namespace mlx::core::cpu
