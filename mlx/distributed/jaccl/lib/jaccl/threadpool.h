// This code was modified from https://github.com/progschj/ThreadPool
// The original License is copied below:
//
// Copyright (c) 2012 Jakob Progsch, Václav Zeman
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
//    1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would be
//    appreciated but is not required.
//
//    2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
//
//    3. This notice may not be removed or altered from any source
//    distribution.
#pragma once

#include <atomic>
#include <concepts>
#include <condition_variable>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace jaccl {

class ThreadPool {
 public:
  ThreadPool(size_t);
  template <class F, class... Args>
  auto enqueue(F&& f, Args&&... args)
      -> std::future<typename std::invoke_result_t<F, Args...>>;
  void resize(size_t);
  ~ThreadPool();

 private:
  void stop_and_wait();
  void start_threads(size_t);

  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};

inline ThreadPool::ThreadPool(size_t threads) : stop(false) {
  start_threads(threads);
}

template <class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::invoke_result_t<F, Args...>> {
  using return_type = typename std::invoke_result_t<F, Args...>;

  auto task = std::make_shared<std::packaged_task<return_type()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<return_type> res = task->get_future();
  {
    std::unique_lock<std::mutex> lock(queue_mutex);

    if (stop) {
      throw std::runtime_error(
          "[ThreadPool::enqueue] Not allowed on stopped ThreadPool");
    }

    tasks.emplace([task]() { (*task)(); });
  }
  condition.notify_one();
  return res;
}

inline void ThreadPool::resize(size_t threads) {
  if (workers.size() == threads) {
    return;
  }

  if (workers.size() > threads) {
    stop_and_wait();
  }
  start_threads(threads - workers.size());
}

inline ThreadPool::~ThreadPool() {
  stop_and_wait();
}

inline void ThreadPool::stop_and_wait() {
  // Stop the current threads and wait until they finish
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    stop = true;
  }
  condition.notify_all();
  for (std::thread& worker : workers) {
    worker.join();
  }

  // Reset the member variables so that the threadpool is reusable
  stop = false;
  workers.clear();
}

inline void ThreadPool::start_threads(size_t threads) {
  for (size_t i = 0; i < threads; ++i) {
    workers.emplace_back([this] {
      for (;;) {
        std::function<void()> task;

        {
          std::unique_lock<std::mutex> lock(this->queue_mutex);
          this->condition.wait(
              lock, [this] { return this->stop || !this->tasks.empty(); });
          if (this->stop && this->tasks.empty())
            return;
          task = std::move(this->tasks.front());
          this->tasks.pop();
        }

        task();
      }
    });
  }
}

// Run fn(lw) for each wire, the first n_wires - 1 on the pool and the last
// inline. Wait for all of them, then rethrow the first failure.
template <std::invocable<int> Fn>
void dispatch_wires(ThreadPool* pool, int n_wires, Fn&& fn) {
  if (n_wires <= 1 || !pool) {
    for (int lw = 0; lw < n_wires; lw++) {
      fn(lw);
    }
    return;
  }

  std::atomic<bool> failed{false};
  std::exception_ptr first_error;
  auto run = [&failed, &first_error, &fn](int lw) {
    try {
      fn(lw);
    } catch (...) {
      if (!failed.exchange(true)) {
        first_error = std::current_exception();
      }
    }
  };

  std::vector<std::future<void>> futures;
  futures.reserve(n_wires - 1);
  for (int lw = 0; lw < n_wires - 1; lw++) {
    futures.emplace_back(pool->enqueue(run, lw));
  }
  run(n_wires - 1);
  for (auto& f : futures) {
    f.wait();
  }

  if (first_error) {
    std::rethrow_exception(first_error);
  }
}

} // namespace jaccl
