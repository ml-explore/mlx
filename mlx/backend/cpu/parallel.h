// Copyright © 2025 Apple Inc.

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace mlx::core::cpu {

// A minimal persistent worker pool for splitting hot CPU kernels over
// independent output slices. The calling thread participates in the work,
// so a pool of size n spawns n - 1 threads. Sized by MLX_CPU_THREADS
// (default 8, clamped to hardware concurrency).
class KernelPool {
 public:
  static KernelPool& instance() {
    static KernelPool pool;
    return pool;
  }

  KernelPool(const KernelPool&) = delete;
  KernelPool& operator=(const KernelPool&) = delete;

  int size() const {
    return n_threads_;
  }

  // Calls fn(t) for every t in [0, size()) and waits for completion. Only
  // call from one thread at a time (the CPU stream thread).
  void run(const std::function<void(int)>& fn) {
    if (n_threads_ == 1) {
      fn(0);
      return;
    }
    {
      std::lock_guard<std::mutex> lk(mtx_);
      fn_ = &fn;
      pending_.store(n_threads_ - 1, std::memory_order_release);
      generation_.fetch_add(1, std::memory_order_release);
    }
    // Workers spin between back-to-back kernels; only pay the wake syscall
    // for the ones that actually parked.
    if (parked_.load(std::memory_order_acquire) > 0) {
      cv_.notify_all();
    }
    fn(0);
    // Kernel slices run for microseconds to milliseconds; spin for the tail.
    while (pending_.load(std::memory_order_acquire) > 0) {
      std::this_thread::yield();
    }
  }

 private:
  KernelPool() {
    int n = 8;
    if (const char* env = std::getenv("MLX_CPU_THREADS")) {
      n = std::atoi(env);
    }
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    n_threads_ = std::max(1, std::min(n, hw > 0 ? hw : 1));
    for (int t = 1; t < n_threads_; t++) {
      workers_.emplace_back([this, t]() { worker(t); });
    }
  }

  ~KernelPool() {
    {
      std::lock_guard<std::mutex> lk(mtx_);
      stop_ = true;
      generation_.fetch_add(1, std::memory_order_release);
    }
    cv_.notify_all();
    for (auto& w : workers_) {
      w.join();
    }
  }

  void worker(int tid) {
    uint64_t seen = 0;
    while (true) {
      // Spin briefly to catch back-to-back kernels without a syscall, then
      // park on the condition variable. Longer windows and WFE waits both
      // regress: they compete for shared resources with the serial sections
      // between kernels.
      for (int spins = 0;
           generation_.load(std::memory_order_acquire) == seen &&
           spins < 20000;
           spins++) {
      }
      const std::function<void(int)>* fn;
      {
        std::unique_lock<std::mutex> lk(mtx_);
        if (generation_.load(std::memory_order_acquire) == seen) {
          parked_.fetch_add(1, std::memory_order_release);
          cv_.wait(lk, [&] {
            return generation_.load(std::memory_order_acquire) != seen;
          });
          parked_.fetch_sub(1, std::memory_order_release);
        }
        seen = generation_.load(std::memory_order_acquire);
        if (stop_) {
          return;
        }
        fn = fn_;
      }
      (*fn)(tid);
      pending_.fetch_sub(1, std::memory_order_release);
    }
  }

  std::mutex mtx_;
  std::condition_variable cv_;
  const std::function<void(int)>* fn_{nullptr};
  std::atomic<uint64_t> generation_{0};
  std::atomic<int> pending_{0};
  std::atomic<int> parked_{0};
  bool stop_{false};
  int n_threads_{1};
  std::vector<std::thread> workers_;
};

// Runs fn(begin, end) over [0, n) rows with dynamic chunk stealing so faster
// cores take more chunks (performance cores finish ahead of efficiency
// cores). work_per_row is an operation-count estimate used to size chunks
// and to skip the pool entirely for small problems.
template <typename F>
void parallel_for_rows(int n, size_t work_per_row, F&& fn) {
  // 128k-op chunks balance the tail straggler against chunk-pull atomics
  // (64k chunks regress the biggest mats; 256k lengthens the tail).
  constexpr size_t min_total_work = size_t(1) << 21;
  constexpr size_t min_chunk_work = size_t(1) << 17;
  auto& pool = KernelPool::instance();
  size_t total = work_per_row * size_t(n);
  if (pool.size() == 1 || total < min_total_work) {
    fn(0, n);
    return;
  }
  int chunk = int(std::max<size_t>(
      8, min_chunk_work / std::max<size_t>(1, work_per_row)));
  int n_chunks = (n + chunk - 1) / chunk;
  if (n_chunks <= 1) {
    fn(0, n);
    return;
  }
  std::atomic<int> next{0};
  std::function<void(int)> job = [&](int) {
    while (true) {
      int c = next.fetch_add(1, std::memory_order_relaxed);
      if (c >= n_chunks) {
        break;
      }
      int begin = c * chunk;
      int end = std::min(n, begin + chunk);
      fn(begin, end);
    }
  };
  pool.run(job);
}

} // namespace mlx::core::cpu
