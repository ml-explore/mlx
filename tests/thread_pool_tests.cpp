// Copyright © 2026 Apple Inc.

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "doctest/doctest.h"

#include "mlx/backend/cpu/threading/common.h"

using namespace mlx::core::cpu;

TEST_CASE("thread_pool max_threads is positive") {
  auto& pool = ThreadPool::instance();
  CHECK(pool.max_threads() >= 1);
}

TEST_CASE("thread_pool parallel_for with 1 thread") {
  auto& pool = ThreadPool::instance();
  int called = 0;
  pool.parallel_for(1, [&](int tid, int nth) {
    CHECK_EQ(tid, 0);
    CHECK_EQ(nth, 1);
    called++;
  });
  CHECK_EQ(called, 1);
}

TEST_CASE("thread_pool parallel_for covers all elements") {
  auto& pool = ThreadPool::instance();
  const int N = 10000;
  std::vector<std::atomic<int>> counts(N);
  for (int i = 0; i < N; i++) {
    counts[i].store(0);
  }

  int nth = pool.max_threads();
  pool.parallel_for(nth, [&](int tid, int nth) {
    int chunk = (N + nth - 1) / nth;
    int start = chunk * tid;
    int end = std::min(start + chunk, N);
    for (int i = start; i < end; i++) {
      counts[i].fetch_add(1, std::memory_order_relaxed);
    }
  });

  for (int i = 0; i < N; i++) {
    CHECK_EQ(counts[i].load(), 1);
  }
}

TEST_CASE("thread_pool parallel_for with max threads") {
  auto& pool = ThreadPool::instance();
  int nth_req = pool.max_threads();
  std::vector<std::atomic<int>> seen(nth_req);
  for (int i = 0; i < nth_req; i++) {
    seen[i].store(0);
  }

  pool.parallel_for(nth_req, [&](int tid, int nth) {
    CHECK_EQ(nth, nth_req);
    CHECK(tid >= 0);
    CHECK(tid < nth);
    seen[tid].fetch_add(1, std::memory_order_relaxed);
  });

  for (int i = 0; i < nth_req; i++) {
    CHECK_EQ(seen[i].load(), 1);
  }
}

TEST_CASE("thread_pool parallel_for clamps n_threads to max_threads") {
  auto& pool = ThreadPool::instance();
  int max_t = pool.max_threads();
  std::atomic<int> max_nth{0};

  pool.parallel_for(max_t + 100, [&](int tid, int nth) {
    // nth should be clamped to max_threads
    int cur = max_nth.load(std::memory_order_relaxed);
    while (nth > cur && !max_nth.compare_exchange_weak(cur, nth)) {
    }
  });

  CHECK_EQ(max_nth.load(), max_t);
}

TEST_CASE("thread_pool parallel_for with n_threads 0 returns immediately") {
  auto& pool = ThreadPool::instance();
  std::atomic<int> called{0};
  pool.parallel_for(0, [&](int tid, int nth) {
    called.fetch_add(1, std::memory_order_relaxed);
  });
  // ThreadPool::parallel_for returns immediately for n_threads <= 0
  CHECK_EQ(called.load(), 0);
}

TEST_CASE("thread_pool rapid serial parallel_for") {
  auto& pool = ThreadPool::instance();
  int nth = std::min(pool.max_threads(), 4);
  std::atomic<int> total{0};

  for (int iter = 0; iter < 10000; iter++) {
    pool.parallel_for(nth, [&](int tid, int nth) {
      total.fetch_add(1, std::memory_order_relaxed);
    });
  }

  CHECK_EQ(total.load(), 10000 * nth);
}

TEST_CASE("thread_pool lopsided work") {
  auto& pool = ThreadPool::instance();
  int nth = std::min(pool.max_threads(), 4);
  if (nth < 2) {
    return; // Need at least 2 threads for this test
  }

  std::atomic<int> done_count{0};

  pool.parallel_for(nth, [&](int tid, int nth) {
    if (tid == 0) {
      // Simulate heavy work on thread 0
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    done_count.fetch_add(1, std::memory_order_relaxed);
  });

  CHECK_EQ(done_count.load(), nth);
}

TEST_CASE("thread_pool concurrent parallel_for from multiple threads") {
  auto& pool = ThreadPool::instance();
  const int N_CALLERS = 4;
  const int ITERS = 1000;
  int nth = std::min(pool.max_threads(), 4);

  std::atomic<int> total{0};

  std::vector<std::thread> callers;
  callers.reserve(N_CALLERS);
  for (int c = 0; c < N_CALLERS; c++) {
    callers.emplace_back([&] {
      for (int i = 0; i < ITERS; i++) {
        pool.parallel_for(nth, [&](int tid, int nth) {
          total.fetch_add(1, std::memory_order_relaxed);
        });
      }
    });
  }

  for (auto& t : callers) {
    t.join();
  }

  CHECK_EQ(total.load(), N_CALLERS * ITERS * nth);
}

TEST_CASE("thread_pool parallel_for with exceptions does not deadlock") {
  // Current impl doesn't propagate exceptions, but it must not hang.
  // We use a watchdog thread to detect deadlock.
  auto& pool = ThreadPool::instance();
  int nth = std::min(pool.max_threads(), 4);

  std::atomic<bool> finished{false};

  std::thread watchdog([&] {
    std::this_thread::sleep_for(std::chrono::seconds(5));
    if (!finished.load()) {
      // Deadlock detected - force fail
      FAIL("parallel_for deadlocked after exception");
    }
  });
  watchdog.detach();

  // Only throw from slot 0 (main thread) so it propagates directly
  try {
    pool.parallel_for(nth, [&](int tid, int nth) {
      if (tid == 0) {
        throw std::runtime_error("test exception");
      }
    });
  } catch (const std::runtime_error&) {
    // Expected
  }

  finished.store(true);

  // Verify pool still works after exception
  std::atomic<int> count{0};
  pool.parallel_for(nth, [&](int tid, int nth) {
    count.fetch_add(1, std::memory_order_relaxed);
  });
  CHECK_EQ(count.load(), nth);
}

TEST_CASE("thread_pool parallel_for with varying thread counts") {
  auto& pool = ThreadPool::instance();
  int max_t = pool.max_threads();
  std::atomic<int> total{0};

  // Alternate between different thread counts to exercise session
  // mode activation/deactivation
  int counts[] = {1, max_t, 2, max_t, 1, max_t};
  for (int nth : counts) {
    nth = std::min(nth, max_t);
    pool.parallel_for(nth, [&](int tid, int nth) {
      total.fetch_add(1, std::memory_order_relaxed);
    });
  }

  int expected = 0;
  for (int nth : counts) {
    expected += std::min(nth, max_t);
  }
  CHECK_EQ(total.load(), expected);
}

TEST_CASE("thread_pool effective_threads") {
  int max_t = 16;

  // Below MIN_TOTAL_ELEMENTS -> 1 thread
  CHECK_EQ(effective_threads(100, max_t), 1);
  CHECK_EQ(effective_threads(0, max_t), 1);

  // At threshold
  CHECK_EQ(
      effective_threads(MIN_TOTAL_ELEMENTS, max_t),
      std::min(
          static_cast<int>(MIN_TOTAL_ELEMENTS / MIN_ELEMENTS_PER_THREAD),
          max_t));

  // Large workload -> capped by max_threads
  CHECK_EQ(effective_threads(1000000000, max_t), max_t);
}
