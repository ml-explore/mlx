// Copyright © 2023-2026 Apple Inc.

#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>

#include "doctest/doctest.h"

#include "mlx/allocator.h"
#include "mlx/device.h"
#include "mlx/memory.h"
#include "mlx/ops.h"
#include "mlx/scheduler.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

using namespace mlx::core;

TEST_CASE("test simple allocations") {
  {
    auto buffer = allocator::malloc(sizeof(float));
    auto fptr = static_cast<float*>(buffer.raw_ptr());
    *fptr = 0.5f;
    CHECK_EQ(*fptr, 0.5f);
    allocator::free(buffer);
  }

  {
    auto buffer = allocator::malloc(128 * sizeof(int));
    int* ptr = static_cast<int*>(buffer.raw_ptr());
    for (int i = 0; i < 128; ++i) {
      ptr[i] = i;
    }
    allocator::free(buffer);
  }

  {
    auto buffer = allocator::malloc(0);
    allocator::free(buffer);
  }
}

TEST_CASE("test large allocations") {
  size_t size = 1 << 30;
  for (int i = 0; i < 100; ++i) {
    auto buffer = allocator::malloc(size);
    allocator::free(buffer);
  }
}

TEST_CASE("test cached allocation keeps capacity") {
  auto old_limit = set_cache_limit(1 << 20);
  clear_cache();

  auto large = allocator::malloc(8192);
  allocator::free(large);
  auto cached = get_cache_memory();
  CHECK_GE(cached, 8192);

  auto small = allocator::malloc(6000);
  CHECK_GE(allocator::allocator().size(small), cached);
  allocator::free(small);
  CHECK_GE(get_cache_memory(), cached);

  clear_cache();
  set_cache_limit(old_limit);
}

TEST_CASE("test array buffer size") {
  auto a = array({1.0f, 2.0f, 3.0f, 4.0f});
  auto view = reshape(a, {2, 2});

  CHECK_THROWS_AS(get_array_buffer_size({view}), std::invalid_argument);

  eval(view);
  auto a_size = a.buffer_size();
  CHECK_EQ(get_array_buffer_size({}), 0);
  CHECK_EQ(get_array_buffer_size({a}), a_size);
  CHECK_EQ(get_array_buffer_size({a, a}), a_size);
  CHECK_EQ(get_array_buffer_size({a, view}), a_size);

  auto b = array({5.0f, 6.0f});
  CHECK_EQ(get_array_buffer_size({a, b}), a_size + b.buffer_size());
  CHECK_EQ(get_array_buffer_size({array({})}), 0);
}

TEST_CASE("test clear cache synchronizes cpu streams") {
  if (is_available(Device{Device::gpu})) {
    return;
  }

  auto old_limit = set_cache_limit(1 << 20);
  clear_cache();

  auto cached = allocator::malloc(8192);
  allocator::free(cached);
  CHECK_GE(get_cache_memory(), 8192);

  auto task_started = std::make_shared<std::promise<void>>();
  auto task_started_future = task_started->get_future();
  auto release_task = std::make_shared<std::promise<void>>();
  auto release_task_future = release_task->get_future().share();
  auto task_finished = std::make_shared<std::promise<void>>();
  auto task_finished_future = task_finished->get_future();
  auto clear_finished = std::make_shared<std::promise<void>>();
  auto clear_finished_future = clear_finished->get_future();

  auto stream = new_stream(Device{Device::cpu});
  scheduler::enqueue(
      stream, [task_started, release_task_future, task_finished] {
        task_started->set_value();
        release_task_future.wait();
        task_finished->set_value();
      });

  task_started_future.wait();

  std::thread clear_thread([clear_finished] {
    clear_cache();
    clear_finished->set_value();
  });

  CHECK_EQ(
      clear_finished_future.wait_for(std::chrono::milliseconds(50)),
      std::future_status::timeout);

  release_task->set_value();

  CHECK_EQ(
      task_finished_future.wait_for(std::chrono::seconds(10)),
      std::future_status::ready);
  CHECK_EQ(
      clear_finished_future.wait_for(std::chrono::seconds(10)),
      std::future_status::ready);
  clear_thread.join();

  set_cache_limit(old_limit);
}
