// Copyright © 2026 Apple Inc.

#include <chrono>

#include "doctest/doctest.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/event.h"
#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test public event requires gpu") {
  auto stream = default_stream(Device::cpu);
  Event internal_event(stream);
  CHECK(internal_event.valid());
  CHECK_THROWS_AS(Event(stream, false), std::invalid_argument);
  CHECK_THROWS_AS(Event(stream, true), std::invalid_argument);
}

TEST_CASE("test metal event synchronization") {
  if (!gpu::is_available()) {
    return;
  }

  auto record_stream = new_stream(Device::gpu);
  auto wait_stream = new_stream(Device::gpu);
  Event event(record_stream, false);
  Event end_event(record_stream, false);

  CHECK(query(record_stream));
  CHECK(query(wait_stream));
  CHECK(event.query());
  CHECK_NOTHROW(event.wait(wait_stream));
  CHECK_NOTHROW(event.synchronize());

  auto out = exp(arange(1 << 20, float32, record_stream), record_stream);
  async_eval(out);
  event.record(record_stream);
  event.wait(wait_stream);
  CHECK_FALSE(query(wait_stream));
  auto waited_out = exp(arange(1 << 20, float32, wait_stream), wait_stream);
  async_eval(waited_out);
  synchronize(wait_stream);

  CHECK(query(wait_stream));
  CHECK(event.query());
  event.record(record_stream);
  event.synchronize();
  CHECK(event.query());

  end_event.record(record_stream);
  CHECK_THROWS_AS(event.elapsed_time(end_event), std::runtime_error);
}

TEST_CASE("test metal event elapsed time") {
  if (!gpu::is_available()) {
    return;
  }

  auto stream = default_stream(Device::gpu);
  Event start(stream, true);
  Event end(stream, true);
  CHECK_THROWS_AS(start.elapsed_time(end), std::runtime_error);

  auto before = std::chrono::steady_clock::now();
  start.record(stream);
  auto out = exp(arange(1 << 20, float32, stream), stream);
  async_eval(out);
  end.record(stream);
  auto elapsed = start.elapsed_time(end);
  auto wall_time = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - before)
                       .count();

  CHECK(elapsed > 0.0);
  CHECK(elapsed <= wall_time);
}
