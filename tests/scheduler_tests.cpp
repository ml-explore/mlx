// Copyright © 2023 Apple Inc.

#include "doctest/doctest.h"

#include "mlx/mlx.h"
#include "mlx/scheduler.h"

using namespace mlx::core;

TEST_CASE("test stream management") {
  auto s1 = default_stream(default_device());
  CHECK_EQ(s1.device, default_device());

  auto s2 = new_stream(default_device());
  CHECK_EQ(s2.device, default_device());
  CHECK_NE(s1, s2);

  // Check that default streams have the correct devices
  if (gpu::is_available()) {
    auto s_gpu = default_stream(Device::gpu);
    CHECK_EQ(s_gpu.device, Device::gpu);
  } else {
    CHECK_THROWS_AS(default_stream(Device::gpu), std::invalid_argument);
  }
  auto s_cpu = default_stream(Device::cpu);
  CHECK_EQ(s_cpu.device, Device::cpu);

  s_cpu = new_stream(Device::cpu);
  CHECK_EQ(s_cpu.device, Device::cpu);

  if (gpu::is_available()) {
    auto s_gpu = new_stream(Device::gpu);
    CHECK_EQ(s_gpu.device, Device::gpu);
  } else {
    CHECK_THROWS_AS(new_stream(Device::gpu), std::invalid_argument);
  }
}

TEST_CASE("test asynchronous launch") {
  auto s1 = default_stream(Device::cpu);
  auto s2 = new_stream(Device::cpu);

  // Make sure streams execute asynchronously
  int x = 1;
  auto p1 = std::make_shared<std::promise<void>>();
  auto p2 = std::make_shared<std::promise<void>>();
  auto f1 = p1->get_future().share();
  auto f2 = p2->get_future().share();
  auto fn1 = [&x, p = std::move(p1)]() {
    x++;
    p->set_value();
  };
  auto fn2 = [&x, p = std::move(p2), f = std::move(f1)]() {
    f.wait();
    x *= 5;
    p->set_value();
  };

  // fn2 is launched first and is waiting on fn1 but since
  // they are on different streams there is no deadlock.
  scheduler::enqueue(s2, std::move(fn2));
  scheduler::enqueue(s1, std::move(fn1));

  f2.wait();

  CHECK_EQ(x, 10);
}

TEST_CASE("test stream placement") {
  auto s1 = default_stream(Device::cpu);
  auto s2 = new_stream(Device::cpu);

  {
    // Wait on stream 1
    auto p = std::make_shared<std::promise<void>>();
    auto f = p->get_future().share();
    scheduler::enqueue(s1, [f = std::move(f)]() { f.wait(); });

    // Do some work on stream 2
    auto x = zeros({100}, float32, s2);
    auto y = ones({100}, float32, s2);
    auto z = add(x, y, s2);
    eval(z);
    p->set_value();
  }

  {
    // Wait on stream 1
    auto p = std::make_shared<std::promise<void>>();
    auto f = p->get_future().share();
    scheduler::enqueue(s1, [f = std::move(f)]() { f.wait(); });

    // Do some work on stream 2
    auto fn = [&s2](array a) { return add(a, add(a, a, s2), s2); };
    auto x = zeros({100}, s2);

    // The whole vjp computation should happen
    // on the second stream otherwise this will hang.
    auto [out, dout] = vjp(fn, x, ones({100}, s2));

    // The whole jvp computation should happen on the
    // second stream.
    std::tie(out, dout) = jvp(fn, x, ones({100}, s2));
    eval(out, dout);

    p->set_value();
  }
}

TEST_CASE("test scheduler races") {
  auto x = zeros({1});
  auto y = zeros({100});
  eval(x, y);
  auto a = exp(x);
  eval(a);
  a = exp(x);
  for (int i = 0; i < 10000; ++i) {
    y = exp(y);
  }
  eval(a, y);
}
