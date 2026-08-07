// Copyright © 2026 Apple Inc.

#include <barrier>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "doctest/doctest.h"
#include "mlx/backend/metal/device.h"

using namespace mlx::core;

TEST_CASE("test concurrent metal kernel cache misses") {
  constexpr int n_threads = 16;
  constexpr auto kernel_name = "concurrent_cache_kernel";
  constexpr auto kernel_source = R"(
    #include <metal_stdlib>
    using namespace metal;

    kernel void concurrent_cache_kernel(
        device float* out [[buffer(0)]],
        uint index [[thread_position_in_grid]]) {
      out[index] = 1.0f;
    }
  )";

  metal::Device device;
  std::vector<MTL::Library*> libraries;
  libraries.reserve(n_threads);
  for (int i = 0; i < n_threads; ++i) {
    auto source =
        std::string(kernel_source) + "\n// library " + std::to_string(i);
    libraries.push_back(device.get_library(
        "concurrent_cache_library_" + std::to_string(i),
        [source = std::move(source)] { return source; }));
  }

  std::barrier start(n_threads);
  std::vector<MTL::ComputePipelineState*> kernels(n_threads);
  std::vector<std::thread> threads;
  std::exception_ptr error;
  std::mutex error_mutex;
  threads.reserve(n_threads);

  for (int i = 0; i < n_threads; ++i) {
    threads.emplace_back([&, i] {
      start.arrive_and_wait();
      try {
        kernels[i] = device.get_kernel(kernel_name, libraries[i]);
      } catch (...) {
        std::lock_guard lock(error_mutex);
        if (!error) {
          error = std::current_exception();
        }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  if (error) {
    std::rethrow_exception(error);
  }

  for (int i = 0; i < n_threads; ++i) {
    CHECK(kernels[i] != nullptr);
    CHECK_EQ(kernels[i], device.get_kernel(kernel_name, libraries[i]));
  }
}
