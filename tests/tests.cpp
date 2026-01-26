// Copyright © 2023 Apple Inc.

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"

#include <cstdlib>

#include "mlx/mlx.h"

// Use regular fp32 precision for tests (matches Python test setup)
#ifdef _WIN32
#define SET_ENV(name, value) _putenv_s(name, value)
#else
#define SET_ENV(name, value) setenv(name, value, 0)
#endif

namespace {
struct EnvInit {
  EnvInit() {
    SET_ENV("MLX_ENABLE_TF32", "0");
  }
} env_init;
} // namespace

using namespace mlx::core;

// Global test listener to reset GPU state after each test.
// This prevents errors from one test cascading to subsequent tests.
struct GpuCleanupListener : doctest::IReporter {
  explicit GpuCleanupListener(const doctest::ContextOptions&) {}

  void test_case_end(const doctest::CurrentTestCaseStats&) override {
    // Synchronize to ensure all GPU operations complete before starting
    // the next test. This prevents incomplete operations from affecting
    // subsequent tests.
    if (is_available(Device::gpu)) {
      synchronize();
    }
  }

  // Required virtual methods (minimal implementation)
  void report_query(const doctest::QueryData&) override {}
  void test_run_start() override {}
  void test_run_end(const doctest::TestRunStats&) override {}
  void test_case_start(const doctest::TestCaseData&) override {}
  void test_case_reenter(const doctest::TestCaseData&) override {}
  void test_case_exception(const doctest::TestCaseException&) override {}
  void subcase_start(const doctest::SubcaseSignature&) override {}
  void subcase_end() override {}
  void log_assert(const doctest::AssertData&) override {}
  void log_message(const doctest::MessageData&) override {}
  void test_case_skipped(const doctest::TestCaseData&) override {}
};

REGISTER_LISTENER("gpu_cleanup", 1, GpuCleanupListener);

int main(int argc, char** argv) {
  doctest::Context context;

  const char* device = std::getenv("DEVICE");
  if (device != nullptr && std::string(device) == "cpu") {
    set_default_device(Device::cpu);
  } else if (is_available(Device::gpu)) {
    // Use generic GPU availability check (works for Metal on macOS, or CUDA on
    // Linux/Windows)
    set_default_device(Device::gpu);
  }

  context.applyCommandLine(argc, argv);
  return context.run();
}
