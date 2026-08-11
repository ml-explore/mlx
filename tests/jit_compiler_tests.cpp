// Copyright © 2026 Apple Inc.

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <tuple>

#include "doctest/doctest.h"

#include "mlx/backend/cpu/jit_compiler.h"

using namespace mlx::core;

TEST_CASE("test CPU JIT compiler supports C++20") {
  if (!JitCompiler::available()) {
    return;
  }

  namespace fs = std::filesystem;
  auto suffix =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  auto test_dir =
      fs::temp_directory_path() / ("mlx-cxx20-jit-" + std::to_string(suffix));
  std::error_code error;
  REQUIRE(fs::create_directory(test_dir, error));

  struct Cleanup {
    fs::path path;
    ~Cleanup() {
      std::error_code error;
      fs::remove_all(path, error);
    }
  } cleanup{test_dir};

  const std::string source_name = "cxx20_probe.cpp";
#ifdef _WIN32
  const std::string library_name = "cxx20_probe.dll";
#else
  const std::string library_name = "libcxx20_probe.so";
#endif

  std::ofstream source(test_dir / source_name);
  REQUIRE(source.good());
  source << std::get<2>(JitCompiler::get_preamble());
  source << R"code(
template <typename T>
concept Addable = requires(T value) {
  value + value;
};

static_assert(Addable<int>);

extern "C" int mlx_cxx20_probe(int value) {
  return value;
}
)code";
  source.close();
  REQUIRE(source.good());

  JitCompiler::exec(
      JitCompiler::build_command(test_dir, source_name, library_name));
  CHECK(fs::is_regular_file(test_dir / library_name));
}
