// Copyright © 2026 Apple Inc.

#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

#include "doctest/doctest.h"

#include "mlx/backend/cuda/runtime_headers.h"

namespace {

class TempDirectory {
 public:
  TempDirectory() {
    std::mt19937_64 random(std::random_device{}());
    for (int attempt = 0; attempt < 100; ++attempt) {
      path_ = std::filesystem::temp_directory_path() /
          ("mlx-cuda-runtime-headers-" + std::to_string(random()));
      if (std::filesystem::create_directory(path_)) {
        return;
      }
    }
    throw std::runtime_error("Failed to create a temporary directory");
  }

  ~TempDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  const std::filesystem::path& path() const {
    return path_;
  }

 private:
  std::filesystem::path path_;
};

void create_cuda_headers(const std::filesystem::path& include_dir) {
  std::filesystem::create_directories(include_dir);
  std::ofstream(include_dir / "cuda.h");
  std::ofstream(include_dir / "cuda_runtime.h");
}

} // namespace

using namespace mlx::core::cu::detail;

TEST_CASE("test CUDA runtime header package layouts") {
  TempDirectory temp;
  auto mlx_root = temp.path() / "site-packages" / "mlx";

  SUBCASE("binary directory resolves to the MLX package root") {
#if defined(_WIN32)
    auto binary_dir = mlx_root;
#else
    auto binary_dir = mlx_root / "lib";
#endif
    CHECK_EQ(mlx_package_root(binary_dir), mlx_root);
  }

  SUBCASE("CUDA 13 uses the cu13 package") {
    auto expected =
        temp.path() / "site-packages" / "nvidia" / "cu13" / "include";
    create_cuda_headers(expected);

    CHECK_EQ(find_cuda_runtime_include_dir(mlx_root, 13, {}), expected);
  }

  SUBCASE("CUDA 12 uses the legacy cuda_runtime package") {
    auto expected =
        temp.path() / "site-packages" / "nvidia" / "cuda_runtime" / "include";
    create_cuda_headers(expected);

    CHECK_EQ(find_cuda_runtime_include_dir(mlx_root, 12, {}), expected);
  }

  SUBCASE("a toolkit root is used when the package is incomplete") {
    auto package_include =
        temp.path() / "site-packages" / "nvidia" / "cu13" / "include";
    std::filesystem::create_directories(package_include);
    std::ofstream(package_include / "cuda_runtime.h");

    auto toolkit_root = temp.path() / "toolkit";
    create_cuda_headers(toolkit_root / "include");
    std::vector<std::filesystem::path> toolkit_roots{toolkit_root};

    CHECK_EQ(
        find_cuda_runtime_include_dir(mlx_root, 13, toolkit_roots),
        toolkit_root / "include");
  }

  SUBCASE("no candidates returns an empty path") {
    CHECK(find_cuda_runtime_include_dir(mlx_root, 13, {}).empty());
  }
}
