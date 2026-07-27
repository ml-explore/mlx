// Copyright © 2026 Apple Inc.

#pragma once

#include <filesystem>
#include <span>
#include <string>

namespace mlx::core::cu::detail {

inline std::filesystem::path mlx_package_root(
    const std::filesystem::path& binary_dir) {
#if defined(_WIN32)
  return binary_dir;
#else
  return binary_dir.parent_path();
#endif
}

inline bool has_cuda_runtime_headers(const std::filesystem::path& path) {
  return std::filesystem::exists(path / "cuda.h") &&
      std::filesystem::exists(path / "cuda_runtime.h");
}

inline std::filesystem::path find_cuda_runtime_include_dir(
    const std::filesystem::path& root_dir,
    int cuda_major_version,
    std::span<const std::filesystem::path> toolkit_roots) {
  auto path = root_dir.parent_path() / "nvidia";
  if (cuda_major_version >= 13) {
    path /= "cu" + std::to_string(cuda_major_version);
  } else {
    path /= "cuda_runtime";
  }
  path /= "include";
  if (has_cuda_runtime_headers(path)) {
    return path;
  }

  for (const auto& root : toolkit_roots) {
    path = root / "include";
    if (has_cuda_runtime_headers(path)) {
      return path;
    }
  }
  return {};
}

} // namespace mlx::core::cu::detail
