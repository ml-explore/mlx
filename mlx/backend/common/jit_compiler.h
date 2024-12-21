// Copyright © 2024 Apple Inc.
#pragma once

#include <string>

namespace mlx::core {

class JitCompiler {
 public:
  // Build a shell command that compiles |source_file_path| to a shared library
  // at |shared_lib_path|.
  static std::string build_command(
      const std::string& source_file_path,
      const std::string& shared_lib_path);
};

} // namespace mlx::core
