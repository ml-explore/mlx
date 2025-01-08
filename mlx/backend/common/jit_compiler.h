// Copyright © 2024 Apple Inc.
#pragma once

#include <filesystem>

namespace mlx::core {

class JitCompiler {
 public:
  // Build a shell command that compiles a source code file to a shared library.
  static std::string build_command(
      const std::filesystem::path& dir,
      const std::string& source_file_name,
      const std::string& shared_lib_name);

  // Run a command and get its output.
  static std::string exec(const std::string& cmd);
};

} // namespace mlx::core
