// Copyright © 2024-2026 Apple Inc.
#pragma once

#include <filesystem>

namespace mlx::core {

class JitCompiler {
 public:
  // Check if a JIT compiler is available on this system.
  // On Windows, this probes for Visual Studio and a usable C++ compiler
  // (MSVC cl.exe or clang-cl). On Linux/macOS, checks for g++ in PATH.
  // Returns false (rather than throwing) if no compiler is found.
  static bool available();

  // Build a shell command that compiles a source code file to a shared library.
  static std::string build_command(
      const std::filesystem::path& dir,
      const std::string& source_file_name,
      const std::string& shared_lib_name);

  // Run a command and get its output.
  static std::string exec(const std::string& cmd);
};

} // namespace mlx::core
