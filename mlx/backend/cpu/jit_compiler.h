// Copyright © 2024-2026 Apple Inc.
#pragma once

#include <filesystem>

namespace mlx::core {

class JitCompiler {
 public:
  // Return the includes that should be prepended to the source code.
  static const std::tuple<bool, std::string, std::string>& get_preamble();

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

  // Identify everything besides the source text that determines the machine
  // code: which compiler was selected and the ISA flags it is given. Both are
  // resolved at runtime, so the same MLX build can compile the same source two
  // different ways. Callers must fold this into any on-disk cache key.
  static std::string toolchain_id();

  // Run a command and get its output.
  static std::string exec(const std::string& cmd);
};

} // namespace mlx::core
