// Copyright © 2024-2026 Apple Inc.

#include "mlx/backend/cpu/jit_compiler.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>

#include <fmt/format.h>

namespace mlx::core {

#ifdef _MSC_VER

namespace {

// Split string into array.
std::vector<std::string> str_split(const std::string& str, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(str);
  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

// Get path information about MSVC.
struct VisualStudioInfo {
  VisualStudioInfo() {
#ifdef _M_ARM64
    arch = "arm64";
#else
    arch = "x64";
#endif
    // Get path of Visual Studio.
    // Use -latest to get only the most recent installation when multiple
    // versions are installed, avoiding path concatenation issues.
    auto pf86 = std::getenv("ProgramFiles(x86)");
    if (!pf86) {
      throw std::runtime_error(
          "ProgramFiles(x86) environment variable not set.");
    }
    std::string vs_path = JitCompiler::exec(
        fmt::format(
            "\"{0}\\Microsoft Visual Studio\\Installer\\vswhere.exe\""
            " -latest -property installationPath 2>&1",
            pf86));
    if (vs_path.empty()) {
      throw std::runtime_error("Can not find Visual Studio.");
    }
    // Trim any trailing whitespace/newlines from the path
    vs_path.erase(
        std::find_if(
            vs_path.rbegin(),
            vs_path.rend(),
            [](unsigned char ch) { return !std::isspace(ch); })
            .base(),
        vs_path.end());
    // Read the envs from vcvarsall.
    std::string envs = JitCompiler::exec(
        fmt::format(
            "\"{0}\\VC\\Auxiliary\\Build\\vcvarsall.bat\" {1} >NUL 2>&1 && set",
            vs_path,
            arch));
    for (const std::string& line : str_split(envs, '\n')) {
      // Each line is in the format "ENV_NAME=values".
      auto pos = line.find_first_of('=');
      if (pos == std::string::npos || pos == 0 || pos == line.size() - 1)
        continue;
      std::string name = line.substr(0, pos);
      std::string value = line.substr(pos + 1);
      if (name == "LIB") {
        libpaths = str_split(value, ';');
      } else if (name == "VCToolsInstallDir" || name == "VCTOOLSINSTALLDIR") {
        msvc_cl = fmt::format("{0}\\bin\\Host{1}\\{1}\\cl.exe", value, arch);
      }
    }

    // Check for clang-cl bundled with Visual Studio.
    std::string clang_cl_path = fmt::format(
        "{0}\\VC\\Tools\\Llvm\\{1}\\bin\\clang-cl.exe", vs_path, arch);
    {
      std::ifstream f(clang_cl_path);
      if (f.good()) {
        clang_cl = clang_cl_path;
      }
    }

    // Select the JIT compiler. The preamble was preprocessed at build time
    // by whichever compiler built the library -- it contains compiler-specific
    // builtins (e.g. __builtin_fpclassify for Clang, __is_same for MSVC) that
    // are only valid for the same compiler family. Prefer the matching one.
#ifdef __clang__
    cl_exe = !clang_cl.empty() ? clang_cl : msvc_cl;
#else
    cl_exe = !msvc_cl.empty() ? msvc_cl : clang_cl;
#endif
  }
  std::string arch;
  std::string cl_exe;
  std::string msvc_cl;
  std::string clang_cl;
  std::vector<std::string> libpaths;
};

const VisualStudioInfo& GetVisualStudioInfo() {
  static VisualStudioInfo info;
  return info;
}

} // namespace

#endif // _MSC_VER

bool JitCompiler::available() {
#ifdef _MSC_VER
  static int result = -1; // -1 = not probed yet
  if (result == -1) {
    try {
      const auto& info = GetVisualStudioInfo();
      // The preamble is preprocessed at build time by the build compiler.
      // It contains compiler-specific intrinsics that only the same compiler
      // family can parse, so we must have the matching compiler at runtime.
#ifdef __clang__
      result = !info.clang_cl.empty() ? 1 : 0;
#else
      result = !info.msvc_cl.empty() ? 1 : 0;
#endif
    } catch (...) {
      result = 0;
    }
  }
  return result == 1;
#else
  static int result = -1;
  if (result == -1) {
    // The build command uses g++, and the preamble was preprocessed by the
    // build compiler (GCC or Clang). The preprocessed output contains
    // compiler-specific intrinsics (e.g. __remove_reference for GCC,
    // __builtin_* for Clang) that only the same compiler family can parse.
    // Note: on some distros (e.g. macOS), g++ may be a symlink to Clang.
    // Check that g++ is available in PATH.
#ifdef _WIN32
    result = (std::system("g++ --version > NUL 2>&1") == 0) ? 1 : 0;
#else
    result = (std::system("g++ --version > /dev/null 2>&1") == 0) ? 1 : 0;
#endif
  }
  return result == 1;
#endif
}

std::string JitCompiler::build_command(
    const std::filesystem::path& dir,
    const std::string& source_file_name,
    const std::string& shared_lib_name) {
#ifdef _MSC_VER
  const VisualStudioInfo& info = GetVisualStudioInfo();
  std::string libpaths;
  for (const std::string& lib : info.libpaths) {
    libpaths += fmt::format(" /libpath:\"{0}\"", lib);
  }
  // clang-cl accepts the same flags as cl.exe (/LD, /EHsc, etc.)
  // but we add -Wno-everything to suppress warnings from the preprocessed
  // preamble, which may contain pragmas or builtins from a different compiler
  // (e.g. MSVC pragmas when compiling with clang-cl, or vice versa).
  std::string extra_flags;
  if (!info.clang_cl.empty() && info.cl_exe == info.clang_cl) {
    extra_flags = " -Wno-everything";
  }
#ifdef __AVX2__
  extra_flags += " /arch:AVX2";
#endif
  auto cmd = fmt::format(
      "\""
      "cd /D \"{0}\" && "
      "\"{1}\" /LD /EHsc /MD /Ox /nologo /std:c++17{5} \"{2}\" "
      "/link /out:\"{3}\" {4} 2>&1"
      "\"",
      dir.string(),
      info.cl_exe,
      source_file_name,
      shared_lib_name,
      libpaths,
      extra_flags);
  return cmd;
#else
  return fmt::format(
      "g++ -std=c++17 -O3 -Wall -fPIC -shared"
#ifdef __AVX2__
      " -mavx2 -mfma -mf16c"
#endif
      " \"{0}\" -o \"{1}\" 2>&1",
      (dir / source_file_name).string(),
      (dir / shared_lib_name).string());
#endif
}

std::string JitCompiler::exec(const std::string& cmd) {
#ifdef _MSC_VER
  FILE* pipe = _popen(cmd.c_str(), "r");
#else
  FILE* pipe = popen(cmd.c_str(), "r");
#endif
  if (!pipe) {
    throw std::runtime_error("popen() failed.");
  }
  char buffer[128];
  std::string ret;
  while (fgets(buffer, sizeof(buffer), pipe)) {
    ret += buffer;
  }
  // Trim trailing whitespace.
  ret.erase(
      std::find_if(
          ret.rbegin(),
          ret.rend(),
          [](unsigned char ch) { return !std::isspace(ch); })
          .base(),
      ret.end());

#ifdef _MSC_VER
  int status = _pclose(pipe);
#else
  int status = pclose(pipe);
#endif
  if (status == -1) {
    throw std::runtime_error("pclose() failed.");
  }
#if defined(_WIN32) || defined(__FreeBSD__)
  int code = status;
#else
  int code = WEXITSTATUS(status);
#endif
  if (code != 0) {
    throw std::runtime_error(
        fmt::format(
            "Failed to execute command with return code {0}: \"{1}\", "
            "the output is: {2}",
            code,
            cmd,
            ret));
  }
  return ret;
}

} // namespace mlx::core
