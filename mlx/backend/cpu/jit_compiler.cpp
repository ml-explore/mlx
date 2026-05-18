// Copyright © 2024 Apple Inc.

#include "mlx/backend/cpu/jit_compiler.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/compiled_preamble.h"

#include <algorithm>
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
    std::string vs_path = JitCompiler::exec(
        fmt::format(
            "\"{0}\\Microsoft Visual Studio\\Installer\\vswhere.exe\""
            " -latest -property installationPath",
            std::getenv("ProgramFiles(x86)")));
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
            "\"{0}\\VC\\Auxiliary\\Build\\vcvarsall.bat\" {1} >NUL && set",
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
        cl_exe = fmt::format("{0}\\bin\\Host{1}\\{1}\\cl.exe", value, arch);
      }
    }
  }
  std::string arch;
  std::string cl_exe;
  std::vector<std::string> libpaths;
};

const VisualStudioInfo& GetVisualStudioInfo() {
  static VisualStudioInfo info;
  return info;
}

} // namespace

#endif // _MSC_VER

const std::tuple<bool, std::string, std::string>& JitCompiler::get_preamble() {
  static auto preamble = []() -> std::tuple<bool, std::string, std::string> {
    // Check whether the headers are shipped with the binary, if so use the
    // preamble from the headers, otherwise use the prebuilt one embeded in
    // binary, which may not work with all compilers.
    auto root_dir = current_binary_dir();
#if !defined(_WIN32)
    root_dir = root_dir.parent_path();
#endif
    auto include_dir = root_dir / "include";
    if (std::filesystem::exists(include_dir / "mlx")) {
      return std::make_tuple(
          true,
          include_dir.string(),
          "#include \"mlx/backend/cpu/compiled_preamble.h\"\n");
    } else {
      return std::make_tuple(false, "", get_prebuilt_preamble());
    }
  }();
  return preamble;
}

std::string JitCompiler::build_command(
    const std::filesystem::path& dir,
    const std::string& source_file_name,
    const std::string& shared_lib_name) {
  auto& [use_include, include_dir, preamble] = get_preamble();
#ifdef _MSC_VER
  std::string extra_flags;
  if (use_include) {
    extra_flags += fmt::format("/I \"{}\"", include_dir);
  }
  const VisualStudioInfo& info = GetVisualStudioInfo();
  for (const std::string& lib : info.libpaths) {
    extra_flags += fmt::format(" /libpath:\"{}\"", lib);
  }
  return fmt::format(
      "\""
      "cd /D \"{}\" && "
      "\"{}\" /LD /EHsc /MD /Ox /nologo /std:c++17 {} \"{}\" "
      "/link /out:\"{}\" 2>&1"
      "\"",
      dir.string(),
      info.cl_exe,
      extra_flags,
      source_file_name,
      shared_lib_name);
#else
  std::string extra_flags;
  if (use_include) {
    extra_flags = fmt::format("-I \"{}\"", include_dir);
  }
  return fmt::format(
      "g++ -std=c++17 -O3 -Wall -fPIC -shared {} \"{}\" -o \"{}\" 2>&1",
      extra_flags,
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
  // Trim trailing spaces.
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
