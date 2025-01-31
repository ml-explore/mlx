// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/jit_compiler.h"

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
    std::string vs_path = JitCompiler::exec(fmt::format(
        "\"{0}\\Microsoft Visual Studio\\Installer\\vswhere.exe\""
        " -property installationPath",
        std::getenv("ProgramFiles(x86)")));
    if (vs_path.empty()) {
      throw std::runtime_error("Can not find Visual Studio.");
    }
    // Read the envs from vcvarsall.
    std::string envs = JitCompiler::exec(fmt::format(
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
      } else if (name == "VCToolsInstallDir") {
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
  return fmt::format(
      "\""
      "cd /D \"{0}\" && "
      "\"{1}\" /LD /EHsc /MD /Ox /nologo /std:c++17 \"{2}\" "
      "/link /out:\"{3}\" {4} 2>&1"
      "\"",
      dir.string(),
      info.cl_exe,
      source_file_name,
      shared_lib_name,
      libpaths);
#else
  return fmt::format(
      "g++ -std=c++17 -O3 -Wall -fPIC -shared \"{0}\" -o \"{1}\" 2>&1",
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
#ifdef _WIN32
  int code = status;
#else
  int code = WEXITSTATUS(status);
#endif
  if (code != 0) {
    throw std::runtime_error(fmt::format(
        "Failed to execute command with return code {0}: \"{1}\", "
        "the output is: {2}",
        code,
        cmd,
        ret));
  }
  return ret;
}

} // namespace mlx::core
