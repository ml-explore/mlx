// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/jit_compiler.h"

#include <sstream>

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

// Run a command and get its output.
std::string exec(const std::string& cmd) {
  std::unique_ptr<FILE, decltype(&_pclose)> pipe(
      _popen(cmd.c_str(), "r"), _pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed.");
  }
  char buffer[128];
  std::string ret;
  while (fgets(buffer, sizeof(buffer), pipe.get())) {
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
  return ret;
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
    std::string vs_path = exec(fmt::format(
        "\"{0}\\Microsoft Visual Studio\\Installer\\vswhere.exe\""
        " -property installationPath",
        std::getenv("ProgramFiles(x86)")));
    if (vs_path.empty()) {
      throw std::runtime_error("Can not find Visual Studio.");
    }
    // Read the envs from vcvarsall.
    std::string envs = exec(fmt::format(
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
    const std::string& source_file_path,
    const std::string& shared_lib_path) {
#ifdef _MSC_VER
  const VisualStudioInfo& info = GetVisualStudioInfo();
  std::string libpaths;
  for (const std::string& lib : info.libpaths) {
    libpaths += fmt::format(" /libpath:\"{0}\"", lib);
  }
  std::string command = fmt::format(
      "\""
      "\"{0}\" /LD /EHsc /nologo /std:c++17 \"{1}\" /link /out:\"{2}\"{3}"
      "\"",
      info.cl_exe,
      source_file_path,
      shared_lib_path,
      libpaths);
#else
  std::string command = fmt::format(
      "g++ -std=c++17 -O3 -Wall -fPIC -shared '{0}' -o '{1}'",
      source_file_path,
      shared_lib_path);
#endif
  return command;
}

} // namespace mlx::core
