// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/jit_module.h"
#include "mlx/backend/rocm/utils.h"

#include <fmt/format.h>
#include <mutex>
#include <sstream>

namespace mlx::core::rocm {

JitModule::JitModule(
    const std::string& kernel_name,
    const std::string& kernel_source,
    const std::vector<std::string>& template_args,
    const std::vector<std::string>& compiler_flags,
    bool verbose) {
  compile(kernel_name, kernel_source, template_args, compiler_flags, verbose);
}

JitModule::~JitModule() {
  if (kernel_) {
    // No hipFunctionDestroy equivalent in HIP
  }
  if (module_) {
    CHECK_HIP_ERROR(hipModuleUnload(module_));
  }
  if (program_) {
    hiprtcDestroyProgram(&program_);
  }
}

void JitModule::compile(
    const std::string& kernel_name,
    const std::string& kernel_source,
    const std::vector<std::string>& template_args,
    const std::vector<std::string>& compiler_flags,
    bool verbose) {
  // Create HIPRTC program
  CHECK_HIP_ERROR(hiprtcCreateProgram(
      &program_,
      kernel_source.c_str(),
      kernel_name.c_str(),
      0,
      nullptr,
      nullptr));

  // Build compiler options
  std::vector<const char*> options;
  std::vector<std::string> option_strings;

  // Add default options
  option_strings.push_back("--std=c++17");
  option_strings.push_back("-O3");
  option_strings.push_back("-DMLX_USE_ROCM");

  // Add user-provided flags
  for (const auto& flag : compiler_flags) {
    option_strings.push_back(flag);
  }

  // Add template arguments
  for (const auto& arg : template_args) {
    option_strings.push_back("-D" + arg);
  }

  // Convert to char* array
  for (const auto& option : option_strings) {
    options.push_back(option.c_str());
  }

  // Compile the program
  hiprtcResult compile_result =
      hiprtcCompileProgram(program_, options.size(), options.data());

  // Get compilation log
  size_t log_size;
  CHECK_HIP_ERROR(hiprtcGetProgramLogSize(program_, &log_size));

  if (log_size > 1) {
    std::vector<char> log(log_size);
    CHECK_HIP_ERROR(hiprtcGetProgramLog(program_, log.data()));

    if (verbose || compile_result != HIPRTC_SUCCESS) {
      fmt::print(
          "HIPRTC compilation log for {}:\n{}\n", kernel_name, log.data());
    }
  }

  if (compile_result != HIPRTC_SUCCESS) {
    throw std::runtime_error(
        fmt::format("HIPRTC compilation failed for kernel {}", kernel_name));
  }

  // Get compiled code
  size_t code_size;
  CHECK_HIP_ERROR(hiprtcGetCodeSize(program_, &code_size));

  std::vector<char> code(code_size);
  CHECK_HIP_ERROR(hiprtcGetCode(program_, code.data()));

  // Load module
  CHECK_HIP_ERROR(hipModuleLoadData(&module_, code.data()));

  // Get kernel function
  CHECK_HIP_ERROR(hipModuleGetFunction(&kernel_, module_, kernel_name.c_str()));
}

JitCache& JitCache::instance() {
  static JitCache cache;
  return cache;
}

std::shared_ptr<JitModule> JitCache::get_or_create(
    const std::string& kernel_name,
    const std::string& kernel_source,
    const std::vector<std::string>& template_args,
    const std::vector<std::string>& compiler_flags) {
  std::string key =
      make_key(kernel_name, kernel_source, template_args, compiler_flags);

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = cache_.find(key);
  if (it != cache_.end()) {
    if (auto module = it->second.lock()) {
      return module;
    } else {
      cache_.erase(it);
    }
  }

  auto module = std::make_shared<JitModule>(
      kernel_name, kernel_source, template_args, compiler_flags);
  cache_[key] = module;
  return module;
}

std::string JitCache::make_key(
    const std::string& kernel_name,
    const std::string& kernel_source,
    const std::vector<std::string>& template_args,
    const std::vector<std::string>& compiler_flags) const {
  std::ostringstream oss;
  oss << kernel_name << "|" << kernel_source;

  for (const auto& arg : template_args) {
    oss << "|" << arg;
  }

  for (const auto& flag : compiler_flags) {
    oss << "|" << flag;
  }

  return oss.str();
}

std::shared_ptr<JitModule> make_jit_kernel(
    const std::string& kernel_name,
    const std::string& kernel_source,
    const std::vector<std::string>& template_args,
    const std::vector<std::string>& compiler_flags) {
  return JitCache::instance().get_or_create(
      kernel_name, kernel_source, template_args, compiler_flags);
}

} // namespace mlx::core::rocm