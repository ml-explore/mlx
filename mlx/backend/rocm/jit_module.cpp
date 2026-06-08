// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/jit_module.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/utils.h"
#include "mlx/version.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <sstream>

#include <fcntl.h>
#include <hip/hiprtc.h>
#include <unistd.h>

namespace mlx::core::rocm {

namespace {

// RAII helper that silences stderr during hipRTC compilation.
// AMD's comgr library (used by hipRTC) unconditionally writes preprocessed
// source and internal diagnostics to fd 2. This floods the terminal with
// thousands of lines of compiler-internal defines every time a new fused
// kernel is JIT-compiled.
struct StderrSuppressor {
  StderrSuppressor() {
    saved_fd_ = dup(STDERR_FILENO);
    if (saved_fd_ >= 0) {
      int devnull = open("/dev/null", O_WRONLY);
      if (devnull >= 0) {
        dup2(devnull, STDERR_FILENO);
        close(devnull);
        active_ = true;
      } else {
        // Could not open /dev/null — leave stderr alone.
        close(saved_fd_);
        saved_fd_ = -1;
      }
    }
  }
  ~StderrSuppressor() {
    restore();
  }
  void restore() {
    if (active_) {
      fflush(stderr);
      dup2(saved_fd_, STDERR_FILENO);
      close(saved_fd_);
      saved_fd_ = -1;
      active_ = false;
    }
  }
  StderrSuppressor(const StderrSuppressor&) = delete;
  StderrSuppressor& operator=(const StderrSuppressor&) = delete;

 private:
  int saved_fd_ = -1;
  bool active_ = false;
};

// Extract the last N lines from a compiler log.  AMD comgr prepends the
// entire preprocessed source to the error log, making it enormous.  The
// actual compiler errors are always at the end.
std::string tail_lines(const std::string& text, size_t n = 60) {
  if (text.empty()) {
    return text;
  }
  // Walk backwards to find the start of the last `n` lines.
  size_t count = 0;
  size_t pos = text.size();
  while (pos > 0 && count < n) {
    --pos;
    if (text[pos] == '\n') {
      ++count;
    }
  }
  if (pos > 0) {
    // Skip past the newline we stopped on.
    return "... [preprocessed source truncated] ...\n" + text.substr(pos + 1);
  }
  return text;
}

// Truncate long kernel names to avoid exceeding filesystem 255-byte limit.
// Names > 200 chars are replaced with a prefix + hash.
std::string safe_filename(const std::string& name) {
  constexpr size_t kMaxLen = 200;
  if (name.size() <= kMaxLen) {
    return name;
  }
  auto h = std::hash<std::string>{}(name);
  std::ostringstream oss;
  oss << name.substr(0, 64) << "_" << std::hex << h;
  return oss.str();
}

#define CHECK_HIPRTC_ERROR(cmd) check_hiprtc_error(#cmd, (cmd))

void check_hiprtc_error(const char* name, hiprtcResult err) {
  if (err != HIPRTC_SUCCESS) {
    std::ostringstream oss;
    oss << name << " failed: " << hiprtcGetErrorString(err);
    throw std::runtime_error(oss.str());
  }
}

// Return the location of the ROCm toolkit.
const std::string& rocm_home() {
  static std::string home = []() -> std::string {
    const char* home = std::getenv("ROCM_HOME");
    if (home) {
      return home;
    }
    home = std::getenv("ROCM_PATH");
    if (home) {
      return home;
    }
#if defined(__linux__)
    home = "/opt/rocm";
    if (std::filesystem::exists(home)) {
      return home;
    }
#endif
    throw std::runtime_error(
        "Environment variable ROCM_HOME or ROCM_PATH is not set.");
  }();
  return home;
}

// Get the cache directory for storing compiled results.
const std::filesystem::path& hsaco_cache_dir() {
  static std::filesystem::path cache = []() -> std::filesystem::path {
    std::filesystem::path cache;
    if (auto c = std::getenv("MLX_HSACO_CACHE_DIR"); c) {
      cache = c;
    } else {
      cache =
          std::filesystem::temp_directory_path() / "mlx" / version() / "hsaco";
    }
    if (!std::filesystem::exists(cache)) {
      std::error_code error;
      if (!std::filesystem::create_directories(cache, error)) {
        return std::filesystem::path();
      }
    }
    return cache;
  }();
  return cache;
}

// Get the path for HSACO file, splitting long names into nested directories.
// This mirrors the CUDA backend approach to handle long kernel names that
// would otherwise exceed filesystem filename limits (typically 255 chars).
std::filesystem::path get_hsaco_path(
    const std::filesystem::path& cache_dir,
    const std::string& module_name,
    const std::string& extension) {
  constexpr int max_file_name_length = 245;
  if (module_name.size() <= max_file_name_length) {
    return cache_dir / (module_name + extension);
  }

  auto hsaco_path = cache_dir;
  int offset = 0;
  while (module_name.size() - offset > max_file_name_length) {
    hsaco_path /= module_name.substr(offset, max_file_name_length);
    offset += max_file_name_length;
  }
  hsaco_path /= module_name.substr(offset) + extension;

  return hsaco_path;
}

// Try to read the cached |hsaco| and |hsaco_kernels| from |cache_dir|.
bool read_cached_hsaco(
    const std::filesystem::path& cache_dir,
    const std::string& module_name,
    std::string& hsaco,
    std::vector<std::pair<std::string, std::string>>& hsaco_kernels) {
  if (cache_dir.empty()) {
    return false;
  }

  auto hsaco_path = get_hsaco_path(cache_dir, module_name, ".hsaco");
  std::error_code error;
  auto hsaco_size = std::filesystem::file_size(hsaco_path, error);
  if (error) {
    return false;
  }
  std::ifstream hsaco_file(hsaco_path, std::ios::binary);
  if (!hsaco_file.good()) {
    return false;
  }
  hsaco.resize(hsaco_size);
  hsaco_file.read(hsaco.data(), hsaco_size);

  auto txt_path = get_hsaco_path(cache_dir, module_name, ".txt");
  std::ifstream txt_file(txt_path, std::ios::binary);
  std::string line;
  while (std::getline(txt_file, line)) {
    auto tab = line.find('\t');
    if (tab != std::string::npos) {
      hsaco_kernels.emplace_back(line.substr(0, tab), line.substr(tab + 1));
    }
  }
  return true;
}

// Write the |hsaco| and |hsaco_kernels| to |cache_dir| with |name|.
void write_cached_hsaco(
    const std::filesystem::path& cache_dir,
    const std::string& module_name,
    const std::string& hsaco,
    const std::vector<std::pair<std::string, std::string>>& hsaco_kernels,
    const std::string& source_code) {
  if (cache_dir.empty()) {
    return;
  }

  auto hsaco_path = get_hsaco_path(cache_dir, module_name, ".hsaco");

  // Create parent directories if they don't exist (for long module names)
  std::error_code error;
  std::filesystem::create_directories(hsaco_path.parent_path(), error);
  if (error) {
    return;
  }

  std::ofstream hsaco_file(hsaco_path, std::ios::binary);
  if (!hsaco.empty()) {
    hsaco_file.write(&hsaco.front(), hsaco.size());
  }

  auto txt_path = get_hsaco_path(cache_dir, module_name, ".txt");
  std::ofstream txt_file(txt_path, std::ios::binary);
  for (const auto& [name, mangled] : hsaco_kernels) {
    txt_file << name << "\t" << mangled << std::endl;
  }

  auto source_path = get_hsaco_path(cache_dir, module_name, ".hip");
  std::ofstream source_file(source_path);
  source_file << source_code;
}

// Get GPU architecture string for the current device
std::string get_gpu_arch() {
  hipDeviceProp_t props;
  int device_id;
  CHECK_HIP_ERROR(hipGetDevice(&device_id));
  CHECK_HIP_ERROR(hipGetDeviceProperties(&props, device_id));
  // gcnArchName already contains the full architecture name like "gfx1011"
  return std::string(props.gcnArchName);
}

void compile(
    Device& device,
    const std::string& module_name,
    const std::string& source,
    const std::vector<std::string>& kernel_names,
    std::string& hsaco,
    std::vector<std::pair<std::string, std::string>>& hsaco_kernels) {
  // Create the program
  // Use a hash of the module name to avoid "File name too long" errors
  // from hiprtc creating temporary files with the program name.
  auto program_name = "kernel_" +
      std::to_string(std::hash<std::string>{}(module_name)) + ".hip";
  hiprtcProgram prog;
  CHECK_HIPRTC_ERROR(hiprtcCreateProgram(
      &prog, source.c_str(), program_name.c_str(), 0, nullptr, nullptr));

  std::unique_ptr<hiprtcProgram, void (*)(hiprtcProgram*)> prog_freer(
      &prog,
      [](hiprtcProgram* p) { CHECK_HIPRTC_ERROR(hiprtcDestroyProgram(p)); });

  for (const auto& name : kernel_names) {
    CHECK_HIPRTC_ERROR(hiprtcAddNameExpression(prog, name.c_str()));
  }

  // Compile program.
  std::vector<const char*> args;
  std::vector<std::string> arg_strings;

  // Add standard flags
  arg_strings.push_back("--std=c++17");
  arg_strings.push_back("-O3");
  arg_strings.push_back("-DMLX_USE_ROCM");

  // Add GPU architecture
  std::string gpu_arch = get_gpu_arch();
  std::string arch_flag = "--offload-arch=" + gpu_arch;
  arg_strings.push_back(arch_flag);

  // Add include paths
  std::string rocm_include = "-I" + rocm_home() + "/include";
  arg_strings.push_back(rocm_include);

  for (const auto& arg : arg_strings) {
    args.push_back(arg.c_str());
  }

  // Suppress stderr during hipRTC compilation.  AMD's comgr backend
  // unconditionally dumps the entire preprocessed source to fd 2, flooding
  // the terminal with thousands of lines of compiler-internal defines.
  StderrSuppressor suppressor;
  hiprtcResult compile_result =
      hiprtcCompileProgram(prog, args.size(), args.data());
  suppressor.restore(); // restore stderr before any error reporting

  if (compile_result != HIPRTC_SUCCESS) {
    size_t log_size;
    CHECK_HIPRTC_ERROR(hiprtcGetProgramLogSize(prog, &log_size));
    std::vector<char> log(log_size + 1, 0);
    CHECK_HIPRTC_ERROR(hiprtcGetProgramLog(prog, log.data()));
    // The comgr log prepends the entire preprocessed source before the
    // actual error messages.  Truncate to only the trailing error lines.
    std::string truncated = tail_lines(std::string(log.data()));
    std::ostringstream oss;
    oss << "Failed to compile kernel '" << module_name << "': " << truncated;
    throw std::runtime_error(oss.str());
  }

  // Get mangled names of kernel names.
  for (const auto& name : kernel_names) {
    const char* mangled;
    CHECK_HIPRTC_ERROR(hiprtcGetLoweredName(prog, name.c_str(), &mangled));
    hsaco_kernels.emplace_back(name, mangled);
  }

  // Get code data.
  size_t code_size;
  CHECK_HIPRTC_ERROR(hiprtcGetCodeSize(prog, &code_size));
  hsaco.resize(code_size);
  CHECK_HIPRTC_ERROR(hiprtcGetCode(prog, hsaco.data()));
}

void load_module(
    const std::string& module_name,
    const std::string& hsaco,
    const std::vector<std::pair<std::string, std::string>>& hsaco_kernels,
    hipModule_t& module_,
    std::unordered_map<std::string, std::pair<hipFunction_t, bool>>& kernels) {
  // Load module.
  hipError_t load_result = hipModuleLoadData(&module_, hsaco.data());
  if (load_result != hipSuccess) {
    std::ostringstream oss;
    oss << "Failed to load compiled " << module_name
        << " kernel: " << hipGetErrorString(load_result) << ".";
    throw std::runtime_error(oss.str());
  }

  // Load kernels.
  for (const auto& [name, mangled] : hsaco_kernels) {
    hipFunction_t kernel;
    CHECK_HIP_ERROR(hipModuleGetFunction(&kernel, module_, mangled.c_str()));
    kernels[name] = std::make_pair(kernel, false);
  }
}

} // namespace

JitModule::JitModule(
    Device& device,
    const std::string& module_name,
    const KernelBuilder& builder,
    bool use_disk_cache) {
  // Will hold the actual device executable source code and kernel names
  std::string hsaco;
  std::vector<std::pair<std::string, std::string>> hsaco_kernels;

  // Use a safe filename for disk cache to avoid exceeding 255-byte limit
  std::string cache_name = safe_filename(module_name);

  // Try to load them from the file cache
  if (!read_cached_hsaco(hsaco_cache_dir(), cache_name, hsaco, hsaco_kernels)) {
    auto [precompiled, source_code, kernel_names] = builder();

    // Get the HSACO (AMD GPU binary)
    if (precompiled) {
      hsaco = std::move(source_code);
      for (auto& name : kernel_names) {
        hsaco_kernels.emplace_back(name, name);
      }
    } else {
      compile(
          device, module_name, source_code, kernel_names, hsaco, hsaco_kernels);
    }

    // If requested save them in the file cache for the next launch
    if (use_disk_cache) {
      write_cached_hsaco(
          hsaco_cache_dir(), cache_name, hsaco, hsaco_kernels, source_code);
    }
  }

  // Load the module
  load_module(module_name, hsaco, hsaco_kernels, module_, kernels_);
}

JitModule::~JitModule() {
  if (module_) {
    (void)hipModuleUnload(module_);
  }
}

hipFunction_t JitModule::get_kernel(
    const std::string& kernel_name,
    std::function<void(hipFunction_t)> configure_kernel) {
  auto it = kernels_.find(kernel_name);
  if (it == kernels_.end()) {
    throw std::runtime_error(
        std::string("There is no kernel named ") + kernel_name + ".");
  }

  // If it is the first time we run this kernel then configure it. Do it only
  // once!
  if (!it->second.second) {
    if (configure_kernel) {
      configure_kernel(it->second.first);
    }
    it->second.second = true;
  }

  return it->second.first;
}

std::unordered_map<std::string, JitModule>& get_jit_module_cache() {
  static std::unordered_map<std::string, JitModule> map;
  return map;
}

JitModule& get_jit_module(
    const mlx::core::Device& mlx_device,
    const std::string& name,
    const KernelBuilder& builder,
    bool cache) {
  auto& map = get_jit_module_cache();
  auto it = map.find(name);
  if (it == map.end()) {
    it = map.try_emplace(name, device(mlx_device), name, builder, cache).first;
  }
  return it->second;
}

} // namespace mlx::core::rocm
