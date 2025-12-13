// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/jit_module.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/version.h"

#include "cuda_jit_sources.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>

#include <fmt/format.h>
#include <nvrtc.h>
#include <unistd.h>

namespace mlx::core::cu {

namespace {

#define CHECK_NVRTC_ERROR(cmd) check_nvrtc_error(#cmd, (cmd))

void check_nvrtc_error(const char* name, nvrtcResult err) {
  if (err != NVRTC_SUCCESS) {
    throw std::runtime_error(
        fmt::format("{} failed: {}", name, nvrtcGetErrorString(err)));
  }
}

// Return the --include-path args used for invoking NVRTC.
const std::vector<std::string>& include_path_args() {
  static std::vector<std::string> cached_args = []() {
    std::vector<std::string> args;
    // Add path to bundled CCCL headers.
    auto root_dir = current_binary_dir().parent_path();
    auto path = root_dir / "include" / "cccl";
#if defined(MLX_CCCL_DIR)
    if (!std::filesystem::exists(path)) {
      path = MLX_CCCL_DIR;
    }
#endif
    if (std::filesystem::exists(path)) {
      args.push_back(fmt::format("--include-path={}", path.string()));
    }
    // Add path to CUDA runtime headers, try local-installed python package
    // first and then system-installed headers.
    path = root_dir.parent_path() / "nvidia" / "cuda_runtime" / "include";
    if (std::filesystem::exists(path)) {
      args.push_back(fmt::format("--include-path={}", path.string()));
    } else {
      const char* home = std::getenv("CUDA_HOME");
      if (!home) {
        home = std::getenv("CUDA_PATH");
      }
#if defined(__linux__)
      if (!home) {
        home = "/usr/local/cuda";
      }
#endif
      if (home && std::filesystem::exists(home)) {
        args.push_back(fmt::format("--include-path={}/include", home));
      } else {
        throw std::runtime_error(
            "Can not find locations of CUDA headers, please set environment "
            "variable CUDA_HOME or CUDA_PATH.");
      }
    }
    return args;
  }();
  return cached_args;
}

// Get the cache directory for storing compiled results.
const std::filesystem::path& ptx_cache_dir() {
  static std::filesystem::path cache = []() -> std::filesystem::path {
    std::filesystem::path cache;
    if (auto c = std::getenv("MLX_PTX_CACHE_DIR"); c) {
      cache = c;
    } else {
      cache =
          std::filesystem::temp_directory_path() / "mlx" / version() / "ptx";
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

std::filesystem::path get_ptx_path(
    const std::filesystem::path& cache_dir,
    const std::string& module_name) {
#ifdef _WIN32
  constexpr int max_file_name_length = 140;
#else
  constexpr int max_file_name_length = 245;
#endif

  if (module_name.size() <= max_file_name_length) {
    return cache_dir / (module_name + ".ptx");
  }

  auto ptx_path = cache_dir;
  int offset = 0;
  while (module_name.size() - offset > max_file_name_length) {
    ptx_path /= module_name.substr(offset, max_file_name_length);
    offset += max_file_name_length;
  }
  ptx_path /= module_name.substr(offset) + ".ptx";

  return ptx_path;
}

// Try to read the cached |ptx| and |ptx_kernels| from |cache_dir|.
bool read_cached_ptx(
    const std::filesystem::path& cache_dir,
    const std::string& module_name,
    std::string& ptx,
    std::vector<std::pair<std::string, std::string>>& ptx_kernels) {
  if (cache_dir.empty()) {
    return false;
  }

  auto ptx_path = get_ptx_path(cache_dir, module_name);
  std::error_code error;
  auto ptx_size = std::filesystem::file_size(ptx_path, error);
  if (error) {
    return false;
  }
  std::ifstream ptx_file(ptx_path, std::ios::binary);
  if (!ptx_file.good()) {
    return false;
  }
  ptx.resize(ptx_size);
  ptx_file.read(ptx.data(), ptx_size);

  std::ifstream txt_file(ptx_path.replace_extension(".txt"), std::ios::binary);
  std::string line;
  while (std::getline(txt_file, line)) {
    auto tab = line.find('\t');
    if (tab != std::string::npos) {
      ptx_kernels.emplace_back(line.substr(0, tab), line.substr(tab + 1));
    }
  }
  return true;
}

// Write the |ptx| and |ptx_kernels| to |cache_dir| with |name|.
void write_cached_ptx(
    const std::filesystem::path& cache_dir,
    const std::string& module_name,
    const std::string& ptx,
    const std::vector<std::pair<std::string, std::string>>& ptx_kernels,
    const std::string& source_code) {
  if (cache_dir.empty()) {
    return;
  }

  auto ptx_path = get_ptx_path(cache_dir, module_name);

  // Ensure that the directory exists
  auto parent = ptx_path.parent_path();
  if (parent != cache_dir) {
    std::filesystem::create_directories(parent);
  }

  // Write the compiled code and mangled names
  std::ofstream ptx_file(ptx_path, std::ios::binary);
  if (!ptx.empty()) {
    ptx_file.write(&ptx.front(), ptx.size());
  }
  std::ofstream txt_file(ptx_path.replace_extension(".txt"), std::ios::binary);
  for (const auto& [name, mangled] : ptx_kernels) {
    txt_file << name << "\t" << mangled << std::endl;
  }

  // Write the generated code
  std::ofstream source_file(ptx_path.replace_extension(".cu"));
  source_file << source_code;
}

// Return if |device|'s version is not newer than |major|.|minor| version.
inline bool version_lower_equal(Device& device, int major, int minor) {
  if (device.compute_capability_major() < major) {
    return true;
  } else if (device.compute_capability_major() == major) {
    return device.compute_capability_minor() <= minor;
  } else {
    return false;
  }
}

// Return whether NVRTC supports compiling to |device|'s SASS code.
bool compiler_supports_device_sass(Device& device) {
  int nvrtc_major, nvrtc_minor;
  CHECK_NVRTC_ERROR(nvrtcVersion(&nvrtc_major, &nvrtc_minor));
  if (nvrtc_major < 9) {
    return false;
  } else if (nvrtc_major == 9) {
    return version_lower_equal(device, 7, 2);
  } else if (nvrtc_major == 10) {
    return version_lower_equal(device, 7, 5);
  } else if (nvrtc_major == 11 && nvrtc_minor == 0) {
    return version_lower_equal(device, 8, 0);
  } else if (nvrtc_major == 11 && nvrtc_minor < 8) {
    return version_lower_equal(device, 8, 6);
  } else {
    return true;
  }
}

#define INCLUDE_PREFIX "mlx/backend/cuda/device/"

constexpr const char* g_include_names[] = {
    INCLUDE_PREFIX "atomic_ops.cuh",
    INCLUDE_PREFIX "binary_ops.cuh",
    INCLUDE_PREFIX "cast_op.cuh",
    INCLUDE_PREFIX "config.h",
    INCLUDE_PREFIX "complex.cuh",
    INCLUDE_PREFIX "fp16_math.cuh",
    INCLUDE_PREFIX "indexing.cuh",
    INCLUDE_PREFIX "scatter_ops.cuh",
    INCLUDE_PREFIX "unary_ops.cuh",
    INCLUDE_PREFIX "ternary_ops.cuh",
    INCLUDE_PREFIX "utils.cuh",
};

#undef INCLUDE_PREFIX

constexpr const char* g_headers[] = {
    jit_source_atomic_ops,
    jit_source_binary_ops,
    jit_source_cast_op,
    jit_source_config,
    jit_source_complex,
    jit_source_fp16_math,
    jit_source_indexing,
    jit_source_scatter_ops,
    jit_source_unary_ops,
    jit_source_ternary_ops,
    jit_source_utils,
};

void compile(
    Device& device,
    const std::string& module_name,
    const std::string& source,
    const std::vector<std::string>& kernel_names,
    std::string& ptx,
    std::vector<std::pair<std::string, std::string>>& ptx_kernels) {
  // Create the program
  nvrtcProgram prog;
  CHECK_NVRTC_ERROR(nvrtcCreateProgram(
      &prog,
      source.c_str(),
      (module_name + ".cu").c_str(),
      std::size(g_headers),
      g_headers,
      g_include_names));
  std::unique_ptr<nvrtcProgram, void (*)(nvrtcProgram*)> prog_freer(
      &prog,
      [](nvrtcProgram* p) { CHECK_NVRTC_ERROR(nvrtcDestroyProgram(p)); });
  for (const auto& name : kernel_names) {
    CHECK_NVRTC_ERROR(nvrtcAddNameExpression(prog, name.c_str()));
  }

  // Compile program.
  std::vector<const char*> args;
  bool use_sass = compiler_supports_device_sass(device);
  auto cc = device.compute_capability_major();
  std::string arch_tag = (cc == 90 || cc == 100 || cc == 121) ? "a" : "";
  std::string compute = fmt::format(
      "--gpu-architecture={}_{}{}{}",
      use_sass ? "sm" : "compute",
      cc,
      device.compute_capability_minor(),
      arch_tag);
  args.push_back(compute.c_str());
  for (const auto& include : include_path_args()) {
    args.push_back(include.c_str());
  }
  nvrtcResult compile_result =
      nvrtcCompileProgram(prog, args.size(), args.data());
  if (compile_result != NVRTC_SUCCESS) {
    size_t log_size;
    CHECK_NVRTC_ERROR(nvrtcGetProgramLogSize(prog, &log_size));
    std::vector<char> log(log_size + 1, 0);
    CHECK_NVRTC_ERROR(nvrtcGetProgramLog(prog, log.data()));
    throw std::runtime_error(
        fmt::format("Failed to compile kernel: {}.", log.data()));
  }

  // Get mangled names of kernel names.
  for (const auto& name : kernel_names) {
    const char* mangled;
    CHECK_NVRTC_ERROR(nvrtcGetLoweredName(prog, name.c_str(), &mangled));
    ptx_kernels.emplace_back(name, mangled);
  }

  // Get ptx data.
  size_t ptx_size;
  if (use_sass) {
    CHECK_NVRTC_ERROR(nvrtcGetCUBINSize(prog, &ptx_size));
  } else {
    CHECK_NVRTC_ERROR(nvrtcGetPTXSize(prog, &ptx_size));
  }
  ptx.resize(ptx_size);
  if (use_sass) {
    CHECK_NVRTC_ERROR(nvrtcGetCUBIN(prog, ptx.data()));
  } else {
    CHECK_NVRTC_ERROR(nvrtcGetPTX(prog, ptx.data()));
  }
}

void load_module(
    const std::string& module_name,
    const std::string& ptx,
    const std::vector<std::pair<std::string, std::string>>& ptx_kernels,
    CUmodule& module_,
    std::unordered_map<std::string, std::tuple<CUfunction, bool, uint>>&
        kernels) {
  // Load module.
  char jit_log[4089] = {};
  CUjit_option options[] = {
      CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  void* values[] = {jit_log, reinterpret_cast<void*>(std::size(jit_log) - 1)};
  CUresult jit_result = cuModuleLoadDataEx(
      &module_, ptx.data(), std::size(options), options, values);
  if (jit_result != CUDA_SUCCESS) {
    throw std::runtime_error(fmt::format(
        "Failed to load compiled {} kernel: {}.", module_name, jit_log));
  }

  // Load kernels.
  for (const auto& [name, mangled] : ptx_kernels) {
    CUfunction kernel;
    CHECK_CUDA_ERROR(cuModuleGetFunction(&kernel, module_, mangled.c_str()));
    kernels[name] = std::make_tuple(kernel, false, 0);
  }
}

} // namespace

JitModule::JitModule(
    Device& device,
    const std::string& module_name,
    const KernelBuilder& builder,
    bool use_disk_cache) {
  // Will hold the actual device executable source code and kernel names
  std::string ptx;
  std::vector<std::pair<std::string, std::string>> ptx_kernels;

  // Try to load them from the file cache
  if (!read_cached_ptx(ptx_cache_dir(), module_name, ptx, ptx_kernels)) {
    auto [precompiled, source_code, kernel_names] = builder();

    // Get the PTX or cubin
    if (precompiled) {
      ptx = std::move(source_code);
      for (auto& name : kernel_names) {
        ptx_kernels.emplace_back(name, name);
      }
    } else {
      compile(device, module_name, source_code, kernel_names, ptx, ptx_kernels);
    }

    // If requested save them in the file cache for the next launch
    if (use_disk_cache) {
      write_cached_ptx(
          ptx_cache_dir(), module_name, ptx, ptx_kernels, source_code);
    }
  }

  // Load the module
  load_module(module_name, ptx, ptx_kernels, module_, kernels_);
}

JitModule::~JitModule() {
  CHECK_CUDA_ERROR(cuModuleUnload(module_));
}

std::pair<CUfunction, uint> JitModule::get_kernel_and_dims(
    const std::string& kernel_name,
    std::function<void(CUfunction)> configure_kernel) {
  auto it = kernels_.find(kernel_name);
  if (it == kernels_.end()) {
    throw std::runtime_error(
        fmt::format("There is no kernel named {}.", kernel_name));
  }

  // If it is the first time we run this kernel then configure it. Do it only
  // once!
  auto kernel = std::get<0>(it->second);
  if (!std::get<1>(it->second)) {
    if (configure_kernel) {
      configure_kernel(kernel);
    }
    std::get<1>(it->second) = true;
    std::get<2>(it->second) = max_occupancy_block_dim(kernel);
  }

  return {kernel, std::get<2>(it->second)};
}

CUfunction JitModule::get_kernel(
    const std::string& kernel_name,
    std::function<void(CUfunction)> configure_kernel) {
  return get_kernel_and_dims(kernel_name, std::move(configure_kernel)).first;
}

std::unordered_map<std::string, JitModule>& get_jit_module_cache() {
  static std::unordered_map<std::string, JitModule> map;
  return map;
}

JitModule& get_jit_module(
    const mlx::core::Device& device,
    const std::string& name,
    const KernelBuilder& builder,
    bool cache) {
  auto& map = get_jit_module_cache();
  auto it = map.find(name);
  if (it == map.end()) {
    it = map.try_emplace(name, cu::device(device), name, builder, cache).first;
  }
  return it->second;
}

} // namespace mlx::core::cu
