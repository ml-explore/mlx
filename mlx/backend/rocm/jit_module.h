// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx::core::rocm {

// JIT compilation module for ROCm
class JitModule {
 public:
  JitModule(
      const std::string& kernel_name,
      const std::string& kernel_source,
      const std::vector<std::string>& template_args = {},
      const std::vector<std::string>& compiler_flags = {},
      bool verbose = false);

  ~JitModule();

  JitModule(const JitModule&) = delete;
  JitModule& operator=(const JitModule&) = delete;

  // Get the compiled kernel function
  hipFunction_t get_kernel() const {
    return kernel_;
  }

  // Launch the kernel with given arguments
  template <typename... Args>
  void launch(
      dim3 grid_dims,
      dim3 block_dims,
      size_t shared_memory,
      hipStream_t stream,
      Args&&... args) {
    void* kernel_args[] = {(void*)&args...};
    CHECK_HIP_ERROR(hipModuleLaunchKernel(
        kernel_,
        grid_dims.x,
        grid_dims.y,
        grid_dims.z,
        block_dims.x,
        block_dims.y,
        block_dims.z,
        shared_memory,
        stream,
        kernel_args,
        nullptr));
  }

 private:
  void compile(
      const std::string& kernel_name,
      const std::string& kernel_source,
      const std::vector<std::string>& template_args,
      const std::vector<std::string>& compiler_flags,
      bool verbose);

  hiprtcProgram program_{nullptr};
  hipModule_t module_{nullptr};
  hipFunction_t kernel_{nullptr};
};

// JIT cache for compiled modules
class JitCache {
 public:
  static JitCache& instance();

  std::shared_ptr<JitModule> get_or_create(
      const std::string& kernel_name,
      const std::string& kernel_source,
      const std::vector<std::string>& template_args = {},
      const std::vector<std::string>& compiler_flags = {});

 private:
  std::unordered_map<std::string, std::weak_ptr<JitModule>> cache_;
  std::mutex mutex_;

  std::string make_key(
      const std::string& kernel_name,
      const std::string& kernel_source,
      const std::vector<std::string>& template_args,
      const std::vector<std::string>& compiler_flags) const;
};

// Helper function to create and cache JIT modules
std::shared_ptr<JitModule> make_jit_kernel(
    const std::string& kernel_name,
    const std::string& kernel_source,
    const std::vector<std::string>& template_args = {},
    const std::vector<std::string>& compiler_flags = {});

} // namespace mlx::core::rocm