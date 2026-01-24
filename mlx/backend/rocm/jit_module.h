// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/rocm/device.h"

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <deque>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <variant>

#include <fmt/format.h>

namespace mlx::core::rocm {

class Device;

// Maximum number of dimensions supported
constexpr int MAX_NDIM = 8;

using KernelBuilderResult = std::tuple<
    /* precompiled */ bool,
    /* source code */ std::string,
    /* kernel names */ std::vector<std::string>>;
using KernelBuilder = std::function<KernelBuilderResult()>;

struct KernelArgs {
  void** args() {
    return args_.data();
  }

  void append(const array& a) {
    append(reinterpret_cast<hipDeviceptr_t>(a.data<void>()));
  }

  template <typename T>
  void append(T val) {
    storage_.emplace_back(val);
    append_ptr(&storage_.back());
  }

  template <typename T>
  void append(SmallVector<T> vec) {
    storage_.emplace_back(std::move(vec));
    append_ptr(std::get<SmallVector<T>>(storage_.back()).data());
  }

  template <typename T>
  void append(const std::vector<T>& vec) {
    append(SmallVector<T>(vec.begin(), vec.end()));
  }

  // Make sure the arg is copied to an array with size of NDIM.
  template <size_t NDIM = MAX_NDIM, typename T>
  void append_ndim(SmallVector<T> vec) {
    if (vec.size() > NDIM) {
      throw std::runtime_error(
          fmt::format("ndim can not be larger than {}.", NDIM));
    }
    vec.resize(NDIM);
    append(std::move(vec));
  }

  void append_ptr(const void* v) {
    args_.push_back(const_cast<void*>(v));
  }

 private:
  std::vector<void*> args_;

  // The hipGraphAddKernelNode API requires passing pointers to arguments so
  // store temporary values until the node is created.
  using Arg = std::variant<
      std::monostate,
      hipDeviceptr_t,
      bool,
      int32_t,
      uint32_t,
      int64_t,
      float,
      SmallVector<const void*>,
      SmallVector<int32_t>,
      SmallVector<int64_t>>;
  std::deque<Arg> storage_;
};

class JitModule {
 public:
  JitModule(
      Device& device,
      const std::string& module_name,
      const KernelBuilder& builder,
      bool cache);
  ~JitModule();

  JitModule(const JitModule&) = delete;
  JitModule& operator=(const JitModule&) = delete;
  
  hipFunction_t get_kernel(
      const std::string& kernel_name,
      std::function<void(hipFunction_t)> configure_kernel = nullptr);

 private:
  hipModule_t module_{nullptr};
  std::unordered_map<std::string, std::pair<hipFunction_t, bool>> kernels_;
};

std::unordered_map<std::string, JitModule>& get_jit_module_cache();

JitModule& get_jit_module(
    const mlx::core::Device& device,
    const std::string& name,
    const KernelBuilder& builder,
    bool use_disk_cache = true);

} // namespace mlx::core::rocm
