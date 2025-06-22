// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/config.h"

#include <deque>
#include <unordered_map>
#include <utility>
#include <variant>

#include <cuda.h>
#include <fmt/format.h>

namespace mlx::core::cu {

class Device;

using KernelBuilderResult = std::pair<
    /* source code */ std::string,
    /* kernel names */ std::vector<std::string>>;
using KernelBuilder = std::function<KernelBuilderResult()>;

struct KernelArgs {
  void** args() {
    return args_.data();
  }

  void append(const array& a) {
    append(reinterpret_cast<CUdeviceptr>(a.data<void>()));
  }

  template <typename T>
  void append(T val) {
    storage_.emplace_back(val);
    append_ptr(&storage_.back());
  }

  template <typename T>
  void append(std::vector<T> vec) {
    if (vec.empty()) {
      // The nullptr can not be used as arg, pass something not null.
      append(std::monostate{});
    } else {
      append_ptr(vec.data());
      storage_.emplace_back(std::move(vec));
    }
  }

  // Make sure the arg is copied to an array with size of NDIM.
  template <size_t NDIM = MAX_NDIM, typename T>
  void append_ndim(std::vector<T> vec) {
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

  // The cuLaunchKernel API requires passing pointers to arguments so store
  // temporary values untill kernel is launched.
  using Arg = std::variant<
      std::monostate,
      CUdeviceptr,
      int32_t,
      uint32_t,
      int64_t,
      std::vector<const void*>,
      std::vector<int32_t>,
      std::vector<int64_t>>;
  std::deque<Arg> storage_;
};

class JitModule {
 public:
  JitModule(
      Device& device,
      const std::string& module_name,
      const KernelBuilder& builder);
  ~JitModule();

  JitModule(const JitModule&) = delete;
  JitModule& operator=(const JitModule&) = delete;
  CUfunction get_kernel(const std::string& kernel_name);

 private:
  CUmodule module_{nullptr};
  std::unordered_map<std::string, CUfunction> kernels_;
};

JitModule& get_jit_module(
    const mlx::core::Device& device,
    const std::string& name,
    const KernelBuilder& builder);

} // namespace mlx::core::cu
