// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/common/utils.h"
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

class JitModule {
 public:
  JitModule(
      Device& device,
      const std::string& module_name,
      const KernelBuilder& builder);
  ~JitModule();

  JitModule(const JitModule&) = delete;
  JitModule& operator=(const JitModule&) = delete;

  void append_arg(const array& a) {
    append_arg(reinterpret_cast<CUdeviceptr>(a.data<void>()));
  }

  template <typename T>
  void append_arg(T val) {
    storage_.emplace_back(val);
    append_ptr_arg(&storage_.back());
  }

  template <typename T>
  void append_arg(std::vector<T> vec) {
    if (vec.empty()) {
      // The nullptr can not be used as arg, pass something not null.
      append_arg(std::monostate{});
    } else {
      append_ptr_arg(vec.data());
      storage_.emplace_back(std::move(vec));
    }
  }

  // Make sure the arg is copied to an array with size of NDIM.
  template <size_t NDIM = MAX_NDIM, typename T>
  void append_ndim_arg(const std::vector<T>& vec) {
    if (vec.size() > NDIM) {
      throw std::runtime_error(
          fmt::format("ndim can not be larger than {}.", NDIM));
    }
    std::vector<T> copied(NDIM);
    std::copy(vec.begin(), vec.end(), copied.data());
    append_arg(std::move(copied));
  }

  // Launch kernel with |kernel_name| that each thread works on
  // |work_per_thread| elements of |arr|.
  void launch_kernel(
      CUstream stream,
      const std::string& kernel_name,
      const array& arr,
      bool large,
      int work_per_thread = 1);

  void launch_kernel(
      CUstream stream,
      CUfunction kernel,
      Dims num_blocks,
      Dims block_dims);

  CUfunction get_kernel(const std::string& kernel_name);

 private:
  void append_ptr_arg(const void* v);

  CUmodule module_{nullptr};
  std::unordered_map<std::string, CUfunction> kernels_;
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

JitModule& get_jit_module(
    const mlx::core::Device& device,
    const std::string& name,
    const KernelBuilder& builder);

} // namespace mlx::core::cu
