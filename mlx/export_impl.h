// Copyright © 2024 Apple Inc.

#include "mlx/io/load.h"
#include "mlx/mlx_export.h"

#pragma once

namespace mlx::core {

struct FunctionTable;

struct MLX_API FunctionExporter {
  void operator()(const std::initializer_list<array>& args) {
    this->operator()(Args(args));
  }
  void operator()(const Args& args);
  void operator()(const Kwargs& kwargs);
  void operator()(const Args& args, const Kwargs& kwargs);

  void close();

  FunctionExporter(const FunctionExporter&) = delete;
  FunctionExporter& operator=(const FunctionExporter&) = delete;
  FunctionExporter(FunctionExporter&& other) = default;

 private:
  friend MLX_API FunctionExporter exporter(
      const std::string&,
      const std::function<std::vector<array>(const Args&)>&,
      bool shapeless);

  friend MLX_API FunctionExporter exporter(
      const std::string&,
      const std::function<std::vector<array>(const Kwargs&)>&,
      bool shapeless);

  friend MLX_API FunctionExporter exporter(
      const std::string&,
      const std::function<std::vector<array>(const Args&, const Kwargs&)>&,
      bool shapeless);

  friend MLX_API FunctionExporter exporter(
      const ExportCallback&,
      const std::function<std::vector<array>(const Args&)>&,
      bool shapeless);

  friend MLX_API FunctionExporter exporter(
      const ExportCallback&,
      const std::function<std::vector<array>(const Kwargs&)>&,
      bool shapeless);

  friend MLX_API FunctionExporter exporter(
      const ExportCallback&,
      const std::function<std::vector<array>(const Args&, const Kwargs&)>&,
      bool shapeless);

  FunctionExporter(
      const std::string& file,
      std::function<std::vector<array>(const Args&, const Kwargs&)> fun,
      bool shapeless);

  FunctionExporter(
      const ExportCallback& callback,
      std::function<std::vector<array>(const Args&, const Kwargs&)> fun,
      bool shapeless);

  io::FileWriter os;
  ExportCallback callback;
  std::function<std::vector<array>(const Args&, const Kwargs& kwargs)> fun;
  void export_function(const Args& args, const Kwargs& kwargs);
  void export_with_callback(
      const std::vector<array>& inputs,
      const std::vector<array>& outputs,
      const std::vector<array>& tape,
      const std::vector<std::string>& kwarg_keys);
  std::unordered_map<std::uintptr_t, array> constants;
  int count{0};
  bool closed{false};
  std::shared_ptr<FunctionTable> ftable;
};

struct MLX_API ImportedFunction {
  std::vector<array> operator()(
      const std::initializer_list<array>& args) const {
    return this->operator()(Args(args));
  }
  std::vector<array> operator()(const Args& args) const;
  std::vector<array> operator()(const Kwargs& kwargs) const;
  std::vector<array> operator()(const Args& args, const Kwargs& kwargs) const;

 private:
  ImportedFunction(const std::string& file);
  friend MLX_API ImportedFunction import_function(const std::string&);
  ImportedFunction();

  std::shared_ptr<FunctionTable> ftable;
};

} // namespace mlx::core
